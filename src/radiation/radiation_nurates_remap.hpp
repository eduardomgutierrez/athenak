#ifndef RADIATION_RADIATION_NURATES_REMAP_HPP_
#define RADIATION_RADIATION_NURATES_REMAP_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_nurates_remap.hpp
//! \brief Moving a per-bin quantity between a ray's comoving frequency bins and the fixed
//!        comoving grid the bns_nurates opacities are tabulated on.
//!
//! ## The geometry
//!
//! i0_ is stored in *lab* frequency bins; the opacities live on freq_grid in *comoving*
//! MeV.  Lab bin ifr covers the comoving interval n0_cm(iang)*[e_lo, e_hi], a different
//! interval on every ray.  On the neutrino grid freq_grid is geometric over the whole of
//! [nu_min, nu_max] -- every bin, with the top closed by FreqBinEdgesMeV -- so in
//! y = ln(nu) the bins are uniform of width h = dlnnu.  n0_cm does not depend on
//! frequency, so ray a's whole spectrum is the fixed grid translated rigidly by
//!
//!     delta_a = ln(n0_cm(a)) / h        bins -- one number per ray, not one per bin.
//!
//! Everything here is that translation, evaluated at a real-valued bin index.
//!
//! ## Why log space
//!
//! Bins uniform in ln(nu) make ln(q) linear in the bin index for any power law in energy
//! -- for bin-integrated quantities too, which is why the stretching of the bin width
//! needs no separate Jacobian.  So interpolating ln(q) against the *index* is exact
//! wherever q is a power law, and its error is relative to the local value rather than to
//! the neighbouring bin's.  That distinction is the whole difficulty of neutrino spectra,
//! whose neighbours differ by decades.
//!
//! ## What is deliberately *not* here, and why
//!
//! The photon path (radiation_multi_freq.cpp) rebins i0_ onto the comoving grid, solves,
//! and rebins back.  That structure cannot be carried over: the round trip is not the
//! identity, and applying it every RK stage accumulates.  Measured, v = 0.5, nfreq = 24,
//! a Fermi-Dirac at 12 MeV on nu in [10, 2000] MeV, drift on the bins carrying more than
//! 1e-12 of the peak:
//!
//!   operator                                    1 trip     100 trips   1000 trips
//!   conservative exponential rebin              5.6e-01    diverges    diverges
//!   log-space cubic + moment restoration        3.5e-03    5.1e-01     3.4e+01
//!   log-space cubic, grid with spectral margin  6.7e-05    5.2e-03     3.3e-02
//!
//! The first is a genuinely conservative rebin -- phi = dE/dy reconstructed as an
//! exponential per bin, overlaps integrated exactly, both moments restored afterwards.
//! It is exact for power laws, but only first order in the curvature of ln(E), and a
//! Fermi-Dirac tail has O(10) curvature per bin, so it inflates the tail by a factor of a
//! few per bin and that compounds.  A production step is two stages and a run is O(1e4)
//! steps, so even the best row is not usable.
//!
//! So MultiFreqRadFluidCouplingNurates transforms the *update*, not the field.  It builds
//! the scattering target on the fixed grid, where every ray shares the bin and the
//! angular mean means something, and then evaluates the opacities, the equilibrium
//! spectrum and that target at each ray's own comoving energy to update the stored lab
//! bins in place.  Two consequences:
//!
//!   * with no source the field is untouched bit-for-bit, so nothing accumulates;
//!   * no rebinned spectrum is ever written back, so the energy and the lepton number
//!     exchanged with the fluid are both differences of the stored field itself.  They
//!     are exactly conservative and exactly consistent with each other, with no remap
//!     anywhere in the conservation path.
//!
//! Templated on the array type so the kernel (team scratch) and the unit test (host
//! mirrors) run the same code; see pgen/unit_tests/rad_freq_remap.cpp.

#include "athena.hpp"

namespace radiation {

//! Stored in the log arrays where the value is not positive, and the threshold that
//! detects it.  Both ends of the gap between them are wide: Kokkos::log bottoms out at
//! -745 on the smallest subnormal, so no real logarithm approaches the threshold from
//! above, and no Lagrange weight -- |w| reaches 26 off the grid -- can carry the sentinel
//! back over it from below.
constexpr Real kRemapLogSentinel = -1.0e30;
constexpr Real kRemapLogLive = 0.1*kRemapLogSentinel;

//! How far outside the two nodes it interpolates between the cubic may land, in e-folds.
//! One is comfortably above what real curvature asks for where a spectrum turns over
//! (measured: 0.06 at the median, 0.57 at the worst), so the bound does not cut into the
//! interpolation.  It must stay a *constant*; a bound scaled to the window is no bound at
//! all, for the reason given in RemapBinValue.
constexpr Real kRemapOvershoot = 1.0;

//! Per-cell workspace, as a count of bin-sized and ray-sized vectors.  **Two** kernels
//! carve up the same array -- MultiFreqRadFluidCouplingNurates and the occupancy pass of
//! CalcOpacityNurates_ -- so the allocation has to cover whichever needs more, and these
//! counts have to track the MakeWorkSlice sequence in both.  Each kernel asserts its own
//! total against the allocation once it has finished slicing, so a count that drifts
//! fails loudly in a debug build instead of writing into the next cell's column.
#define NURATES_MF_WORK_BIN_VECS 13
#define NURATES_MF_WORK_RAY_VECS 3
#define NURATES_OPAC_WORK_BIN_VECS 3
#define NURATES_OPAC_WORK_RAY_VECS 3
#define NURATES_WORK_VEC_MAX(a, b) ((a) > (b) ? (a) : (b))
#define NURATES_MF_WORK_SIZE(nfreq, nang) \
  (NURATES_WORK_VEC_MAX(NURATES_MF_WORK_BIN_VECS, \
                        NURATES_OPAC_WORK_BIN_VECS)*(nfreq) + \
   NURATES_WORK_VEC_MAX(NURATES_MF_WORK_RAY_VECS, \
                        NURATES_OPAC_WORK_RAY_VECS)*(nang))

//----------------------------------------------------------------------------------------
//! \struct WorkSlice
//! \brief One cell's column of a global [nmb, nwork, nk, nj, ni] workspace, presented as
//! a
//!        1-D array.
//!
//! The multi-frequency source term is dispatched with a flat par_for, so it cannot use
//! team
//! scratch and its per-cell vectors live in a preallocated global array instead.  That
//! array
//! is laid out like i0_ -- component-major with i fastest -- so a cell's successive
//! entries
//! are separated by a fixed stride rather than being contiguous.  Wrapping the stride
//! here
//! lets RemapFillLogs and RemapBinValue template over these exactly as they do over team
//! scratch or the unit test's host mirrors, with no second code path.
struct WorkSlice {
  Real *p;
  int stride;
  KOKKOS_INLINE_FUNCTION Real &operator()(const int f) const { return p[f*stride]; }
};

//----------------------------------------------------------------------------------------
//! \fn WorkSlice MakeWorkSlice
//! \brief Carve the next n entries off a workspace column, advancing the offset.  Callers
//!        declare their vectors in sequence and the offsets take care of themselves; the
//!        total must match the nwork the array was allocated with.
KOKKOS_INLINE_FUNCTION
WorkSlice MakeWorkSlice(Real *base, const int stride, int &off, const int n) {
  WorkSlice w{base + static_cast<size_t>(off)*stride, stride};
  off += n;
  return w;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapFillLogs
//! \brief Fill lq with log(q) over [flo, fhi] at offset off, using the sentinel where q
//!        is not positive.  Kept next to RemapBinValue so the two cannot disagree about
//!        the convention.
template <class QT, class LT>
KOKKOS_INLINE_FUNCTION
void RemapFillLogs(const QT &q, LT &lq, const int off, const int flo, const int fhi) {
  for (int f = flo; f <= fhi; ++f) {
    lq(off+f) = (q(off+f) > 0.0) ? Kokkos::log(q(off+f)) : kRemapLogSentinel;
  }
}

//----------------------------------------------------------------------------------------
//! \enum RemapAsymptote
//! \brief Which physical form a quantity takes outside the tabulated grid.
//!
//! The shift asks for values at f +/- delta_a, so bins within |delta_a| of either end
//! read
//! energies the grid does not tabulate.  What to put there is a boundary condition on the
//! frequency domain, and it is not the same condition for every quantity:
//!
//!   kOpacity   sigma_a, sigma_s.  Power laws in energy at both ends -- isoenergetic
//!              scattering is exactly 4 pi E^2 (1 - g) R_iso/c, and the two exponentials
//!              in the beta rates cancel identically against kappa = eta exp((E-mu)/T),
//!              leaving E^2 at high energy with no cutoff at all.  Continued at the slope
//!              measured from the two edge nodes, which is exact for any E^p.
//!
//!   kSpectrum  bin-integrated intensities: the stored spectrum, the equilibrium spectrum
//!              eta/sigma_a, and the scattering target.  The *intensity* goes as nu^3 at
//!              low energy -- I_nu = nu^3 f with f a bounded occupation -- but what this
//!              operator carries is the bin integral, and dE = I_nu dnu with dnu
//!              proportional to nu on a log grid, so E ~ nu^4.  The exponent here is the
//!              bin integral's, one higher than the intensity's.  At the top the
//!              occupation is Boltzmann and the spectrum cuts off exponentially.
//!
//! Applying one rule to both would be wrong by exp(E/T) on the opacities.
enum class RemapAsymptote { kOpacity, kSpectrum };

//----------------------------------------------------------------------------------------
//! \struct RemapStencil
//! \brief A four-point window on the grid and the Lagrange weights that go with it.
//!
//! Split out from the lookup so that lookups sharing a fractional position pay for the
//! weights once.  That is every lookup the solver makes: each is one ray walked across
//! the grid at a fixed Doppler displacement, and with xpos = ifr +- dlt and ifr an
//! integer, floor(xpos) = ifr + floor(+-dlt), so the unclamped origin advances exactly
//! one bin per ifr while t = frac(+-dlt) + 1 does not move at all.  Only a window
//! clamped against an end of the grid breaks that, which is the outermost |dlt| + 3
//! bins or so.  RemapMakeRay and RemapRayStencil build and walk such a ray;
//! RemapStencilAt serves a lookup that stands alone, and several quantities read at one
//! energy can share one window between them.
struct RemapStencil {
  int st;
  Real t, w0, w1, w2, w3;
};

//----------------------------------------------------------------------------------------
//! \fn void RemapSetWeights
//! \brief Cubic Lagrange weights for offset t from the window's first node.
KOKKOS_INLINE_FUNCTION
void RemapSetWeights(RemapStencil &s) {
  const Real t = s.t;
  s.w0 = -(t - 1.0)*(t - 2.0)*(t - 3.0)/6.0;
  s.w1 = t*(t - 2.0)*(t - 3.0)/2.0;
  s.w2 = -t*(t - 1.0)*(t - 3.0)/2.0;
  s.w3 = t*(t - 1.0)*(t - 2.0)/6.0;
}

//----------------------------------------------------------------------------------------
//! \fn RemapStencil RemapStencilAt
//! \brief The window for one real-valued bin index, slid to stay inside [flo, fhi].
//!
//! Requires fhi - flo >= 3, which the <radiation> constructor enforces as nfreq >= 4.
KOKKOS_INLINE_FUNCTION
RemapStencil RemapStencilAt(const int flo, const int fhi, const Real xpos) {
  RemapStencil s;
  int st = static_cast<int>(Kokkos::floor(xpos)) - 1;
  if (st < flo) { st = flo; }
  if (st > fhi - 3) { st = fhi - 3; }
  s.st = st;
  s.t = xpos - static_cast<Real>(st);
  RemapSetWeights(s);
  return s;
}

//----------------------------------------------------------------------------------------
//! \struct RemapRay
//! \brief The window one ray reuses across the whole grid, built once per ray.
//!
//! `in` carries the interior weights; `st0` is the unclamped origin at bin 0, so bin ifr
//! sits at st0 + ifr.  t here is the exact fractional part of the displacement rather
//! than xpos - st recovered from a rounded xpos, which is the same number to within an
//! ulp and the more accurate of the two.
struct RemapRay {
  RemapStencil in;
  int st0;
};

//----------------------------------------------------------------------------------------
//! \fn RemapRay RemapMakeRay
//! \brief Build a ray's window from its displacement, x0 being xpos at bin 0.
KOKKOS_INLINE_FUNCTION
RemapRay RemapMakeRay(const Real x0) {
  RemapRay r;
  r.st0 = static_cast<int>(Kokkos::floor(x0)) - 1;
  r.in.st = r.st0;
  r.in.t = x0 - static_cast<Real>(r.st0);
  RemapSetWeights(r.in);
  return r;
}

//----------------------------------------------------------------------------------------
//! \fn RemapStencil RemapRayStencil
//! \brief The window for bin ifr of a ray: the ray's own wherever it clears both ends of
//!        the grid, rebuilt where the clamp holds it still.
KOKKOS_INLINE_FUNCTION
RemapStencil RemapRayStencil(const RemapRay &r, const int flo, const int fhi,
                             const int ifr, const Real xpos) {
  const int st = r.st0 + ifr;
  if (st >= flo && st <= fhi - 3) {
    RemapStencil s = r.in;
    s.st = st;
    return s;
  }
  return RemapStencilAt(flo, fhi, xpos);
}

//----------------------------------------------------------------------------------------
//! \fn Real RemapBinValue
//! \brief Per-bin quantity at a real-valued bin index, by Lagrange on ln(q).
//!
//! Exact for a power law in energy, for point-sampled and bin-integrated quantities
//! alike: the grid is uniform in ln(nu), so the bin width stretches with the bin and the
//! integrated form carries its own extra power -- the caller applies no Jacobian.  lq
//! holds log(q), or kRemapLogSentinel where q is not positive.  dln is the bin width in
//! ln(nu), since the asymptotic slopes are per unit ln(nu) and this works in bin index.
//!
//! lo and hi are the live nodes bracketing t, lo2 and hi2 the next ones out.  The cubic
//! joins in only where all four are live, so a window straddling the edge of the live
//! spectrum degrades cubic -> log-linear -> asymptote -> zero without leaving log space.
//! Outside the grid the bare cubic is useless: |w| reaches 26 a couple of bins out.
//!
//! Three invariants that are easy to "simplify" away:
//!
//!   * **Between two live nodes, bound against those two nodes, never the window.**  w0
//!     and w3 are negative for the 1 < t < 2 a walked ray lands in, so a node far below
//!     its neighbours enters the sum as a large *positive* contribution, and a bound
//!     scaled to the window's range would grow with the very excursion it must catch.
//!     kRemapOvershoot is a constant for the same reason.
//!   * **A bin-integrated intensity is capped below by nu^4.**  I_nu = nu^3 f with f a
//!     bounded occupation, so the intensity goes as nu^3; the log grid's bin width is
//!     proportional to nu, which puts the integrated form one power higher.  The cap
//!     forces decay rather than merely forbidding growth, which is what is needed where
//!     the whole grid sits on a falling tail and the measured edge slope points the
//!     wrong way.
//!   * **Opacities are not capped at all**: they keep growing as E^2 with no cutoff, so
//!     their measured edge slope is continued as it stands.  RemapAsymptote is where
//!     that choice is made.
//!
//! Above the grid a spectrum gets the same treatment minus the floor, which has no
//! counterpart there: the cubic where it decays faster, the chord as the power-law-exact
//! bound, and no growth past the edge node.  A Boltzmann cutoff was tried there and
//! rejected on measurement; that and the rest of the edge study are in
//! notes/multifreq-comoving-solve.md section 6.4.
template <class LT>
KOKKOS_INLINE_FUNCTION
Real RemapBinValue(const LT &lq, const int off, const RemapStencil &s,
                   const RemapAsymptote asym, const Real dln) {
  const Real t = s.t;
  const int a = off + s.st;
  int lo = -1, lo2 = -1, hi = -1, hi2 = -1, nlive = 0;
  for (int k=0; k<=3; ++k) {
    if (!(lq(a+k) > kRemapLogLive)) { continue; }
    ++nlive;
    if (static_cast<Real>(k) <= t) { lo2 = lo; lo = k; }
    if (static_cast<Real>(k) >= t) {
      if (hi < 0) { hi = k; } else if (hi2 < 0) { hi2 = k; }
    }
  }
  const bool full = (nlive == 4);
  Real lval = 0.0;
  if (full) {
    lval = s.w0*lq(a) + s.w1*lq(a+1) + s.w2*lq(a+2) + s.w3*lq(a+3);
  }
  if (lo >= 0 && hi >= 0) {
    // Between two live nodes.  The cubic leaves their interval only by the curvature it
    // carries -- kRemapOvershoot e-folds, measured.  The bound has to be those two nodes
    // rather than the window, because w0 and w3 are NEGATIVE for the 1 < t < 2 that a ray
    // window always lands in: a node far BELOW its neighbours enters lval as a large
    // POSITIVE contribution, so a window-scaled bound grows with exactly the excursion it
    // is meant to catch.  A neutrino spectrum spanning 69 e-folds across the grid, with
    // tail bins underflowing and a charged-current threshold putting a cliff in the
    // middle, supplies such nodes as a matter of course.
    const Real llo = lq(a+lo), lhi = lq(a+hi);
    if (!full) {
      return (hi > lo) ? Kokkos::exp(llo + (t - static_cast<Real>(lo))*(lhi - llo)
                                           /static_cast<Real>(hi - lo))
                       : Kokkos::exp(llo);
    }
    return Kokkos::exp(
        Kokkos::fmin(Kokkos::fmax(lval, Kokkos::fmin(llo, lhi) - kRemapOvershoot),
                     Kokkos::fmax(llo, lhi) + kRemapOvershoot));
  }
  if (hi >= 0) {
    // Below every live node.  kOpacity continues the measured edge slope: exact for E^p,
    // and where sigma_a has a threshold that slope is steep, so it continues toward zero.
    const Real lref = lq(a+hi);
    const Real slope = (hi2 >= 0) ? (lq(a+hi2) - lref)/static_cast<Real>(hi2 - hi) : 0.0;
    const Real chord = lref + (t - static_cast<Real>(hi))*slope;
    if (asym == RemapAsymptote::kOpacity) { return Kokkos::exp(chord); }
    // The nu^4 line is a cap, not a replacement: a spectrum with a sharper low-energy
    // cutoff than a saturating occupation falls faster than nu^4 well before reaching the
    // limit, and that curvature is real information the cubic carries.  Take whichever
    // decays fastest.
    Real lval_out = Kokkos::fmin(chord, lref + (t - static_cast<Real>(hi))*(4.0*dln));
    return Kokkos::exp(full ? Kokkos::fmin(lval, lval_out) : lval_out);
  }
  if (lo >= 0) {
    // Above every live node.
    const Real lref = lq(a+lo);
    const Real slope = (lo2 >= 0) ? (lref - lq(a+lo2))/static_cast<Real>(lo - lo2) : 0.0;
    const Real chord = lref + (t - static_cast<Real>(lo))*slope;
    if (asym == RemapAsymptote::kOpacity) { return Kokkos::exp(chord); }
    Real lval_out = Kokkos::fmin(chord, lref);
    return Kokkos::exp(full ? Kokkos::fmin(lval, lval_out) : lval_out);
  }
  return 0.0;
}

}  // namespace radiation

#endif  // RADIATION_RADIATION_NURATES_REMAP_HPP_
