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
//! Measured over 2.2e5 evaluations of the single-zone relaxation: the interpolation
//! leaves that interval by 0.058 e-folds at the median and 0.565 at the worst, all of it
//! real curvature where the spectrum turns over, so this keeps the interpolation intact.
//! It must stay a constant.  A bound scaled to the window instead is no bound at all on
//! this data -- see RemapBinValue.
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
//!              eta/sigma_a, and the scattering target.  I_nu = nu^3 f with f a bounded
//!              occupation, so I_nu ~ nu^3 as nu -> 0 and the *bin* integral goes as nu^4
//!              on a log grid; at the top the occupation is Boltzmann and the spectrum
//!              cuts off exponentially.  Note the extra power: eta is integrated over the
//!              bin while kappa is sampled at its midpoint.
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
//! \brief Per-bin quantity on a prepared four-point window, by Lagrange on ln(q).
//!
//! Exact for a power law in energy, for point-sampled quantities (sigma_a, sigma_s) and
//! bin-integrated ones (the equilibrium spectrum, the mean intensity) alike -- the latter
//! pick up the n0_cm^(p+1) stretch of the bin width on their own, so the caller applies
//! no Jacobian.  lq holds log(q), or the sentinel where q is not positive; the raw values
//! are never needed here, which is what keeps the two spaces from disagreeing.
//!
//! The window is slid to stay inside [flo, fhi] rather than dropping to two-point near
//! the ends: a centred window with a linear fallback is wrong by -18% on the peak bin and
//! a factor two above it, because the equilibrium intensity turns over two bins below the
//! top of the grid.  RemapStencilAt and RemapRayStencil build the window.
//!
//! Outside the grid the cubic alone is useless -- |w| reaches 26 for the 2.3 bins the
//! shock grid asks for, and it overshoots a Fermi-Dirac tail by many decades -- so the
//! continuation is bounded, and what bounds it depends on the quantity: see
//! RemapAsymptote.  dln is the grid's bin width in ln(nu), needed because the asymptotic
//! slopes are stated per unit ln(nu) while this function works in bin index.
//!
//! ## One operator, at whatever order the live nodes support
//!
//! lo and hi are the live nodes bracketing t, lo2 and hi2 the next ones out.  Three
//! positions follow -- between two live nodes, below them all, above them all -- and the
//! cubic joins in only where all four nodes are live, so a window that straddles the edge
//! of the live spectrum degrades cubic -> linear -> asymptote -> zero without leaving log
//! space.  That matters because a dead node is not corrupt data: measured over a whole
//! snapshot of the 1D shock, interior holes were 1 in 1536 spectra and every other zero
//! was the high-energy tail past underflow or the floor.  Such a window is straddling a
//! boundary of the same kind as the edge of the tabulated grid, and gets the same
//! treatment, with the error staying relative to the local value rather than absolute
//! against the window's largest node.
//!
//! The low-energy bound is the physics, not a numerical guard.  A bin-integrated
//! intensity decays at least as fast as nu^4 going down -- I_nu = nu^3 f with f a bounded
//! occupation increasing as nu falls, so d ln I/d ln nu <= 3 with equality attained as
//! nu -> 0, and the log bin width adds the fourth power.  Capping there forces decay
//! rather than merely forbidding growth, and it costs nothing: the p = 0 spectral
//! diffusion test is unchanged to seven digits by it.  Continuing the *measured* slope
//! alone points the wrong way on the shock's cold upstream, where the whole grid sits on
//! a falling tail, and reaches Sx = 1e187 with NANS_IN_CONS within a few cycles.
//!
//! Above the grid the opacities keep growing as E^2 with no cutoff at all -- the
//! beta-process exponentials cancel identically against kappa = eta exp((E-mu)/T),
//! leaving Bruenn's E^2 phase space -- so their measured slope is continued uncapped,
//! and must not be capped the way a spectrum is.  The intensities do cut off, and a
//! Boltzmann tail was tried for them: ln q ~ 4 y - nu/T gives a cutoff scale readable
//! straight off the edge slope, nu_edge/T = 4 - s/dln, with no EOS lookup.  It is the
//! right asymptotics for a thermal neutrino spectrum and it is *not* used, because
//! measurement did not support it: it doubled the error of the p = 0 spectral diffusion
//! test (2.480e-2 -> 5.106e-2), the one case here with an exact solution, while changing
//! the scattering-isotropy residual by 0.4% and the production shock not at all.
//! Imposing exp(-nu/T) steepening beyond what the edge slope already shows is an
//! extrapolation the data does not carry, and it over-decays any tail that is not
//! Boltzmann.  So above the grid a spectrum gets the same shape as below minus the
//! physical floor, which has no counterpart there: the cubic where it decays faster, the
//! chord as the power-law-exact bound, and no growth past the edge node.
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

//----------------------------------------------------------------------------------------
//! \enum RemapRecon
//! \brief Which profile a conservative remap reconstructs inside each source bin.
//!
//! kDonor is piecewise constant, kPlm piecewise linear with a monotonised-central slope,
//! kPpm the piecewise parabola of Colella & Woodward with their monotonicity constraint,
//! kExp piecewise exponential with a minmod-limited slope of ln(q).  The reconstruction
//! sets the accuracy and nothing else: the integral is conserved for any profile
//! whatsoever, because each source bin hands its two target bins a partition of its own
//! content.
//!
//! On this data only kExp is usable.  A neutrino spectrum on the production grid falls by
//! up to fifteen orders of magnitude from one bin to the next, so no profile polynomial
//! in q can represent it: every monotonicity limiter flattens, kPlm and kPpm become
//! kDonor,
//! and donor-cell smears the peak into a tail bin many decades below it -- a negligible
//! absolute error and a catastrophic relative one, which is what the lepton source, whose
//! bins are weighted by 1/E, is most sensitive to.  ln(q) is the variable that is smooth:
//! it is *linear* in the bin index for any power law in energy, because the bins are
//! uniform in ln(nu), so the exponential profile is exact there and its slope is what the
//! limiter sees.  kPlm and kPpm are kept because they are what tells this story on a
//! spectrum that a polynomial can represent.
//!
//! kMoment is not a reconstruction at all.  A partition into two adjacent bins has one
//! degree of freedom, the head fraction, and demanding that the map preserve **both** the
//! energy sum and the number sum uses it up exactly -- so the split is determined rather
//! than chosen, and no profile enters.  On a geometric grid with ratio r the two targets
//! sit at energies e and e*r while the source sits at e*r^s, so
//!
//!     head + tail/r = E*r^-s   together with   head + tail = E
//!
//! gives head/E = (r^-s - r^-1)/(1 - r^-1), a single number per ray, bounded in [0, 1]
//! and monotone in s, so positivity comes for free.  The consequence worth stating: this
//! is the *only* two-moment-conserving two-band partition there is, its accuracy is
//! therefore donor-cell's, and buying accuracy back would mean spreading each source bin
//! over more than two targets.
enum class RemapRecon { kDonor, kPlm, kPpm, kExp, kExpC, kMoment };

//----------------------------------------------------------------------------------------
//! \enum RemapEdge
//! \brief What a conservative remap does with content whose image leaves the grid.
//!
//! kOpen drops it.  A Doppler shift genuinely carries spectrum past nu_min and nu_max, so
//! this is the physical condition, and the price is that the remap conserves only what
//! stays on the grid.
//!
//! kClosed deposits it in the nearest bin that is on the grid, which is the discrete
//! no-flux condition at both edges and makes the operator conserve exactly.  Note what
//! that costs: the bottom of this grid is nu_min, not zero, so no-flux there is false as
//! physics -- it is the same assumption that makes freq_fluxes unusable on this path --
//! and the deposited content has no way back out.  Whether that builds a reservoir is a
//! question about the whole relaxation rather than about one map; rad_freq_remap's
//! repeated-application check is what measures it.
enum class RemapEdge { kOpen, kClosed };

//----------------------------------------------------------------------------------------
//! \fn Real RemapInterfaceValue
//! \brief The reconstructed value at the interface below bin e, on the four surrounding
//!        bin averages, clamped into the range of the two bins it separates.
//!
//! The clamp is what makes the parabola's endpoints non-negative on a non-negative
//! spectrum; the unclamped quartic interpolant undershoots badly where neighbouring bins
//! differ by decades, which on a neutrino spectrum is everywhere.  Outside the grid there
//! is no information, so the edge bin's own average is used and the outermost profile
//! degrades to a slope of zero.
template <class QT>
KOKKOS_INLINE_FUNCTION
Real RemapInterfaceValue(const QT &q, const int off, const int nfreq, const int e) {
  if (e <= 0) { return q(off); }
  if (e >= nfreq) { return q(off + nfreq - 1); }
  const Real qm = q(off + e - 1), qp = q(off + e);
  Real v = 0.5*(qm + qp);
  if (e >= 2 && e <= nfreq - 2) {
    v = (7.0*(qm + qp) - (q(off + e - 2) + q(off + e + 1)))/12.0;
  }
  return Kokkos::fmin(Kokkos::fmax(v, Kokkos::fmin(qm, qp)), Kokkos::fmax(qm, qp));
}

//----------------------------------------------------------------------------------------
//! \fn void RemapLimitParabola
//! \brief Colella & Woodward's monotonicity constraint on a bin's two edge values.
KOKKOS_INLINE_FUNCTION
void RemapLimitParabola(const Real qbar, Real &ql, Real &qr) {
  if ((qr - qbar)*(qbar - ql) <= 0.0) { ql = qbar; qr = qbar; return; }
  const Real dq = qr - ql;
  const Real d6 = dq*(qbar - 0.5*(ql + qr));
  if (d6 > dq*dq/6.0) {
    ql = 3.0*qbar - 2.0*qr;
  } else if (d6 < -dq*dq/6.0) {
    qr = 3.0*qbar - 2.0*ql;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real RemapProfileHead
//! \brief Integral of the reconstructed profile over the first fraction a of a bin.
//!
//! In Colella & Woodward's parameterisation phi(x) = ql + x*(dq + q6*(1 - x)), for which
//! the whole-bin integral is the bin average identically -- so the tail is the bin
//! average minus this, and the partition is exact whatever the coefficients are.
KOKKOS_INLINE_FUNCTION
Real RemapProfileHead(const Real ql, const Real dq, const Real q6, const Real a) {
  return a*(ql + 0.5*a*dq + q6*a*(0.5 - a/3.0));
}

//----------------------------------------------------------------------------------------
//! \fn void RemapShiftConservative
//! \brief Rebin a whole spectrum by a rigid shift of delta bins, conserving its integral.
//!
//! src is indexed by source bin f, dst by target bin g, and the two index spaces are
//! related by g = f + delta.  Because the grid is uniform in ln(nu) and n0_cm carries no
//! frequency dependence, that shift is rigid (see the header of this file), so with
//! delta = n + s every source bin overlaps exactly two target bins, in the same
//! proportions 1 - s and s, and the operator is two-banded with ray-constant weights.
//! Nothing here evaluates a logarithm or an exponential.
//!
//! Three properties this is built for, none of which RemapBinValue has:
//!
//!   * the integral is conserved exactly -- each source bin's content is partitioned, not
//!     interpolated -- so an un-shift followed by a re-shift moves no energy between
//!     comoving groups even though it is not the identity;
//!   * s = 0 returns src bin for bin, by algebra rather than by a branch, since every
//!     profile correction is multiplied by the head fraction and the tail vanishes;
//!   * content whose image leaves [0, nfreq-1] is dropped under RemapEdge::kOpen and
//!     deposited in the nearest on-grid bin under kClosed, which is the difference
//!     between conserving what stays on the grid and conserving everything.
//!
//! The head is clamped into [0, qbar], which is a partition into two non-negative pieces
//! whatever the reconstruction did, so the output is non-negative on non-negative input
//! and the clamp costs conservation nothing.  src and dst must not alias.
template <class QT, class RT>
KOKKOS_INLINE_FUNCTION
void RemapShiftConservative(const QT &src, RT &dst, const int off, const int nfreq,
                            const Real delta, const Real dln, const RemapRecon recon,
                            const RemapEdge edge) {
  for (int g = 0; g < nfreq; ++g) { dst(off + g) = 0.0; }

  const int n = static_cast<int>(Kokkos::floor(delta));
  const Real s = delta - static_cast<Real>(n);
  const Real a = 1.0 - s;
  // Ray-constant, so the two exponentials are paid once per ray and not once per bin.
  Real hfix = a;
  if (recon == RemapRecon::kMoment && dln > 1.0e-8) {
    const Real em = Kokkos::exp(-dln);
    hfix = (Kokkos::exp(-s*dln) - em)/(1.0 - em);
  }

  Real qil = RemapInterfaceValue(src, off, nfreq, 0);
  for (int f = 0; f < nfreq; ++f) {
    const Real qbar = src(off + f);
    const Real qir = RemapInterfaceValue(src, off, nfreq, f + 1);
    const int gl = f + n, gr = f + n + 1;

    Real head = qbar*a;
    if (recon == RemapRecon::kMoment) {
      head = qbar*hfix;
    } else if ((recon == RemapRecon::kExp || recon == RemapRecon::kExpC) &&
               qbar > 0.0) {
      // phi = A exp(b xi) has the closed-form head fraction (e^{ab} - 1)/(e^b - 1), whose
      // A cancels -- so the profile never has to be normalised.  b is the minmod of the
      // two one-sided slopes of ln(q), which for a power law are equal and exact, making
      // the whole remap exact on a power law as well as conservative.
      const Real qm = (f > 0) ? src(off + f - 1) : 0.0;
      const Real qp = (f < nfreq - 1) ? src(off + f + 1) : 0.0;
      Real b = 0.0;
      if (qm > 0.0 && qp > 0.0) {
        const Real bm = Kokkos::log(qbar/qm), bp = Kokkos::log(qp/qbar);
        if (recon == RemapRecon::kExpC) {
          b = 0.5*(bm + bp);
        } else if (bm*bp > 0.0) {
          b = Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(bm), Kokkos::fabs(bp)), bm);
        }
      }
      Real hfrac;
      if (Kokkos::fabs(b) < 1.0e-6) {
        hfrac = a;
      } else if (b > 0.0) {
        // Scaled by e^{-b} so a steeply rising bin cannot overflow the exponential.
        const Real eb = Kokkos::exp(-b);
        hfrac = (Kokkos::exp(-s*b) - eb)/(1.0 - eb);
      } else {
        const Real eb = Kokkos::exp(b);
        hfrac = (Kokkos::exp(a*b) - 1.0)/(eb - 1.0);
      }
      head = qbar*Kokkos::fmin(Kokkos::fmax(hfrac, 0.0), 1.0);
    } else if (recon != RemapRecon::kDonor && qbar > 0.0) {
      Real ql = qbar, qr = qbar;
      if (recon == RemapRecon::kPpm) {
        ql = qil; qr = qir;
        RemapLimitParabola(qbar, ql, qr);
      } else {
        // Monotonised central slope, zero at either end of the grid where one of the
        // one-sided differences does not exist.
        const Real dm = (f > 0) ? (qbar - src(off + f - 1)) : 0.0;
        const Real dp = (f < nfreq - 1) ? (src(off + f + 1) - qbar) : 0.0;
        Real m = 0.0;
        if (dm*dp > 0.0) {
          const Real lim = 2.0*Kokkos::fmin(Kokkos::fabs(dm), Kokkos::fabs(dp));
          m = Kokkos::copysign(Kokkos::fmin(0.5*Kokkos::fabs(dm + dp), lim), dm);
        }
        ql = qbar - 0.5*m; qr = qbar + 0.5*m;
      }
      const Real q6 = 6.0*(qbar - 0.5*(ql + qr));
      head = RemapProfileHead(ql, qr - ql, q6, a);
      head = Kokkos::fmin(Kokkos::fmax(head, 0.0), qbar);
    }

    if (edge == RemapEdge::kClosed) {
      dst(off + Kokkos::min(Kokkos::max(gl, 0), nfreq - 1)) += head;
      dst(off + Kokkos::min(Kokkos::max(gr, 0), nfreq - 1)) += qbar - head;
    } else {
      if (gl >= 0 && gl < nfreq) { dst(off + gl) += head; }
      if (gr >= 0 && gr < nfreq) { dst(off + gr) += qbar - head; }
    }
    qil = qir;
  }
}

}  // namespace radiation

#endif  // RADIATION_RADIATION_NURATES_REMAP_HPP_
