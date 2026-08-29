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
//! The derivation, the fluid coupling and every measurement quoted here are in
//! notes/multifreq-comoving-solve.md in the ChiralDynamo superproject.
//!
//! Templated on the array type so the kernel (team scratch) and the unit test (host
//! mirrors) run the same code; see pgen/unit_tests/rad_freq_remap.cpp.

#include "athena.hpp"

namespace radiation {

//! Sentinel stored in the log arrays where the value is not positive.
#define NURATES_REMAP_LOG_SENTINEL (-1.0e30)

//! Size of the multi-frequency source term's per-cell workspace, as a count of bin-sized
//! and ray-sized vectors.  Defined here so the allocation in Radiation::Radiation and the
//! sequence of MakeWorkSlice calls in the kernel cannot drift apart.
#define NURATES_MF_WORK_BIN_VECS 13
#define NURATES_MF_WORK_RAY_VECS 3
#define NURATES_MF_WORK_SIZE(nfreq, nang) \
  (NURATES_MF_WORK_BIN_VECS*(nfreq) + NURATES_MF_WORK_RAY_VECS*(nang))

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
    lq(off+f) = (q(off+f) > 0.0) ? Kokkos::log(q(off+f)) : NURATES_REMAP_LOG_SENTINEL;
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
//! \fn Real RemapBinValue
//! \brief Per-bin quantity at a real-valued bin index, by four-point Lagrange on ln(q).
//!
//! Exact for a power law in energy, for point-sampled quantities (sigma_a, sigma_s) and
//! bin-integrated ones (the equilibrium spectrum, the mean intensity) alike -- the latter
//! pick up the n0_cm^(p+1) stretch of the bin width on their own, so the caller applies
//! no Jacobian.  lq holds log(q), or the sentinel where q is not positive; if any stencil
//! node is flagged, q itself is interpolated and the result is clamped to the tabulated
//! range, which costs accuracy in an empty bin but never produces a NaN.
//!
//! The window is slid to stay inside [flo, fhi] rather than dropping to two-point near
//! the ends: a centred window with a linear fallback is wrong by -18% on the peak bin and
//! a factor two above it, because the equilibrium intensity turns over two bins below the
//! top of the grid.
//!
//! Outside the grid the cubic alone is useless -- |w| reaches 26 for the 2.3 bins the
//! shock
//! grid asks for, and it overshoots a Fermi-Dirac tail by many decades -- so the
//! continuation is bounded, and what bounds it depends on the quantity: see
//! RemapAsymptote.  dln is the grid's bin width in ln(nu), needed because the asymptotic
//! slopes are stated per unit ln(nu) while this function works in bin index.
//!
//! The low-energy bound is the physics, not a numerical guard.  A bin-integrated
//! intensity decays at least as fast as nu^4 going down -- I_nu = nu^3 f with f a bounded
//! occupation increasing as nu falls, so d ln I/d ln nu <= 3 with equality attained as
//! nu -> 0, and the log bin width adds the fourth power.  Capping there forces decay,
//! where the old rule only forbade growth, and it costs nothing: measured, the p = 0
//! spectral diffusion test is unchanged to seven digits.
template <class QT>
KOKKOS_INLINE_FUNCTION
Real RemapBinValue(const QT &q, const QT &lq,
                   const int off, const int flo, const int fhi, const Real xpos,
                   const RemapAsymptote asym, const Real dln) {
  if (fhi - flo < 3) {
    int jb = static_cast<int>(Kokkos::floor(xpos));
    if (jb < flo) { jb = flo; }
    if (jb > fhi - 1) { jb = fhi - 1; }
    Real s = xpos - static_cast<Real>(jb);
    int a = off + jb, b = off + jb + 1;
    if (lq(a) > -1.0e29 && lq(b) > -1.0e29) {
      return Kokkos::exp(lq(a) + s*(lq(b) - lq(a)));
    }
    return Kokkos::fmax(q(a) + s*(q(b) - q(a)), 0.0);
  }
  int st = static_cast<int>(Kokkos::floor(xpos)) - 1;
  if (st < flo) { st = flo; }
  if (st > fhi - 3) { st = fhi - 3; }
  Real t = xpos - static_cast<Real>(st);
  Real w0 = -(t - 1.0)*(t - 2.0)*(t - 3.0)/6.0;
  Real w1 = t*(t - 2.0)*(t - 3.0)/2.0;
  Real w2 = -t*(t - 1.0)*(t - 3.0)/2.0;
  Real w3 = t*(t - 1.0)*(t - 2.0)/6.0;
  int a = off + st, b = off + st + 1, c = off + st + 2, d = off + st + 3;
  if (lq(a) > -1.0e29 && lq(b) > -1.0e29 && lq(c) > -1.0e29 && lq(d) > -1.0e29) {
    Real lval = w0*lq(a) + w1*lq(b) + w2*lq(c) + w3*lq(d);
    Real lo = Kokkos::fmin(Kokkos::fmin(lq(a), lq(b)), Kokkos::fmin(lq(c), lq(d)));
    Real hi = Kokkos::fmax(Kokkos::fmax(lq(a), lq(b)), Kokkos::fmax(lq(c), lq(d)));
    if (t < 0.0) {
      // Below the grid.  kOpacity continues the measured edge slope: exact for E^p, and
      // where sigma_a has a threshold that slope is steep, so it continues toward zero.
      Real chord = lq(a) + t*(lq(b) - lq(a));
      if (asym == RemapAsymptote::kOpacity) { return Kokkos::exp(chord); }
      // kSpectrum decays *at least* as fast as nu^4.  That line is the physical limit --
      // I_nu = nu^3 f with f a bounded occupation increasing as nu falls, so the bin
      // integral's log-slope tends to 4 from below -- and because its slope is positive
      // the continuation can only fall going down, whatever the data does.  That is what
      // makes it safe, and it is why there is no longer a hand-imposed "may not grow
      // outward" clamp here: continuing the *measured* slope downward on the shock's cold
      // upstream, where the whole grid sits on a falling tail, points the wrong way and
      // reached Sx = 1e187 with NANS_IN_CONS from cycle 4.  The nu^4 line forbids that by
      // construction rather than by fiat, and it is strictly stronger than the old clamp,
      // which only forbade growth and did not require decay.
      //
      // It is a cap, not a replacement: a spectrum with a sharper low-energy cutoff than
      // a saturating occupation falls faster than nu^4 well before reaching the limit,
      // and that curvature is real information the cubic carries.  Take whichever decays
      // fastest.  Measured, this whole branch costs nothing -- the p = 0 diffusion test
      // reproduces its pre-existing 2.480037e-02 to every digit.
      Real limit = lq(a) + t*(4.0*dln);
      return Kokkos::exp(Kokkos::fmin(Kokkos::fmin(lval, chord), limit));
    }
    if (t > 3.0) {
      // Above the grid.  The opacities keep growing as E^2 with no cutoff at all -- the
      // beta-process exponentials cancel identically against kappa = eta exp((E-mu)/T),
      // leaving Bruenn's E^2 phase space -- so continuing their measured slope is both
      // exact and correct, and must not be capped the way a spectrum is.
      Real u = t - 3.0;
      Real slope = lq(d) - lq(c);
      if (asym == RemapAsymptote::kOpacity) { return Kokkos::exp(lq(d) + u*slope); }
      // The intensities do cut off, and a Boltzmann tail was tried here: ln q ~ 4 y -
      // nu/T
      // gives a cutoff scale readable straight off the edge slope, nu_edge/T = 4 - s/dln,
      // with no EOS lookup.  It is the right asymptotics for a thermal neutrino spectrum
      // and it is *not* used, because measurement did not support it: it doubled the
      // error
      // of the p = 0 spectral diffusion test (2.480e-2 -> 5.106e-2), the one case here
      // with an exact solution, while changing the scattering-isotropy residual by 0.4%
      // and the production shock not at all.  The reason is that imposing exp(-nu/T)
      // steepening beyond what the edge slope already shows is an extrapolation the data
      // does not carry, and it over-decays anything whose tail is not Boltzmann.
      //
      // So the continuation here is the same shape as the low end minus the physical
      // floor, which has no counterpart above: the cubic where it decays faster, the
      // chord as the power-law-exact bound, and no growth past the edge node.
      return Kokkos::exp(Kokkos::fmin(Kokkos::fmin(lval, lq(d) + u*slope), lq(d)));
    }
    // Inside the stencil a cubic through four nodes may legitimately overshoot between
    // them; the pad is scaled to the stencil's own spread so a steep spectrum is not
    // clipped where the interpolation is meaningful.
    Real pad = (hi - lo) + 1.0;
    return Kokkos::exp(Kokkos::fmin(Kokkos::fmax(lval, lo - pad), hi + pad));
  }
  // Sentinel path: a non-positive node means the log-space form is unavailable, so the
  // cubic would run in linear space where its |w| ~ 26 has nothing to temper it.  Nothing
  // here is a power law any more, so clamp to the tabulated range outright.
  Real qlo = Kokkos::fmin(Kokkos::fmin(q(a), q(b)), Kokkos::fmin(q(c), q(d)));
  Real qhi = Kokkos::fmax(Kokkos::fmax(q(a), q(b)), Kokkos::fmax(q(c), q(d)));
  Real lin = w0*q(a) + w1*q(b) + w2*q(c) + w3*q(d);
  return Kokkos::fmin(Kokkos::fmax(lin, Kokkos::fmax(qlo, 0.0)), qhi);
}

}  // namespace radiation

#endif  // RADIATION_RADIATION_NURATES_REMAP_HPP_
