//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_nurates.hpp
//  \brief structs and functions for bns_nurates integration in gray radiation

#ifndef RADIATION_RADIATION_NURATES_HPP_
#define RADIATION_RADIATION_NURATES_HPP_

#include <string>

#include "config.hpp"

#if ENABLE_NURATES

#include "athena.hpp"
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_fermi.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \struct NuratesParams
//! \brief Parameters for bns_nurates opacity library
struct NuratesParams {
  Real nb_min;           // minimum baryon number density [fm^-3, i.e. EOS number
                         //   density units] below which opacities are set to zero
  Real temp_min_mev;     // minimum temperature [MeV] below which opacities are set to zero

  bool use_abs_em;           // include absorption/emission (beta processes)
  bool use_pair;             // include pair processes
  bool use_brem;             // include bremsstrahlung
  bool use_iso;              // include isoenergetic scattering
  bool use_inelastic_scatt;  // include inelastic scattering
  bool use_WM_ab;            // use weak magnetism correction for absorption
  bool use_WM_sc;            // use weak magnetism correction for scattering
  bool use_dU;               // include nuclear interaction correction dU
  bool use_dm_eff;           // include effective mass correction dm_eff
  bool use_NN_medium_corr;   // include nucleon-nucleon medium correction
  bool neglect_blocking;     // neglect Pauli blocking
  bool use_decay;            // include muon decay
  bool use_BRT_brem;         // use BRT06 bremsstrahlung (instead of HR98)
  bool use_equilibrium_distribution;  // assume neutrinos in thermal equilibrium
  bool use_kirchhoff_law;    // replace gray emissivities by kappa times equilibrium density

  int quad_nx;               // number of quadrature points for 1d integration
  int quad_nx_2;             // number of quadrature points for 2d integration (-1 = same)
  MyQuadrature quadrature;   // 1d quadrature for bns_nurates
  MyQuadrature quadrature_2; // 2d quadrature for bns_nurates
};

//----------------------------------------------------------------------------------------
//! \fn void FreqBinEdgesMeV
//! \brief Lower/upper edge of frequency bin ifr, in MeV.
//!
//! freq_grid holds bin lower edges in MeV, with freq_grid(0) = 0 and
//! freq_grid(nfreq-1) = nu_max.  The open top bin is closed by continuing the
//! grid's own spacing -- geometric on a log grid, linear otherwise.
//!
//! Deliberately independent of temperature.  e_hi sets the energy at which the
//! bin's neutrinos are counted (see FreqBinMidMeV), and dN = dE/e_mid is the one
//! place a number is inferred from an energy.  A cutoff that tracked the local
//! temperature would charge energy entering the bin and energy leaving it to
//! different numbers of neutrinos, injecting spurious lepton number wherever the
//! fluid heats or cools.  Widening the bin to span the thermal tail is not worth
//! that: if 20*T exceeds nu_max the grid is under-resolved and should be extended.
//!
//! Single definition on purpose.  The opacity module, the source term and the
//! initial data must agree on where the bins are; they used to disagree by 26%
//! on the top bin's midpoint.
KOKKOS_INLINE_FUNCTION
void FreqBinEdgesMeV(const DvceArray1D<Real> freq_grid, int ifr, int nfreq,
                     int freq_scale, Real &e_lo, Real &e_hi) {
  e_lo = freq_grid(ifr);
  if (ifr < nfreq - 1) {
    e_hi = freq_grid(ifr+1);
  } else {
    e_hi = (freq_scale == 1 && freq_grid(ifr-1) > 0.0) ?
           e_lo*e_lo/freq_grid(ifr-1) : e_lo + (e_lo - freq_grid(ifr-1));
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FreqBinMidMeV
//! \brief Representative energy of a frequency bin, in MeV: the geometric mean on a
//!        log grid (freq_scale == 1), the arithmetic mean otherwise.

KOKKOS_INLINE_FUNCTION
Real FreqBinMidMeV(Real e_lo, Real e_hi, int freq_scale) {
  return (freq_scale == 1 && e_lo > 0.0) ? sqrt(e_lo*e_hi) : 0.5*(e_lo + e_hi);
}

//----------------------------------------------------------------------------------------
//! \fn Real ShiftedBinValue
//! \brief Per-bin quantity at a real-valued bin index.
//!
//! n0_cm is frequency independent, so on a log grid one ray's bins are the grid
//! translated rigidly by delta = ln(n0_cm)/dlnnu.  Bins are uniform in ln(nu), so
//! interpolating ln(q) against the index also carries the bin-width stretching and
//! is exact for a power law.  Cubic, window slid to stay in range; lq holds log(q)
//! or -1e30 where q is not positive, in which case q itself is interpolated.
//! Linear grids need xpos per bin and the n0_cm width factor from the caller.
//! Templated on the array type so the same code serves the kernel (team scratch)
//! and the unit test (host mirrors); see pgen/unit_tests/rad_freq_shift.cpp.
//!
//! Extrapolation past either end is bounded by a log-space chord continued from the
//! edge: a rigorous upper bound for a spectrum concave in ln q (every physical one here)
//! and exact for a power law.  Unbounded, the cubic reaches |w| ~ 26 for the shifts the
//! shock grid needs and overshoots a Fermi-Dirac tail by many decades.
template <class QT>
KOKKOS_INLINE_FUNCTION
Real ShiftedBinValue(const QT &q, const QT &lq,
                     const int off, const int flo, const int fhi, const Real xpos) {
  if (fhi - flo < 3) {
    int jb = static_cast<int>(floor(xpos));
    if (jb < flo) { jb = flo; }
    if (jb > fhi - 1) { jb = fhi - 1; }
    Real s = xpos - static_cast<Real>(jb);
    int a = off + jb, b = off + jb + 1;
    if (lq(a) > -1.0e29 && lq(b) > -1.0e29) {
      return exp(lq(a) + s*(lq(b) - lq(a)));
    }
    return fmax(q(a) + s*(q(b) - q(a)), 0.0);
  }
  // Slide the window to stay inside [flo, fhi]; falling back to linear near the ends
  // straddles the equilibrium peak and is wrong by tens of percent there.
  int st = static_cast<int>(floor(xpos)) - 1;
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
    // Outside the stencil the cubic is unbounded, and a steep spectrum turns that into
    // nonsense: un-shifting the top bin of a Fermi-Dirac tail 2.3 bins past the end
    // overshoots by 1e7 (nfreq = 24, nu in [10, 2000] MeV, T = 12 MeV, v = 0.5), that
    // value enters the angular average, and the forward lookup of the corrupted
    // spectrum then reaches 1e92.
    //
    // A log-space chord continued from the edge is a rigorous upper bound whenever
    // ln q is concave in the bin index, which every physical spectrum is here: a
    // Fermi-Dirac tail has ln q ~ -C exp(b dlnnu), and the E^3 rise below the peak is
    // linear in b.  It is exact for a power law, so nothing the shift legitimately asks
    // for is clipped.  In a convex region it clamps low rather than high, which is the
    // safe direction for a relaxation target.
    if (t > 3.0) {
      lval = fmin(lval, lq(d) + (t - 3.0)*(lq(d) - lq(c)));
    } else if (t < 0.0) {
      lval = fmin(lval, lq(a) - t*(lq(a) - lq(b)));
    }
    Real lo = fmin(fmin(lq(a), lq(b)), fmin(lq(c), lq(d)));
    Real hi = fmax(fmax(lq(a), lq(b)), fmax(lq(c), lq(d)));
    Real pad = (hi - lo) + 1.0;
    return exp(fmin(fmax(lval, lo - pad), hi + pad));
  }
  // Sentinel path: a non-positive node means the log-space form is unavailable, so the
  // cubic runs in linear space where its |w| ~ 26 has nothing to temper it.  Nothing
  // here is a power law any more, so clamp to the tabulated range outright.
  Real qlo = fmin(fmin(q(a), q(b)), fmin(q(c), q(d)));
  Real qhi = fmax(fmax(q(a), q(b)), fmax(q(c), q(d)));
  return fmin(fmax(w0*q(a) + w1*q(b) + w2*q(c) + w3*q(d), fmax(qlo, 0.0)), qhi);
}

//----------------------------------------------------------------------------------------
//! \fn Real ShiftedSpectrum
//! \brief qbar(ifr + delta): the isotropic, bin-integrated comoving spectrum at *this
//!        ray's* comoving energies rather than at the grid's own bin.
//!
//! This is the target the isoenergetic scattering operator relaxes each ray toward.
//! Elastic scattering acts at fixed comoving energy, so for ray iang the target is
//!
//!     integral over gamma*[e_lo, e_hi] of J_cm(eps)/(4 pi) d eps  =  qbar(ifr + delta),
//!
//! and nothing else.  qbar must come from the radiation field -- it is the un-shifted
//! angular mean built by the first pass of MultiFreqRadFluidCouplingNurates -- never
//! from an equilibrium spectrum.  Returns qbar(ifr) itself at rest, for bin 0 (which
//! spans [0, nu_min] and sits outside the log family), and when the shift is disabled.
//!
//! This replaced a shape factor S = qbar(ifr + delta)/qbar(ifr), whose product with an
//! implicitly solved bin-mean jr_bin ~ qbar(ifr) reinstated the qbar(ifr) that S had
//! just divided out.  Two things came of that cancellation being exact in algebra and
//! not in floating point: 0*inf NaNs wherever qbar(ifr) underflowed while the shifted
//! value did not, and -- because jr_bin was an angular average weighted by sigma_s and
//! vncsigma, which vary over the rays in a bin once sigma_s depends on energy -- an
//! O(v) violation of isoenergeticity that took the whole run non-finite for
//! sigma_s ~ E^2 at v = 0.3.  Neither is reachable here: there is no division and no
//! angular average.  See notes/multifreq-frame-consistent-review.md sections 2 and 12
//! in the ChiralDynamo superproject.
//!
//! [flo, fhi] must be the range over which qbar was reconstructed by the un-shift pass.
//! The caller enables the shift only for bins whose whole ray set lands inside it, and
//! falls back to the lab-bin treatment elsewhere -- which is what bin 0 always does and
//! what the baseline did everywhere.  Three other treatments of the ends were measured
//! and are all worse: extrapolating the cubic is wrong by 1e7 on a Fermi-Dirac tail and
//! the corrupted spectrum then reaches 1e92; clamping the position corrupts qbar at the
//! edge and the corruption propagates max|delta| bins inward; dropping the ray makes the
//! bin a half-sky average, which carries the flux into what is meant to be the isotropic
//! part and drives those bins to 1e6 within a few hundred cycles.
//!
//! The shift therefore costs about 2*max|delta| + 4 bins of margin, i.e.
//! 2*ln(gamma_max)/dlnnu + 4.  Widen [nu_min, nu_max] if the bins it gives up carry
//! anything (see the note on FreqBinEdgesMeV).
//!
//! No nfreq floor beyond what shift_ok_ already enforces (nfreq >= 3): ShiftedBinValue
//! falls back to two points when the window is short, and returning the unshifted value
//! instead would leave the opacities shifted while the target was not -- the half-scheme
//! that measured 21x worse than the unshifted baseline.
template <class QT, class WT>
KOKKOS_INLINE_FUNCTION
Real ShiftedSpectrum(const QT &qbar, const QT &lqbar,
                     const int off, const int ifr, const int iang, const int nfreq,
                     const bool enabled, const int freq_scale,
                     const WT &dlt_iang, const Real n0_cm,
                     const WT &emid_f, const Real grid_dlin,
                     const int flo, const int fhi) {
  if (!enabled || ifr <= 0 || nfreq < 3) { return qbar(off+ifr); }
  Real xpos, jac = 1.0;
  if (freq_scale == 1) {
    xpos = static_cast<Real>(ifr) + dlt_iang(iang);
  } else {
    xpos = static_cast<Real>(ifr) + (n0_cm - 1.0)*emid_f(ifr)/grid_dlin;
    jac = n0_cm;
  }
  // A ray at rest reads the tabulated value back bit for bit.
  if (fabs(xpos - static_cast<Real>(ifr)) <= 1.0e-12) { return qbar(off+ifr); }
  // [flo, fhi] is the range over which qbar was actually reconstructed by the un-shift;
  // outside it qbar is the plain lab-bin mean, a different quantity, and reading it here
  // is what leaks an O(1) error inward from the ends.  The caller only enables the shift
  // where xpos stays inside, so the clamp is a guard, not a mechanism.
  Real lo = static_cast<Real>(flo), hi = static_cast<Real>(fhi);
  if (xpos < lo) { xpos = lo; }
  if (xpos > hi) { xpos = hi; }
  // Deliberately no qbar(ifr) > 0 guard: a bin that is empty on the fixed grid can
  // still be populated at a ray's own comoving energy, and that ray now gets the right
  // target instead of the bin's own value.  The old guard existed only to keep a
  // division safe.
  return jac*ShiftedBinValue(qbar, lqbar, off, flo, fhi, xpos);
}

//----------------------------------------------------------------------------------------
//! \fn void AmplitudeAccumulate
//! \brief One ray's contribution to the bin's amplitude factor g = gnum/gden.
//!
//! The scattering target for ray iang is g(ifr)*qbar(ifr + delta_iang): the field's own
//! isotropic spectrum at that ray's comoving energy, times one O(1) number per bin.  g is
//! the implicit unknown -- the new-time amplitude of the spectrum -- and it is what the
//! isoenergeticity requirement is a statement about: elastic scattering cannot move
//! energy between comoving groups, so a comoving-isotropic field must give g == 1.
//!
//! Every term of gnum is (n0*I + ...)/q_sh, and I ~ q_sh for a field consistent with its
//! own spectrum, so the terms are all the same order and neither sum can collapse.  In
//! particular there is no qbar(ifr) in either sum, which is what the previous form had:
//! it multiplied an implicit bin mean jr_bin ~ qbar(ifr) by a shape factor
//! S = qbar(ifr + delta)/qbar(ifr), reinstating the qbar(ifr) that S divided out.  That
//! cancellation is exact in algebra and not in floating point -- hence 0*inf NaNs
//! wherever qbar(ifr) underflowed -- and it put S inside the angular average, which
//! biased it by 47x to 137x the grey L1 error on
//! inputs/tests/rad_diffusion_spectral_nurates.athinput.  See sections 2 and 12 of
//! notes/multifreq-frame-consistent-review.md in the ChiralDynamo superproject.
//!
//! Dropping g altogether -- freezing the target at the previous step, which looks
//! attractive because isoenergetic scattering leaves J_nu alone and so needs no implicit
//! solve -- costs three orders of magnitude on that same test (4.5e+00 against 4.3e-03 at
//! v = 0.1).  g is not the implicitness, it is the ray-dependence of gamma_a*vncsigma:
//! without it the discrete operator stops conserving the comoving angular mean as soon as
//! the field is anisotropic and the medium moves, and the error accumulates.  Solving for
//! it is free, since it is linear.
KOKKOS_INLINE_FUNCTION
void AmplitudeAccumulate(const Real omega_cm, const Real ss, const Real vncsigma,
                         const Real n0, const Real n0_cm, const Real dtcsiga,
                         const Real eq_j, const Real intensity_cm, const Real q_sh,
                         Real &gnum, Real &gden) {
  if (!(q_sh > 0.0)) { return; }
  Real w = omega_cm*ss*vncsigma;
  gnum += w*(n0*intensity_cm + n0_cm*dtcsiga*eq_j)/q_sh;
  gden += w*(n0 + n0_cm*dtcsiga);
}

KOKKOS_INLINE_FUNCTION
Primitive::UnitSystem MakeNuratesUnitSystem() {
  // bns_nurates internal unit system: energy = MeV, length = nm, time = s.
  // Keep this device-callable; Primitive::MakeNGS() is a host-side factory.
  constexpr Real c_cgs    = 2.99792458e10;
  constexpr Real g_cgs    = 6.67408e-8;
  constexpr Real kb_cgs   = 1.38064852e-16;
  constexpr Real msun_cgs = 1.98848e33;
  constexpr Real mev_cgs  = 1.6021766208e-6;

  return Primitive::UnitSystem{
    c_cgs * 1e7,
    g_cgs * mev_cgs / (c_cgs * c_cgs * c_cgs * c_cgs) * 1e7,
    1.0,
    msun_cgs * (c_cgs * c_cgs) / mev_cgs,
    1.0,

    1e7,
    1.0,
    1e-21,
    1e21,
    1.0,
    1.0 / mev_cgs,
    1e-21 / mev_cgs,
    kb_cgs / mev_cgs,
    1.0 / mev_cgs,
  };
}

KOKKOS_INLINE_FUNCTION
MyQuadratureIntegrand RadiationSpectralIntegrand(
    const BS_REAL nu_bar, GreyOpacityParams &grey_op_params) {
  constexpr BS_REAL zero = 0.0;
  constexpr BS_REAL one  = 1.0;

  MyEOSParams my_eos_params  = grey_op_params.eos_pars;
  OpacityFlags opacity_flags = grey_op_params.opacity_flags;
  OpacityParams opacity_pars = grey_op_params.opacity_pars;

  BS_REAL nu = grey_op_params.kernel_pars.pair_kernel_params.omega;
  BS_REAL block_factor[total_num_species];
  BS_REAL g_nu[total_num_species], g_nu_bar[total_num_species];

  for (int idx = 0; idx < total_num_species; ++idx) {
    g_nu[idx] = TotalNuF(nu, &grey_op_params.distr_pars, idx);
    g_nu_bar[idx] = TotalNuF(nu_bar, &grey_op_params.distr_pars, idx);
  }

  MyKernelOutput pair_kernels_m1 = {0};
  if (opacity_flags.use_pair) {
    grey_op_params.kernel_pars.pair_kernel_params.omega_prime = nu_bar;
    grey_op_params.kernel_pars.pair_kernel_params.cos_theta = one;
    grey_op_params.kernel_pars.pair_kernel_params.filter    = zero;
    grey_op_params.kernel_pars.pair_kernel_params.lmax      = zero;
    grey_op_params.kernel_pars.pair_kernel_params.mu        = one;
    grey_op_params.kernel_pars.pair_kernel_params.mu_prime  = one;
    pair_kernels_m1 =
        PairKernels(&my_eos_params, &grey_op_params.kernel_pars.pair_kernel_params);
  }

  MyKernelOutput brem_kernels_m1 = {0};
  if (opacity_flags.use_brem) {
    grey_op_params.kernel_pars.brem_kernel_params.omega_prime = nu_bar;
    if (opacity_pars.brem_implementation == BREM_BRT06) {
      brem_kernels_m1 =
          BremKernelsBRT06(&grey_op_params.kernel_pars.brem_kernel_params,
                           &my_eos_params);
    } else if (opacity_pars.brem_implementation == BREM_HR98) {
      grey_op_params.kernel_pars.brem_kernel_params.l = 0;
      grey_op_params.kernel_pars.brem_kernel_params.use_NN_medium_corr =
          grey_op_params.opacity_pars.use_NN_medium_corr;
      brem_kernels_m1 =
          BremKernelsLegCoeff(&grey_op_params.kernel_pars.brem_kernel_params,
                              &my_eos_params);
    } else if (opacity_pars.brem_implementation == BREM_GP19) {
      brem_kernels_m1 =
          BremKernelAbsGP19(&grey_op_params.kernel_pars.brem_kernel_params,
                            &my_eos_params);
    }
  }

  MyKernelOutput inelastic_kernels_m1 = {0};
  if (opacity_flags.use_inelastic_scatt) {
    grey_op_params.kernel_pars.inelastic_kernel_params.omega_prime = nu_bar;
    inelastic_kernels_m1 =
        InelasticScattKernels(&grey_op_params.kernel_pars.inelastic_kernel_params,
                              &grey_op_params.eos_pars);
  }

  for (int idx = 0; idx < total_num_species; ++idx) {
    block_factor[idx] = opacity_pars.neglect_blocking ? one : one - g_nu_bar[idx];
  }

  BS_REAL pro_term[total_num_species] = {0};
  pro_term[id_nue] =
      (pair_kernels_m1.em[id_nue] + brem_kernels_m1.em[id_nue]) *
      block_factor[id_anue];
  pro_term[id_anue] =
      (pair_kernels_m1.em[id_anue] + brem_kernels_m1.em[id_anue]) *
      block_factor[id_nue];
  pro_term[id_nux] =
      (pair_kernels_m1.em[id_nux] + brem_kernels_m1.em[id_nux]) *
      block_factor[id_anux];
  pro_term[id_anux] =
      (pair_kernels_m1.em[id_anux] + brem_kernels_m1.em[id_anux]) *
      block_factor[id_nux];

  for (int idx = 0; idx < total_num_species; ++idx) {
    pro_term[idx] += inelastic_kernels_m1.em[idx] * g_nu_bar[idx];
  }

  BS_REAL ann_term[total_num_species] = {0};
  ann_term[id_nue] =
      (pair_kernels_m1.abs[id_nue] + brem_kernels_m1.abs[id_nue]) *
      g_nu_bar[id_anue];
  ann_term[id_anue] =
      (pair_kernels_m1.abs[id_anue] + brem_kernels_m1.abs[id_anue]) *
      g_nu_bar[id_nue];
  ann_term[id_nux] =
      (pair_kernels_m1.abs[id_nux] + brem_kernels_m1.abs[id_nux]) *
      g_nu_bar[id_anux];
  ann_term[id_anux] =
      (pair_kernels_m1.abs[id_anux] + brem_kernels_m1.abs[id_anux]) *
      g_nu_bar[id_nux];

  if (kirchoff_flag) {
    ann_term[id_nue] +=
        pair_kernels_m1.em[id_nue] * g_nu_bar[id_anue] / g_nu[id_nue];
    ann_term[id_anue] +=
        pair_kernels_m1.em[id_anue] * g_nu_bar[id_nue] / g_nu[id_anue];
    ann_term[id_nux] +=
        pair_kernels_m1.em[id_nux] * g_nu_bar[id_anux] / g_nu[id_nux];
    ann_term[id_anux] +=
        pair_kernels_m1.em[id_anux] * g_nu_bar[id_nux] / g_nu[id_anux];
  }

  for (int idx = 0; idx < total_num_species; ++idx) {
    ann_term[idx] += inelastic_kernels_m1.abs[idx] * block_factor[idx];
  }

  MyQuadratureIntegrand result = {0};
  result.n = 8;
  result.integrand[0] = POW2(nu_bar) * pro_term[id_nue];
  result.integrand[1] = POW2(nu_bar) * pro_term[id_anue];
  result.integrand[2] = POW2(nu_bar) * pro_term[id_nux];
  result.integrand[3] = POW2(nu_bar) * pro_term[id_anux];
  result.integrand[4] = POW2(nu_bar) * ann_term[id_nue];
  result.integrand[5] = POW2(nu_bar) * ann_term[id_anue];
  result.integrand[6] = POW2(nu_bar) * ann_term[id_nux];
  result.integrand[7] = POW2(nu_bar) * ann_term[id_anux];

  return result;
}

KOKKOS_INLINE_FUNCTION
MyQuadratureIntegrand RadiationIntegrateSpectral1D(
    MyQuadrature* quad, GreyOpacityParams &grey_op_params, BS_REAL* t) {
  constexpr int num_integrands = 8;
  MyQuadratureIntegrand result = {0};

  result.n = num_integrands;
  for (int k = 0; k < num_integrands; ++k) {
    BS_REAL f1_sum = 0.0;
    BS_REAL f2_sum = 0.0;
    for (int i = 0; i < quad->nx; ++i) {
      const BS_REAL x = quad->points[i];
      const BS_REAL w = quad->w[i];
      MyQuadratureIntegrand f1_vals =
          RadiationSpectralIntegrand(t[k] * x, grey_op_params);
      f1_sum += w * f1_vals.integrand[k];

      MyQuadratureIntegrand f2_vals =
          RadiationSpectralIntegrand(t[k] / x, grey_op_params);
      f2_sum += w * f2_vals.integrand[k] / (x * x);
    }

    result.integrand[k] = t[k] * (f1_sum + f2_sum);
  }

  return result;
}

KOKKOS_INLINE_FUNCTION
SpectralOpacities RadiationComputeSpectralOpacitiesNotStimulatedAbs(
    const BS_REAL nu, MyQuadrature* quad_1d, GreyOpacityParams* grey_op_params) {
  constexpr BS_REAL zero    = 0.0;
  constexpr BS_REAL one     = 1.0;
  constexpr BS_REAL four_pi = 4.0 * kBS_Pi;
  constexpr BS_REAL c_light = kBS_Clight;

  grey_op_params->kernel_pars.pair_kernel_params.omega      = nu;
  grey_op_params->kernel_pars.brem_kernel_params.omega      = nu;
  grey_op_params->kernel_pars.inelastic_kernel_params.omega = nu;

  GreyOpacityParams local_grey_params = *grey_op_params;
  local_grey_params.opacity_flags.use_inelastic_scatt = 0;

  BS_REAL g_nu[total_num_species];
  for (int idx = 0; idx < total_num_species; ++idx) {
    g_nu[idx] = TotalNuF(nu, &grey_op_params->distr_pars, idx);
  }

  constexpr BS_REAL temp_multiple = 0.5 * 4.364;
  BS_REAL s_pair[8];
  BS_REAL s_neps[8];
  for (int i = 0; i < 8; ++i) {
    s_pair[i] = temp_multiple * grey_op_params->eos_pars.temp;
    s_neps[i] = nu;
  }

  MyQuadratureIntegrand integrals_pair_1d = {0};
  if (grey_op_params->opacity_flags.use_pair || grey_op_params->opacity_flags.use_brem) {
    integrals_pair_1d =
        RadiationIntegrateSpectral1D(quad_1d, local_grey_params, s_pair);
  }

  MyQuadratureIntegrand integrals_neps_1d = {0};
  if (grey_op_params->opacity_flags.use_inelastic_scatt == 1) {
    local_grey_params.opacity_flags                     = {0};
    local_grey_params.opacity_flags.use_inelastic_scatt = 1;
    integrals_neps_1d =
        RadiationIntegrateSpectral1D(quad_1d, local_grey_params, s_neps);
  }

  MyOpacity abs_em_beta = {0};
  if (grey_op_params->opacity_flags.use_abs_em) {
    abs_em_beta = AbsOpacity(nu, &grey_op_params->opacity_pars,
                             &grey_op_params->eos_pars);
  }

  BS_REAL iso_scatt = zero;
  if (grey_op_params->opacity_flags.use_iso) {
    iso_scatt = IsoScattLegCoeff(nu, &grey_op_params->opacity_pars,
                                 &grey_op_params->eos_pars, 0);
  }

  SpectralOpacities sp_opacities = {0};
  sp_opacities.j[id_nue] =
      abs_em_beta.em[id_nue] +
      kBS_FourPi_hc3 * (integrals_pair_1d.integrand[0] +
                        integrals_neps_1d.integrand[0]);
  sp_opacities.j[id_anue] =
      abs_em_beta.em[id_anue] +
      kBS_FourPi_hc3 * (integrals_pair_1d.integrand[1] +
                        integrals_neps_1d.integrand[1]);
  sp_opacities.j[id_nux] =
      abs_em_beta.em[id_nux] +
      kBS_FourPi_hc3 * (integrals_pair_1d.integrand[2] +
                        integrals_neps_1d.integrand[2]);
  sp_opacities.j[id_anux] =
      abs_em_beta.em[id_anux] +
      kBS_FourPi_hc3 * (integrals_pair_1d.integrand[3] +
                        integrals_neps_1d.integrand[3]);

  sp_opacities.kappa[id_nue] =
      (abs_em_beta.abs[id_nue] +
       kBS_FourPi_hc3 * (integrals_pair_1d.integrand[4] +
                         integrals_neps_1d.integrand[4])) / c_light;
  sp_opacities.kappa[id_anue] =
      (abs_em_beta.abs[id_anue] +
       kBS_FourPi_hc3 * (integrals_pair_1d.integrand[5] +
                         integrals_neps_1d.integrand[5])) / c_light;
  sp_opacities.kappa[id_nux] =
      (abs_em_beta.abs[id_nux] +
       kBS_FourPi_hc3 * (integrals_pair_1d.integrand[6] +
                         integrals_neps_1d.integrand[6])) / c_light;
  sp_opacities.kappa[id_anux] =
      (abs_em_beta.abs[id_anux] +
       kBS_FourPi_hc3 * (integrals_pair_1d.integrand[7] +
                         integrals_neps_1d.integrand[7])) / c_light;

  sp_opacities.j_s[id_nue]  = four_pi * POW2(nu) * g_nu[id_nue] * iso_scatt;
  sp_opacities.j_s[id_anue] = four_pi * POW2(nu) * g_nu[id_anue] * iso_scatt;
  sp_opacities.j_s[id_nux]  = four_pi * POW2(nu) * g_nu[id_nux] * iso_scatt;
  sp_opacities.j_s[id_anux] = four_pi * POW2(nu) * g_nu[id_anux] * iso_scatt;

  sp_opacities.kappa_s[id_nue] =
      four_pi * POW2(nu) * (one - g_nu[id_nue]) * iso_scatt / c_light;
  sp_opacities.kappa_s[id_anue] =
      four_pi * POW2(nu) * (one - g_nu[id_anue]) * iso_scatt / c_light;
  sp_opacities.kappa_s[id_nux] =
      four_pi * POW2(nu) * (one - g_nu[id_nux]) * iso_scatt / c_light;
  sp_opacities.kappa_s[id_anux] =
      four_pi * POW2(nu) * (one - g_nu[id_anux]) * iso_scatt / c_light;

  return sp_opacities;
}

KOKKOS_INLINE_FUNCTION
SpectralOpacities RadiationComputeSpectralOpacitiesStimulatedAbs(
    const BS_REAL nu, MyQuadrature* quad_1d, GreyOpacityParams* grey_op_params) {
  constexpr BS_REAL c_light = kBS_Clight;
  SpectralOpacities op =
      RadiationComputeSpectralOpacitiesNotStimulatedAbs(nu, quad_1d, grey_op_params);

  for (int idx = 0; idx < total_num_species; ++idx) {
    op.kappa[idx]   += op.j[idx] / c_light;
    op.kappa_s[idx] += op.j_s[idx] / c_light;
  }

  return op;
}

//----------------------------------------------------------------------------------------
//! \fn void bns_nurates_gray
//! \brief Wrapper for bns_nurates grey opacity call for gray (frequency-integrated)
//!        neutrino transport.
//!
//! \param[in]  nb          baryon number density (in code units)
//! \param[in]  temp        temperature (MeV)
//! \param[in]  yp          proton fraction  (dimensionless)
//! \param[in]  yn          neutron fraction (dimensionless)
//! \param[in]  mu_n        neutron chemical potential (MeV)
//! \param[in]  mu_p        proton chemical potential (MeV)
//! \param[in]  mu_e        electron chemical potential (MeV)
//! \param[in]  nudens_0    neutrino number densities [4] (fm^-3, per species)
//! \param[in]  nudens_1    neutrino energy densities [4] (code units, per species)
//! \param[out] eta_0       number emissivity [nspecies] (fm^-3 per code time)
//! \param[out] eta_1       energy emissivity [nspecies] (code units)
//! \param[out] abs_0       number absorption opacity [nspecies] (code units)
//! \param[out] abs_1       energy absorption opacity [nspecies] (code units)
//! \param[out] scat_1      energy scattering opacity [nspecies] (code units)
//! \param[in]  nurates_params   nurates parameters
//! \param[in]  code_units       code unit system
//! \param[in]  eos_units        EOS unit system
KOKKOS_INLINE_FUNCTION
void bns_nurates_gray(Real nb, Real temp, Real yp, Real yn,
                      Real mu_n, Real mu_p, Real mu_e,
                      Real nudens_0[4], Real nudens_1[4],
                      Real eta_0[4], Real eta_1[4],
                      Real abs_0[4], Real abs_1[4],
                      Real scat_0[4], Real scat_1[4],
                      NuratesParams const &nurates_params,
                      Primitive::UnitSystem &code_units,
                      Primitive::UnitSystem &eos_units) {
  // bns_nurates internal unit system: energy = MeV, length = nm, time = s.
  // Verified from bns_nurates/include/constants.hpp:
  //   kBS_Clight       = 2.998e17  nm/s  (dimensional speed of light, not c=1)
  //   kBS_Saturation_n = 0.15e18   nm^-3 (nuclear saturation density)
  // And mwe.cpp: eos_pars.nb = nb_cm3 * 1e-21  (cm^-3 -> nm^-3 conversion).
  Primitive::UnitSystem nurates_units = MakeNuratesUnitSystem();

  const Real unit_length    = code_units.LengthConversion(nurates_units);
  const Real unit_time      = code_units.TimeConversion(nurates_units);
  // Every number density crossing this interface -- the baryon density nb, the
  // neutrino densities nudens_0 and the number emissivity eta_0 -- is in the EOS
  // number-density unit (fm^-3), as in radiation_m1/.  Energy densities stay in
  // code units, per the G = c = 1 convention.
  const Real unit_num_dens  = eos_units.NumberDensityConversion(nurates_units);
  const Real unit_ene_dens  = code_units.EnergyDensityConversion(nurates_units);

  // zero outputs if below floor values
  // nb_min is in fm^-3 (the EOS number-density unit), so compare nb before the
  // conversion to bns_nurates units -- this is what radiation_m1_nurates.hpp does.
  if ((nb < nurates_params.nb_min) ||
      (temp < nurates_params.temp_min_mev)) {
    for (int idx = 0; idx < 4; ++idx) {
      eta_0[idx]  = 0.;
      eta_1[idx]  = 0.;
      abs_0[idx]  = 0.;
      abs_1[idx]  = 0.;
      scat_0[idx] = 0.;
      scat_1[idx] = 0.;
    }
    return;
  }

  // populate GreyOpacityParams
  GreyOpacityParams grey_op_params = {0};

  // reaction flags
  grey_op_params.opacity_flags.use_abs_em           = nurates_params.use_abs_em;
  grey_op_params.opacity_flags.use_brem             = nurates_params.use_brem;
  grey_op_params.opacity_flags.use_pair             = nurates_params.use_pair;
  grey_op_params.opacity_flags.use_iso              = nurates_params.use_iso;
  grey_op_params.opacity_flags.use_inelastic_scatt  = nurates_params.use_inelastic_scatt;

  // other opacity flags
  grey_op_params.opacity_pars.use_WM_ab            = nurates_params.use_WM_ab;
  grey_op_params.opacity_pars.use_WM_sc            = nurates_params.use_WM_sc;
  grey_op_params.opacity_pars.use_dU               = nurates_params.use_dU;
  grey_op_params.opacity_pars.use_dm_eff           = nurates_params.use_dm_eff;
  grey_op_params.opacity_pars.use_NN_medium_corr   = nurates_params.use_NN_medium_corr;
  grey_op_params.opacity_pars.neglect_blocking     = nurates_params.neglect_blocking;
  grey_op_params.opacity_pars.use_decay            = nurates_params.use_decay;
  grey_op_params.opacity_pars.brem_implementation  =
      nurates_params.use_BRT_brem ? BREM_BRT06 : BREM_HR98;

  // EOS quantities (convert to bns_nurates/nuclear units)
  grey_op_params.eos_pars.nb   = nb * unit_num_dens;  // [baryon/nm^3]
  grey_op_params.eos_pars.temp = temp;                 // [MeV]
  grey_op_params.eos_pars.yp   = yp;                  // [dimensionless]
  grey_op_params.eos_pars.yn   = yn;                  // [dimensionless]
  grey_op_params.eos_pars.mu_e = mu_e;                 // [MeV]
  grey_op_params.eos_pars.mu_p = mu_p;                 // [MeV]
  grey_op_params.eos_pars.mu_n = mu_n;                 // [MeV]
  grey_op_params.eos_pars.dU      = 0.;
  grey_op_params.eos_pars.dm_eff  = 0.;

  // neutrino distribution: equilibrium or reconstructed from M1 moments
  if (nurates_params.use_equilibrium_distribution) {
    // equilibrium distribution: assume full LTE for all species
    grey_op_params.distr_pars = NuEquilibriumParams(&grey_op_params.eos_pars);
    ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                         &grey_op_params.distr_pars,
                         &grey_op_params.m1_pars);
    grey_op_params.m1_pars.chi[id_nue]  = 1./3.;
    grey_op_params.m1_pars.chi[id_anue] = 1./3.;
    grey_op_params.m1_pars.chi[id_nux]  = 1./3.;
    grey_op_params.m1_pars.chi[id_anux] = 1./3.;
  } else {
    // reconstruct from gray radiation moments (frequency-integrated)
    // nudens_0[isp] = number density, nudens_1[isp] = energy density
    // factor 1/2 for nux/anux: bns_nurates uses "mu or tau", gray rad uses "mu+tau"
    grey_op_params.m1_pars.n[id_nue]  = nudens_0[0] * unit_num_dens;
    grey_op_params.m1_pars.n[id_anue] = nudens_0[1] * unit_num_dens;
    grey_op_params.m1_pars.n[id_nux]  = 0.5 * nudens_0[2] * unit_num_dens;
    grey_op_params.m1_pars.n[id_anux] = 0.5 * nudens_0[3] * unit_num_dens;

    grey_op_params.m1_pars.J[id_nue]  = nudens_1[0] * unit_ene_dens;
    grey_op_params.m1_pars.J[id_anue] = nudens_1[1] * unit_ene_dens;
    grey_op_params.m1_pars.J[id_nux]  = 0.5 * nudens_1[2] * unit_ene_dens;
    grey_op_params.m1_pars.J[id_anux] = 0.5 * nudens_1[3] * unit_ene_dens;

    // eddington factor: assume isotropic (1/3) for gray
    grey_op_params.m1_pars.chi[id_nue]  = 1./3.;
    grey_op_params.m1_pars.chi[id_anue] = 1./3.;
    grey_op_params.m1_pars.chi[id_nux]  = 1./3.;
    grey_op_params.m1_pars.chi[id_anux] = 1./3.;

    grey_op_params.distr_pars =
        CalculateDistrParamsFromM1(&grey_op_params.m1_pars, &grey_op_params.eos_pars);
  }

  // compute opacities
  M1Opacities opacities = ComputeM1Opacities(&nurates_params.quadrature,
                                              &nurates_params.quadrature_2,
                                              &grey_op_params);

  // extract emissivities (factors of 2 for nux/anux: bns_nurates "mu OR tau",
  // gray rad "mu AND tau")
  eta_0[0] = opacities.eta_0[id_nue];
  eta_0[1] = opacities.eta_0[id_anue];
  eta_0[2] = opacities.eta_0[id_nux]  * 2.;
  eta_0[3] = opacities.eta_0[id_anux] * 2.;

  eta_1[0] = opacities.eta[id_nue];
  eta_1[1] = opacities.eta[id_anue];
  eta_1[2] = opacities.eta[id_nux]  * 2.;
  eta_1[3] = opacities.eta[id_anux] * 2.;

  // extract absorption inverse mean-free paths
  abs_0[0] = opacities.kappa_0_a[id_nue];
  abs_0[1] = opacities.kappa_0_a[id_anue];
  abs_0[2] = opacities.kappa_0_a[id_nux];
  abs_0[3] = opacities.kappa_0_a[id_anux];

  abs_1[0] = opacities.kappa_a[id_nue];
  abs_1[1] = opacities.kappa_a[id_anue];
  abs_1[2] = opacities.kappa_a[id_nux];
  abs_1[3] = opacities.kappa_a[id_anux];

  // extract scattering inverse mean-free paths
  scat_0[0] = 0.;  // number scattering not used in gray scheme
  scat_0[1] = 0.;
  scat_0[2] = 0.;
  scat_0[3] = 0.;

  scat_1[0] = opacities.kappa_s[id_nue];
  scat_1[1] = opacities.kappa_s[id_anue];
  scat_1[2] = opacities.kappa_s[id_nux];
  scat_1[3] = opacities.kappa_s[id_anux];

  // Convert outputs from bns_nurates nuclear units back to code units.
  //
  // Convention: unit_X = code_units.XConversion(nurates_units)
  //             means  x_nuclear = x_code * unit_X
  //
  //   opacity [1/length]:
  //     kappa_nuclear = kappa_code * unit_length  (length appears in denominator)
  //     => kappa_code = kappa_nuclear * unit_length
  //
  //   energy emissivity [energy_density / time]:
  //     eta_nuclear = eta_code * (unit_ene_dens / unit_time)
  //     => eta_code = eta_nuclear * unit_time / unit_ene_dens
  //
  //   number emissivity [fm^-3 / time]:
  //     eta_0_nuclear = eta_0_eos * (unit_num_dens / unit_time)
  //     => eta_0_eos = eta_0_nuclear * unit_time / unit_num_dens
  const Real kap_to_code  = unit_length;
  const Real eta1_to_code = unit_time / unit_ene_dens;
  const Real eta0_to_eos  = unit_time / unit_num_dens;

  Real eq_n_eos[4] = {0., 0., 0., 0.};
  Real eq_J_code[4] = {0., 0., 0., 0.};
  if (nurates_params.use_kirchhoff_law) {
    M1Quantities eq_m1_pars = {0};
    NuDistributionParams eq_distr_pars =
        NuEquilibriumParams(&grey_op_params.eos_pars);
    ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                         &eq_distr_pars,
                         &eq_m1_pars);

    eq_n_eos[0] = eq_m1_pars.n[id_nue] / unit_num_dens;
    eq_n_eos[1] = eq_m1_pars.n[id_anue] / unit_num_dens;
    eq_n_eos[2] = 2.0 * eq_m1_pars.n[id_nux] / unit_num_dens;
    eq_n_eos[3] = 2.0 * eq_m1_pars.n[id_anux] / unit_num_dens;

    eq_J_code[0] = eq_m1_pars.J[id_nue] / unit_ene_dens;
    eq_J_code[1] = eq_m1_pars.J[id_anue] / unit_ene_dens;
    eq_J_code[2] = 2.0 * eq_m1_pars.J[id_nux] / unit_ene_dens;
    eq_J_code[3] = 2.0 * eq_m1_pars.J[id_anux] / unit_ene_dens;
  }

  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  *= eta0_to_eos;
    eta_1[idx]  *= eta1_to_code;
    abs_0[idx]  *= kap_to_code;
    abs_1[idx]  *= kap_to_code;
    scat_1[idx] *= kap_to_code;
    // scat_0 is set to zero above; conversion unnecessary

    if (nurates_params.use_kirchhoff_law) {
      eta_0[idx] = (abs_0[idx] > 0.0) ? abs_0[idx] * eq_n_eos[idx] : eta_0[idx];
      eta_1[idx] = (abs_1[idx] > 0.0) ? abs_1[idx] * eq_J_code[idx] : eta_1[idx];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void bns_nurates_spectral_bin
//! \brief Midpoint/quadrature wrapper for bns_nurates spectral opacity coefficients.
//!        The frequency bin bounds are in MeV, as freq_grid holds them; outputs are
//!        bin-integrated emissivities and bin-centered opacities in code units, except
//!        the number emissivity eta_0, which is in fm^-3 per code time (as is the
//!        input nudens_0).
KOKKOS_INLINE_FUNCTION
void bns_nurates_spectral_bin(Real e_lo_mev, Real e_hi_mev,
                              int freq_scale,
                              Real nb, Real temp, Real yp, Real yn,
                              Real mu_n, Real mu_p, Real mu_e,
                              Real nudens_0[4], Real nudens_1[4],
                              Real eta_0[4], Real eta_1[4],
                              Real abs_0[4], Real abs_1[4], Real scat_1[4],
                              NuratesParams const &nurates_params,
                              Primitive::UnitSystem &code_units,
                              Primitive::UnitSystem &eos_units) {
  Primitive::UnitSystem nurates_units = MakeNuratesUnitSystem();

  const Real unit_length    = code_units.LengthConversion(nurates_units);
  const Real unit_time      = code_units.TimeConversion(nurates_units);
  // Every number density crossing this interface -- the baryon density nb, the
  // neutrino densities nudens_0 and the number emissivity eta_0 -- is in the EOS
  // number-density unit (fm^-3), as in radiation_m1/.  Energy densities stay in
  // code units, per the G = c = 1 convention.
  const Real unit_num_dens  = eos_units.NumberDensityConversion(nurates_units);
  const Real unit_ene_dens  = code_units.EnergyDensityConversion(nurates_units);

  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  = 0.;
    eta_1[idx]  = 0.;
    abs_0[idx]  = 0.;
    abs_1[idx]  = 0.;
    scat_1[idx] = 0.;
  }

  // nb_min is in fm^-3 (the EOS number-density unit), so compare nb before the
  // conversion to bns_nurates units -- this is what radiation_m1_nurates.hpp does.
  if ((nb < nurates_params.nb_min) ||
      (temp < nurates_params.temp_min_mev) ||
      (e_hi_mev <= e_lo_mev)) {
    return;
  }

  GreyOpacityParams grey_op_params = {0};
  grey_op_params.opacity_flags.use_abs_em           = nurates_params.use_abs_em;
  grey_op_params.opacity_flags.use_brem             = nurates_params.use_brem;
  grey_op_params.opacity_flags.use_pair             = nurates_params.use_pair;
  grey_op_params.opacity_flags.use_iso              = nurates_params.use_iso;
  grey_op_params.opacity_flags.use_inelastic_scatt  = nurates_params.use_inelastic_scatt;

  grey_op_params.opacity_pars.use_WM_ab            = nurates_params.use_WM_ab;
  grey_op_params.opacity_pars.use_WM_sc            = nurates_params.use_WM_sc;
  grey_op_params.opacity_pars.use_dU               = nurates_params.use_dU;
  grey_op_params.opacity_pars.use_dm_eff           = nurates_params.use_dm_eff;
  grey_op_params.opacity_pars.use_NN_medium_corr   = nurates_params.use_NN_medium_corr;
  grey_op_params.opacity_pars.neglect_blocking     = nurates_params.neglect_blocking;
  grey_op_params.opacity_pars.use_decay            = nurates_params.use_decay;
  grey_op_params.opacity_pars.brem_implementation  =
      nurates_params.use_BRT_brem ? BREM_BRT06 : BREM_HR98;

  grey_op_params.eos_pars.nb   = nb * unit_num_dens;
  grey_op_params.eos_pars.temp = temp;
  grey_op_params.eos_pars.yp   = yp;
  grey_op_params.eos_pars.yn   = yn;
  grey_op_params.eos_pars.mu_e = mu_e;
  grey_op_params.eos_pars.mu_p = mu_p;
  grey_op_params.eos_pars.mu_n = mu_n;
  grey_op_params.eos_pars.dU      = 0.;
  grey_op_params.eos_pars.dm_eff  = 0.;

  // TODO(@dradice): are these needed for spectral opacities?
  if (nurates_params.use_equilibrium_distribution) {
    grey_op_params.distr_pars = NuEquilibriumParams(&grey_op_params.eos_pars);
    ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                         &grey_op_params.distr_pars,
                         &grey_op_params.m1_pars);
    for (int idx = 0; idx < 4; ++idx) {
      grey_op_params.m1_pars.chi[idx] = 1./3.;
    }
  } else {
    grey_op_params.m1_pars.n[id_nue]  = nudens_0[0] * unit_num_dens;
    grey_op_params.m1_pars.n[id_anue] = nudens_0[1] * unit_num_dens;
    grey_op_params.m1_pars.n[id_nux]  = 0.5 * nudens_0[2] * unit_num_dens;
    grey_op_params.m1_pars.n[id_anux] = 0.5 * nudens_0[3] * unit_num_dens;

    grey_op_params.m1_pars.J[id_nue]  = nudens_1[0] * unit_ene_dens;
    grey_op_params.m1_pars.J[id_anue] = nudens_1[1] * unit_ene_dens;
    grey_op_params.m1_pars.J[id_nux]  = 0.5 * nudens_1[2] * unit_ene_dens;
    grey_op_params.m1_pars.J[id_anux] = 0.5 * nudens_1[3] * unit_ene_dens;

    for (int idx = 0; idx < 4; ++idx) {
      grey_op_params.m1_pars.chi[idx] = 1./3.;
    }
    grey_op_params.distr_pars =
        CalculateDistrParamsFromM1(&grey_op_params.m1_pars, &grey_op_params.eos_pars);
  }

  const Real e_lo = e_lo_mev;
  const Real e_hi = e_hi_mev;
  const Real de = e_hi - e_lo;
  const Real e_mid = (freq_scale == 1 && e_lo > 0.0) ? sqrt(e_lo*e_hi) :
                                                         0.5*(e_lo + e_hi);

  MyQuadrature quad = nurates_params.quadrature;
  SpectralOpacities op_mid =
      RadiationComputeSpectralOpacitiesStimulatedAbs(e_mid, &quad, &grey_op_params);

  for (int iq = 0; iq < quad.nx; ++iq) {
    const Real x = quad.points[iq];
    const Real w = quad.w[iq];
    const Real e = e_lo + de*x;
    SpectralOpacities op =
        RadiationComputeSpectralOpacitiesStimulatedAbs(e, &quad, &grey_op_params);
    for (int idx = 0; idx < 4; ++idx) {
      eta_0[idx] += w * de * kBS_FourPi_hc3 * SQR(e) * op.j[idx];
      eta_1[idx] += w * de * kBS_FourPi_hc3 * SQR(e) * e * op.j[idx];
    }
  }

  eta_0[2] *= 2.;
  eta_0[3] *= 2.;
  eta_1[2] *= 2.;
  eta_1[3] *= 2.;

  const Real kap_to_code  = unit_length;
  const Real eta0_to_eos  = unit_time / unit_num_dens;
  const Real eta1_to_code = unit_time / unit_ene_dens;

  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  *= eta0_to_eos;
    eta_1[idx]  *= eta1_to_code;
    abs_0[idx]   = op_mid.kappa[idx] * kap_to_code;
    abs_1[idx]   = op_mid.kappa[idx] * kap_to_code;
    scat_1[idx]  = op_mid.kappa_s[idx] * kap_to_code;
  }
}

} // namespace radiation

#endif  // ENABLE_NURATES

#endif  // RADIATION_RADIATION_NURATES_HPP_
