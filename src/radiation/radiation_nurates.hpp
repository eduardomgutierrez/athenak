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
#include "bns_nurates_ns.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"

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

  // Partially-equilibrated (T*, Ye*) emissivity predictor, and the only equilibrium
  // closure this module has: Kirchhoff's law against the blackbody at the state the
  // matter is predicted to reach over the step, not the state it starts from.  Off, the
  // emissivity is bns_nurates' own.  Same names, same defaults and the same meaning as
  // in radiation_m1/radiation_m1_params.hpp, so one <bns_nurates> block drives both.
  bool use_partial_equilibrium;
  bool peq_heavy;      // the predictor also sets the heavy-lepton emissivity
  bool heavy_eps_midpoint;  // heavy emissivity from the bin-centre evaluation, not the quadrature
  Real peq_w_floor;    // tier-0 gate: skip the cell outright below this weight
  Real peq_dlnT_tol;   // tier-1 gate: skip when the predicted |dlnT| is below this
  Real peq_dYe_tol;    // tier-1 gate: ... and the predicted |dYe| below this

  int quad_nx;               // number of quadrature points for 1d integration
  int quad_nx_2;             // number of quadrature points for 2d integration (-1 = same)
  bns_nurates::MyQuadrature quadrature;   // 1d quadrature for bns_nurates
  bns_nurates::MyQuadrature quadrature_2; // 2d quadrature for bns_nurates

  int nspecies;              // <radiation>/nspecies; 3 merges the heavy flavours
};

//----------------------------------------------------------------------------------------
//! \brief The heavy-flavour slot convention, and the fold between the two choices.
//!
//! bns_nurates always resolves four species.  This module carries them in as many slots
//! as <radiation>/nspecies asks for:
//!
//!   nspecies = 4   slot 2 = nu_x (mu + tau), slot 3 = nubar_x (mubar + taubar)
//!   nspecies = 3   slot 2 = all four heavy states, slot 3 unused
//!
//! so slot 2's multiplicity is 2 or 4 and the per-state occupancy is j/2 or j/4.
//! Extensive per-slot quantities -- emissivities and equilibrium densities -- add; the
//! intensive ones -- absorption and scattering coefficients, which are per-particle
//! inverse mean free paths -- average.
//!
//! Given the same inputs the fold is exact: it reproduces the four-slot sum term for
//! term, differing only in summation order.  The approximation lives in the inputs, in
//! two places of the same order, both quadratic in a nu_x/nubar_x difference that only
//! the electron-positron pair kernel produces at all:
//!
//!   extinction  kappa_bar*(J_x + J_xbar) vs kappa_x*J_x + kappa_xbar*J_xbar,
//!               short by the covariance (kappa_x - kappa_xbar)(J_x - J_xbar)/2
//!   blocking    the kernels block each heavy species against its PARTNER, so feeding
//!               both ids the merged mean fbar costs (em_x - em_xbar)(f_x - f_xbar)/2
//!
//! The second reaches the emissivity even under use_partial_equilibrium, where
//! eta = kappa*J_eq: J_eq[2] == J_eq[3] identically (both carry eta_t = 0), so J_eq is
//! untouched, but kappa is not.  It is inert only when the occupancy comes from the
//! equilibrium distribution rather than the evolved field.
//!
//! Fold* writes this module's slots; Mirror writes bns_nurates' species ids.  The two
//! run in opposite directions and are named apart for that reason.

KOKKOS_INLINE_FUNCTION
bool NuratesMergeHeavy(const int nspecies) {
  return (nspecies == 3);
}

//! \brief Reciprocal of a slot's state count, which turns its energy into an occupancy.
KOKKOS_INLINE_FUNCTION
Real NuratesInvMultiplicity(const int nspecies, const int isp) {
  if (isp < 2) { return 1.0; }
  return NuratesMergeHeavy(nspecies) ? 0.25 : 0.5;
}

//! \brief Collapse an extensive per-slot quantity onto the merged heavy slot.
KOKKOS_INLINE_FUNCTION
void NuratesFoldHeavySum(const int nspecies, Real q[4]) {
  if (NuratesMergeHeavy(nspecies)) {
    q[2] += q[3];
    q[3] = 0.0;
  }
}

//! \brief Collapse an intensive per-slot quantity onto the merged heavy slot.
KOKKOS_INLINE_FUNCTION
void NuratesFoldHeavyMean(const int nspecies, Real q[4]) {
  if (NuratesMergeHeavy(nspecies)) {
    q[2] = 0.5*(q[2] + q[3]);
    q[3] = 0.0;
  }
}

//! \brief Give bns_nurates' antineutrino heavy id the merged slot's occupancy.
//!
//! The pair and bremsstrahlung kernels block each heavy species against its partner
//! (pro_term[id_nux] carries block_factor[id_anux]), so a merged species blocks against
//! itself and both ids must see the same number.
KOKKOS_INLINE_FUNCTION
void NuratesMirrorHeavy(const int nspecies, bns_nurates::BS_REAL f[total_num_species]) {
  if (NuratesMergeHeavy(nspecies)) {
    f[id_anux] = f[id_nux];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FreqBinEdgesMeV
//! \brief Lower/upper edge of frequency bin ifr, in MeV.
//!
//! freq_grid holds bin lower edges in MeV, with freq_grid(0) = nu_min and
//! freq_grid(nfreq-1) = nu_max on the neutrino grid -- every bin geometric, no
//! [0, nu_min] catch-all.  The open top bin is closed by continuing the grid's own
//! spacing -- geometric on a log grid, linear otherwise.
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

//----------------------------------------------------------------------------------------
//! \struct NuratesGroupOccupancy
//! \brief One cell's evolved neutrino occupancy as the partner-energy integrals need to
//!        see it: a flat view of f(species, bin), plus the uniform coordinate the bin
//!        midpoints are spaced on.
//!
//! A null `f` means the cell carries no evolved spectrum, which is what
//! `use_equilibrium_distribution` asks for; GreyOpacityParams::distr_pars is used then.

struct NuratesGroupOccupancy {
  const Real *f;    // f[isp*sp_stride + ifr*fr_stride], clamped to [0, 1]
  int sp_stride;
  int fr_stride;
  int nspecies;
  int nfreq;
  Real x0;          // grid coordinate of bin 0's midpoint
  Real inv_dx;      // reciprocal spacing of that coordinate
  bool log_grid;    // the coordinate is log(E/MeV) rather than E
};

//----------------------------------------------------------------------------------------
//! \fn void NuratesOccupancyAt
//! \brief Occupancy of every species at energy nu (MeV), interpolated linearly between
//!        the bins whose midpoints bracket it.
//!
//! Outside the grid the two ends take different values, matching the asymptotes the
//! source term's frequency remap is built on (RemapAsymptote::kSpectrum,
//! radiation_nurates_remap.hpp): below mid(0) a fermion occupation saturates, so 1;
//! above mid(nfreq-1) it is Boltzmann-suppressed, so 0.  A non-positive or NaN energy
//! takes the zero branch.

KOKKOS_INLINE_FUNCTION
void NuratesOccupancyAt(const NuratesGroupOccupancy &spec, const Real nu,
                        bns_nurates::BS_REAL f[total_num_species]) {
  for (int idx = 0; idx < total_num_species; ++idx) {
    f[idx] = 0.0;
  }
  if (nu <= 0.0) {
    return;
  }
  const Real s = ((spec.log_grid ? log(nu) : nu) - spec.x0)*spec.inv_dx;
  if (!(s < static_cast<Real>(spec.nfreq - 1))) {
    return;
  }
  const int nsp = (spec.nspecies < total_num_species) ? spec.nspecies
                                                      : total_num_species;
  if (s <= 0.0) {
    for (int idx = 0; idx < nsp; ++idx) {
      f[idx] = 1.0;
    }
  } else {
    const int il = static_cast<int>(s);
    const Real w = s - static_cast<Real>(il);
    for (int idx = 0; idx < nsp; ++idx) {
      const Real *p = spec.f + idx*spec.sp_stride + il*spec.fr_stride;
      f[idx] = p[0] + w*(p[spec.fr_stride] - p[0]);
    }
  }
  // Single exit past this point: every slot bns_nurates resolves has to be filled, and
  // the merged run stores only one heavy slot.  The two zero returns above need no
  // mirror -- f = 0 for all four is what the mirror would produce there anyway.
  NuratesMirrorHeavy(spec.nspecies, f);
}

//----------------------------------------------------------------------------------------
//! \fn void NuratesOccupancy
//! \brief Occupancy of every species at energy nu (MeV), from the evolved spectrum where
//!        the cell has one and from the equilibrium distribution otherwise.

KOKKOS_INLINE_FUNCTION
void NuratesOccupancy(const NuratesGroupOccupancy &spec,
                      bns_nurates::NuDistributionParams &distr_pars,
                      bns_nurates::MyEOSParams &eos_pars,
                      const Real nu, bns_nurates::BS_REAL f[total_num_species]) {
  if (spec.f != nullptr) {
    NuratesOccupancyAt(spec, nu, f);
  } else {
    for (int idx = 0; idx < total_num_species; ++idx) {
      f[idx] = TotalNuF(nu, &distr_pars, &eos_pars, idx);
    }
  }
}

KOKKOS_INLINE_FUNCTION
bns_nurates::MyQuadratureIntegrand RadiationSpectralIntegrand(
    const bns_nurates::BS_REAL nu_bar, const NuratesGroupOccupancy &spec,
    bns_nurates::GreyOpacityParams &grey_op_params) {
  constexpr bns_nurates::BS_REAL zero = 0.0;
  constexpr bns_nurates::BS_REAL one  = 1.0;

  bns_nurates::MyEOSParams &my_eos_params        = grey_op_params.eos_pars;
  const bns_nurates::OpacityFlags &opacity_flags = grey_op_params.opacity_flags;
  const bns_nurates::OpacityParams &opacity_pars = grey_op_params.opacity_pars;

  bns_nurates::BS_REAL block_factor[total_num_species];
  bns_nurates::BS_REAL g_nu_bar[total_num_species];

  NuratesOccupancy(spec, grey_op_params.distr_pars, grey_op_params.eos_pars,
                   nu_bar, g_nu_bar);

  bns_nurates::MyKernelOutput pair_kernels_m1 = {0};
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

  bns_nurates::MyKernelOutput brem_kernels_m1 = {0};
  if (opacity_flags.use_brem) {
    grey_op_params.kernel_pars.brem_kernel_params.omega_prime = nu_bar;
    if (opacity_pars.brem_implementation == bns_nurates::BREM_BRT06) {
      brem_kernels_m1 =
          BremKernelsBRT06(&grey_op_params.kernel_pars.brem_kernel_params,
                           &my_eos_params);
    } else if (opacity_pars.brem_implementation == bns_nurates::BREM_HR98) {
      grey_op_params.kernel_pars.brem_kernel_params.l = 0;
      grey_op_params.kernel_pars.brem_kernel_params.use_NN_medium_corr =
          grey_op_params.opacity_pars.use_NN_medium_corr;
      brem_kernels_m1 =
          BremKernelsLegCoeff(&grey_op_params.kernel_pars.brem_kernel_params,
                              &my_eos_params);
    } else if (opacity_pars.brem_implementation == bns_nurates::BREM_GP19) {
      brem_kernels_m1 =
          BremKernelAbsGP19(&grey_op_params.kernel_pars.brem_kernel_params,
                            &my_eos_params);
    }
  }

  bns_nurates::MyKernelOutput inelastic_kernels_m1 = {0};
  if (opacity_flags.use_inelastic_scatt) {
    grey_op_params.kernel_pars.inelastic_kernel_params.omega_prime = nu_bar;
    inelastic_kernels_m1 =
        InelasticScattKernels(&grey_op_params.kernel_pars.inelastic_kernel_params,
                              &grey_op_params.eos_pars);
  }

  for (int idx = 0; idx < total_num_species; ++idx) {
    block_factor[idx] = opacity_pars.neglect_blocking ? one : one - g_nu_bar[idx];
  }

  bns_nurates::BS_REAL pro_term[total_num_species] = {0};
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

  bns_nurates::BS_REAL ann_term[total_num_species] = {0};
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

  for (int idx = 0; idx < total_num_species; ++idx) {
    ann_term[idx] += inelastic_kernels_m1.abs[idx] * block_factor[idx];
  }

  bns_nurates::MyQuadratureIntegrand result = {0};
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
bns_nurates::MyQuadratureIntegrand RadiationIntegrateSpectral1D(
    bns_nurates::MyQuadrature* quad,
    bns_nurates::GreyOpacityParams &grey_op_params,
    const NuratesGroupOccupancy &spec,
    const bns_nurates::BS_REAL t) {
  constexpr int num_integrands = 8;
  bns_nurates::MyQuadratureIntegrand result = {0};

  result.n = num_integrands;

  // The branch sums stay separate and the x*x division follows the weight;
  // reassociating either changes the result in floating point.
  bns_nurates::BS_REAL f1_sum[num_integrands] = {0.0};
  bns_nurates::BS_REAL f2_sum[num_integrands] = {0.0};

  for (int i = 0; i < quad->nx; ++i) {
    const bns_nurates::BS_REAL x = quad->points[i];
    const bns_nurates::BS_REAL w = quad->w[i];
    const bns_nurates::MyQuadratureIntegrand f1_vals =
        RadiationSpectralIntegrand(t * x, spec, grey_op_params);
    const bns_nurates::MyQuadratureIntegrand f2_vals =
        RadiationSpectralIntegrand(t / x, spec, grey_op_params);

    for (int k = 0; k < num_integrands; ++k) {
      f1_sum[k] += w * f1_vals.integrand[k];
      f2_sum[k] += w * f2_vals.integrand[k] / (x * x);
    }
  }

  for (int k = 0; k < num_integrands; ++k) {
    result.integrand[k] = t * (f1_sum[k] + f2_sum[k]);
  }

  return result;
}

//----------------------------------------------------------------------------------------
//! \fn bns_nurates::SpectralOpacities RadiationComputeSpectralOpacities
//! \brief Spectral coefficients at energy nu (MeV): the emissivity `j`, the bare
//!        absorption coefficient `kappa`, and the elastic scattering coefficient
//!        `kappa_s`.
//!
//! `kappa_s` is already in the stimulated form the solver consumes -- see the elastic
//! block below -- so `j_s` is left at zero and nothing adds it back.  `kappa` is the one
//! coefficient still missing its stimulated correction, which
//! RadiationComputeSpectralOpacitiesStimulatedAbs supplies.

KOKKOS_INLINE_FUNCTION
bns_nurates::SpectralOpacities RadiationComputeSpectralOpacities(
    const bns_nurates::BS_REAL nu, bns_nurates::MyQuadrature* quad_1d,
    const NuratesGroupOccupancy &spec,
    bns_nurates::GreyOpacityParams* grey_op_params) {
  constexpr bns_nurates::BS_REAL zero    = 0.0;
  constexpr bns_nurates::BS_REAL four_pi = 4.0 * bns_nurates::kBS_Pi;
  constexpr bns_nurates::BS_REAL c_light = bns_nurates::kBS_Clight;

  grey_op_params->kernel_pars.pair_kernel_params.omega      = nu;
  grey_op_params->kernel_pars.brem_kernel_params.omega      = nu;
  grey_op_params->kernel_pars.inelastic_kernel_params.omega = nu;

  bns_nurates::GreyOpacityParams local_grey_params = *grey_op_params;
  local_grey_params.opacity_flags.use_inelastic_scatt = 0;

  constexpr bns_nurates::BS_REAL temp_multiple = 0.5 * 4.364;
  const bns_nurates::BS_REAL s_pair = temp_multiple * grey_op_params->eos_pars.temp;
  const bns_nurates::BS_REAL s_neps = nu;

  bns_nurates::MyQuadratureIntegrand integrals_pair_1d = {0};
  if (grey_op_params->opacity_flags.use_pair || grey_op_params->opacity_flags.use_brem) {
    integrals_pair_1d =
        RadiationIntegrateSpectral1D(quad_1d, local_grey_params, spec, s_pair);
  }

  bns_nurates::MyQuadratureIntegrand integrals_neps_1d = {0};
  if (grey_op_params->opacity_flags.use_inelastic_scatt == 1) {
    local_grey_params.opacity_flags                     = {0};
    local_grey_params.opacity_flags.use_inelastic_scatt = 1;
    // NES wants a thermal final state for its out-scattering blocking and the evolved
    // one for its in-scattering, which this single source cannot give it.  The
    // constructor makes the channel a fatal error for that reason; this branch is
    // unreachable and kept only so the integrand stays complete.
    integrals_neps_1d =
        RadiationIntegrateSpectral1D(quad_1d, local_grey_params, spec, s_neps);
  }

  bns_nurates::MyOpacity abs_em_beta = {0};
  if (grey_op_params->opacity_flags.use_abs_em) {
    abs_em_beta = AbsOpacity(nu, &grey_op_params->opacity_pars,
                             &grey_op_params->eos_pars);
  }

  bns_nurates::BS_REAL iso_scatt = zero;
  if (grey_op_params->opacity_flags.use_iso) {
    iso_scatt = IsoScattLegCoeff(nu, &grey_op_params->opacity_pars,
                                 &grey_op_params->eos_pars, 0);
  }

  bns_nurates::SpectralOpacities sp_opacities = {0};
  sp_opacities.j[id_nue] =
      abs_em_beta.em[id_nue] +
      bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[0] +
                        integrals_neps_1d.integrand[0]);
  sp_opacities.j[id_anue] =
      abs_em_beta.em[id_anue] +
      bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[1] +
                        integrals_neps_1d.integrand[1]);
  sp_opacities.j[id_nux] =
      abs_em_beta.em[id_nux] +
      bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[2] +
                        integrals_neps_1d.integrand[2]);
  sp_opacities.j[id_anux] =
      abs_em_beta.em[id_anux] +
      bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[3] +
                        integrals_neps_1d.integrand[3]);

  sp_opacities.kappa[id_nue] =
      (abs_em_beta.abs[id_nue] +
       bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[4] +
                         integrals_neps_1d.integrand[4])) / c_light;
  sp_opacities.kappa[id_anue] =
      (abs_em_beta.abs[id_anue] +
       bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[5] +
                         integrals_neps_1d.integrand[5])) / c_light;
  sp_opacities.kappa[id_nux] =
      (abs_em_beta.abs[id_nux] +
       bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[6] +
                         integrals_neps_1d.integrand[6])) / c_light;
  sp_opacities.kappa[id_anux] =
      (abs_em_beta.abs[id_anux] +
       bns_nurates::kBS_FourPi_hc3 * (integrals_pair_1d.integrand[7] +
                         integrals_neps_1d.integrand[7])) / c_light;

  // In- and out-scattering sit at the same energy for an elastic process, so the
  // occupancy cancels: j_s/c + kappa_s is this, for any distribution.  It must stay
  // exactly independent of the radiation field, so keep it as one product.
  const bns_nurates::BS_REAL kappa_s_elastic = four_pi * POW2(nu) * iso_scatt / c_light;
  for (int idx = 0; idx < total_num_species; ++idx) {
    sp_opacities.kappa_s[idx] = kappa_s_elastic;
  }

  return sp_opacities;
}

KOKKOS_INLINE_FUNCTION
bns_nurates::SpectralOpacities RadiationComputeSpectralOpacitiesStimulatedAbs(
    const bns_nurates::BS_REAL nu, bns_nurates::MyQuadrature* quad_1d,
    const NuratesGroupOccupancy &spec,
    bns_nurates::GreyOpacityParams* grey_op_params) {
  constexpr bns_nurates::BS_REAL c_light = bns_nurates::kBS_Clight;
  bns_nurates::SpectralOpacities op =
      RadiationComputeSpectralOpacities(nu, quad_1d, spec, grey_op_params);

  for (int idx = 0; idx < total_num_species; ++idx) {
    op.kappa[idx] += op.j[idx] / c_light;
  }

  return op;
}

//----------------------------------------------------------------------------------------
//! \fn void NuratesEqDensities
//! \brief Frequency-integrated equilibrium neutrino number and energy densities of a
//!        matter state, in this module's per-slot convention.
//!
//! The state is given by its chemical potentials and temperature rather than by
//! (nb, T, Ye), so the caller can hand it a *predicted* state without rebuilding the
//! opacity parameters.  Slots 2 and 3 carry mu+tau combined while bns_nurates' id_nux is
//! one species, hence the factors of two -- the same convention as the emissivities
//! everywhere else in this file.
//!
//! \param[in]  temp      temperature (MeV)
//! \param[in]  mu_n      neutron chemical potential (MeV)
//! \param[in]  mu_p      proton chemical potential (MeV)
//! \param[in]  mu_e      electron chemical potential (MeV)
//! \param[in]  unit_num_dens  eos_units.NumberDensityConversion(nurates_units)
//! \param[in]  unit_ene_dens  code_units.EnergyDensityConversion(nurates_units)
//! \param[out] n_eq      equilibrium number densities [4] (fm^-3)
//! \param[out] J_eq      equilibrium energy densities [4] (code units)
//!
//! The two conversions are passed in rather than rebuilt from the unit systems: every
//! caller already has them, and recomputing them here made the arithmetic differ from
//! the inline version this replaced in the last couple of bits, purely through where
//! the compiler chose to contract multiply-adds.
KOKKOS_INLINE_FUNCTION
void NuratesEqDensities(Real temp, Real mu_n, Real mu_p, Real mu_e,
                        Real unit_num_dens, Real unit_ene_dens,
                        Real n_eq[4], Real J_eq[4],
                        NuratesParams const &nurates_params) {
  // A live screen on the INPUTS, not an assert on the outputs, and not optional.
  // ComputeM1DensitiesEq below goes through bns_nurates' FDI_p2/FDI_p3, which return a
  // clean 0.0 for a non-finite argument -- so a NaN chemical potential or temperature
  // would come back here as a finite ZERO blackbody.  Kirchhoff's law would then emit
  // exactly nothing for that species, and every finiteness screen downstream would see
  // nothing wrong.  An assert would not do either: -DNDEBUG is on in every build that
  // does not ask for Debug.  Propagating puts those screens back in charge.
  if (!Kokkos::isfinite(temp) || !Kokkos::isfinite(mu_n) ||
      !Kokkos::isfinite(mu_p) || !Kokkos::isfinite(mu_e)) {
    const Real bad = Kokkos::Experimental::quiet_NaN_v<Real>;
    for (int idx = 0; idx < 4; ++idx) {
      n_eq[idx] = bad;
      J_eq[idx] = bad;
    }
    return;
  }

  bns_nurates::MyEOSParams eq_eos_pars = {0};
  eq_eos_pars.temp = temp;
  eq_eos_pars.mu_n = mu_n;
  eq_eos_pars.mu_p = mu_p;
  eq_eos_pars.mu_e = mu_e;

  bns_nurates::M1Quantities eq_m1_pars = {0};
  bns_nurates::NuDistributionParams eq_distr_pars = NuEquilibriumParams(&eq_eos_pars);
  ComputeM1DensitiesEq(&eq_eos_pars, &eq_distr_pars, &eq_m1_pars);

  n_eq[0] = eq_m1_pars.n[id_nue] / unit_num_dens;
  n_eq[1] = eq_m1_pars.n[id_anue] / unit_num_dens;
  n_eq[2] = 2.0 * eq_m1_pars.n[id_nux] / unit_num_dens;
  n_eq[3] = 2.0 * eq_m1_pars.n[id_anux] / unit_num_dens;

  J_eq[0] = eq_m1_pars.J[id_nue] / unit_ene_dens;
  J_eq[1] = eq_m1_pars.J[id_anue] / unit_ene_dens;
  J_eq[2] = 2.0 * eq_m1_pars.J[id_nux] / unit_ene_dens;
  J_eq[3] = 2.0 * eq_m1_pars.J[id_anux] / unit_ene_dens;

  NuratesFoldHeavySum(nurates_params.nspecies, n_eq);
  NuratesFoldHeavySum(nurates_params.nspecies, J_eq);
}

//----------------------------------------------------------------------------------------
//! \fn void NuratesEqSpectrumBin
//! \brief Bin-integrated equilibrium neutrino number and energy densities: the spectral
//!        counterpart of NuratesEqDensities.
//!
//! Deliberately the same bin, the same quadrature and the same 4pi/(hc)^3 E^2 measure
//! that bns_nurates_spectral_bin integrates the emissivity over, so that
//! eta_1 = kappa*J_eq holds bin by bin instead of only in the frequency-integrated sense.
//! That exactness is the point: the source term reads back eq_j = eta_1/kappa as the
//! spectrum it relaxes towards, so any mismatch between the two is a shift of the
//! equilibrium it converges to.
//!
//! \param[in]  e_lo_mev  lower bin edge (MeV)
//! \param[in]  e_hi_mev  upper bin edge (MeV)
//! \param[out] n_eq      bin-integrated equilibrium number densities [4] (fm^-3)
//! \param[out] J_eq      bin-integrated equilibrium energy densities [4] (code units)
KOKKOS_INLINE_FUNCTION
void NuratesEqSpectrumBin(Real e_lo_mev, Real e_hi_mev,
                          Real temp, Real mu_n, Real mu_p, Real mu_e,
                          bns_nurates::MyQuadrature const &quad,
                          Real unit_num_dens, Real unit_ene_dens,
                          Real n_eq[4], Real J_eq[4],
                          NuratesParams const &nurates_params) {
  for (int idx = 0; idx < 4; ++idx) {
    n_eq[idx] = 0.;
    J_eq[idx] = 0.;
  }
  if (e_hi_mev <= e_lo_mev) {
    return;
  }

  bns_nurates::MyEOSParams eq_eos_pars = {0};
  eq_eos_pars.temp = temp;
  eq_eos_pars.mu_n = mu_n;
  eq_eos_pars.mu_p = mu_p;
  eq_eos_pars.mu_e = mu_e;
  bns_nurates::NuDistributionParams eq_distr_pars = NuEquilibriumParams(&eq_eos_pars);

  const Real de = e_hi_mev - e_lo_mev;
  for (int iq = 0; iq < quad.nx; ++iq) {
    const Real e = e_lo_mev + de*quad.points[iq];
    const Real wde = quad.w[iq] * de * bns_nurates::kBS_FourPi_hc3 * SQR(e);
    for (int idx = 0; idx < 4; ++idx) {
      const Real f_eq = TotalNuF(e, &eq_distr_pars, &eq_eos_pars, idx);
      n_eq[idx] += wde * f_eq;
      J_eq[idx] += wde * e * f_eq;
    }
  }

  // Slots 2 and 3 are mu+tau combined; see NuratesEqDensities.
  n_eq[2] *= 2.;
  n_eq[3] *= 2.;
  J_eq[2] *= 2.;
  J_eq[3] *= 2.;

  for (int idx = 0; idx < 4; ++idx) {
    n_eq[idx] /= unit_num_dens;
    J_eq[idx] /= unit_ene_dens;
  }

  NuratesFoldHeavySum(nurates_params.nspecies, n_eq);
  NuratesFoldHeavySum(nurates_params.nspecies, J_eq);
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
  // Every number density crossing this interface -- the baryon density nb and the
  // number emissivity eta_0 -- is in the EOS number-density unit (fm^-3), as in
  // radiation_m1/.  Energy densities stay in code units, per the G = c = 1 convention.
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
  bns_nurates::GreyOpacityParams grey_op_params = {0};

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
      nurates_params.use_BRT_brem ? bns_nurates::BREM_BRT06 : bns_nurates::BREM_HR98;

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

  // Full LTE, which is the only distribution the grey variables can support: they carry
  // no independent neutrino number density to reconstruct a non-equilibrium one from,
  // and the constructor rejects use_equilibrium_distribution = false here for that
  // reason.  m1_pars is load-bearing here: ComputeM1Opacities normalises its
  // energy-averaged opacities by n and J.
  grey_op_params.distr_pars = NuEquilibriumParams(&grey_op_params.eos_pars);
  ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                       &grey_op_params.distr_pars,
                       &grey_op_params.m1_pars);
  for (int idx = 0; idx < total_num_species; ++idx) {
    grey_op_params.m1_pars.chi[idx] = 1./3.;
  }

  // compute opacities
  bns_nurates::M1Opacities opacities = ComputeM1Opacities(&nurates_params.quadrature,
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

  // The emissivity is bns_nurates' own, at (T^n, Ye^n).  The predictor is the module's
  // only equilibrium closure and it acts in the caller, overwriting eta with kappa times
  // the blackbody at the predicted (T*, Ye*); a local-blackbody Kirchhoff here would only
  // be overwritten when it is on, and is the wrong law when it is off -- bns_nurates
  // already enforces detailed balance bin by bin, and re-imposing it on frequency-
  // integrated coefficients is not equivalent out of equilibrium.
  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  *= eta0_to_eos;
    eta_1[idx]  *= eta1_to_code;
    abs_0[idx]  *= kap_to_code;
    abs_1[idx]  *= kap_to_code;
    scat_1[idx] *= kap_to_code;
    // scat_0 is set to zero above; conversion unnecessary
  }

  NuratesFoldHeavySum(nurates_params.nspecies, eta_0);
  NuratesFoldHeavySum(nurates_params.nspecies, eta_1);
  NuratesFoldHeavyMean(nurates_params.nspecies, abs_0);
  NuratesFoldHeavyMean(nurates_params.nspecies, abs_1);
  NuratesFoldHeavyMean(nurates_params.nspecies, scat_1);
}

//----------------------------------------------------------------------------------------
//! \struct NuratesCellState
//! \brief bns_nurates inputs that depend on the cell but not on the energy bin.
//!        `active` is false where the density or temperature floor rejects the cell.

struct NuratesCellState {
  bns_nurates::GreyOpacityParams grey_op_params;
  NuratesGroupOccupancy spec;
  Real unit_length;
  Real unit_time;
  Real unit_num_dens;
  Real unit_ene_dens;
  bool active;
};

//----------------------------------------------------------------------------------------
//! \fn NuratesCellState NuratesSpectralCellState
//! \brief Builds the per-cell bns_nurates state for bns_nurates_spectral_bin. Call once
//!        per cell, then pass the result to every energy bin.
//!
//! The partner-neutrino occupancy the kernels need comes from `spec`, so no moment
//! reconstruction happens here and `m1_pars` is left empty: nothing on this path reads
//! it, unlike the grey wrapper above.

KOKKOS_INLINE_FUNCTION
NuratesCellState NuratesSpectralCellState(
    Real nb, Real temp, Real yp, Real yn,
    Real mu_n, Real mu_p, Real mu_e,
    NuratesGroupOccupancy const &spec,
    NuratesParams const &nurates_params,
    Primitive::UnitSystem &code_units,
    Primitive::UnitSystem &eos_units) {
  NuratesCellState cell;
  cell.grey_op_params = bns_nurates::GreyOpacityParams{0};
  cell.spec = spec;

  Primitive::UnitSystem nurates_units = MakeNuratesUnitSystem();

  const Real unit_length    = code_units.LengthConversion(nurates_units);
  const Real unit_time      = code_units.TimeConversion(nurates_units);
  // Every number density crossing this interface -- the baryon density nb and the
  // number emissivity eta_0 -- is in the EOS number-density unit (fm^-3), as in
  // radiation_m1/.  Energy densities stay in code units, per the G = c = 1 convention.
  const Real unit_num_dens  = eos_units.NumberDensityConversion(nurates_units);
  const Real unit_ene_dens  = code_units.EnergyDensityConversion(nurates_units);

  cell.unit_length   = unit_length;
  cell.unit_time     = unit_time;
  cell.unit_num_dens = unit_num_dens;
  cell.unit_ene_dens = unit_ene_dens;

  // nb_min is in fm^-3 (the EOS number-density unit), so compare nb before the
  // conversion to bns_nurates units -- this is what radiation_m1_nurates.hpp does.
  cell.active = !((nb < nurates_params.nb_min) ||
                  (temp < nurates_params.temp_min_mev));
  if (!cell.active) {
    return cell;
  }

  bns_nurates::GreyOpacityParams &grey_op_params = cell.grey_op_params;
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
      nurates_params.use_BRT_brem ? bns_nurates::BREM_BRT06 : bns_nurates::BREM_HR98;

  grey_op_params.eos_pars.nb   = nb * unit_num_dens;
  grey_op_params.eos_pars.temp = temp;
  grey_op_params.eos_pars.yp   = yp;
  grey_op_params.eos_pars.yn   = yn;
  grey_op_params.eos_pars.mu_e = mu_e;
  grey_op_params.eos_pars.mu_p = mu_p;
  grey_op_params.eos_pars.mu_n = mu_n;
  grey_op_params.eos_pars.dU      = 0.;
  grey_op_params.eos_pars.dm_eff  = 0.;

  // The one case with no evolved spectrum to take the partner occupancy from.
  if (spec.f == nullptr) {
    grey_op_params.distr_pars = NuEquilibriumParams(&grey_op_params.eos_pars);
  }
  return cell;
}

//----------------------------------------------------------------------------------------
//! \fn void bns_nurates_spectral_bin
//! \brief Midpoint/quadrature wrapper for bns_nurates spectral opacity coefficients.
//!        The frequency bin bounds are in MeV, as freq_grid holds them; outputs are
//!        bin-integrated emissivities and bin-centered opacities in code units, except
//!        the number emissivity eta_0, which is in fm^-3 per code time.

KOKKOS_INLINE_FUNCTION
void bns_nurates_spectral_bin(NuratesCellState &cell,
                              Real e_lo_mev, Real e_hi_mev, int freq_scale,
                              Real eta_0[4], Real eta_1[4],
                              Real abs_0[4], Real abs_1[4], Real scat_1[4],
                              NuratesParams const &nurates_params) {
  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  = 0.;
    eta_1[idx]  = 0.;
    abs_0[idx]  = 0.;
    abs_1[idx]  = 0.;
    scat_1[idx] = 0.;
  }

  if (!cell.active || (e_hi_mev <= e_lo_mev)) {
    return;
  }

  bns_nurates::GreyOpacityParams &grey_op_params = cell.grey_op_params;
  const Real unit_length   = cell.unit_length;
  const Real unit_time     = cell.unit_time;
  const Real unit_num_dens = cell.unit_num_dens;
  const Real unit_ene_dens = cell.unit_ene_dens;

  const Real e_lo = e_lo_mev;
  const Real e_hi = e_hi_mev;
  const Real de = e_hi - e_lo;
  const Real e_mid = (freq_scale == 1 && e_lo > 0.0) ? sqrt(e_lo*e_hi) :
                                                         0.5*(e_lo + e_hi);

  bns_nurates::MyQuadrature quad = nurates_params.quadrature;
  bns_nurates::SpectralOpacities op_mid =
      RadiationComputeSpectralOpacitiesStimulatedAbs(e_mid, &quad, cell.spec,
                                                     &grey_op_params);

  // The emissivity, integrated over the bin.  Skipped outright when the partially-
  // equilibrated predictor sets the emissivity of every species: the caller then sets
  // eta from kappa times the blackbody at (T*, Ye*), so every one of these quadrature
  // points -- a full spectral opacity evaluation each -- would be thrown away.  eta_*
  // are left at the zeros set above; the caller overwrites them.
  if (!nurates_params.use_partial_equilibrium || !nurates_params.peq_heavy) {
    if (nurates_params.heavy_eps_midpoint) {
      for (int idx = 0; idx < 4; ++idx) {
        eta_0[idx] = de * bns_nurates::kBS_FourPi_hc3 * SQR(e_mid) * op_mid.j[idx];
        eta_1[idx] = de * bns_nurates::kBS_FourPi_hc3 * SQR(e_mid) * e_mid * op_mid.j[idx];
      }
    } else {
      for (int iq = 0; iq < quad.nx; ++iq) {
        const Real x = quad.points[iq];
        const Real w = quad.w[iq];
        const Real e = e_lo + de*x;
        bns_nurates::SpectralOpacities op =
            RadiationComputeSpectralOpacitiesStimulatedAbs(e, &quad, cell.spec,
                                                           &grey_op_params);
        for (int idx = 0; idx < 4; ++idx) {
          eta_0[idx] += w * de * bns_nurates::kBS_FourPi_hc3 * SQR(e) * op.j[idx];
          eta_1[idx] += w * de * bns_nurates::kBS_FourPi_hc3 * SQR(e) * e * op.j[idx];
        }
      }
    }

    eta_0[2] *= 2.;
    eta_0[3] *= 2.;
    eta_1[2] *= 2.;
    eta_1[3] *= 2.;
  }

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

  NuratesFoldHeavySum(nurates_params.nspecies, eta_0);
  NuratesFoldHeavySum(nurates_params.nspecies, eta_1);
  NuratesFoldHeavyMean(nurates_params.nspecies, abs_0);
  NuratesFoldHeavyMean(nurates_params.nspecies, abs_1);
  NuratesFoldHeavyMean(nurates_params.nspecies, scat_1);
}

} // namespace radiation

#endif  // ENABLE_NURATES

#endif  // RADIATION_RADIATION_NURATES_HPP_
