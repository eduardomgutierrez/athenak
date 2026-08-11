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
  Real nb_min;           // minimum baryon number density [bns_nurates units] below which
                         //   opacities are set to zero
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
//! \param[in]  nudens_0    neutrino number densities [4] (code units, per species)
//! \param[in]  nudens_1    neutrino energy densities [4] (code units, per species)
//! \param[out] eta_0       number emissivity [nspecies] (code units)
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
  const Real unit_eos_num_dens  = eos_units.DensityConversion(nurates_units);
  const Real unit_code_num_dens = code_units.DensityConversion(nurates_units);
  const Real unit_ene_dens  = code_units.EnergyDensityConversion(nurates_units);

  // zero outputs if below floor values
  if ((nb * unit_eos_num_dens < nurates_params.nb_min) ||
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
  grey_op_params.eos_pars.nb   = nb * unit_eos_num_dens;  // [baryon/nm^3]
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
    grey_op_params.m1_pars.n[id_nue]  = nudens_0[0] * unit_code_num_dens;
    grey_op_params.m1_pars.n[id_anue] = nudens_0[1] * unit_code_num_dens;
    grey_op_params.m1_pars.n[id_nux]  = 0.5 * nudens_0[2] * unit_code_num_dens;
    grey_op_params.m1_pars.n[id_anux] = 0.5 * nudens_0[3] * unit_code_num_dens;

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
  //   number emissivity [code_number_density / time]:
  //     eta_0_nuclear = eta_0_code * (unit_code_num_dens / unit_time)
  //     => eta_0_code = eta_0_nuclear * unit_time / unit_code_num_dens
  const Real kap_to_code  = unit_length;
  const Real eta1_to_code = unit_time / unit_ene_dens;
  const Real eta0_to_code = unit_time / unit_code_num_dens;

  Real eq_n_code[4] = {0., 0., 0., 0.};
  Real eq_J_code[4] = {0., 0., 0., 0.};
  if (nurates_params.use_kirchhoff_law) {
    M1Quantities eq_m1_pars = {0};
    NuDistributionParams eq_distr_pars =
        NuEquilibriumParams(&grey_op_params.eos_pars);
    ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                         &eq_distr_pars,
                         &eq_m1_pars);

    eq_n_code[0] = eq_m1_pars.n[id_nue] / unit_code_num_dens;
    eq_n_code[1] = eq_m1_pars.n[id_anue] / unit_code_num_dens;
    eq_n_code[2] = 2.0 * eq_m1_pars.n[id_nux] / unit_code_num_dens;
    eq_n_code[3] = 2.0 * eq_m1_pars.n[id_anux] / unit_code_num_dens;

    eq_J_code[0] = eq_m1_pars.J[id_nue] / unit_ene_dens;
    eq_J_code[1] = eq_m1_pars.J[id_anue] / unit_ene_dens;
    eq_J_code[2] = 2.0 * eq_m1_pars.J[id_nux] / unit_ene_dens;
    eq_J_code[3] = 2.0 * eq_m1_pars.J[id_anux] / unit_ene_dens;
  }

  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  *= eta0_to_code;
    eta_1[idx]  *= eta1_to_code;
    abs_0[idx]  *= kap_to_code;
    abs_1[idx]  *= kap_to_code;
    scat_1[idx] *= kap_to_code;
    // scat_0 is set to zero above; conversion unnecessary

    if (nurates_params.use_kirchhoff_law) {
      eta_0[idx] = (abs_0[idx] > 0.0) ? abs_0[idx] * eq_n_code[idx] : eta_0[idx];
      eta_1[idx] = (abs_1[idx] > 0.0) ? abs_1[idx] * eq_J_code[idx] : eta_1[idx];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void bns_nurates_spectral_bin
//! \brief Midpoint/quadrature wrapper for bns_nurates spectral opacity coefficients.
//!        The frequency bin bounds are code-energy values; outputs are bin-integrated
//!        emissivities and bin-centered opacities in code units.
KOKKOS_INLINE_FUNCTION
void bns_nurates_spectral_bin(Real e_lo_code, Real e_hi_code,
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
  const Real unit_eos_num_dens  = eos_units.DensityConversion(nurates_units);
  const Real unit_code_num_dens = code_units.DensityConversion(nurates_units);
  const Real unit_ene_dens  = code_units.EnergyDensityConversion(nurates_units);
  const Real unit_energy    = code_units.EnergyConversion(nurates_units);

  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  = 0.;
    eta_1[idx]  = 0.;
    abs_0[idx]  = 0.;
    abs_1[idx]  = 0.;
    scat_1[idx] = 0.;
  }

  if ((nb * unit_eos_num_dens < nurates_params.nb_min) ||
      (temp < nurates_params.temp_min_mev) ||
      (e_hi_code <= e_lo_code)) {
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

  grey_op_params.eos_pars.nb   = nb * unit_eos_num_dens;
  grey_op_params.eos_pars.temp = temp;
  grey_op_params.eos_pars.yp   = yp;
  grey_op_params.eos_pars.yn   = yn;
  grey_op_params.eos_pars.mu_e = mu_e;
  grey_op_params.eos_pars.mu_p = mu_p;
  grey_op_params.eos_pars.mu_n = mu_n;
  grey_op_params.eos_pars.dU      = 0.;
  grey_op_params.eos_pars.dm_eff  = 0.;

  if (nurates_params.use_equilibrium_distribution) {
    grey_op_params.distr_pars = NuEquilibriumParams(&grey_op_params.eos_pars);
    ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                         &grey_op_params.distr_pars,
                         &grey_op_params.m1_pars);
    for (int idx = 0; idx < 4; ++idx) {
      grey_op_params.m1_pars.chi[idx] = 1./3.;
    }
  } else {
    grey_op_params.m1_pars.n[id_nue]  = nudens_0[0] * unit_code_num_dens;
    grey_op_params.m1_pars.n[id_anue] = nudens_0[1] * unit_code_num_dens;
    grey_op_params.m1_pars.n[id_nux]  = 0.5 * nudens_0[2] * unit_code_num_dens;
    grey_op_params.m1_pars.n[id_anux] = 0.5 * nudens_0[3] * unit_code_num_dens;

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

  const Real e_lo = e_lo_code * unit_energy;
  const Real e_hi = e_hi_code * unit_energy;
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
  const Real eta0_to_code = unit_time / unit_code_num_dens;
  const Real eta1_to_code = unit_time / unit_ene_dens;

  for (int idx = 0; idx < 4; ++idx) {
    eta_0[idx]  *= eta0_to_code;
    eta_1[idx]  *= eta1_to_code;
    abs_0[idx]   = op_mid.kappa[idx] * kap_to_code;
    abs_1[idx]   = op_mid.kappa[idx] * kap_to_code;
    scat_1[idx]  = op_mid.kappa_s[idx] * kap_to_code;
  }
}

} // namespace radiation

#endif  // ENABLE_NURATES

#endif  // RADIATION_RADIATION_NURATES_HPP_
