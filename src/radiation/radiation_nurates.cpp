//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_nurates.cpp
//! \brief Calculates gray neutrino opacities using the bns_nurates library.
//!        Opacities are computed once per RK stage and stored in arrays.

#include "athena.hpp"
#include "config.hpp"

#if ENABLE_NURATES

#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation.hpp"
#include "radiation_nurates.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn void PredictPartialEquilibrium
//! \brief The partially-equilibrated (T*, Ye*) state the matter is predicted to reach
//!        over one step, given the neutrino field it is in contact with.
//!
//! A one-parameter family in w = a/(1+a), a = dtau*kappa, evaluated per cell and per
//! channel (nu_e pair energy, heavy pair energy, net lepton number).  At w = 1 the
//! residuals are the fully-trapped weak equilibrium's exactly; at w = 0 they return
//! (T, Ye), the local blackbody.  The interpolation is fixed by the step and the local
//! opacity, not by a tuned threshold, and every intermediate point is a valid scheme --
//! which is what lets a rejected root retry with the weights halved.
//!
//! Ported from radiation_m1/radiation_m1_calc_opacities_nurates.cpp; the gates, the
//! trust region and the fallback ladder are deliberately identical, so the two solvers
//! predict the same state from the same inputs.
//!
//! On entry T_star, Ye_star must hold (T, Ye); they are left untouched for a cell that
//! is gated out or that never produces an accepted root.
//!
//! \param[in]  dtau      proper time of the step, alpha*dt/W
//! \param[in]  J_e,J_x   comoving energy density of the electron / heavy pairs (code)
//! \param[in]  N_L       comoving net electron lepton number density (fm^-3)
//! \param[in]  *_eq      the same three, for the local blackbody at (T, Ye)
//! \param[in]  kbar_*    field-weighted mean absorption opacities, one per channel
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void PredictPartialEquilibrium(const Primitive::EOS<EOSPolicy, ErrorPolicy> &eos,
                               Real nb, Real T, Real Y, Real dtau,
                               Real J_e, Real J_x, Real N_L,
                               Real J_e_eq, Real J_x_eq, Real N_L_eq,
                               Real kbar_1e, Real kbar_1x, Real kbar_0e,
                               NuratesParams const &nurates_params,
                               Primitive::UnitSystem &code_units,
                               Primitive::UnitSystem &eos_units,
                               Real &T_star, Real &Ye_star) {
  // Half-width of the tier-1 c_v secant, relative to T; a gate, not a Jacobian, so a
  // crude c_v is enough.
  const Real peq_cv_eps = 1.0e-2;
  // How far outside the tier-1 linear bound a root may land and still be believed.
  const Real peq_trust_c = 10.0;
  // Halvings of the weights allowed before the cell is declared unusable.
  const int peq_max_halvings = 4;

  const Real a_1e = dtau*kbar_1e;
  const Real a_1x = dtau*kbar_1x;
  const Real a_0e = dtau*kbar_0e;
  const Real w_1e = a_1e/(1.0 + a_1e);
  const Real w_1x = a_1x/(1.0 + a_1x);
  const Real w_0e = a_0e/(1.0 + a_0e);

  // Tier-0 gate: no EOS calls at all.  An optically thin cell has nothing to
  // equilibrate with and must cost nothing.  Ternaries not fmax, per
  // eos_compose.hpp:186 (SYCL's fmax(x, NaN) = NaN), so a NaN weight gates the cell out
  // deliberately, not by luck.
  const bool w_finite = Kokkos::isfinite(w_1e) && Kokkos::isfinite(w_1x) &&
                        Kokkos::isfinite(w_0e);
  Real w_max = (w_1e > w_1x) ? w_1e : w_1x;
  w_max = (w_max > w_0e) ? w_max : w_0e;
  if (!(w_finite && w_max >= nurates_params.peq_w_floor)) {
    return;
  }

  Real Y_part[3] = {Y, 0.0, 0.0};

  // Tier-1 gate: first-order bounds on the excursion the solve would produce, from the
  // blackbody already in hand and c_v.  A cell already at equilibrium must also cost
  // nothing.  T is clamped to the table because weight_idx_lt clamps the index but not
  // the interpolation weight, so an out-of-range T extrapolates; dividing by the actual
  // T_hi - T_lo keeps a one-sided secant at the edge correct.
  const Real T_tab_min =
      eos.GetMinimumTemperature()*eos_units.TemperatureConversion(code_units);
  const Real T_tab_max =
      eos.GetMaximumTemperature()*eos_units.TemperatureConversion(code_units);
  Real T_lo = T*(1.0 - peq_cv_eps);
  Real T_hi = T*(1.0 + peq_cv_eps);
  T_lo = (T_lo > T_tab_min) ? T_lo : T_tab_min;
  T_hi = (T_hi < T_tab_max) ? T_hi : T_tab_max;
  const Real cv = (T_hi > T_lo)
                      ? (eos.GetEnergy(nb, T_hi, Y_part) -
                         eos.GetEnergy(nb, T_lo, Y_part))/(T_hi - T_lo)
                      : 0.0;
  const bool cv_ok = Kokkos::isfinite(cv) && cv > 0.0;

  const Real dlnT_hat = cv_ok ? (w_1e*Kokkos::fabs(J_e_eq - J_e) +
                                 w_1x*Kokkos::fabs(J_x_eq - J_x))/(T*cv)
                              : 0.0;
  const Real dYe_hat = w_0e*Kokkos::fabs(N_L_eq - N_L)/nb;

  // A bad c_v removes the gate and the trust region both -- everything below divides by
  // T*cv.  Predict nothing instead.
  if (!(cv_ok && !(dlnT_hat < nurates_params.peq_dlnT_tol &&
                   dYe_hat < nurates_params.peq_dYe_tol))) {
    return;
  }

  // Trust region.  The gate estimates do double duty: a root far outside the linear
  // bound is a converged-but-wrong root -- the failure mode that matters, since a small
  // energy residual admits a badly wrong T wherever the thermal energy is a small
  // fraction of the total.  The floor at the gate tolerances lets an already-
  // equilibrated cell still move imperceptibly, and is where the ternaries' NaN branch
  // lands: reject, not admit.
  const Real dlnT_trust = peq_trust_c*dlnT_hat;
  const Real dYe_trust = peq_trust_c*dYe_hat;
  const Real dlnT_max = (dlnT_trust > nurates_params.peq_dlnT_tol)
                            ? dlnT_trust : nurates_params.peq_dlnT_tol;
  const Real dYe_max = (dYe_trust > nurates_params.peq_dYe_tol)
                           ? dYe_trust : nurates_params.peq_dYe_tol;

  const Real e_mat = eos.GetEnergy(nb, T, Y_part);

  // On failure, halve all three weights and retry: that slides the problem along the
  // same one-parameter family toward the trivial one, so every intermediate point is
  // still a valid scheme.  A cell that never produces an accepted root keeps (T, Y_e).
  Real f_soft = 1.0;
  for (int n_soft = 0; n_soft <= peq_max_halvings; ++n_soft, f_soft *= 0.5) {
    const Real u_1e = f_soft*w_1e;
    const Real u_1x = f_soft*w_1x;
    const Real u_0e = f_soft*w_0e;

    const Real e_rhs = e_mat + u_1e*J_e + u_1x*J_x;
    Real Yl_rhs[3] = {Y + u_0e*N_L/nb, 0.0, 0.0};

    Real T_try = T;
    Real Ye_try[3] = {Y, 0.0, 0.0};
    bool ok = eos.GetBetaEquilibriumPartial(nb, e_rhs, Yl_rhs, u_1e, u_1x, u_0e,
                                            T_try, &Ye_try[0], T, Y_part);

    if (ok && Kokkos::fabs(Kokkos::log(T_try/T)) <= dlnT_max &&
        Kokkos::fabs(Ye_try[0] - Y) <= dYe_max) {
      T_star = T_try;
      Ye_star = Ye_try[0];
      return;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::CalcOpacityNurates
//! \brief Entry point for nurates opacity calculation (dispatches to templated version)

TaskStatus Radiation::CalcOpacityNurates(Driver *pdrive, int stage) {
  if (!use_nurates) {
    return TaskStatus::complete;
  }

  // Opacities are constant within a timestep; only compute at stage 1
  if (stage > 1) {
    return TaskStatus::complete;
  }

  // Toy opacities need no EOS, so they short-circuit the dispatch below.
  if (nurates_toy_scattering >= 0.0) {
    return CalcOpacityNuratesToy(pdrive, stage);
  }

  auto *pmy_pack = this->pmy_pack;

  // Dispatch based on EOS type (mirrors the pattern in dyn_grmhd.cpp constructor)
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                      Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                       Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  // If no supported EOS is found, abort: nurates requires a tabulated CompOSE EOS.
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "use_nurates=true but no supported EOSCompOSE found. "
            << "Check that DynGRMHD is using EOSCompOSE<NQTLogs> or "
            << "EOSCompOSE<NormalLogs>." << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::CalcOpacityNuratesToy
//! \brief Elastic scattering with a power-law energy dependence, no absorption or
//! emission.  Lets the nurates source terms be driven by a prescribed opacity instead of
//! the library, which is what the diffusion test needs; mirrors
//! <radiation_m1>/opacity_type = toy.
//!
//! sigma_s(e) = scat*(e_mid/e_ref)^p on the comoving grid.  p = 0 reproduces the
//! constant opacity exactly and leaves the grey diffusion test untouched.  The grey
//! slot always carries the p = 0 value: with p != 0 there is no single grey opacity
//! that reproduces the per-group answer, which is the point of the test.

TaskStatus Radiation::CalcOpacityNuratesToy(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nsp_ = nspecies;
  int nfreq_ = nfreq;
  bool multi_freq_ = multi_freq;
  int fscale_ = flag_fscale;
  Real scat_ = nurates_toy_scattering;
  Real scat_p_ = nurates_toy_scat_p;
  Real scat_eref_ = nurates_toy_scat_eref;
  auto &nu_tet_ = freq_grid;

  auto &eta_0_ = nurates_eta_0;
  auto &eta_1_ = nurates_eta_1;
  auto &abs_0_ = nurates_abs_0;
  auto &abs_1_ = nurates_abs_1;
  auto &scat_1_ = nurates_scat_1;

  par_for("rad_nurates_toy_opacity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int isp = 0; isp < nsp_; ++isp) {
      eta_0_(m,isp,k,j,i) = 0.0;
      eta_1_(m,isp,k,j,i) = 0.0;
      abs_0_(m,isp,k,j,i) = 0.0;
      abs_1_(m,isp,k,j,i) = 0.0;
      scat_1_(m,isp,k,j,i) = scat_;
    }
  });

  if (multi_freq_) {
    auto &eta_0_f_ = nurates_eta_0_freq;
    auto &eta_1_f_ = nurates_eta_1_freq;
    auto &abs_0_f_ = nurates_abs_0_freq;
    auto &abs_1_f_ = nurates_abs_1_freq;
    auto &scat_1_f_ = nurates_scat_1_freq;
    par_for("rad_nurates_toy_opacity_freq", DevExeSpace(), 0, nmb1, ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int isp = 0; isp < nsp_; ++isp) {
        for (int ifr = 0; ifr < nfreq_; ++ifr) {
          Real ss = scat_;
          if (scat_p_ != 0.0) {
            Real e_lo = 0.0, e_hi = 0.0;
            FreqBinEdgesMeV(nu_tet_, ifr, nfreq_, fscale_, e_lo, e_hi);
            Real e_mid = FreqBinMidMeV(e_lo, e_hi, fscale_);
            ss = (e_mid > 0.0)
                 ? scat_*Kokkos::pow(e_mid/scat_eref_, scat_p_) : scat_;
          }
          eta_0_f_(m,isp,ifr,k,j,i) = 0.0;
          eta_1_f_(m,isp,ifr,k,j,i) = 0.0;
          abs_0_f_(m,isp,ifr,k,j,i) = 0.0;
          abs_1_f_(m,isp,ifr,k,j,i) = 0.0;
          scat_1_f_(m,isp,ifr,k,j,i) = ss;
        }
      }
    });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::CalcOpacityNurates_<EOSPolicy, ErrorPolicy>
//! \brief Templated implementation: fills eta_0_, abs_1_, scat_1_ arrays

template <class EOSPolicy, class ErrorPolicy>
TaskStatus Radiation::CalcOpacityNurates_(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &nspecies_ = nspecies;
  int nfreq_ = nfreq;
  bool multi_freq_ = multi_freq;
  int freq_scale_ = flag_fscale;
  int nang_ = prgeo->nangles;
  int nang1 = nang_ - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;

  // EOS reference (needed for T, yp, yn, mu_* in kernel)
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();
  nurates_baryon_mass = mb;
  nurates_code_edens_to_eos =
      eos.GetCodeUnitSystem().EnergyDensityConversion(eos.GetEOSUnitSystem());

  // Unit systems
  auto code_units    = eos.GetCodeUnitSystem();
  auto eos_units_loc = eos.GetEOSUnitSystem();
  // The two conversions the equilibrium helpers need, built once on the host: the
  // number densities crossing the nurates interface are in fm^-3 and the energy
  // densities in code units, so these are the only two the blackbody has to cross.
  auto nurates_units_h = MakeNuratesUnitSystem();
  const Real unit_num_dens_ = eos_units_loc.NumberDensityConversion(nurates_units_h);
  const Real unit_ene_dens_ = code_units.EnergyDensityConversion(nurates_units_h);
  // Neutrino number densities are carried in the EOS number-density unit (fm^-3)
  // throughout the nurates interface, as in radiation_m1/.  The multi-frequency
  // solver builds them from the intensity moments, so the conversion is applied
  // once, at the single production site below: a code-unit energy density in
  // MeV/fm^3, over a bin energy in MeV, is a number density in fm^-3.
  const Real code_edens_to_eos = nurates_code_edens_to_eos;
  bool debug_opacity_ = nurates_debug_opacity;

  // Nurates params (captured by value for use inside KOKKOS_LAMBDA)
  auto nurates_params_ = nurates_params;
  const bool peq_on_ = nurates_params_.use_partial_equilibrium;
  // A FULL step, not the stage's beta*dt: this function returns early for stage > 1, so
  // the emissivity it computes serves the whole cycle.  With rk2, beta = {1, 1/2} and
  // gam0 = {0, 1/2}, so the source applied over a cycle is exactly dt.
  const Real dt_ = pmy_pack->pmesh->dt;

  // Primitives: for DynGRMHD with MHD, primitives are in pmhd->w0; with hydro in phydro->w0
  DvceArray5D<Real> w0_arr;
  if (this->is_mhd_enabled) {
    w0_arr = pmy_pack->pmhd->w0;
  } else if (this->is_hydro_enabled) {
    w0_arr = pmy_pack->phydro->w0;
  }

  // Output opacity arrays
  auto &eta_0_   = nurates_eta_0;
  auto &eta_1_   = nurates_eta_1;
  auto &abs_0_   = nurates_abs_0;
  auto &abs_1_   = nurates_abs_1;
  auto &scat_1_  = nurates_scat_1;
  auto &eta_0_f_   = nurates_eta_0_freq;
  auto &eta_1_f_   = nurates_eta_1_freq;
  auto &abs_0_f_   = nurates_abs_0_freq;
  auto &abs_1_f_   = nurates_abs_1_freq;
  auto &scat_1_f_  = nurates_scat_1_freq;
  auto &i0_ = i0;
  auto &freq_grid_ = freq_grid;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;

  par_for("radiation_calc_opacity_nurates", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // fluid primitives
    Real wdn = w0_arr(m, IDN, k, j, i);
    Real wen = w0_arr(m, IEN, k, j, i);  // pressure for DynGR (Valencia)

    // baryon number density (code units -> EOS)
    Real nb = wdn / mb;

    // temperature [MeV]: use pressure stored in IEN for DynGR Valencia formulation
    Real p  = wen;  // DynGR: IEN = pressure
    Real Ye = w0_arr(m, IYF, k, j, i);
    Real T  = eos.GetTemperatureFromP(nb, p, &Ye);

    // proton and neutron fractions from EOS table
    Real yp = eos.GetProtonFraction(nb, T, &Ye);
    Real yn = eos.GetNeutronFraction(nb, T, &Ye);

    // chemical potentials [MeV]
    Real mu_b  = eos.GetBaryonChemicalPotential(nb, T, &Ye);
    Real mu_q  = eos.GetChargeChemicalPotential(nb, T, &Ye);
    Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &Ye);
    Real mu_n  = mu_b;
    Real mu_p  = mu_b + mu_q;
    Real mu_e  = mu_le - mu_q;

    // Neutrino number densities in fm^-3, energy densities in code units.  For
    // multifrequency transport, derive number density directly from the
    // bin-integrated energy.
    Real nudens_0[4] = {0., 0., 0., 0.};
    Real nudens_1[4] = {0., 0., 0., 0.};

    // Make local mutable copies for unit conversion calls
    auto code_units_l   = code_units;
    auto eos_units_loc_l = eos_units_loc;

    // The equilibrium at the START of the step, (T^n, Ye^n).  Three jobs: the tier-1
    // gate's reference state, the fallback for a cell the predictor declines, and --
    // through the mean energy J_eq/n_eq -- the only handle the grey path has on a
    // neutrino number density, since its variables carry no independent one.
    Real n_eq_loc[4] = {0., 0., 0., 0.};
    Real J_eq_loc[4] = {0., 0., 0., 0.};
    if (peq_on_) {
      NuratesEqDensities(T, mu_n, mu_p, mu_e, unit_num_dens_, unit_ene_dens_,
                         n_eq_loc, J_eq_loc);
    }

    // The comoving moments of the ACTUAL neutrino field, and the geometry they need.
    // The predictor wants them whatever use_equilibrium_distribution says -- they are
    // what the matter is in contact with -- while that flag governs only whether the
    // reconstruction is handed on to bns_nurates below.
    Real alpha_cell = 1.0;
    Real w_lorentz = 1.0;
    // Hoisted out of the block below: the multi-frequency weights re-sum the angles
    // bin by bin, once the opacities are known, and need the same ray geometry.
    Real u_tet[4] = {1.0, 0.0, 0.0, 0.0};
    Real n0 = 1.0;
    Real wght_sum = 1.0;
    const bool need_moments =
        peq_on_ || (multi_freq_ && !nurates_params_.use_equilibrium_distribution);
    if (need_moments) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
      Real wvx = w0_arr(m, IVX, k, j, i);
      Real wvy = w0_arr(m, IVY, k, j, i);
      Real wvz = w0_arr(m, IVZ, k, j, i);
      Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy +
               2.0*glower[1][3]*wvx*wvz + glower[2][2]*wvy*wvy +
               2.0*glower[2][3]*wvy*wvz + glower[3][3]*wvz*wvz;

      Real gamma = sqrt(fmax(1.0, 1.0 + q));
      alpha_cell = sqrt(-1.0/gupper[0][0]);
      w_lorentz = gamma;
      u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                  norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
      u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                  norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
      u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                  norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
      u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                  norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);

      n0 = tt(m,0,0,k,j,i);
      wght_sum = 0.0;
      for (int iang = 0; iang <= nang1; ++iang) {
        Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                      u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
        wght_sum += solid_angles_.d_view(iang)/SQR(n0_cm);
      }
      if (multi_freq_) {
        for (int isp = 0; isp < nspecies_; ++isp) {
          int sp_off = isp*nfreq_*nang_;
          for (int ifr = 0; ifr < nfreq_; ++ifr) {
            Real e_lo = 0.0, e_hi = 0.0;
            FreqBinEdgesMeV(freq_grid_, ifr, nfreq_, freq_scale_, e_lo, e_hi);
            Real e_mid = FreqBinMidMeV(e_lo, e_hi, freq_scale_);
            for (int iang = 0; iang <= nang1; ++iang) {
              int n = sp_off + ifr*nang_ + iang;
              Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                         tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                         tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                         tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
              Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) -
                            u_tet[1]*nh_c_.d_view(iang,1) -
                            u_tet[2]*nh_c_.d_view(iang,2) -
                            u_tet[3]*nh_c_.d_view(iang,3));
              Real omega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
              Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
              nudens_1[isp] += intensity_cm*omega_cm;
              nudens_0[isp] += intensity_cm*omega_cm/fmax(n0_cm*e_mid, 1.0e-100);
            }
          }
          nudens_1[isp] /= wght_sum;
          nudens_0[isp] /= wght_sum;
          nudens_0[isp] *= code_edens_to_eos;
        }
      } else {
        // Grey: an angular sum and nothing else.  The bin midpoint that turns energy
        // into number above has no counterpart here, so the number density comes from
        // the mean energy of the local blackbody, J_eq/n_eq -- the same proxy
        // RadFluidCouplingNurates uses to turn its own energy update into a number
        // update, so the predictor and the source term agree on what N means.  Taken at
        // (T^n, Ye^n) deliberately: drawing it from (T*, Ye*) would make the weights
        // depend on the state they are used to predict.
        for (int isp = 0; isp < nspecies_; ++isp) {
          int sp_off = isp*nang_;
          for (int iang = 0; iang <= nang1; ++iang) {
            int n = sp_off + iang;
            Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                       tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                       tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                       tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
            Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) -
                          u_tet[1]*nh_c_.d_view(iang,1) -
                          u_tet[2]*nh_c_.d_view(iang,2) -
                          u_tet[3]*nh_c_.d_view(iang,3));
            Real omega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
            Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
            nudens_1[isp] += intensity_cm*omega_cm;
          }
          nudens_1[isp] /= wght_sum;
          // n_eq and J_eq are in fm^-3 and code units respectively, exactly as
          // nudens_0 and nudens_1 are, so the ratio needs no conversion factor.
          nudens_0[isp] = (J_eq_loc[isp] > 0.0)
                          ? nudens_1[isp]*n_eq_loc[isp]/J_eq_loc[isp] : 0.0;
        }
      }
    }

    if (!multi_freq_) {
      // output arrays
      Real loc_eta_0[4]  = {0.};
      Real loc_eta_1[4]  = {0.};
      Real loc_abs_0[4]  = {0.};
      Real loc_abs_1[4]  = {0.};
      Real loc_scat_0[4] = {0.};
      Real loc_scat_1[4] = {0.};

      bns_nurates_gray(nb, T, yp, yn, mu_n, mu_p, mu_e,
                       nudens_0, nudens_1,
                       loc_eta_0, loc_eta_1,
                       loc_abs_0, loc_abs_1,
                       loc_scat_0, loc_scat_1,
                       nurates_params_, code_units_l, eos_units_loc_l);

      // assert no NaN/Inf in output opacities
      for (int idx = 0; idx < 4; ++idx) {
        assert(Kokkos::isfinite(loc_eta_0[idx]));
        assert(Kokkos::isfinite(loc_eta_1[idx]));
        assert(Kokkos::isfinite(loc_abs_0[idx]));
        assert(Kokkos::isfinite(loc_abs_1[idx]));
        assert(Kokkos::isfinite(loc_scat_1[idx]));
      }

      // Partially-equilibrated (T*, Ye*) predictor, and Kirchhoff's law against the
      // blackbody there.  The opacities keep the state they were tabulated at; only the
      // emissivity moves.  That is what makes the exchange saturate as dt grows,
      // instead of emitting at the start-of-step rate for the whole step.
      if (peq_on_) {
        // Pair-averaged ABSORPTION opacities.  Elastic scattering neither thermalises
        // the energy nor changes the number density, so scat_1 has no business here.
        // The field-weighted mean makes kappa_bar*J equal the sum of the per-species
        // kappa_x*J_x exactly at t^n; with an empty field there is nothing to weight
        // with, so the arithmetic mean stands in.  These are the same coefficients the
        // source term relaxes at, so the weight a = dtau*kappa is the step measured in
        // units of the relaxation time the solver actually uses.
        Real J_e = nudens_1[0] + nudens_1[1];
        Real n_e = nudens_0[0] + nudens_0[1];
        Real N_L = nudens_0[0] - nudens_0[1];
        Real kJ_e = loc_abs_1[0]*nudens_1[0] + loc_abs_1[1]*nudens_1[1];
        Real kN_e = loc_abs_0[0]*nudens_0[0] + loc_abs_0[1]*nudens_0[1];
        Real J_x = 0.0, kJ_x = 0.0, ks_x = 0.0, J_x_eq = 0.0;
        int n_x = 0;
        for (int isp = 2; isp < nspecies_; ++isp) {
          J_x += nudens_1[isp];
          kJ_x += loc_abs_1[isp]*nudens_1[isp];
          ks_x += loc_abs_1[isp];
          J_x_eq += J_eq_loc[isp];
          ++n_x;
        }
        Real kbar_1e = (J_e > 0.0) ? kJ_e/J_e : 0.5*(loc_abs_1[0] + loc_abs_1[1]);
        Real kbar_0e = (n_e > 0.0) ? kN_e/n_e : 0.5*(loc_abs_0[0] + loc_abs_0[1]);
        Real kbar_1x = (n_x == 0) ? 0.0 : ((J_x > 0.0) ? kJ_x/J_x : ks_x/n_x);

        Real T_star = T;
        Real Ye_star = Ye;
        PredictPartialEquilibrium(eos, nb, T, Ye, dt_*alpha_cell/w_lorentz,
                                  J_e, J_x, N_L,
                                  J_eq_loc[0] + J_eq_loc[1], J_x_eq,
                                  n_eq_loc[0] - n_eq_loc[1],
                                  kbar_1e, kbar_1x, kbar_0e,
                                  nurates_params_, code_units_l, eos_units_loc_l,
                                  T_star, Ye_star);

        // The equilibrium the cell is predicted to be radiating towards.  Evaluated
        // unconditionally: a gated or declined cell has (T*, Ye*) = (T, Ye), so this
        // reproduces the local blackbody bit for bit and the w -> 0 limit costs no
        // special case.
        Real Ye_arr[3] = {Ye_star, 0.0, 0.0};
        Real mu_b_s  = eos.GetBaryonChemicalPotential(nb, T_star, Ye_arr);
        Real mu_q_s  = eos.GetChargeChemicalPotential(nb, T_star, Ye_arr);
        Real mu_le_s = eos.GetElectronLeptonChemicalPotential(nb, T_star, Ye_arr);
        Real n_eq_s[4] = {0., 0., 0., 0.};
        Real J_eq_s[4] = {0., 0., 0., 0.};
        NuratesEqDensities(T_star, mu_b_s, mu_b_s + mu_q_s, mu_le_s - mu_q_s,
                           unit_num_dens_, unit_ene_dens_, n_eq_s, J_eq_s);

        // Finiteness screen.  Nothing downstream catches this: kappa*NaN would go
        // straight into eta and on into the source term.  Fall back to the local
        // blackbody, the w -> 0 answer.
        bool eq_finite = true;
        for (int idx = 0; idx < 4; ++idx) {
          eq_finite = eq_finite && Kokkos::isfinite(n_eq_s[idx]) &&
                      Kokkos::isfinite(J_eq_s[idx]);
        }

        // Kirchhoff's law, and the only place the emissivity is set on this path --
        // bns_nurates_gray leaves it alone whenever the predictor is on.
        for (int isp = 0; isp < nspecies_; ++isp) {
          Real my_n = eq_finite ? n_eq_s[isp] : n_eq_loc[isp];
          Real my_J = eq_finite ? J_eq_s[isp] : J_eq_loc[isp];
          loc_eta_0[isp] = (loc_abs_0[isp] > 0.0) ? loc_abs_0[isp]*my_n : loc_eta_0[isp];
          loc_eta_1[isp] = (loc_abs_1[isp] > 0.0) ? loc_abs_1[isp]*my_J : loc_eta_1[isp];
        }
      }

      // store per-species opacities
      for (int isp = 0; isp < nspecies_; ++isp) {
        eta_0_(m, isp, k, j, i)  = loc_eta_0[isp];
        eta_1_(m, isp, k, j, i)  = loc_eta_1[isp];
        abs_0_(m, isp, k, j, i)  = loc_abs_0[isp];
        abs_1_(m, isp, k, j, i)  = loc_abs_1[isp];
        scat_1_(m, isp, k, j, i) = loc_scat_1[isp];
      }
    }

    if (multi_freq_) {
      // Field-weighted mean absorption opacities for the predictor.  kappa varies bin
      // to bin, so the means run over (isp, ifr) weighted by the per-bin energy and
      // number densities -- the spectral generalisation of the grey averages above.
      // The per-bin angular sum is fused into the opacity loop rather than stored: one
      // extra pass over the angles, and no workspace, against a loop already dominated
      // by the bns_nurates kernel integrals.
      Real J_e = nudens_1[0] + nudens_1[1];
      Real n_e = nudens_0[0] + nudens_0[1];
      Real N_L = nudens_0[0] - nudens_0[1];
      Real J_x = 0.0;
      for (int isp = 2; isp < nspecies_; ++isp) { J_x += nudens_1[isp]; }
      Real kJ_e = 0.0, kJ_x = 0.0, kN_e = 0.0;
      Real ka_e = 0.0, ka_x = 0.0;
      int nka_e = 0, nka_x = 0;

      // freq_grid is in MeV, which is what bns_nurates_spectral_bin wants
      for (int ifr = 0; ifr < nfreq_; ++ifr) {
        Real e_lo = 0.0, e_hi = 0.0;
        FreqBinEdgesMeV(freq_grid_, ifr, nfreq_, freq_scale_, e_lo, e_hi);

        Real loc_eta_0_f[4]  = {0.};
        Real loc_eta_1_f[4]  = {0.};
        Real loc_abs_0_f[4]  = {0.};
        Real loc_abs_1_f[4]  = {0.};
        Real loc_scat_1_f[4] = {0.};
        bns_nurates_spectral_bin(e_lo, e_hi, freq_scale_,
                                 nb, T, yp, yn, mu_n, mu_p, mu_e,
                                 nudens_0, nudens_1,
                                 loc_eta_0_f, loc_eta_1_f,
                                 loc_abs_0_f, loc_abs_1_f, loc_scat_1_f,
                                 nurates_params_, code_units_l, eos_units_loc_l);
        for (int isp = 0; isp < nspecies_; ++isp) {
          eta_0_f_(m, isp, ifr, k, j, i)  = loc_eta_0_f[isp];
          eta_1_f_(m, isp, ifr, k, j, i)  = loc_eta_1_f[isp];
          abs_0_f_(m, isp, ifr, k, j, i)  = loc_abs_0_f[isp];
          abs_1_f_(m, isp, ifr, k, j, i)  = loc_abs_1_f[isp];
          scat_1_f_(m, isp, ifr, k, j, i) = loc_scat_1_f[isp];
        }

        if (peq_on_) {
          Real e_mid = FreqBinMidMeV(e_lo, e_hi, freq_scale_);
          for (int isp = 0; isp < nspecies_; ++isp) {
            Real J_f = 0.0;
            Real N_f = 0.0;
            int sp_off = isp*nfreq_*nang_;
            for (int iang = 0; iang <= nang1; ++iang) {
              int n = sp_off + ifr*nang_ + iang;
              Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                         tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                         tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                         tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
              Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) -
                            u_tet[1]*nh_c_.d_view(iang,1) -
                            u_tet[2]*nh_c_.d_view(iang,2) -
                            u_tet[3]*nh_c_.d_view(iang,3));
              Real omega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
              Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
              J_f += intensity_cm*omega_cm;
              // Divided by the ray's OWN comoving bin energy, exactly as the species
              // totals above are: the lab bin sits at n0_cm*[e_lo, e_hi] on this ray.
              N_f += intensity_cm*omega_cm/fmax(n0_cm*e_mid, 1.0e-100);
            }
            J_f /= wght_sum;
            // Same conversion the species totals get: a code-unit energy density over a
            // bin energy in MeV is a number density in fm^-3.
            N_f *= code_edens_to_eos/wght_sum;
            Real ka = loc_abs_1_f[isp];
            if (isp < 2) {
              kJ_e += ka*J_f;
              kN_e += ka*N_f;
              ka_e += ka;
              ++nka_e;
            } else {
              kJ_x += ka*J_f;
              ka_x += ka;
              ++nka_x;
            }
          }
        }
      }

      if (peq_on_) {
        Real kbar_1e = (J_e > 0.0) ? kJ_e/J_e
                                   : ((nka_e > 0) ? ka_e/nka_e : 0.0);
        Real kbar_0e = (n_e > 0.0) ? kN_e/n_e
                                   : ((nka_e > 0) ? ka_e/nka_e : 0.0);
        Real kbar_1x = (J_x > 0.0) ? kJ_x/J_x
                                   : ((nka_x > 0) ? ka_x/nka_x : 0.0);

        Real J_e_eq = J_eq_loc[0] + J_eq_loc[1];
        Real N_L_eq = n_eq_loc[0] - n_eq_loc[1];
        Real J_x_eq = 0.0;
        for (int isp = 2; isp < nspecies_; ++isp) { J_x_eq += J_eq_loc[isp]; }

        Real T_star = T;
        Real Ye_star = Ye;
        PredictPartialEquilibrium(eos, nb, T, Ye, dt_*alpha_cell/w_lorentz,
                                  J_e, J_x, N_L, J_e_eq, J_x_eq, N_L_eq,
                                  kbar_1e, kbar_1x, kbar_0e,
                                  nurates_params_, code_units_l, eos_units_loc_l,
                                  T_star, Ye_star);

        Real Ye_arr[3] = {Ye_star, 0.0, 0.0};
        Real mu_b_s  = eos.GetBaryonChemicalPotential(nb, T_star, Ye_arr);
        Real mu_q_s  = eos.GetChargeChemicalPotential(nb, T_star, Ye_arr);
        Real mu_le_s = eos.GetElectronLeptonChemicalPotential(nb, T_star, Ye_arr);
        Real mu_n_s = mu_b_s;
        Real mu_p_s = mu_b_s + mu_q_s;
        Real mu_e_s = mu_le_s - mu_q_s;
        bool mu_finite = Kokkos::isfinite(T_star) && Kokkos::isfinite(mu_n_s) &&
                         Kokkos::isfinite(mu_p_s) && Kokkos::isfinite(mu_e_s);
        if (!mu_finite) {
          T_star = T;
          mu_n_s = mu_n;
          mu_p_s = mu_p;
          mu_e_s = mu_e;
        }

        // Kirchhoff's law, bin by bin, and the only place the spectral emissivity is
        // set when the predictor is on -- bns_nurates_spectral_bin skips its own
        // emissivity quadrature entirely in that case.  Exact per bin rather than
        // approximate: the source term reads eq_j = eta_1/kappa back as the spectrum it
        // relaxes towards, so kappa*J_eq is the only pairing whose fixed point is the
        // equilibrium it is meant to reach.
        for (int ifr = 0; ifr < nfreq_; ++ifr) {
          Real e_lo = 0.0, e_hi = 0.0;
          FreqBinEdgesMeV(freq_grid_, ifr, nfreq_, freq_scale_, e_lo, e_hi);
          Real n_eq_f[4] = {0., 0., 0., 0.};
          Real J_eq_f[4] = {0., 0., 0., 0.};
          NuratesEqSpectrumBin(e_lo, e_hi, T_star, mu_n_s, mu_p_s, mu_e_s,
                               nurates_params_.quadrature,
                               unit_num_dens_, unit_ene_dens_, n_eq_f, J_eq_f);
          for (int isp = 0; isp < nspecies_; ++isp) {
            Real ka0 = abs_0_f_(m, isp, ifr, k, j, i);
            Real ka1 = abs_1_f_(m, isp, ifr, k, j, i);
            Real e0 = ka0*n_eq_f[isp];
            Real e1 = ka1*J_eq_f[isp];
            eta_0_f_(m, isp, ifr, k, j, i) = Kokkos::isfinite(e0) ? e0 : 0.0;
            eta_1_f_(m, isp, ifr, k, j, i) = Kokkos::isfinite(e1) ? e1 : 0.0;
          }
        }
      }
    }
  });

  if (debug_opacity_) {
    if (multi_freq_) {
      const int ncell = (nmb1 + 1)*(ke - ks + 1)*(je - js + 1)*(ie - is + 1);
      const int ntot = ncell*nspecies_*nfreq_;
      Real eta0_sum = 0.0, eta1_sum = 0.0, abs0_sum = 0.0, abs1_sum = 0.0, scat_sum = 0.0;
      Kokkos::parallel_reduce("nurates_mf_eta0_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int &idx, Real &sum) {
        int q = idx;
        int ifr = q % nfreq_;
        q /= nfreq_;
        int isp = q % nspecies_;
        q /= nspecies_;
        int i = is + (q % (ie - is + 1));
        q /= (ie - is + 1);
        int j = js + (q % (je - js + 1));
        q /= (je - js + 1);
        int k = ks + (q % (ke - ks + 1));
        int m = q / (ke - ks + 1);
        sum += eta_0_f_(m, isp, ifr, k, j, i);
      }, eta0_sum);
      Kokkos::parallel_reduce("nurates_mf_eta1_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int &idx, Real &sum) {
        int q = idx;
        int ifr = q % nfreq_;
        q /= nfreq_;
        int isp = q % nspecies_;
        q /= nspecies_;
        int i = is + (q % (ie - is + 1));
        q /= (ie - is + 1);
        int j = js + (q % (je - js + 1));
        q /= (je - js + 1);
        int k = ks + (q % (ke - ks + 1));
        int m = q / (ke - ks + 1);
        sum += eta_1_f_(m, isp, ifr, k, j, i);
      }, eta1_sum);
      Kokkos::parallel_reduce("nurates_mf_abs0_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int &idx, Real &sum) {
        int q = idx;
        int ifr = q % nfreq_;
        q /= nfreq_;
        int isp = q % nspecies_;
        q /= nspecies_;
        int i = is + (q % (ie - is + 1));
        q /= (ie - is + 1);
        int j = js + (q % (je - js + 1));
        q /= (je - js + 1);
        int k = ks + (q % (ke - ks + 1));
        int m = q / (ke - ks + 1);
        sum += abs_0_f_(m, isp, ifr, k, j, i);
      }, abs0_sum);
      Kokkos::parallel_reduce("nurates_mf_abs1_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int &idx, Real &sum) {
        int q = idx;
        int ifr = q % nfreq_;
        q /= nfreq_;
        int isp = q % nspecies_;
        q /= nspecies_;
        int i = is + (q % (ie - is + 1));
        q /= (ie - is + 1);
        int j = js + (q % (je - js + 1));
        q /= (je - js + 1);
        int k = ks + (q % (ke - ks + 1));
        int m = q / (ke - ks + 1);
        sum += abs_1_f_(m, isp, ifr, k, j, i);
      }, abs1_sum);
      Kokkos::parallel_reduce("nurates_mf_scat_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int &idx, Real &sum) {
        int q = idx;
        int ifr = q % nfreq_;
        q /= nfreq_;
        int isp = q % nspecies_;
        q /= nspecies_;
        int i = is + (q % (ie - is + 1));
        q /= (ie - is + 1);
        int j = js + (q % (je - js + 1));
        q /= (je - js + 1);
        int k = ks + (q % (ke - ks + 1));
        int m = q / (ke - ks + 1);
        sum += scat_1_f_(m, isp, ifr, k, j, i);
      }, scat_sum);
      Kokkos::fence();
      auto freq_grid_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), freq_grid);
      std::cout << "### Nurates opacity debug: nspecies=" << nspecies_
                << " nfreq=" << nfreq_ << " edens_code_to_MeV_per_fm3="
                << nurates_code_edens_to_eos
                << " freq_grid_MeV=[";
      for (int ifr = 0; ifr < nfreq_; ++ifr) {
        std::cout << (ifr == 0 ? "" : ", ") << freq_grid_h(ifr);
      }
      std::cout << "] sum(eta_0_freq)=" << eta0_sum
                << " sum(eta_1_freq)=" << eta1_sum
                << " sum(abs_0_freq)=" << abs0_sum
                << " sum(abs_1_freq)=" << abs1_sum
                << " sum(scat_1_freq)=" << scat_sum << std::endl;
    } else {
      const int ncell = (nmb1 + 1)*(ke - ks + 1)*(je - js + 1)*(ie - is + 1);
      const int ntot = ncell*nspecies_;
      Real eta0_sum = 0.0, eta1_sum = 0.0, abs0_sum = 0.0, abs1_sum = 0.0, scat1_sum = 0.0;
      Kokkos::parallel_reduce("nurates_gray_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int &idx, Real &s0, Real &s1, Real &s2, Real &s3, Real &s4) {
        int q = idx;
        int isp = q % nspecies_;
        q /= nspecies_;
        int i = is + (q % (ie - is + 1));
        q /= (ie - is + 1);
        int j = js + (q % (je - js + 1));
        q /= (je - js + 1);
        int k = ks + (q % (ke - ks + 1));
        int m = q / (ke - ks + 1);
        s0 += eta_0_(m, isp, k, j, i);
        s1 += eta_1_(m, isp, k, j, i);
        s2 += abs_0_(m, isp, k, j, i);
        s3 += abs_1_(m, isp, k, j, i);
        s4 += scat_1_(m, isp, k, j, i);
      }, eta0_sum, eta1_sum, abs0_sum, abs1_sum, scat1_sum);
      Kokkos::fence();
      std::cout << "### Nurates opacity debug: nspecies=" << nspecies_
                << " sum(eta_0)=" << eta0_sum
                << " sum(eta_1)=" << eta1_sum
                << " sum(abs_0)=" << abs0_sum
                << " sum(abs_1)=" << abs1_sum
                << " sum(scat_1)=" << scat1_sum << std::endl;
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation

#endif  // ENABLE_NURATES
