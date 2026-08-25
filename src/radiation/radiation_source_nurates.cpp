//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_source_nurates.cpp
//! \brief Implicit neutrino-fluid coupling using precomputed bns_nurates opacities.
//!        Dispatched from RadFluidCoupling when use_nurates is enabled.

#include "athena.hpp"
#include "config.hpp"

#if ENABLE_NURATES

#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "radiation.hpp"

#include "radiation/radiation_tetrad.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::RadFluidCouplingNurates(Driver *pdriver, int stage)
//! \brief Implicit neutrino-fluid source term using precomputed bns_nurates opacities.
//!        Emissivity, absorption, and scattering opacities are fixed at the start of
//!        each RK stage (see CalcOpacityNurates).

TaskStatus Radiation::RadFluidCouplingNurates(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = prgeo->nangles - 1;
  int nang_ = prgeo->nangles;
  int nsp_ = nspecies;
  auto &size = pmy_pack->pmb->mb_size;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  bool &is_compton_enabled_ = is_compton_enabled;
  bool &fixed_fluid_ = fixed_fluid;
  bool &affect_fluid_ = affect_fluid;
  bool &evolve_ye_ = evolve_ye;
  int ye_source_model_ = ye_source_model;
  Real &source_Ye_min_ = source_Ye_min;
  Real &source_Ye_max_ = source_Ye_max;
  Real &source_limiter_ = source_limiter;
  bool &backreact_chiral_ = backreact_chiral;
  bool is_dyngr = (pmy_pack->pdyngr != nullptr);

  // Extract coordinate/excision data
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;

  // Extract radiation frame and angular mesh data
  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;

  // Extract hydro/mhd quantities
  DvceArray5D<Real> u0_, w0_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }

  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto &nurates_eta_1_ = nurates_eta_1;
  auto &nurates_eta_0_ = nurates_eta_0;
  auto &nurates_abs_0_ = nurates_abs_0;
  auto &nurates_abs_1_ = nurates_abs_1;
  auto &nurates_scat_1_ = nurates_scat_1;
  Real mb_code_ = nurates_baryon_mass;

  // Update primitives before source term application
  if (!(fixed_fluid_)) {
    if (is_dyngr) {
      pmy_pack->pdyngr->ConToPrimBC(is, ie, js, je, ks, ke);
    } else if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is,ie,js,je,ks,ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      auto &bcc0_ = pmy_pack->pmhd->bcc0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is,ie,js,je,ks,ke);
    }
  }

  par_for("radiation_source_nurates", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // compute metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real beta_u[3];
    beta_u[0] = SQR(alpha)*gupper[0][1];
    beta_u[1] = SQR(alpha)*gupper[0][2];
    beta_u[2] = SQR(alpha)*gupper[0][3];

    // fluid velocity components
    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);

    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
           + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
           + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);

    // fluid velocity in tetrad frame
    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);

    // coordinate component n^0
    Real n0 = tt(m,0,0,k,j,i);

    // angle-weight sum (fluid-frame solid angle, independent of species)
    Real wght_sum = 0.0;
    for (int n = 0; n <= nang1; ++n) {
      Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                    u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
      wght_sum += solid_angles_.d_view(n)/SQR(n0_cm);
    }

    // per-species opacities and implicit coefficients
    Real sigma_a_sp[4], sigma_s_sp[4], sigma_p_sp[4];
    Real suma1_sp[4], suma2_sp[4], suma3_sp[4];

    for (int isp = 0; isp < nsp_; ++isp) {
      sigma_a_sp[isp] = nurates_abs_1_(m, isp, k, j, i);
      sigma_s_sp[isp] = nurates_scat_1_(m, isp, k, j, i);
      sigma_p_sp[isp] = 0.0;

      Real dtcsiga_s = dt_*sigma_a_sp[isp];
      Real dtcsigs_s = dt_*sigma_s_sp[isp];
      Real dtcsigp_s = dt_*sigma_p_sp[isp];
      int sp_off = isp * nang_;

      Real sum1 = 0.0, sum2 = 0.0;
      for (int n = 0; n <= nang1; ++n) {
        int nn = sp_off + n;
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                   tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
        Real intensity_cm = 4.0*M_PI*(i0_(m,nn,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0 + (dtcsiga_s + dtcsigs_s)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        sum1 += omega_cm*vncsigma2;
        sum2 += intensity_cm*omega_cm*n0*vncsigma;
      }
      sum1 /= wght_sum;
      sum2 /= wght_sum;
      suma3_sp[isp] = sum1*(dtcsigs_s - dtcsigp_s);
      suma1_sp[isp] = sum1*(dtcsiga_s + dtcsigp_s);
      suma2_sp[isp] = sum2;
    }

    // update intensities using equilibrium energy density = eta_1 / kappa_a
    // (analogous to arad*T^4 in the non-nurates case; intensity_cm = 4pi*I so
    // the equilibrium intensity_cm^eq = eta_1/kappa_a, not eta_1/(4pi))
    Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
    Real dJ_fluid[4] = {0.0};
    Real dN_rad_source[4] = {0.0};
    for (int isp = 0; isp < nsp_; ++isp) {
      Real emission_sp = (sigma_a_sp[isp] > 0.0) ?
                         nurates_eta_1_(m, isp, k, j, i) / sigma_a_sp[isp] : 0.0;
      Real dtcsiga_s = dt_*sigma_a_sp[isp];
      Real dtcsigs_s = dt_*sigma_s_sp[isp];
      Real dtcsigp_s = dt_*sigma_p_sp[isp];
      Real jr_cm_s = (suma1_sp[isp]*emission_sp + suma2_sp[isp])/(1.0 - suma3_sp[isp]);
      int sp_off = isp * nang_;
      Real J_old_sp = 0.0;
      Real J_new_sp = 0.0;
      for (int n = 0; n <= nang1; ++n) {
        int nn = sp_off + n;
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
        Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
        Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

        m_old[0] += (    i0_(m,nn,k,j,i)    *solid_angles_.d_view(n));
        m_old[1] += (n_1*i0_(m,nn,k,j,i)/n_0*solid_angles_.d_view(n));
        m_old[2] += (n_2*i0_(m,nn,k,j,i)/n_0*solid_angles_.d_view(n));
        m_old[3] += (n_3*i0_(m,nn,k,j,i)/n_0*solid_angles_.d_view(n));

        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real intensity_cm = 4.0*M_PI*(i0_(m,nn,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
        J_old_sp += intensity_cm*omega_cm;
        Real vncsigma = 1.0/(n0 + (dtcsiga_s + dtcsigs_s)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real di_cm = ( ((dtcsigs_s-dtcsigp_s)*jr_cm_s
                      + (dtcsiga_s+dtcsigp_s)*emission_sp
                      - (dtcsigs_s+dtcsiga_s)*intensity_cm)*vncsigma2 );
        i0_(m,nn,k,j,i) = n0*n_0*fmax(i0_(m,nn,k,j,i)/(n0*n_0) +
                                     di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

        m_new[0] += (    i0_(m,nn,k,j,i)    *solid_angles_.d_view(n));
        m_new[1] += (n_1*i0_(m,nn,k,j,i)/n_0*solid_angles_.d_view(n));
        m_new[2] += (n_2*i0_(m,nn,k,j,i)/n_0*solid_angles_.d_view(n));
        m_new[3] += (n_3*i0_(m,nn,k,j,i)/n_0*solid_angles_.d_view(n));

        Real intensity_cm_new = 4.0*M_PI*(i0_(m,nn,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        J_new_sp += intensity_cm_new*omega_cm;

        if (excise) {
          bool apply_excision = (rad_mask_(m,k,j,i) ||
                                 (!(is_compton_enabled_) && fabs(n_0) < n_0_floor_));
          if (apply_excision) { i0_(m,nn,k,j,i) = 0.0; }
        }
      }
      if (isp < 4) {
        Real J_old = J_old_sp/wght_sum;
        Real J_new = J_new_sp/wght_sum;
        dJ_fluid[isp] = J_new - J_old;

        Real eta0 = nurates_eta_0_(m, isp, k, j, i);
        Real eta1 = nurates_eta_1_(m, isp, k, j, i);
        Real abs0 = nurates_abs_0_(m, isp, k, j, i);
        Real abs1 = nurates_abs_1_(m, isp, k, j, i);
        // Estimate gray number density so J=eta1/abs1 implies N=eta0/abs0.
        // eta_0 and abs_0 come from nurates with eta_0 in fm^-3 per code time,
        // so eps_eq is a code energy density per fm^-3 and N is in fm^-3.
        Real eps_eq = (eta0 > 0.0 && abs0 > 0.0 && abs1 > 0.0) ?
                      (eta1/abs1)/(eta0/abs0) : 0.0;
        Real N_old = (eps_eq > 0.0) ? J_old/eps_eq : 0.0;
        Real N_new = (N_old + dt_*eta0)/(1.0 + dt_*abs0);
        dN_rad_source[isp] = N_new - N_old;
      }
    }

    // feedback on fluid conserved variables
    if (affect_fluid_) {
      Real dm0 = m_old[0] - m_new[0];
      Real dm1 = m_old[1] - m_new[1];
      Real dm2 = m_old[2] - m_new[2];
      Real dm3 = m_old[3] - m_new[3];
      if (is_dyngr) {
        u0_(m,IEN,k,j,i) += (1.0/alpha)*(-dm0+beta_u[0]*dm1+beta_u[1]*dm2+beta_u[2]*dm3);
      } else {
        u0_(m,IEN,k,j,i) += dm0;
      }
      u0_(m,IM1,k,j,i) += dm1;
      u0_(m,IM2,k,j,i) += dm2;
      u0_(m,IM3,k,j,i) += dm3;
      if (evolve_ye_ && nsp_ > 1 && is_mhd_enabled_) {
        Real dN_nue = dN_rad_source[0];
        Real dN_anue = dN_rad_source[1];
        if (ye_source_model_ == 1) {
          dN_nue = 0.0;
          dN_anue = 0.0;
        }
        Real eps_nue = (nurates_eta_0_(m, 0, k, j, i) > 0.0) ?
                       nurates_eta_1_(m, 0, k, j, i)/nurates_eta_0_(m, 0, k, j, i) : 0.0;
        Real eps_anue = (nurates_eta_0_(m, 1, k, j, i) > 0.0) ?
                        nurates_eta_1_(m, 1, k, j, i)/nurates_eta_0_(m, 1, k, j, i) : 0.0;
        if (ye_source_model_ == 1) {
          if (eps_nue > 0.0) {
            dN_nue = gamma*dJ_fluid[0]/eps_nue;
          }
          if (eps_anue > 0.0) {
            dN_anue = gamma*dJ_fluid[1]/eps_anue;
          }
        }
        // dN_* are in fm^-3, matching the EOS number-density unit that
        // GetBaryonMass() is defined against.
        Real dDYe = mb_code_*(-dN_nue + dN_anue);

        Real cons_dens = u0_(m,IDN,k,j,i);
        if (cons_dens > 0.0) {
          Real ye_old = w0_(m,IYF,k,j,i);
          // u0_(IYF) stores the conserved scalar D*Ye, so convert the
          // number-density source through the conserved density.
          Real raw_dDYe = dDYe;
          if (source_limiter_ >= 0.0) {
            Real raw_dYe = raw_dDYe/cons_dens;
            Real theta = 1.0;
            if (raw_dYe > 0.0) {
              theta = fmin(theta, source_limiter_*
                                  fmax(source_Ye_max_ - ye_old, 0.0)/raw_dYe);
            } else if (raw_dYe < 0.0) {
              theta = fmin(theta, source_limiter_*
                                  fmin(source_Ye_min_ - ye_old, 0.0)/raw_dYe);
            }
            raw_dDYe *= fmax(theta, 0.0);
          }
          Real ye_new = ye_old + raw_dDYe/cons_dens;
          ye_new = fmin(fmax(ye_new, source_Ye_min_), source_Ye_max_);
          Real dDYe_applied = cons_dens*(ye_new - ye_old);
          u0_(m,IYF,k,j,i) += dDYe_applied;
          // The weak reactions that convert protons to neutrons also flip
          // electron chirality, so the chiral imbalance Y5 is sourced with the
          // opposite sign to Ye.  Using the increment that was actually applied
          // means Y5 inherits the source_limiter / source_Ye_min / source_Ye_max
          // protections for free.
          if (backreact_chiral_) {
            u0_(m,IYF+1,k,j,i) -= dDYe_applied;
          }
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::MultiFreqRadFluidCouplingNurates(Driver *pdriver, int stage)
//! \brief Implicit multifrequency neutrino-fluid coupling using spectral bns_nurates
//!        coefficients.  The Ye update is computed from the change in neutrino number
//!        density, derived directly from the frequency-resolved intensities.

TaskStatus Radiation::MultiFreqRadFluidCouplingNurates(Driver *pdriver, int stage) {
  if (!(rad_source)) {
    return TaskStatus::complete;
  }
  if (!(multi_freq)) {
    return RadFluidCouplingNurates(pdriver, stage);
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang_ = prgeo->nangles;
  int nang1 = nang_ - 1;
  int nfrq_ = nfreq;
  int nsp_ = nspecies;
  int freq_scale_ = flag_fscale;
  auto &size = pmy_pack->pmb->mb_size;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  bool &fixed_fluid_ = fixed_fluid;
  bool &affect_fluid_ = affect_fluid;
  bool &evolve_ye_ = evolve_ye;
  Real &source_Ye_min_ = source_Ye_min;
  Real &source_Ye_max_ = source_Ye_max;
  Real &source_limiter_ = source_limiter;
  bool &backreact_chiral_ = backreact_chiral;
  bool is_dyngr = (pmy_pack->pdyngr != nullptr);

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;

  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;
  auto &nu_tet = freq_grid;

  DvceArray5D<Real> u0_, w0_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }

  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eta_1_f_ = nurates_eta_1_freq;
  auto &abs_1_f_ = nurates_abs_1_freq;
  auto &scat_1_f_ = nurates_scat_1_freq;
  Real mb_code_ = nurates_baryon_mass;
  Real code_edens_to_eos_ = nurates_code_edens_to_eos;

  if (!(fixed_fluid_)) {
    if (is_dyngr) {
      pmy_pack->pdyngr->ConToPrimBC(is, ie, js, je, ks, ke);
    } else if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is,ie,js,je,ks,ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      auto &bcc0_ = pmy_pack->pmhd->bcc0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is,ie,js,je,ks,ke);
    }
  }

  size_t scr_size = ScrArray1D<Real>::shmem_size(nsp_*nfrq_) * 6;
  int scr_level = 0;
  par_for_outer("multi_freq_source_nurates", DevExeSpace(), scr_size, scr_level,
  0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(TeamMember_t member, int m, int k, int j, int i) {
    ScrArray1D<Real> sigma_a(member.team_scratch(scr_level), nsp_*nfrq_);
    ScrArray1D<Real> sigma_s(member.team_scratch(scr_level), nsp_*nfrq_);
    ScrArray1D<Real> sum1_a(member.team_scratch(scr_level), nsp_*nfrq_);
    ScrArray1D<Real> sum2_a(member.team_scratch(scr_level), nsp_*nfrq_);
    ScrArray1D<Real> sum3_a(member.team_scratch(scr_level), nsp_*nfrq_);
    ScrArray1D<Real> eq_j(member.team_scratch(scr_level), nsp_*nfrq_);

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
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real beta_u[3];
    beta_u[0] = SQR(alpha)*gupper[0][1];
    beta_u[1] = SQR(alpha)*gupper[0][2];
    beta_u[2] = SQR(alpha)*gupper[0][3];

    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);
    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy +
             2.0*glower[1][3]*wvx*wvz + glower[2][2]*wvy*wvy +
             2.0*glower[2][3]*wvy*wvz + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);

    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);
    Real n0 = tt(m,0,0,k,j,i);

    Real wght_sum = 0.0;
    for (int iang=0; iang<=nang1; ++iang) {
      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                    u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
      wght_sum += solid_angles_.d_view(iang)/SQR(n0_cm);
    }

    for (int isp=0; isp<nsp_; ++isp) {
      for (int ifr=0; ifr<nfrq_; ++ifr) {
        int idx = isp*nfrq_ + ifr;
        sigma_a(idx) = abs_1_f_(m, isp, ifr, k, j, i);
        sigma_s(idx) = scat_1_f_(m, isp, ifr, k, j, i);
        eq_j(idx) = (sigma_a(idx) > 0.0) ? eta_1_f_(m, isp, ifr, k, j, i)/sigma_a(idx) : 0.0;
        Real dtcsiga = dt_*sigma_a(idx);
        Real dtcsigs = dt_*sigma_s(idx);
        Real sum1 = 0.0, sum2 = 0.0;
        int sp_off = isp*nfrq_*nang_;
        for (int iang=0; iang<=nang1; ++iang) {
          int nn = sp_off + ifr*nang_ + iang;
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                     tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                     tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                     tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
          Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                        u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
          Real omega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
          Real intensity_cm = 4.0*M_PI*(i0_(m,nn,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
          Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
          sum1 += omega_cm*n0_cm*vncsigma;
          sum2 += intensity_cm*omega_cm*n0*vncsigma;
        }
        sum1_a(idx) = sum1/wght_sum;
        sum2_a(idx) = sum2/wght_sum;
        sum3_a(idx) = sum1_a(idx)*dtcsigs;
      }
    }

    Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
    Real dN_rad_moment[4] = {0.0};
    for (int isp=0; isp<nsp_; ++isp) {
      int sp_off = isp*nfrq_*nang_;
      for (int ifr=0; ifr<nfrq_; ++ifr) {
        int idx = isp*nfrq_ + ifr;
        Real dtcsiga = dt_*sigma_a(idx);
        Real dtcsigs = dt_*sigma_s(idx);
        Real jr_cm = (sum1_a(idx)*dtcsiga*eq_j(idx) + sum2_a(idx))/(1.0 - sum3_a(idx));
        Real e_mid = 0.0;
        if (ifr < nfrq_ - 1) {
          Real e_lo = nu_tet(ifr);
          Real e_hi = nu_tet(ifr+1);
          e_mid = (freq_scale_ == 1 && e_lo > 0.0) ? sqrt(e_lo*e_hi) :
                                                      0.5*(e_lo + e_hi);
        } else {
          Real e_hi = (freq_scale_ == 1 && nu_tet(ifr-1) > 0.0) ?
                      nu_tet(ifr)*nu_tet(ifr)/nu_tet(ifr-1) :
                      nu_tet(ifr) + (nu_tet(ifr) - nu_tet(ifr-1));
          e_mid = (freq_scale_ == 1 && nu_tet(ifr) > 0.0) ? sqrt(nu_tet(ifr)*e_hi) :
                                                            0.5*(nu_tet(ifr) + e_hi);
        }
        for (int iang=0; iang<=nang1; ++iang) {
          int nn = sp_off + ifr*nang_ + iang;
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                     tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                     tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                     tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(iang,0) +
                     tc(m,1,1,k,j,i)*nh_c_.d_view(iang,1) +
                     tc(m,2,1,k,j,i)*nh_c_.d_view(iang,2) +
                     tc(m,3,1,k,j,i)*nh_c_.d_view(iang,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(iang,0) +
                     tc(m,1,2,k,j,i)*nh_c_.d_view(iang,1) +
                     tc(m,2,2,k,j,i)*nh_c_.d_view(iang,2) +
                     tc(m,3,2,k,j,i)*nh_c_.d_view(iang,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(iang,0) +
                     tc(m,1,3,k,j,i)*nh_c_.d_view(iang,1) +
                     tc(m,2,3,k,j,i)*nh_c_.d_view(iang,2) +
                     tc(m,3,3,k,j,i)*nh_c_.d_view(iang,3);
          Real domega = solid_angles_.d_view(iang);

          m_old[0] += i0_(m,nn,k,j,i)*domega;
          m_old[1] += n_1*i0_(m,nn,k,j,i)/n_0*domega;
          m_old[2] += n_2*i0_(m,nn,k,j,i)/n_0*domega;
          m_old[3] += n_3*i0_(m,nn,k,j,i)/n_0*domega;

          Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                        u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
          Real intensity_cm_old = 4.0*M_PI*(i0_(m,nn,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
          Real omega_cm = domega/SQR(n0_cm);
          Real e_cm = n0_cm*e_mid;

          Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
          Real di_cm = ((dtcsigs*jr_cm + dtcsiga*eq_j(idx) -
                         (dtcsigs + dtcsiga)*intensity_cm_old)*n0_cm*vncsigma);
          i0_(m,nn,k,j,i) = n0*n_0*fmax(i0_(m,nn,k,j,i)/(n0*n_0) +
                             di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

          if (isp < 4) {
            Real intensity_cm_new = 4.0*M_PI*(i0_(m,nn,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
            dN_rad_moment[isp] += (intensity_cm_new - intensity_cm_old)*omega_cm/
                                  fmax(e_cm, 1.0e-100);
          }

          m_new[0] += i0_(m,nn,k,j,i)*domega;
          m_new[1] += n_1*i0_(m,nn,k,j,i)/n_0*domega;
          m_new[2] += n_2*i0_(m,nn,k,j,i)/n_0*domega;
          m_new[3] += n_3*i0_(m,nn,k,j,i)/n_0*domega;

          if (excise) {
            bool apply_excision = (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_);
            if (apply_excision) { i0_(m,nn,k,j,i) = 0.0; }
          }
        }
      }
    }
    for (int isp=0; isp<4; ++isp) {
      dN_rad_moment[isp] /= wght_sum;
      // the sum above is a code-unit energy density over a bin energy in MeV;
      // rescaling the energy density to MeV/fm^3 makes it a number density in
      // fm^-3, which is the convention throughout the nurates interface
      dN_rad_moment[isp] *= code_edens_to_eos_;
    }

    if (affect_fluid_) {
      Real dm0 = m_old[0] - m_new[0];
      Real dm1 = m_old[1] - m_new[1];
      Real dm2 = m_old[2] - m_new[2];
      Real dm3 = m_old[3] - m_new[3];
      if (is_dyngr) {
        u0_(m,IEN,k,j,i) += (1.0/alpha)*(-dm0+beta_u[0]*dm1+beta_u[1]*dm2+beta_u[2]*dm3);
      } else {
        u0_(m,IEN,k,j,i) += dm0;
      }
      u0_(m,IM1,k,j,i) += dm1;
      u0_(m,IM2,k,j,i) += dm2;
      u0_(m,IM3,k,j,i) += dm3;

      if (evolve_ye_ && nsp_ > 1 && is_mhd_enabled_) {
        Real dN_nue = dN_rad_moment[0];
        Real dN_anue = dN_rad_moment[1];
        Real dDYe = mb_code_*(-dN_nue + dN_anue);
        Real cons_dens = u0_(m,IDN,k,j,i);
        if (cons_dens > 0.0) {
          Real ye_old = w0_(m,IYF,k,j,i);
          if (source_limiter_ >= 0.0) {
            Real raw_dYe = dDYe/cons_dens;
            Real theta = 1.0;
            if (raw_dYe > 0.0) {
              theta = fmin(theta, source_limiter_*
                                  fmax(source_Ye_max_ - ye_old, 0.0)/raw_dYe);
            } else if (raw_dYe < 0.0) {
              theta = fmin(theta, source_limiter_*
                                  fmin(source_Ye_min_ - ye_old, 0.0)/raw_dYe);
            }
            dDYe *= fmax(theta, 0.0);
          }
          Real ye_new = ye_old + dDYe/cons_dens;
          ye_new = fmin(fmax(ye_new, source_Ye_min_), source_Ye_max_);
          Real dDYe_applied = cons_dens*(ye_new - ye_old);
          u0_(m,IYF,k,j,i) += dDYe_applied;
          // Y5 mirrors the applied Ye increment; see the grey path above.
          if (backreact_chiral_) {
            u0_(m,IYF+1,k,j,i) -= dDYe_applied;
          }
        }
      }
    }
  });

  return TaskStatus::complete;
}

} // namespace radiation

#endif  // ENABLE_NURATES
