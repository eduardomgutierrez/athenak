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
#include "radiation/radiation_nurates.hpp"
#include "radiation/radiation_nurates_remap.hpp"

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
      // dm_mu is the undensitised coordinate-frame -Delta R^t_mu, so the fluid gains
      // Delta T^t_mu = dm_mu.  Valencia carries sqrt(gamma)*(E-D) and sqrt(gamma)*S_k,
      // and E = -T^t_t + beta^k T^t_k, which is where the minus sign comes from; HARM
      // carries T^t_t + D and T^t_k instead, hence the branch.  The general Valencia
      // increments are sqrt(gamma)*(-dm0 + beta^k dm_k) and sqrt(gamma)*alpha*dm_k;
      // they take the form below only because the radiation module runs on the
      // analytic Cartesian Kerr-Schild metric, where det g = -1 so alpha*sqrt(gamma)
      // = 1: the 1/alpha IS the sqrt(gamma), and the momentum needs no factor.  On any
      // metric with sqrt(-g) != 1 this is wrong (as is taking alpha, beta and the
      // tetrad from the analytic background at all).
      if (is_dyngr) {
        u0_(m,IEN,k,j,i) += (1.0/alpha)*(-dm0+beta_u[0]*dm1+beta_u[1]*dm2+beta_u[2]*dm3);
      } else {
        u0_(m,IEN,k,j,i) += dm0;
      }
      u0_(m,IM1,k,j,i) += dm1;
      u0_(m,IM2,k,j,i) += dm2;
      u0_(m,IM3,k,j,i) += dm3;
      if (evolve_ye_ && nsp_ > 1 && is_mhd_enabled_) {
        // Both routes produce dN_*, the *comoving* neutrino number density change in
        // fm^-3 -- the EOS number-density unit that GetBaryonMass() is defined against.
        // 'opacity' integrates the number source directly; 'moment' divides the energy
        // density change by a mean energy.
        Real dN_nue = dN_rad_source[0];
        Real dN_anue = dN_rad_source[1];
        if (ye_source_model_ == 1) {
          Real eta0_nue = nurates_eta_0_(m, 0, k, j, i);
          Real eta0_anue = nurates_eta_0_(m, 1, k, j, i);
          Real eps_nue = (eta0_nue > 0.0) ?
                         nurates_eta_1_(m, 0, k, j, i)/eta0_nue : 0.0;
          Real eps_anue = (eta0_anue > 0.0) ?
                          nurates_eta_1_(m, 1, k, j, i)/eta0_anue : 0.0;
          dN_nue = (eps_nue > 0.0) ? dJ_fluid[0]/eps_nue : 0.0;
          dN_anue = (eps_anue > 0.0) ? dJ_fluid[1]/eps_anue : 0.0;
        }
        // u0_(IYF) holds the conserved D*Ye = rho*W*Ye, so a comoving number density
        // change enters it with one factor of W = gamma.  Applied here, once, for both
        // routes.  It used to be applied inside the 'moment' branch, which left
        // 'opacity' -- the default -- short by exactly W, so the two routes disagreed
        // on a moving fluid and only one of them was right.  Same form as the
        // multifrequency path below; keep them textually identical.
        // Applied in full.  Any cap on this increment that is not matched by the same
        // cap on the neutrino field destroys lepton number: the neutrinos have already
        // been updated by dN_nue and dN_anue by this point, so whatever the fluid does
        // not accept is simply lost -- and the energy and momentum above are not capped
        // either, which would leave the lepton exchange inconsistent with the energy
        // exchange that produced it.
        Real dDYe = gamma*mb_code_*(-dN_nue + dN_anue);
        u0_(m,IYF,k,j,i) += dDYe;
        // The weak reactions that convert protons to neutrons also flip electron
        // chirality, so the chiral imbalance Y5 is sourced with the opposite sign to Ye.
        if (backreact_chiral_) {
          u0_(m,IYF+1,k,j,i) -= dDYe;
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::MultiFreqRadFluidCouplingNurates(Driver *pdriver, int stage)
//! \brief Implicit multifrequency neutrino-fluid coupling.  The scattering target is
//!        built on the fixed comoving grid; the update is applied to the stored lab bins
//!        at each ray's own comoving energy.
//!
//! ## The frame problem
//!
//! i0_ is stored in *lab* frequency bins; bns_nurates tabulates sigma_a, sigma_s and eta
//! on freq_grid in *comoving* MeV.  Lab bin ifr covers the comoving interval
//! n0_cm(iang)*[e_lo, e_hi], a different interval on every ray, so the two cannot be
//! combined as they stand.
//!
//! ## The passes
//!
//!  1. **Un-shift, and accumulate.** Each ray's spectrum is evaluated at the fixed grid's
//!     comoving energies, E*_a(g - delta_a), by RemapBinValue.  On a log grid that is a
//!     rigid translation, one number per ray, and it is exact for a power law.  The
//!     result is not stored: its only consumer is the angular sum of step 2, to which
//!     each ray contributes once, so the two are fused and the workspace is two numbers
//!     per bin rather than one per (ray, bin).
//!  2. **Solve, at a common comoving energy.** The implicit angular system for the
//!     scattering mean intensity is solved *on the fixed grid*, where sigma_a, sigma_s
//!     and eta are tabulated and every ray shares the bin:
//!
//!         vnc_a    = 1/(n0 + n0_cm_a dt (sigma_a + sigma_s))
//!         jr_cm(g) = [sum_a w_a vnc_a (n0 I_a + n0_cm_a dt eta)]
//!                  / [1 - sum_a w_a vnc_a n0_cm_a dt sigma_s]
//!
//!     Because these intensities are all at the same energy this is the plain unweighted
//!     angular mean, and there is nothing to choose.  Comoving isotropy, the grey limit,
//!     finiteness at sigma_s = 0 and isoenergeticity all follow from that and need no
//!     construction.
//!  3. **Update the stored bins.** For ray a's bin ifr, sitting at fixed-grid position
//!     x = ifr + delta_a, the opacities, the equilibrium spectrum and jr_cm are read at
//!     x and the implicit update is applied to i0_ in place.
//!
//! ## Why the field itself is never rebinned
//!
//! The photon path rebins i0_ onto the comoving grid, solves, and rebins back.  The round
//! trip is not the identity and applying it twice per step accumulates -- measured for
//! three different operators, see radiation_nurates_remap.hpp.  Transforming the update
//! instead makes the scheme exactly inert when the source vanishes, so nothing can build
//! up, and it keeps every remap out of the conservation path: the energy, the momentum
//! and the lepton number handed to the fluid are all differences of the stored field.
//!
//! ## The whole grid is in the family
//!
//! On the neutrino grid freq_grid(0) = nu_min, so every bin is geometric and every bin is
//! displaced by the same delta_a.  There is no exception to carve out.  The photon grid
//! keeps its [0, nu_min] bin, which needs one and which SetFrequencyGrid still builds;
//! this path does not.  A bin outside the geometric family has no delta_a and could only
//! be solved unshifted in its own lab bin, which is frame-inconsistent and puts an
//! nfreq-independent floor under the isotropy test.

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

  // Width of a bin in ln(nu).  Every bin is geometric on the neutrino grid, so this is
  // one number for the whole grid and a ray's Doppler factor becomes a rigid displacement
  // ln(n0_cm)/grid_dln_ in bin index -- for every bin, with no exceptions to carve out.
  // Taken from grid_nu_min/grid_nu_max rather than the raw nu_min/nu_max: freq_grid may
  // have been rescaled by SetFrequencyGrid and any spacing has to match what it actually
  // holds.  The constructor has already refused a non-log grid, nfreq < 4 and nu_min <=
  // 0.
  Real grid_dln_ = log(grid_nu_max/grid_nu_min)/static_cast<Real>(nfrq_-1);

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

  // Flat par_for, one thread per cell: the body is serial over the whole cell and owns
  // it outright.  par_for_outer would be wrong here -- it requests Kokkos::AUTO team
  // size, and with no inner parallel region every team member would run this same body,
  // so i0_ would be read-modify-written and the fluid source applied once per member.
  // Team scratch goes with it, so the workspace is a preallocated global array sliced
  // per cell; nurates_mf_work is laid out to match i0_, component-major with i fastest.
  auto &work_ = nurates_mf_work;
  const int wstride = static_cast<int>(work_.stride(1));
  par_for("multi_freq_source_nurates", DevExeSpace(),
  0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // This cell's slice of the workspace.  WorkSlice presents a strided column as a 1-D
    // array so RemapFillLogs/RemapBinValue template over it unchanged.
    Real *wbase = &work_(m, 0, k, j, i);
    int woff = 0;
    // The angular sums of the implicit solve, accumulated ray by ray.  Two numbers per
    // bin, not the whole un-shifted field: see the fused pass below.
    WorkSlice acc1 = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice acc2 = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice sigma_a = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice sigma_s = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice eq_j = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice lsa = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice lss = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice lej = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice emid_f = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice jr_cm = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice ljr = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice espec = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice lspec = MakeWorkSlice(wbase, wstride, woff, nfrq_);
    WorkSlice n_0_iang = MakeWorkSlice(wbase, wstride, woff, nang_);
    WorkSlice n0_cm_iang = MakeWorkSlice(wbase, wstride, woff, nang_);
    WorkSlice dlt_iang = MakeWorkSlice(wbase, wstride, woff, nang_);

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

    // Per-ray geometry.  n0_cm does not depend on frequency, so one number per ray
    // displaces every bin of that ray.
    Real wght_sum = 0.0;
    for (int iang=0; iang<=nang1; ++iang) {
      n_0_iang(iang) = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                       tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                       tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                       tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                    u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
      n0_cm_iang(iang) = n0_cm;
      dlt_iang(iang) = (n0_cm > 0.0) ? log(n0_cm)/grid_dln_ : 0.0;
      wght_sum += solid_angles_.d_view(iang)/SQR(n0_cm);
    }
    for (int ifr=0; ifr<nfrq_; ++ifr) {
      Real e_lo = 0.0, e_hi = 0.0;
      FreqBinEdgesMeV(nu_tet, ifr, nfrq_, freq_scale_, e_lo, e_hi);
      emid_f(ifr) = FreqBinMidMeV(e_lo, e_hi, freq_scale_);
    }

    Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
    Real dN_rad_moment[4] = {0.0};
    for (int isp=0; isp<nsp_; ++isp) {
      // One species at a time: passes 1-3 all run inside this loop, so the opacity
      // workspace holds the current species only and carries no species index.
      for (int ifr=0; ifr<nfrq_; ++ifr) {
        Real sa = abs_1_f_(m, isp, ifr, k, j, i);
        sigma_a(ifr) = sa;
        sigma_s(ifr) = scat_1_f_(m, isp, ifr, k, j, i);
        eq_j(ifr) = (sa > 0.0) ? eta_1_f_(m, isp, ifr, k, j, i)/sa : 0.0;
      }
      RemapFillLogs(sigma_a, lsa, 0, 0, nfrq_-1);
      RemapFillLogs(sigma_s, lss, 0, 0, nfrq_-1);
      RemapFillLogs(eq_j, lej, 0, 0, nfrq_-1);

      // ---- Pass 1+2: un-shift each ray onto the fixed grid and accumulate the angular
      // sums of the implicit solve as we go.  The un-shifted field is never stored: its
      // only consumer is the sum below, and each ray contributes to it exactly once, so
      // holding all of it would cost nang*nfreq of workspace and buy nothing.  Ray-major
      // so one ray's whole spectrum is built once and read straight through.
      for (int ifr=0; ifr<nfrq_; ++ifr) { acc1(ifr) = 0.0; acc2(ifr) = 0.0; }
      for (int iang=0; iang<=nang1; ++iang) {
        Real n_0 = n_0_iang(iang);
        Real n0_cm = n0_cm_iang(iang);
        Real dlt = dlt_iang(iang);
        Real sfac = 4.0*M_PI*SQR(SQR(n0_cm))/(n0*n_0);
        for (int ifr=0; ifr<nfrq_; ++ifr) {
          espec(ifr) = sfac*i0_(m, isp*nfrq_*nang_ + ifr*nang_ + iang, k, j, i);
        }
        // Exactly at rest the ray already sits on the grid, and reading the stored
        // values straight back is what makes v = 0 bit-identical to no frame treatment.
        bool shift = (fabs(dlt) > 1.0e-12);
        if (shift) { RemapFillLogs(espec, lspec, 0, 0, nfrq_-1); }
        Real omega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
        for (int ifr=0; ifr<nfrq_; ++ifr) {
          Real icm = shift ? RemapBinValue(espec, lspec, 0, 0, nfrq_-1,
                                           static_cast<Real>(ifr) - dlt,
                                           RemapAsymptote::kSpectrum, grid_dln_)
                           : espec(ifr);
          int idx = ifr;
          Real dtcsiga = dt_*sigma_a(idx);
          Real dtcsigs = dt_*sigma_s(idx);
          Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
          acc1(ifr) += omega_cm*n0_cm*vncsigma;
          acc2(ifr) += icm*omega_cm*n0*vncsigma;
        }
      }
      for (int ifr=0; ifr<nfrq_; ++ifr) {
        int idx = ifr;
        Real dtcsiga = dt_*sigma_a(idx);
        Real dtcsigs = dt_*sigma_s(idx);
        Real sum1 = acc1(ifr)/wght_sum;
        Real sum2 = acc2(ifr)/wght_sum;
        // The angular mean of the *new* intensity, which is what makes the update
        // implicit in the scattering term.  sigma_s is the same on every ray at this
        // energy, so this is the unweighted mean and nothing has to be chosen -- which
        // is exactly what the shape-factor closure could not arrange.
        Real den = 1.0 - sum1*dtcsigs;
        Real jr = (den > 0.0) ? (sum2 + sum1*dtcsiga*eq_j(idx))/den : 0.0;
        jr_cm(ifr) = (jr > 0.0) ? jr : 0.0;
      }
      RemapFillLogs(jr_cm, ljr, 0, 0, nfrq_-1);

      // ---- Pass 3: update the stored lab bins at each ray's own comoving energy ------
      for (int iang=0; iang<=nang1; ++iang) {
        Real n_0 = n_0_iang(iang);
        Real n0_cm = n0_cm_iang(iang);
        Real dlt = dlt_iang(iang);
        bool shift = (fabs(dlt) > 1.0e-12);
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
        Real omega_cm = domega/SQR(n0_cm);
        Real sfac = 4.0*M_PI*SQR(SQR(n0_cm))/(n0*n_0);

        for (int ifr=0; ifr<nfrq_; ++ifr) {
          int nn = isp*nfrq_*nang_ + ifr*nang_ + iang;
          Real i0_old = i0_(m,nn,k,j,i);
          m_old[0] += i0_old*domega;
          m_old[1] += n_1*i0_old/n_0*domega;
          m_old[2] += n_2*i0_old/n_0*domega;
          m_old[3] += n_3*i0_old/n_0*domega;

          // This ray's bin ifr covers comoving n0_cm*[e_lo, e_hi]; on a log grid that is
          // the fixed grid's position ifr + dlt.
          Real sa = sigma_a(ifr);
          Real ss = sigma_s(ifr);
          Real ej = eq_j(ifr);
          Real jr = jr_cm(ifr);
          if (shift) {
            Real xpos = static_cast<Real>(ifr) + dlt;
            sa = RemapBinValue(sigma_a, lsa, 0, 0, nfrq_-1, xpos,
                               RemapAsymptote::kOpacity, grid_dln_);
            ss = RemapBinValue(sigma_s, lss, 0, 0, nfrq_-1, xpos,
                               RemapAsymptote::kOpacity, grid_dln_);
            ej = RemapBinValue(eq_j, lej, 0, 0, nfrq_-1, xpos,
                               RemapAsymptote::kSpectrum, grid_dln_);
            jr = RemapBinValue(jr_cm, ljr, 0, 0, nfrq_-1, xpos,
                               RemapAsymptote::kSpectrum, grid_dln_);
          }

          Real dtcsiga = dt_*sa;
          Real dtcsigs = dt_*ss;
          Real e_cm = n0_cm*emid_f(ifr);
          Real i_old = sfac*i0_old;
          Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
          // A comoving-isotropic field makes jr the common spectrum, so jr read at this
          // ray's own comoving energy is this ray's own bin content and the source
          // vanishes ray by ray, not merely in the angular sum.
          Real di_cm = ((dtcsigs*jr + dtcsiga*ej -
                         (dtcsigs + dtcsiga)*i_old)*n0_cm*vncsigma);
          // Apply the increment to the *stored* value rather than dividing the updated
          // comoving value back.  (sfac*x)/sfac differs from x for ~11% of arguments in
          // IEEE double, so the round trip would perturb every bin by an ulp on every
          // stage even where the source vanishes exactly -- and a vanishing source
          // leaving the field alone is the invariant the whole scheme rests on.  sfac
          // carries the sign of 1/(n0*n_0) and i0_ carries the sign of n0*n_0, so the
          // floor belongs on the comoving intensity, which is what is tested here.
          Real i_new = i_old + di_cm;
          Real i0_new = (i_new > 0.0) ? (i0_old + di_cm/sfac) : 0.0;
          i0_(m,nn,k,j,i) = i0_new;

          if (isp < 4) {
            // Differenced on the stored field, at the bin's own comoving energy, so
            // this is the *same* difference the energy moments below take -- including
            // where the floor binds, which the raw increment di_cm would miss.  No
            // rebinned spectrum is involved anywhere in either.
            dN_rad_moment[isp] +=
                sfac*(i0_new - i0_old)*omega_cm/fmax(e_cm, 1.0e-100);
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
      // dm_mu is the undensitised coordinate-frame -Delta R^t_mu, so the fluid gains
      // Delta T^t_mu = dm_mu.  Valencia carries sqrt(gamma)*(E-D) and sqrt(gamma)*S_k,
      // and E = -T^t_t + beta^k T^t_k, which is where the minus sign comes from; HARM
      // carries T^t_t + D and T^t_k instead, hence the branch.  The general Valencia
      // increments are sqrt(gamma)*(-dm0 + beta^k dm_k) and sqrt(gamma)*alpha*dm_k;
      // they take the form below only because the radiation module runs on the
      // analytic Cartesian Kerr-Schild metric, where det g = -1 so alpha*sqrt(gamma)
      // = 1: the 1/alpha IS the sqrt(gamma), and the momentum needs no factor.  On any
      // metric with sqrt(-g) != 1 this is wrong (as is taking alpha, beta and the
      // tetrad from the analytic background at all).
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
        // dN_rad_moment is comoving; IYF holds D*Ye = rho*W*Ye, so the increment
        // carries a factor W.  Matches the grey path above.
        // Applied in full; see the grey path above for why it is not capped.
        Real dDYe = gamma*mb_code_*(-dN_nue + dN_anue);
        u0_(m,IYF,k,j,i) += dDYe;
        // Y5 mirrors the Ye increment; see the grey path above.
        if (backreact_chiral_) {
          u0_(m,IYF+1,k,j,i) -= dDYe;
        }
      }
    }
  });

  return TaskStatus::complete;
}


} // namespace radiation

#endif  // ENABLE_NURATES
