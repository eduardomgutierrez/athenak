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

  // Unit systems
  auto code_units    = eos.GetCodeUnitSystem();
  auto eos_units_loc = eos.GetEOSUnitSystem();
  bool debug_opacity_ = nurates_debug_opacity;

  // Nurates params (captured by value for use inside KOKKOS_LAMBDA)
  auto nurates_params_ = nurates_params;

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
  auto &eta_1_f_   = nurates_eta_1_freq;
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

    // Neutrino number/energy densities in code units.  For multifrequency
    // transport, derive number density directly from bin-integrated energy.
    Real nudens_0[4] = {0., 0., 0., 0.};
    Real nudens_1[4] = {0., 0., 0., 0.};
    if (multi_freq_) {
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
      for (int iang = 0; iang <= nang1; ++iang) {
        Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                      u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
        wght_sum += solid_angles_.d_view(iang)/SQR(n0_cm);
      }
      for (int isp = 0; isp < nspecies_; ++isp) {
        int sp_off = isp*nfreq_*nang_;
        for (int ifr = 0; ifr < nfreq_; ++ifr) {
          Real e_mid = 0.0;
          if (ifr < nfreq_ - 1) {
            Real e_lo = freq_grid_(ifr);
            Real e_hi = freq_grid_(ifr+1);
            e_mid = (freq_scale_ == 1 && e_lo > 0.0) ? sqrt(e_lo*e_hi) :
                                                        0.5*(e_lo + e_hi);
          } else {
            Real e_hi = freq_grid_(ifr) + fmax(freq_grid_(ifr) - freq_grid_(ifr-1),
                                               20.0*T - freq_grid_(ifr));
            e_mid = (freq_scale_ == 1 && freq_grid_(ifr) > 0.0) ?
                    sqrt(freq_grid_(ifr)*e_hi) : 0.5*(freq_grid_(ifr) + e_hi);
          }
          for (int iang = 0; iang <= nang1; ++iang) {
            int n = sp_off + ifr*nang_ + iang;
            Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) +
                       tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                       tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) +
                       tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
            Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                          u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
            Real omega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
            Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
            nudens_1[isp] += intensity_cm*omega_cm;
            nudens_0[isp] += intensity_cm*omega_cm/fmax(n0_cm*e_mid, 1.0e-100);
          }
        }
        nudens_1[isp] /= wght_sum;
        nudens_0[isp] /= wght_sum;
      }
    }

    // Make local mutable copies for unit conversion calls
    auto code_units_l   = code_units;
    auto eos_units_loc_l = eos_units_loc;

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
      for (int ifr = 0; ifr < nfreq_; ++ifr) {
        Real e_lo = freq_grid_(ifr);
        Real e_hi = 0.0;
        if (ifr < nfreq_ - 1) {
          e_hi = freq_grid_(ifr+1);
        } else {
          e_hi = freq_grid_(ifr) + fmax(freq_grid_(ifr) - freq_grid_(ifr-1),
                                        20.0*T - freq_grid_(ifr));
        }

        Real loc_eta_1_f[4]  = {0.};
        Real loc_abs_1_f[4]  = {0.};
        Real loc_scat_1_f[4] = {0.};
        bns_nurates_spectral_bin(e_lo, e_hi, freq_scale_,
                                 nb, T, yp, yn, mu_n, mu_p, mu_e,
                                 nudens_0, nudens_1,
                                 loc_eta_1_f, loc_abs_1_f, loc_scat_1_f,
                                 nurates_params_, code_units_l, eos_units_loc_l);
        for (int isp = 0; isp < nspecies_; ++isp) {
          eta_1_f_(m, isp, ifr, k, j, i)  = loc_eta_1_f[isp];
          abs_1_f_(m, isp, ifr, k, j, i)  = loc_abs_1_f[isp];
          scat_1_f_(m, isp, ifr, k, j, i) = loc_scat_1_f[isp];
        }
      }
    }
  });

  if (debug_opacity_) {
    if (multi_freq_) {
      const int ncell = (nmb1 + 1)*(ke - ks + 1)*(je - js + 1)*(ie - is + 1);
      const int ntot = ncell*nspecies_*nfreq_;
      Real eta_sum = 0.0, abs_sum = 0.0, scat_sum = 0.0;
      Kokkos::parallel_reduce("nurates_mf_eta_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
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
      }, eta_sum);
      Kokkos::parallel_reduce("nurates_mf_abs_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
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
      }, abs_sum);
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
      Primitive::UnitSystem nurates_units = Primitive::MakeNGS();
      std::cout << "### Nurates opacity debug: nspecies=" << nspecies_
                << " nfreq=" << nfreq_ << " energy_code_to_MeV="
                << code_units.EnergyConversion(nurates_units)
                << " freq_grid_code=[";
      for (int ifr = 0; ifr < nfreq_; ++ifr) {
        std::cout << (ifr == 0 ? "" : ", ") << freq_grid_h(ifr);
      }
      std::cout << "] freq_grid_MeV=[";
      for (int ifr = 0; ifr < nfreq_; ++ifr) {
        std::cout << (ifr == 0 ? "" : ", ")
                  << freq_grid_h(ifr)*code_units.EnergyConversion(nurates_units);
      }
      std::cout << "] sum(eta_1_freq)=" << eta_sum
                << " sum(abs_1_freq)=" << abs_sum
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
