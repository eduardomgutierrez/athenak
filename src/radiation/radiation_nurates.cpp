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

  // EOS reference (needed for T, yp, yn, mu_* in kernel)
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();
  nurates_baryon_mass = mb;

  // Unit systems
  auto code_units    = eos.GetCodeUnitSystem();
  auto eos_units_loc = eos.GetEOSUnitSystem();

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

    // neutrino number/energy densities (zero for now — opacities computed in
    // equilibrium or without radiation back-reaction on distribution)
    Real nudens_0[4] = {0., 0., 0., 0.};
    Real nudens_1[4] = {0., 0., 0., 0.};

    // output arrays
    Real loc_eta_0[4]  = {0.};
    Real loc_eta_1[4]  = {0.};
    Real loc_abs_0[4]  = {0.};
    Real loc_abs_1[4]  = {0.};
    Real loc_scat_0[4] = {0.};
    Real loc_scat_1[4] = {0.};

    // Make local mutable copies for unit conversion calls
    auto code_units_l   = code_units;
    auto eos_units_loc_l = eos_units_loc;

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
  });

  return TaskStatus::complete;
}

} // namespace radiation

#endif  // ENABLE_NURATES
