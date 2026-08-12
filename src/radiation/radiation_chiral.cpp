//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_chiral.cpp
//! \brief chirality-flip sink and E.B anomaly source for the chiral imbalance Y5
//!
//! This is the multi-frequency solver's counterpart to the chiral_gamma_m kernel
//! in radiation_m1/radiation_m1_update.cpp.  Both call the same functions in
//! mhd/chiral_dynamo.hpp, so the two solvers apply identical chiral physics and
//! any difference between them is attributable to the transport.
//!
//! The URCA source for Y5 is not here -- it rides along with the Ye update in
//! radiation_source_nurates.cpp, so that it inherits the same limiter.

#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mesh/mesh.hpp"
#include "mhd/chiral_dynamo.hpp"
#include "mhd/mhd.hpp"
#include "radiation.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::ChiralSources
//! \brief Entry point; dispatches on the EOS type like CalcOpacityNurates does.

TaskStatus Radiation::ChiralSources(Driver *pdriver, int stage) {
  if (!backreact_chiral || !chiral_gamma_m) {
    return TaskStatus::complete;
  }

  // Apply once per cycle, on the final stage, with the full timestep -- exactly
  // as the M1 kernel does (radiation_m1_update.cpp guards on stage == 2).
  //
  // This is not cosmetic.  The Gamma_m sink is a multiplicative implicit update,
  // U <- U/(1 + dt*alpha*Gamma_m), and that does not compose with the RK stage
  // weighting: at stage 2 the integrator forms u0 = gam0*u0 + gam1*u1 + ...,
  // which discards whatever stage 1 applied to u0 and mixes back in an undamped
  // gam1*u1.  Running at every stage therefore gives an effective per-cycle
  // factor of gam1/(1 + beta*dt*alpha*Gamma_m) instead of
  // 1/(1 + dt*alpha*Gamma_m).  The two agree asymptotically for
  // dt*Gamma_m >> 1, which is why the shock problems do not show it, but in the
  // weakly-damped limit the former spuriously decays Y5 by gam1 every cycle.
  if (stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }

  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return ChiralSources_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                          Primitive::ResetFloor>(pdriver, stage);
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>,
      Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return ChiralSources_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                          Primitive::ResetFloor>(pdriver, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "backreact_chiral=true but no supported EOSCompOSE found. "
            << "Check that DynGRMHD is using EOSCompOSE<NQTLogs> or "
            << "EOSCompOSE<NormalLogs>." << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::ChiralSources_<EOSPolicy, ErrorPolicy>
//! \brief Apply the E.B anomaly source and the Gamma_m sink to u0_(IYF+1).
//!
//! Numerics follow the M1 kernel exactly: E.B explicit, Gamma_m implicit, in one
//! combined step, U <- (U + eb_src)/(1 + dt*alpha*Gamma_m), which is
//! unconditionally stable for any Gamma_m > 0.

template <class EOSPolicy, class ErrorPolicy>
TaskStatus Radiation::ChiralSources_(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // EOS reference captured at function scope, not a raw host pointer -- this is
  // the pattern radiation_nurates.cpp uses, and it is what avoids the CUDA
  // illegal-memory access that motivated M1's separate kernel.
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb_ = eos.GetBaryonMass();
  const Primitive::UnitSystem code_units_ = eos.GetCodeUnitSystem();

  auto &adm = pmy_pack->padm->adm;
  auto &w0_ = pmy_pack->pmhd->w0;
  auto &u0_ = pmy_pack->pmhd->u0;
  auto &bcc0_ = pmy_pack->pmhd->bcc0;

  // full timestep: this runs once per cycle, after the last RK stage
  const Real dt_ = pmy_pack->pmesh->dt;

  par_for("rad_chiral_sources", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Non-degenerate proton limit (T > T_P ~ 13 MeV at n0, Ye=0.1);
    // revisit for the degenerate regime if needed.
    const Real nb    = w0_(m, IDN, k, j, i) / mb_;
    const Real p     = w0_(m, IPR, k, j, i);
    Real Y           = w0_(m, IYF, k, j, i);
    const Real T     = eos.GetTemperatureFromP(nb, p, &Y);
    const Real mu_q  = eos.GetChargeChemicalPotential(nb, T, &Y);
    const Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &Y);
    const Real mu_e  = mu_le - mu_q;  // MeV

    const Real Gamma_m =
        chiral::GammaM(mu_e, code_units_.time, chiral::kAlphaEM);

#ifdef CHIRAL_DEBUG
    if (m == 0 && k == ks && j == js && i == is) {
      Kokkos::printf("[chiral] T=%.9e mu_e=%.9e code_time=%.9e Gamma_m=%.9e "
                     "dt=%.9e dt*Gamma_m=%.9e\n",
                     T, mu_e, code_units_.time, Gamma_m, dt_, dt_*Gamma_m);
    }
#endif

    const Real g11 = adm.g_dd(m,0,0,k,j,i), g12 = adm.g_dd(m,0,1,k,j,i);
    const Real g13 = adm.g_dd(m,0,2,k,j,i), g22 = adm.g_dd(m,1,1,k,j,i);
    const Real g23 = adm.g_dd(m,1,2,k,j,i), g33 = adm.g_dd(m,2,2,k,j,i);
    const Real detg    = adm::SpatialDet(g11,g12,g13,g22,g23,g33);
    const Real sqrtgam = Kokkos::sqrt(detg);
    const Real ivol    = 1.0/sqrtgam;
    const Real alpha   = adm.alpha(m, k, j, i);

    const Real ux = w0_(m,IVX,k,j,i);
    const Real uy = w0_(m,IVY,k,j,i);
    const Real uz = w0_(m,IVZ,k,j,i);
    const Real iW2 = 1.0/(1.0 + g11*ux*ux + 2.0*g12*ux*uy
                          + 2.0*g13*ux*uz + g22*uy*uy
                          + 2.0*g23*uy*uz + g33*uz*uz);

    // densitized B: cB^i = sqrt(gamma)*B^i (stored in bcc0_)
    const Real Bx = bcc0_(m,IBX,k,j,i);
    const Real By = bcc0_(m,IBY,k,j,i);
    const Real Bz = bcc0_(m,IBZ,k,j,i);

    // physical covariant B: Bd_i = g_ij B^j
    const Real Bd1 = (g11*Bx + g12*By + g13*Bz)*ivol;
    const Real Bd2 = (g12*Bx + g22*By + g23*Bz)*ivol;
    const Real Bd3 = (g13*Bx + g23*By + g33*Bz)*ivol;

    const Real Bsq = (Bx*Bd1 + By*Bd2 + Bz*Bd3)*ivol;  // g_ij B^i B^j
    const Real Bv  = Bd1*ux + Bd2*uy + Bd3*uz;         // W g_ij B^i v^j
    const Real bsq = (Bsq + Bv*Bv)*iW2;                // fluid-frame b^2

    const Real xi = chiral::Xi(w0_(m, IYF+1, k, j, i),
                               w0_(m, IYF, k, j, i), chiral::kAlphaEM);

    const Real eb_src = - dt_ * alpha * sqrtgam * mb_
                        * chiral::AnomalySource(xi, bsq, chiral::kAlphaEM);

#ifdef CHIRAL_DEBUG
    if (m == 0 && k == ks && j == js && i == is) {
      Kokkos::printf("[chiralEB] Y5=%.17e Ye=%.17e xi=%.17e bsq=%.17e "
                     "mb=%.17e sqrtgam=%.17e alpha=%.17e eb_src=%.17e "
                     "U_before=%.17e\n",
                     w0_(m,IYF+1,k,j,i), w0_(m,IYF,k,j,i), xi, bsq, mb_,
                     sqrtgam, alpha, eb_src, u0_(m,IYF+1,k,j,i));
    }
#endif
    u0_(m, IYF + 1, k, j, i) =
        (u0_(m, IYF + 1, k, j, i) + eb_src) / (1.0 + dt_ * alpha * Gamma_m);
  });

  return TaskStatus::complete;
}

}  // namespace radiation
