//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_neutrino_singlezone.cpp
//! \brief Single-zone neutrino equilibration test for gray radiation with bns_nurates.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation/radiation.hpp"
#ifdef ENABLE_NURATES
#include "radiation/radiation_nurates.hpp"
#endif

namespace {
template <class EOSPolicy, class ErrorPolicy>
void SingleZoneImpl(Mesh *pmesh, ParameterInput *pin, const bool restart);

//----------------------------------------------------------------------------------------
//! \fn Real PlanckLikeIntegral
//! \brief Indefinite integral of E^3 exp(-E/T), used to seed an exactly
//!        comoving-isotropic radiation field with a non-trivial spectrum.
//!
//! d/dE [ -T exp(-E/T) (E^3 + 3T E^2 + 6T^2 E + 6T^3) ] = E^3 exp(-E/T), and the
//! integral from 0 to infinity is 6 T^4.
KOKKOS_INLINE_FUNCTION
Real PlanckLikeIntegral(Real e, Real t) {
  return -t*exp(-e/t)*(e*e*e + 3.0*t*e*e + 6.0*t*t*e + 6.0*t*t*t);
}
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nqt != nullptr) {
    return SingleZoneImpl<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                          Primitive::ResetFloor>(pmy_mesh_, pin, restart);
  }

  auto *ptest_nlog =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nlog != nullptr) {
    return SingleZoneImpl<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                          Primitive::ResetFloor>(pmy_mesh_, pin, restart);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "rad_neutrino_singlezone requires DynGRMHD with EOSCompOSE" << std::endl;
  std::exit(EXIT_FAILURE);
}

namespace {

template <class EOSPolicy, class ErrorPolicy>
void SingleZoneImpl(Mesh *pmesh, ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmesh->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_neutrino_singlezone requires <mhd> with DynGRMHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->prad == nullptr || !pmbp->prad->use_nurates) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_neutrino_singlezone requires <radiation>/use_nurates = true"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmbp->prad->is_neutrino || pmbp->prad->nspecies < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_neutrino_singlezone requires <radiation>/radiation_type = neutrino "
              << "and nspecies >= 2" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmbp->nmb_thispack - 1;
  int nang = pmbp->prad->prgeo->nangles;
  int nspecies = pmbp->prad->nspecies;
  int nfreq = pmbp->prad->nfreq;

  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  Real vy = pin->GetOrAddReal("problem", "vy", 0.0);
  Real vz = pin->GetOrAddReal("problem", "vz", 0.0);
  Real ye = pin->GetReal("problem", "Y_e");
  Real erad = pin->GetOrAddReal("problem", "erad", 0.0);
  // Seed an exactly comoving-isotropic field with a non-trivial spectrum:
  // S(E) = A E^3 exp(-E/spec_temp), normalised so the comoving energy density
  // is erad.  Each ray is given the integral of S over *its own* comoving bin
  // n0_cm*[e_lo, e_hi], which is what a comoving-isotropic field actually looks
  // like in lab-frame bins.  The default (spec_temp <= 0) keeps the old flat
  // erad/nfreq per bin, which is isotropic in the lab-bin sense but does not
  // correspond to any single comoving spectrum once the fluid moves.
  Real spec_temp = pin->GetOrAddReal("problem", "spec_temp", 0.0);
  // Chiral imbalance seed and a uniform seed field, for the chiral unit tests.
  // Both default to zero, so the plain equilibration test is unaffected.
  Real y5 = pin->GetOrAddReal("problem", "Y_5", 0.0);
  Real bx = pin->GetOrAddReal("problem", "bx", 0.0);
  Real by = pin->GetOrAddReal("problem", "by", 0.0);
  Real bz = pin->GetOrAddReal("problem", "bz", 0.0);
  if (y5 != 0.0 && pmbp->pmhd->nscalars < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem/Y_5 requires mhd/nscalars >= 2." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real wlor = 1.0/std::sqrt(1.0 - vx*vx - vy*vy - vz*vz);

  auto &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();
  Real nb = rho/mb;

  auto &w0 = pmbp->pmhd->w0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  const int nscalars_ = pmbp->pmhd->nscalars;
  par_for("pgen_neutrino_singlezone_mhd", DevExeSpace(), 0, nmb1, 0, n3 - 1, 0,
          n2 - 1, 0, n1 - 1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            Real ye_ = ye;
            w0(m, IDN, k, j, i) = rho;
            w0(m, IVX, k, j, i) = wlor*vx;
            w0(m, IVY, k, j, i) = wlor*vy;
            w0(m, IVZ, k, j, i) = wlor*vz;
            w0(m, IPR, k, j, i) = eos.GetPressure(nb, temp, &ye_);
            w0(m, IYF, k, j, i) = ye;
            if (nscalars_ > 1) {
              w0(m, IYF + 1, k, j, i) = y5;
            }

            // uniform field: divergence-free by construction
            bcc0(m, IBX, k, j, i) = bx;
            bcc0(m, IBY, k, j, i) = by;
            bcc0(m, IBZ, k, j, i) = bz;
            b0.x1f(m, k, j, i) = bx;
            b0.x2f(m, k, j, i) = by;
            b0.x3f(m, k, j, i) = bz;
            if (i == n1 - 1) b0.x1f(m, k, j, i + 1) = bx;
            if (j == n2 - 1) b0.x2f(m, k, j + 1, i) = by;
            if (k == n3 - 1) b0.x3f(m, k + 1, j, i) = bz;
          });

  if (pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_neutrino_singlezone requires ADM variables (padm must not be null)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(0, n1 - 1, 0, n2 - 1, 0, n3 - 1);

  auto &i0 = pmbp->prad->i0;
  auto &norm_to_tet = pmbp->prad->norm_to_tet;
  auto &nh_c = pmbp->prad->nh_c;
  auto &tet_c = pmbp->prad->tet_c;
  auto &tetcov_c = pmbp->prad->tetcov_c;
  auto &freq_grid = pmbp->prad->freq_grid;
  const int freq_scale = pmbp->prad->flag_fscale;
  const Real spec_norm = (spec_temp > 0.0) ?
                         erad/(6.0*SQR(SQR(spec_temp))) : 0.0;
  par_for("pgen_neutrino_singlezone_rad", DevExeSpace(), 0, nmb1, 0, n3 - 1, 0,
          n2 - 1, 0, n1 - 1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            Real u_tet[4];
            u_tet[0] = norm_to_tet(m,0,0,k,j,i)*wlor +
                       norm_to_tet(m,0,1,k,j,i)*wlor*vx +
                       norm_to_tet(m,0,2,k,j,i)*wlor*vy +
                       norm_to_tet(m,0,3,k,j,i)*wlor*vz;
            u_tet[1] = norm_to_tet(m,1,0,k,j,i)*wlor +
                       norm_to_tet(m,1,1,k,j,i)*wlor*vx +
                       norm_to_tet(m,1,2,k,j,i)*wlor*vy +
                       norm_to_tet(m,1,3,k,j,i)*wlor*vz;
            u_tet[2] = norm_to_tet(m,2,0,k,j,i)*wlor +
                       norm_to_tet(m,2,1,k,j,i)*wlor*vx +
                       norm_to_tet(m,2,2,k,j,i)*wlor*vy +
                       norm_to_tet(m,2,3,k,j,i)*wlor*vz;
            u_tet[3] = norm_to_tet(m,3,0,k,j,i)*wlor +
                       norm_to_tet(m,3,1,k,j,i)*wlor*vx +
                       norm_to_tet(m,3,2,k,j,i)*wlor*vy +
                       norm_to_tet(m,3,3,k,j,i)*wlor*vz;

            Real erad_freq = erad/static_cast<Real>(nfreq);
            for (int isp = 0; isp < nspecies; ++isp) {
              for (int ifr = 0; ifr < nfreq; ++ifr) {
                for (int n = 0; n < nang; ++n) {
                  Real un_t = u_tet[1]*nh_c.d_view(n,1) +
                              u_tet[2]*nh_c.d_view(n,2) +
                              u_tet[3]*nh_c.d_view(n,3);
                  Real n0_f = u_tet[0]*nh_c.d_view(n,0) - un_t;
                  Real n0 = tet_c(m,0,0,k,j,i);
                  Real n_0 = 0.0;
                  for (int d = 0; d < 4; ++d) {
                    n_0 += tetcov_c(m,d,0,k,j,i)*nh_c.d_view(n,d);
                  }
                  int nn = (isp*nfreq + ifr)*nang + n;
                  Real intensity_cm = erad_freq;
#ifdef ENABLE_NURATES
                  if (spec_temp > 0.0) {
                    Real e_lo = 0.0, e_hi = 0.0;
                    radiation::FreqBinEdgesMeV(freq_grid, ifr, nfreq, freq_scale,
                                               e_lo, e_hi);
                    intensity_cm = spec_norm*
                        (PlanckLikeIntegral(n0_f*e_hi, spec_temp) -
                         PlanckLikeIntegral(n0_f*e_lo, spec_temp));
                  }
#endif
                  i0(m, nn, k, j, i) =
                      n0*n_0*(intensity_cm/(4.0*M_PI))/SQR(SQR(n0_f));
                }
              }
            }
          });
}

} // namespace
