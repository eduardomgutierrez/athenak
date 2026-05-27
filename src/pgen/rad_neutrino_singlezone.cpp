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

namespace {
template <class EOSPolicy, class ErrorPolicy>
void SingleZoneImpl(Mesh *pmesh, ParameterInput *pin, const bool restart);
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

  auto &indcs = pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmbp->nmb_thispack - 1;
  int nang = pmbp->prad->prgeo->nangles;
  int nspecies = pmbp->prad->nspecies;
  int nfreq = pmbp->prad->nfreq;
  if (nfreq != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_neutrino_singlezone requires gray (nfreq=1) radiation" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  Real vy = pin->GetOrAddReal("problem", "vy", 0.0);
  Real vz = pin->GetOrAddReal("problem", "vz", 0.0);
  Real ye = pin->GetReal("problem", "Y_e");
  Real erad = pin->GetOrAddReal("problem", "erad", 0.0);
  Real wlor = 1.0/std::sqrt(1.0 - vx*vx - vy*vy - vz*vz);

  auto &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();
  Real nb = rho/mb;
  Real y[1] = {ye};
  Real press = eos.GetPressure(nb, temp, y);

  auto &w0 = pmbp->pmhd->w0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_neutrino_singlezone_mhd", DevExeSpace(), 0, nmb1, 0, n3 - 1, 0,
          n2 - 1, 0, n1 - 1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            w0(m, IDN, k, j, i) = rho;
            w0(m, IVX, k, j, i) = wlor*vx;
            w0(m, IVY, k, j, i) = wlor*vy;
            w0(m, IVZ, k, j, i) = wlor*vz;
            w0(m, IPR, k, j, i) = press;
            w0(m, IYF, k, j, i) = ye;

            bcc0(m, IBX, k, j, i) = 0.0;
            bcc0(m, IBY, k, j, i) = 0.0;
            bcc0(m, IBZ, k, j, i) = 0.0;
            b0.x1f(m, k, j, i) = 0.0;
            b0.x2f(m, k, j, i) = 0.0;
            b0.x3f(m, k, j, i) = 0.0;
            if (i == n1 - 1) b0.x1f(m, k, j, i + 1) = 0.0;
            if (j == n2 - 1) b0.x2f(m, k, j + 1, i) = 0.0;
            if (k == n3 - 1) b0.x3f(m, k + 1, j, i) = 0.0;
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

            for (int isp = 0; isp < nspecies; ++isp) {
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
                int nn = isp*nang + n;  // nfreq=1 guaranteed above
                i0(m, nn, k, j, i) =
                    n0*n_0*(erad/(4.0*M_PI))/SQR(SQR(n0_f));
              }
            }
          });
}

} // namespace
