//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_chiral_dynamo.cpp
//! \brief force-free Beltrami field for the chiral dynamo, with an exact growth rate
//!
//! Static, uniform matter threaded by a circularly polarised field varying along
//! x1 only,
//!
//!     B = B0 (0, cos(k x), sigma sin(k x)),   k = 2 pi n / L,  sigma = +-1,
//!
//! with uniform Ye and Y5, so xi is a constant.  The field is Beltrami:
//! curl B = -sigma k B, so the Lorentz force J x B vanishes and b^2 = B0^2 is
//! uniform.  The fluid therefore stays at rest and the only evolution is the
//! dynamo itself.
//!
//! With v = 0 the Ohm's law e^mu = xi b^mu gives E = xi B, so
//!
//!     d_t B = -curl E = -xi curl B = sigma xi k B,
//!
//! i.e. every component grows or decays exponentially at the rate
//!
//!     Gamma = sigma * xi * k,     xi = -(4/pi) alpha^2 ln(1/alpha) (Y5/Ye)^(1/3)
//!
//! with the field pattern held fixed.  Because xi < 0 for Y5, Ye > 0, sigma = +1
//! decays and sigma = -1 grows; the magnitude |xi k| is the same either way, so
//! running both helicities checks the sign of the implementation as well as its
//! amplitude.  This is the standard chiral plasma instability, and it is the
//! sharpest available test of the EMF: it is a pure exponential with no free
//! parameters.
//!
//! Set <mhd>/chiral_dynamo = false to confirm the field is otherwise stationary.
//!
//! History output: max|B^y|, max|B^z|, and the volume-averaged b^2, whose log
//! slope is 2*Gamma.

#include <math.h>

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "mhd/chiral_dynamo.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "pgen/pgen.hpp"

namespace {

// history: max|B^y|, max|B^z|, <b^2>
void ChiralDynamoHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem
//! \brief Beltrami field in a static, uniform, dynamically-GR background

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &ChiralDynamoHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dyngr_chiral_dynamo requires <mhd>" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->nscalars < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dyngr_chiral_dynamo requires <mhd>/nscalars = 2 "
              << "(slot 0 = Ye, slot 1 = Y5)" << std::endl;
    exit(EXIT_FAILURE);
  }

  const Real rho0  = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real pgas0 = pin->GetOrAddReal("problem", "pgas", 1.0);
  const Real b0    = pin->GetOrAddReal("problem", "b0", 1.0e-4);
  const Real ye0   = pin->GetOrAddReal("problem", "Y_e", 0.1);
  const Real y50   = pin->GetOrAddReal("problem", "Y_5", 1.0e-3);
  const Real nwave = pin->GetOrAddReal("problem", "nwave", 1.0);
  // helicity: +1 decays, -1 grows (xi < 0 for Y5, Ye > 0)
  const Real sigma = pin->GetOrAddReal("problem", "helicity", 1.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int &is = indcs.is, &ie = indcs.ie;
  const int &js = indcs.js, &je = indcs.je;
  const int &ks = indcs.ks, &ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // wavenumber from the full mesh extent, so the pattern is periodic on the mesh
  const Real xlen = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  const Real kx = 2.0*M_PI*nwave/xlen;

  // report the analytic rate once, for comparison with the measured slope
  const Real xi_an = chiral::Xi(y50, ye0, chiral::kAlphaEM);
  if (global_variable::my_rank == 0) {
    std::cout << "### dyngr_chiral_dynamo: xi = " << xi_an << ", k = " << kx
              << ", analytic growth rate Gamma = sigma*xi*k = "
              << sigma*xi_an*kx << std::endl;
  }

  // ---- primitives: static, uniform ----
  auto &w0 = pmbp->pmhd->w0;
  par_for("pgen_chiral_w", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = rho0;
    w0(m,IPR,k,j,i) = pgas0;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IYF  ,k,j,i) = ye0;
    w0(m,IYF+1,k,j,i) = y50;
  });

  // ---- magnetic field ----
  // B^x = 0, and B^y, B^z depend on x1 only, so div B = 0 identically.  The
  // x2- and x3-faces sit at the cell centre in x1, so both use x1v.
  auto &b0_ = pmbp->pmhd->b0;
  auto &bcc0_ = pmbp->pmhd->bcc0;
  par_for("pgen_chiral_b", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real &x1min = size.d_view(m).x1min;
    const Real &x1max = size.d_view(m).x1max;
    const int nx1 = indcs.nx1;
    const Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    if (k <= ke && j <= je) { b0_.x1f(m,k,j,i) = 0.0; }
    if (k <= ke && i <= ie) { b0_.x2f(m,k,j,i) =       b0*cos(kx*x1v); }
    if (j <= je && i <= ie) { b0_.x3f(m,k,j,i) = sigma*b0*sin(kx*x1v); }

    if (k <= ke && j <= je && i <= ie) {
      bcc0_(m,IBX,k,j,i) = 0.0;
      bcc0_(m,IBY,k,j,i) =       b0*cos(kx*x1v);
      bcc0_(m,IBZ,k,j,i) = sigma*b0*sin(kx*x1v);
    }
  });

  // ---- conserved variables ----
  // The ADM metric must exist first: PrimToCons needs sqrt(det g), and an
  // unset metric gives det g = 0 and NaN conserved variables.
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void ChiralDynamoHistory
//! \brief max|B^y|, max|B^z| and the volume-averaged b^2
//!
//! d(ln <b^2>)/dt = 2*Gamma, which is what the test measures.

void ChiralDynamoHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 3;
  pdata->label[0] = "by-max";
  pdata->label[1] = "bz-max";
  pdata->label[2] = "bsq-avg";

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  Real by_max = 0.0, bz_max = 0.0, bsq_sum = 0.0, vol_sum = 0.0;
  Kokkos::parallel_reduce("chiral_dynamo_hist", Kokkos::RangePolicy<>(DevExeSpace(),
                          0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_by, Real &mb_bz, Real &mb_bsq, Real &mb_vol) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // bcc0 holds the densitised field; divide by sqrt(det g) for B^i
    const Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                      adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                      adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    const Real sqrtg = sqrt(detg);
    const Real bx = bcc0_(m,IBX,k,j,i)/sqrtg;
    const Real by = bcc0_(m,IBY,k,j,i)/sqrtg;
    const Real bz = bcc0_(m,IBZ,k,j,i)/sqrtg;

    // g_ij B^i B^j
    const Real bsq = adm.g_dd(m,0,0,k,j,i)*bx*bx + 2.0*adm.g_dd(m,0,1,k,j,i)*bx*by
                   + 2.0*adm.g_dd(m,0,2,k,j,i)*bx*bz + adm.g_dd(m,1,1,k,j,i)*by*by
                   + 2.0*adm.g_dd(m,1,2,k,j,i)*by*bz + adm.g_dd(m,2,2,k,j,i)*bz*bz;

    const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3*sqrtg;

    mb_by = fmax(fabs(by), mb_by);
    mb_bz = fmax(fabs(bz), mb_bz);
    mb_bsq += bsq*vol;
    mb_vol += vol;
  }, Kokkos::Max<Real>(by_max), Kokkos::Max<Real>(bz_max),
     Kokkos::Sum<Real>(bsq_sum), Kokkos::Sum<Real>(vol_sum));

#if MPI_PARALLEL_ENABLED
  Real gmax[2] = {by_max, bz_max};
  Real gsum[2] = {bsq_sum, vol_sum};
  MPI_Allreduce(MPI_IN_PLACE, gmax, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, gsum, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  by_max = gmax[0]; bz_max = gmax[1];
  bsq_sum = gsum[0]; vol_sum = gsum[1];
#endif

  pdata->hdata[0] = by_max;
  pdata->hdata[1] = bz_max;
  pdata->hdata[2] = (vol_sum > 0.0) ? bsq_sum/vol_sum : 0.0;
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
}

}  // namespace
