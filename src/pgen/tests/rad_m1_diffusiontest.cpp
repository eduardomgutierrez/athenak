//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_diffusiontest.cpp
//  \brief 1D diffusion test in a moving, purely scattering medium.  Runs with grey M1
//         (<radiation_m1>) or with the discrete-ordinates solver (<radiation>); both are
//         seeded from the same comoving (J, H^x) so the two can be compared directly, and
//         initial_data=diffusion seeds the exact advection-diffusion solution.

#include <iostream>
#include <string>

#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

namespace {

enum InitialData {kGaussian, kStep, kDiffusion};

//----------------------------------------------------------------------------------------
//! \fn void ComovingState
//  \brief Comoving energy density J and flux H^x at lab event (x1, t).
//  kDiffusion is the exact solution of dJ/dt' = D d^2J/dx'^2 with H = -D dJ/dx', pulled
//  back to the lab slice through x' = W(x-vt), t' = W(t-vx); the other two are
//  comoving-isotropic pulses specified through their lab energy density.

KOKKOS_INLINE_FUNCTION
void ComovingState(const int ic, const Real x1, const Real t, const Real vx,
                   const Real wl, const Real dd, const Real nusq, const Real t0,
                   Real &jj, Real &hh) {
  if (ic == kDiffusion) {
    Real xp = wl*(x1 - vx*t);
    Real ss = 1.0 + 4.0*dd*nusq*(t0 + wl*(t - vx*x1));
    jj = Kokkos::exp(-nusq*xp*xp/ss)/Kokkos::sqrt(ss);
    hh = 2.0*dd*nusq*xp*jj/ss;
  } else {
    Real ee = (ic == kGaussian) ? Kokkos::exp(-9.0*x1*x1) : static_cast<Real>(x1 < 0.0);
    jj = 3.0*ee/(4.0*wl*wl - 1.0);
    hh = 0.0;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::RadiationM1DiffusionTest

void ProblemGenerator::RadiationM1DiffusionTest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool use_m1 = (pmbp->pradm1 != nullptr);
  bool use_sn = (pmbp->prad != nullptr);

  if (use_m1 == use_sn) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion test needs exactly one of <radiation_m1> and "
              << "<radiation> in the input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion test problem generator can only be run with one "
                 "dimension, but parfile grid setup is not in 1d" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string ic_str = pin->GetOrAddString("problem", "initial_data", "gaussian");
  int ic = kGaussian;
  if (ic_str == "step") {
    ic = kStep;
  } else if (ic_str == "diffusion") {
    ic = kDiffusion;
  } else if (ic_str != "gaussian") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown problem/initial_data='" << ic_str << "'; use gaussian, step "
              << "or diffusion" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Scattering opacity is owned by the pgen so that both solvers see the same sigma_s.
  Real kappa_s = pin->GetOrAddReal("problem", "kappa_s", 100.0);
  Real vx = pin->GetOrAddReal("problem", "fluid_velocity", 0.0);
  Real nusq = SQR(pin->GetOrAddReal("problem", "nu", 4.0));
  Real t0 = pin->GetOrAddReal("problem", "t0", 0.0);
  Real wl = 1.0/std::sqrt(1.0 - vx*vx);
  Real dd = 1.0/(3.0*kappa_s);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;

  if (use_m1) {
    if (pmbp->pradm1->nspecies != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "The 1d diffusion test problem generator can only be "
                << "run with one neutrino species only!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (pmbp->pradm1->params.src_update == radiationm1::Explicit) {
      pmbp->pradm1->toy_opacity_fn =
          radiationm1::ToyOpacity{radiationm1::ToyOpacityModel::DiffusionExplicit,
                                  kappa_s};
    } else {
      pmbp->pradm1->toy_opacity_fn =
          radiationm1::ToyOpacity{radiationm1::ToyOpacityModel::DiffusionImplicit,
                                  kappa_s};
    }

    auto &w0_ = pmbp->pradm1->w0;
    auto &u0_ = pmbp->pradm1->u0;
    auto &chi_ = pmbp->pradm1->chi;
    adm::ADM::ADM_vars &adm = pmbp->padm->adm;
    auto &params_ = pmbp->pradm1->params;

    par_for("pgen_diffusiontest_m1", DevExeSpace(), 0, nmb1, ksg, keg, jsg, jeg, isg,
            ieg, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
        }
      }
      adm.psi4(m, k, j, i) = 1.;
      adm.alpha(m, k, j, i) = 1.;
      // the initial data is P^ab = (J/3)(g^ab + u^a u^b); the closure is only solved
      // inside a stage, so seed it here to make the cycle-0 diagnostics meaningful
      chi_(m, 0, k, j, i) = 1.0/3.0;

      w0_(m, IVX, k, j, i) = wl*vx;
      w0_(m, IVY, k, j, i) = 0.;
      w0_(m, IVZ, k, j, i) = 0.;

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

      Real jj, hh;
      ComovingState(ic, x1, 0.0, vx, wl, dd, nusq, t0, jj, hh);
      // Lab moments of J u^a u^b + H^a u^b + u^a H^b + (J/3)(eta^ab + u^a u^b)
      Real ee = SQR(wl)*(jj*(1.0 + vx*vx/3.0) + 2.0*vx*hh);
      Real ffx = SQR(wl)*((4.0/3.0)*vx*jj + (1.0 + vx*vx)*hh);

      AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          g_uu(a, b) = 0;
        }
      }
      g_uu(0, 0) = -1;
      g_uu(1, 1) = 1;
      g_uu(2, 2) = 1;
      g_uu(3, 3) = 1;
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
      pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
               adm.beta_u(m, 2, k, j, i), ffx, 0, 0, F_d);
      radiationm1::apply_floor(g_uu, ee, F_d, params_);
      u0_(m, M1_E_IDX, k, j, i) = ee;
      u0_(m, M1_FX_IDX, k, j, i) = F_d(M1_FX_IDX);
      u0_(m, M1_FY_IDX, k, j, i) = F_d(M1_FY_IDX);
      u0_(m, M1_FZ_IDX, k, j, i) = F_d(M1_FZ_IDX);
    });
    return;
  }

  // ---------------------------------------------------------------- discrete ordinates
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion test with <radiation> needs a <hydro> block for the "
              << "background fluid" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->prad->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion test can only be run with one species" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Pure elastic scattering, matched to the M1 toy opacity.
  pmbp->prad->kappa_s = kappa_s;
  pmbp->prad->kappa_a = 0.0;
  pmbp->prad->kappa_p = 0.0;
  pmbp->prad->power_opacity = false;
  // With use_nurates the source terms read the per-species arrays instead, so feed
  // the same sigma_s through the toy hook and skip the library entirely.
  if (pmbp->prad->use_nurates) { pmbp->prad->nurates_toy_scattering = kappa_s; }

  Real uu1 = wl*vx;
  auto &w0 = pmbp->phydro->w0;
  par_for("pgen_diffusiontest_fluid", DevExeSpace(), 0, nmb1, ksg, keg, jsg, jeg, isg,
          ieg, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m, IDN, k, j, i) = 1.0;
    w0(m, IVX, k, j, i) = uu1;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = 1.0;
  });
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u0, 0, n1-1, 0, n2-1, 0, n3-1);

  int nang1 = pmbp->prad->prgeo->nangles - 1;
  int nang = pmbp->prad->prgeo->nangles;
  int nfreq = pmbp->prad->nfreq;
  auto &nh_c_ = pmbp->prad->nh_c;
  auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &tetcov_c_ = pmbp->prad->tetcov_c;
  auto &i0 = pmbp->prad->i0;

  par_for("pgen_diffusiontest_sn", DevExeSpace(), 0, nmb1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

    Real jj, hh;
    ComovingState(ic, x1, 0.0, vx, wl, dd, nusq, t0, jj, hh);

    Real uu0 = std::sqrt(1.0 + SQR(uu1));
    Real u_tet_[4];
    for (int d = 0; d < 4; ++d) {
      u_tet_[d] = norm_to_tet_(m,d,0,k,j,i)*uu0 + norm_to_tet_(m,d,1,k,j,i)*uu1;
    }

    for (int n = 0; n <= nang1; ++n) {
      Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                   u_tet_[3]*nh_c_.d_view(n,3));
      Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t;
      Real n1_f = (-u_tet_[1]*nh_c_.d_view(n,0) + u_tet_[1]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,1));

      // P1 intensity on the comoving unit direction n1_f/n0_f; its second moment is
      // exactly J/3, i.e. the Eddington closure the diffusion solution assumes.
      Real ii_f = fmax((jj + 3.0*hh*n1_f/n0_f)/(4.0*M_PI), 1.0e-30);

      Real n0 = tet_c_(m,0,0,k,j,i);
      Real n_0 = 0.0;
      for (int d = 0; d < 4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
      Real ii_lab = n0*n_0*ii_f/SQR(SQR(n0_f))/static_cast<Real>(nfreq);
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        i0(m, ifr*nang + n, k, j, i) = ii_lab;
      }
    }
  });

  return;
}
