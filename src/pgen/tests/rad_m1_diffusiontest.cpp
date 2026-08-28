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
//
//  With <radiation>/multi_freq the discrete-ordinates branch can carry a comoving
//  spectrum (problem/spec_width > 0) and an energy-dependent scattering opacity
//  (<radiation>/nurates_toy_scat_p != 0).  Both exist to give the frame-consistent
//  multi-frequency source term a problem with an absolute answer and a shape factor
//  S != 1 -- the combination no other test in the suite has, and the only thing that
//  can choose between the competing closure definitions in radiation_nurates.hpp.
//  The spectrum is a Gaussian in ln(E) sampled at bin midpoints, so ln(qbar) is
//  exactly quadratic in the bin index and the log-space cubic in ShiftedBinValue
//  reproduces it, and its shift by ln(n0_cm)/dlnnu, to roundoff.  The frequency
//  machinery is then exact by construction and the run measures the closure alone.
//
//  problem/spec_width = 0 (the default) reproduces the flat seed bit for bit.
//  pgen_final_func reports the L1 error against the analytic solution, per group.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

// error function for the discrete-ordinates branch, installed as pgen_final_func
void RadiationDiffusionErrors(ParameterInput *pin, Mesh *pm);

namespace {

enum InitialData {kGaussian, kStep, kDiffusion};

//----------------------------------------------------------------------------------------
//! \struct DiffusionVars
//  \brief Everything the exact solution needs, shared between the initial data and the
//  error function so the two cannot drift apart.

struct DiffusionVars {
  int ic;
  int nfreq;
  Real tol_l1, tol_l1_group;   // > 0 makes the error function a pass/fail test
  Real vx, wl, nusq, t0;
  Real kappa_s, dd;          // dd = 1/(3*kappa_s), the grey diffusion coefficient
  Real scat_p, scat_eref;    // sigma_s(e) = kappa_s*(e/scat_eref)^scat_p
  Real ln_numin, dln;        // log grid: ln(e_mid(ifr)) = ln_numin + (ifr-0.5)*dln
  Real spec_y0, spec_wid;    // comoving spectrum exp(-(y-y0)^2/(2 wid^2)), y = ln(E)
  Real spec_norm;            // sum of the rest-frame weights over bins 1..nfreq-1
};

DiffusionVars dvars;

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

//----------------------------------------------------------------------------------------
//! \fn Real SpecBinWeight
//  \brief Fraction of the comoving spectrum that falls in lab bin ifr as seen along a
//  ray with Doppler factor n0_cm, whose comoving bin is n0_cm*[e_lo, e_hi].
//
//  Midpoint rule in y = ln(E), so ln(weight) is exactly quadratic in the bin index and
//  the shift by ln(n0_cm)/dlnnu lands on the same parabola.  Bin 0 spans [0, nu_min],
//  sits outside the log family and is left empty.  spec_wid <= 0 gives the flat seed.

KOKKOS_INLINE_FUNCTION
Real SpecBinWeight(const DiffusionVars &d, const int ifr, const Real ln_n0cm) {
  if (d.spec_wid <= 0.0) { return 1.0; }
  if (ifr < 1) { return 0.0; }
  Real y = d.ln_numin + (static_cast<Real>(ifr) - 0.5)*d.dln;
  Real z = (y + ln_n0cm - d.spec_y0)/d.spec_wid;
  return Kokkos::exp(-0.5*z*z)/d.spec_norm;
}

//----------------------------------------------------------------------------------------
//! \fn void RayComoving
//  \brief Doppler factor n0_cm = -u_mu n^mu and comoving direction cosine along x for
//  tetrad ray (nh0..nh3).  Pulled out so the initial data and the error function use the
//  same expression.

KOKKOS_INLINE_FUNCTION
void RayComoving(const Real u_tet[4], const Real nh0, const Real nh1, const Real nh2,
                 const Real nh3, Real &n0_cm, Real &mu_cm) {
  Real un_t = u_tet[1]*nh1 + u_tet[2]*nh2 + u_tet[3]*nh3;
  n0_cm = u_tet[0]*nh0 - un_t;
  Real n1_f = -u_tet[1]*nh0 + u_tet[1]/(u_tet[0] + 1.0)*un_t + nh1;
  mu_cm = n1_f/n0_cm;
}

//----------------------------------------------------------------------------------------
//! \fn void BinComovingState
//  \brief (J, H^x) for lab bin ifr on a ray, with the bin's own diffusion coefficient.
//
//  Elastic scattering does not couple comoving groups, so each comoving energy diffuses
//  independently with D(E) = 1/(3 sigma_s(E)); the bin's comoving energy is n0_cm*e_mid,
//  which is where the whole frame-consistency question lives.  Bin 0 keeps the p = 0
//  opacity, matching CalcOpacityNuratesToy.

KOKKOS_INLINE_FUNCTION
void BinComovingState(const DiffusionVars &d, const int ifr, const Real x1,
                      const Real t, const Real n0_cm, Real &jj, Real &hh) {
  Real ddf = d.dd;
  if (d.scat_p != 0.0 && ifr > 0) {
    Real e_cm = n0_cm*Kokkos::exp(d.ln_numin + (static_cast<Real>(ifr) - 0.5)*d.dln);
    ddf = 1.0/(3.0*d.kappa_s*Kokkos::pow(e_cm/d.scat_eref, d.scat_p));
  }
  ComovingState(d.ic, x1, t, d.vx, d.wl, ddf, d.nusq, d.t0, jj, hh);
}

//----------------------------------------------------------------------------------------
//! \fn Real P1Margin
//  \brief (J + 3 mu H)/J, the P1 intensity in units of its isotropic part.
//
//  The advection-diffusion solution is only a solution while this stays positive: at
//  6 D nu^2 |x'|/s >= 1 the P1 reconstruction of the exact (J, H) is negative on some
//  rays, the floor below truncates it, and the seeded spectrum acquires a kink that the
//  log-space shift lookups turn into garbage.  With an energy-dependent opacity the low
//  groups have the largest D and hit this first, which is why the seed has to be checked
//  rather than assumed.  problem/t0 > 0 buys margin: s = 1 + 4 D nu^2 t0 at t = 0.

KOKKOS_INLINE_FUNCTION
Real P1Margin(const DiffusionVars &d, const int ifr, const Real x1, const Real t,
              const Real n0_cm, const Real mu_cm) {
  Real jj, hh;
  BinComovingState(d, ifr, x1, t, n0_cm, jj, hh);
  return (jj > 0.0) ? (jj + 3.0*hh*mu_cm)/jj : 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real ExactLabIntensity
//  \brief i0 for lab bin ifr on a ray, at lab event (x1, t).

KOKKOS_INLINE_FUNCTION
Real ExactLabIntensity(const DiffusionVars &d, const int ifr, const Real x1,
                       const Real t, const Real n0_cm, const Real mu_cm,
                       const Real n0, const Real n_0) {
  Real wb = SpecBinWeight(d, ifr, (d.spec_wid > 0.0) ? Kokkos::log(n0_cm) : 0.0);
  if (!(wb > 0.0)) { return 0.0; }
  Real jj, hh;
  BinComovingState(d, ifr, x1, t, n0_cm, jj, hh);
  Real ii_f = fmax((jj + 3.0*hh*mu_cm)/(4.0*M_PI), 1.0e-30);
  Real ii_lab = n0*n_0*ii_f/SQR(SQR(n0_cm));
  // Divide rather than multiply by 1/nfreq in the flat case: the pre-spectrum seed did,
  // and the grey/multifrequency bit-identity check is sensitive to the last bit.
  return (d.spec_wid > 0.0) ? ii_lab*wb
                            : ii_lab/static_cast<Real>(d.nfreq);
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

  // ------------------------------------------------------- comoving spectrum, if asked
  // spec_width is the standard deviation of the comoving spectrum in units of the bin
  // spacing dlnnu; 0 keeps the flat seed.  spec_peak is where it peaks, in the units of
  // <radiation>/nu_min (MeV on the nurates path); 0 puts it at the middle of the grid.
  Real spec_width = pin->GetOrAddReal("problem", "spec_width", 0.0);
  Real spec_peak = pin->GetOrAddReal("problem", "spec_peak", 0.0);
  Real scat_p = 0.0, scat_eref = 1.0;
#if ENABLE_NURATES
  scat_p = pmbp->prad->nurates_toy_scat_p;
  scat_eref = pmbp->prad->nurates_toy_scat_eref;
#endif
  int nfreq_in = pmbp->prad->nfreq;
  Real ln_numin = 0.0, dln = 0.0;
  if (spec_width > 0.0 || scat_p != 0.0) {
    if (!pmbp->prad->multi_freq) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "problem/spec_width and nurates_toy_scat_p need "
                << "<radiation>/multi_freq = true" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (pmbp->prad->flag_fscale != 1 || nfreq_in < 5) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "the spectral diffusion test needs "
                << "<radiation>/freq_scale = log and nfreq >= 5" << std::endl;
      exit(EXIT_FAILURE);
    }
    ln_numin = std::log(pmbp->prad->nu_min);
    dln = std::log(pmbp->prad->nu_max/pmbp->prad->nu_min)/(nfreq_in - 2);
  }

  dvars.ic = ic;
  dvars.nfreq = nfreq_in;
  // > 0 turns the final error report into a regression test.  Left at 0 by every
  // pre-existing input, which only ever printed the numbers.
  dvars.tol_l1 = pin->GetOrAddReal("problem", "tol_l1", 0.0);
  dvars.tol_l1_group = pin->GetOrAddReal("problem", "tol_l1_group", 0.0);
  dvars.vx = vx;
  dvars.wl = wl;
  dvars.nusq = nusq;
  dvars.t0 = t0;
  dvars.kappa_s = kappa_s;
  dvars.dd = dd;
  dvars.scat_p = scat_p;
  dvars.scat_eref = scat_eref;
  dvars.ln_numin = ln_numin;
  dvars.dln = dln;
  dvars.spec_wid = spec_width*dln;
  dvars.spec_y0 = (spec_peak > 0.0) ? std::log(spec_peak)
                : ln_numin + (0.5*nfreq_in - 0.5)*dln;
  dvars.spec_norm = 1.0;
  if (spec_width > 0.0) {
    Real acc = 0.0;
    for (int f = 1; f < nfreq_in; ++f) {
      Real z = (ln_numin + (f - 0.5)*dln - dvars.spec_y0)/dvars.spec_wid;
      acc += std::exp(-0.5*z*z);
    }
    dvars.spec_norm = acc;
    // How much of the spectrum a ray loses off the ends of the grid.  This is the seed
    // truncation that made max|F/J| in the scattering test 98% initial data: the sum of
    // the weights has to be ray independent, or the initial data is already anisotropic.
    Real worst = 0.0;
    for (Real ncm : {wl*(1.0 - vx), wl*(1.0 + vx)}) {
      Real acc2 = 0.0;
      for (int f = 1; f < nfreq_in; ++f) {
        Real z = (ln_numin + (f - 0.5)*dln + std::log(ncm) - dvars.spec_y0)
                 /dvars.spec_wid;
        acc2 += std::exp(-0.5*z*z);
      }
      worst = std::max(worst, std::fabs(acc2/acc - 1.0));
    }
    std::cout << "### rad_m1_diffusiontest: comoving spectrum sigma = " << spec_width
              << " bins, peak at exp(" << dvars.spec_y0 << ") = "
              << std::exp(dvars.spec_y0) << ", worst ray-to-ray coverage error "
              << std::scientific << std::setprecision(3) << worst << std::endl;
    // Two ways to lose the ray-independence of the total, and they pull opposite ways:
    // a wide spectrum runs off the ends of the grid, and a narrow one is under-sampled
    // by the bins, so shifting it past the midpoints changes the discrete sum (the
    // error is ~exp(-2 pi^2 (sigma/dlnnu)^2)).  Keep spec_width near or above 1 bin and
    // at least ~8.5 sigma from either end.
    if (worst > 1.0e-6) {
      std::cout << "### WARNING in " << __FILE__ << ": the spectrum is not resolved by "
                << "and contained in the grid; the seed is anisotropic before any source "
                << "term runs.  Adjust spec_width (target ~1-1.5 bins) or widen "
                << "[nu_min, nu_max]." << std::endl;
    }
  }
  pgen_final_func = RadiationDiffusionErrors;

  Real uu1 = wl*vx;
  Real uu0 = std::sqrt(1.0 + SQR(uu1));   // host: keep std:: math out of the kernels
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

  auto dv = dvars;
  par_for("pgen_diffusiontest_sn", DevExeSpace(), 0, nmb1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

    Real u_tet_[4];
    for (int d = 0; d < 4; ++d) {
      u_tet_[d] = norm_to_tet_(m,d,0,k,j,i)*uu0 + norm_to_tet_(m,d,1,k,j,i)*uu1;
    }

    for (int n = 0; n <= nang1; ++n) {
      // P1 intensity on the comoving unit direction; its second moment is exactly J/3,
      // i.e. the Eddington closure the diffusion solution assumes.
      Real n0_f, mu_f;
      RayComoving(u_tet_, nh_c_.d_view(n,0), nh_c_.d_view(n,1), nh_c_.d_view(n,2),
                  nh_c_.d_view(n,3), n0_f, mu_f);
      Real n0 = tet_c_(m,0,0,k,j,i);
      Real n_0 = 0.0;
      for (int d = 0; d < 4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        i0(m, ifr*nang + n, k, j, i) =
            ExactLabIntensity(dv, ifr, x1, 0.0, n0_f, mu_f, n0, n_0);
      }
    }
  });

  // Is the seed actually a solution?  See P1Margin.  Checked over the active cells and
  // every (ray, group) that carries weight; a floored entry is silent otherwise, and
  // with an energy-dependent opacity it poisons the spectrum the shift lookups read.
  {
    Real p1min = 1.0e300;
    int nxa = indcs.nx1;
    const int nmi = (pmbp->nmb_thispack)*nxa;
    Kokkos::parallel_reduce("raddiff-seedcheck",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmi),
    KOKKOS_LAMBDA(const int &idx, Real &lmin) {
      int m = idx/nxa;
      int i = (idx - m*nxa) + is;
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1 = CellCenterX(i - is, nxa, x1min, x1max);
      Real u_tet_[4];
      for (int d = 0; d < 4; ++d) {
        u_tet_[d] = norm_to_tet_(m,d,0,ks,js,i)*uu0 + norm_to_tet_(m,d,1,ks,js,i)*uu1;
      }
      for (int n = 0; n <= nang1; ++n) {
        Real n0_f, mu_f;
        RayComoving(u_tet_, nh_c_.d_view(n,0), nh_c_.d_view(n,1), nh_c_.d_view(n,2),
                    nh_c_.d_view(n,3), n0_f, mu_f);
        for (int ifr = 0; ifr < nfreq; ++ifr) {
          Real wb = SpecBinWeight(dv, ifr, (dv.spec_wid > 0.0) ? log(n0_f) : 0.0);
          if (!(wb > 1.0e-14)) { continue; }
          lmin = fmin(lmin, P1Margin(dv, ifr, x1, 0.0, n0_f, mu_f));
        }
      }
    }, Kokkos::Min<Real>(p1min));
    std::cout << "### rad_m1_diffusiontest: min (J + 3 mu H)/J over the seed = "
              << std::scientific << std::setprecision(3) << p1min << std::endl;
    if (p1min <= 0.0) {
      std::cout << "### WARNING in " << __FILE__ << ": the seed is not a solution -- the "
                << "P1 reconstruction is negative on some rays and has been floored, so "
                << "the comoving spectrum has a kink before any source term runs.  "
                << "Raise problem/t0 or problem/kappa_s, or shrink nu/the domain."
                << std::endl;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RadiationDiffusionErrors
//  \brief L1 error of the discrete-ordinates intensity against the exact solution at the
//  final time, in total and per frequency group.
//
//  The per-group numbers are the point: a closure that returns the wrong bin-mean
//  intensity distorts the comoving spectrum, which shows as an error that varies across
//  groups even where the total is unremarkable.

void RadiationDiffusionErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->prad == nullptr) { return; }

  auto &indcs = pm->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  int nang = pmbp->prad->prgeo->nangles;
  int nfreq = pmbp->prad->nfreq;
  auto &nh_c_ = pmbp->prad->nh_c;
  auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &tetcov_c_ = pmbp->prad->tetcov_c;
  auto &i0 = pmbp->prad->i0;
  auto &w0 = pmbp->phydro->w0;
  Real tnow = pm->time;
  auto dv = dvars;

  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  std::cout << "### rad_m1_diffusiontest errors at t = " << std::scientific
            << std::setprecision(6) << tnow << " (v = " << dv.vx << ", nfreq = "
            << nfreq << ", spec sigma/dlnnu = "
            << ((dv.dln > 0.0) ? dv.spec_wid/dv.dln : 0.0)
            << ", sigma_s index = " << dv.scat_p << ")" << std::endl;
  std::cout << "  group        L1(exact)         L1(err)     rel" << std::endl;

  Real tot_err = 0.0, tot_ref = 0.0, worst_rel = 0.0;
  int worst_grp = -1;
  std::vector<Real> grp_ref(nfreq, 0.0), grp_rel(nfreq, 0.0);
  for (int ifr = 0; ifr < nfreq; ++ifr) {
    Real sum_err = 0.0, sum_ref = 0.0;
    Kokkos::parallel_reduce("raddiff-err", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &lerr, Real &lref) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

      Real uu1 = w0(m, IVX, k, j, i);
      Real uu0 = sqrt(1.0 + SQR(uu1));
      Real u_tet_[4];
      for (int d = 0; d < 4; ++d) {
        u_tet_[d] = norm_to_tet_(m,d,0,k,j,i)*uu0 + norm_to_tet_(m,d,1,k,j,i)*uu1;
      }
      for (int n = 0; n < nang; ++n) {
        Real n0_f, mu_f;
        RayComoving(u_tet_, nh_c_.d_view(n,0), nh_c_.d_view(n,1), nh_c_.d_view(n,2),
                    nh_c_.d_view(n,3), n0_f, mu_f);
        Real n0 = tet_c_(m,0,0,k,j,i);
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) { n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d); }
        Real ex = ExactLabIntensity(dv, ifr, x1, tnow, n0_f, mu_f, n0, n_0);
        lerr += vol*fabs(i0(m, ifr*nang + n, k, j, i) - ex);
        lref += vol*fabs(ex);
      }
    }, Kokkos::Sum<Real>(sum_err), Kokkos::Sum<Real>(sum_ref));

    tot_err += sum_err;
    tot_ref += sum_ref;
    Real rel = (sum_ref > 0.0) ? sum_err/sum_ref : 0.0;
    grp_ref[ifr] = sum_ref;
    grp_rel[ifr] = rel;
    std::printf("  %5d  %14.6e  %14.6e  %10.3e\n", ifr, sum_ref, sum_err, rel);
  }
  // Groups holding a negligible slice of the spectrum have a meaningless ratio, so the
  // worst-group figure is taken only over groups carrying >= 0.1% of the total.  This
  // needs the whole total, not the running one.
  for (int ifr = 0; ifr < nfreq; ++ifr) {
    if (grp_ref[ifr] > 1.0e-3*tot_ref && grp_rel[ifr] > worst_rel) {
      worst_rel = grp_rel[ifr];
      worst_grp = ifr;
    }
  }
  Real tot_rel = (tot_ref > 0.0) ? tot_err/tot_ref : 0.0;
  std::printf("  total  %14.6e  %14.6e  %10.3e\n", tot_ref, tot_err, tot_rel);
  std::printf("### raddiff-summary t=%.6e v=%.3f nfreq=%d p=%.2f "
              "rel_total=%.6e rel_worst=%.6e worst_group=%d\n",
              tnow, dv.vx, nfreq, dv.scat_p, tot_rel, worst_rel, worst_grp);

  if (dv.tol_l1 <= 0.0 && dv.tol_l1_group <= 0.0) { return; }
  // The multi-frequency source term the tolerances are set for is the nurates one; the
  // built-in multifrequency path is a different scheme and would fail for reasons that
  // have nothing to do with what this test pins.
  if (!pmbp->prad->use_nurates) {
    std::cout << "### rad_m1_diffusiontest: tolerance check skipped, not running "
              << "use_nurates" << std::endl;
    return;
  }
  bool bad = (dv.tol_l1 > 0.0 && !(tot_rel <= dv.tol_l1))
          || (dv.tol_l1_group > 0.0 && !(worst_rel <= dv.tol_l1_group));
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << ": diffusion L1 error "
              << std::scientific << std::setprecision(6) << tot_rel << " (worst group "
              << worst_rel << ") exceeds problem/tol_l1 = " << dv.tol_l1
              << " (tol_l1_group = " << dv.tol_l1_group << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "### rad_m1_diffusiontest: within tolerance" << std::endl;
}
