//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_nu_sphere.cpp
//! \brief Homogeneous neutrino-emitting sphere in vacuum, with bns_nurates rates and the
//! multifrequency Boltzmann solver.
//!
//! A sphere of radius R at (rho, T, Ye) sits in a cold, thin atmosphere and starts with
//! no neutrinos.  Run it with frozen matter (<mhd>/fixed = true) so the sphere is a
//! steady source.  With absorption and emission only (<bns_nurates>/use_iso, use_pair,
//! use_brem = false) and the predictor off (use_partial_equilibrium = false), each
//! species and bin has a constant absorption opacity kappa and source B = eta/kappa, and
//! the steady luminosity of a homogeneous sphere is exact (Smit, van den Horn & Bludman
//! 1997; Abdikamalov et al. 2012):
//!
//!   L = pi R^2 B [1 - (1 - (1 + 2 tau) exp(-2 tau))/(2 tau^2)],   tau = kappa R,
//!
//! with B the bin-integrated equilibrium energy density.  The thin limit is the volume
//! emission (4 pi/3) R^3 kappa B; the thick limit is the surface flux B/4.
//!
//! The pgen evaluates kappa and eta per bin with the solver's own bns_nurates_spectral_bin
//! at the sphere's state, so the comparison tests the transport and not the binning, and
//! writes the analytic luminosities to sphere_analytic.txt.  R is the effective radius of
//! the staircase sphere, (3 N_in dV/4 pi)^(1/3).
//!
//! problem/vx, vy, vz give the sphere matter a uniform velocity (the atmosphere stays at
//! rest); the analytic table is then only the rest-frame reference.
//!
//! problem/user_hist = true adds a user history with the energy and number luminosity of
//! each species through the sphere of radius problem/hist_radius (default 12), computed
//! as the torus history does (radiation/radiation_nu_hist.hpp).

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_nu_hist.hpp"
#if ENABLE_NURATES
#include "radiation/radiation_nurates.hpp"
#endif

namespace {
template <class EOSPolicy, class ErrorPolicy>
void SphereImpl(Mesh *pmesh, ParameterInput *pin, const bool restart);
void SphereNuHistory(HistoryData *pdata, Mesh *pm);
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // user history: neutrino luminosities through a sphere (set up on restarts too)
  if (user_hist) {
    const Real rhist = pin->GetOrAddReal("problem", "hist_radius", 12.0);
    spherical_grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rhist));
    user_hist_func = SphereNuHistory;
  }
  using NQTPolicy = Primitive::EOSCompOSE<Primitive::NQTLogs>;
  using LogPolicy = Primitive::EOSCompOSE<Primitive::NormalLogs>;
  if (dynamic_cast<dyngr::DynGRMHDPS<NQTPolicy, Primitive::ResetFloor> *>(pmbp->pdyngr)
      != nullptr) {
    return SphereImpl<NQTPolicy, Primitive::ResetFloor>(pmy_mesh_, pin, restart);
  }
  if (dynamic_cast<dyngr::DynGRMHDPS<LogPolicy, Primitive::ResetFloor> *>(pmbp->pdyngr)
      != nullptr) {
    return SphereImpl<LogPolicy, Primitive::ResetFloor>(pmy_mesh_, pin, restart);
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "rad_nu_sphere requires DynGRMHD with EOSCompOSE" << std::endl;
  std::exit(EXIT_FAILURE);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void SphereNuHistory
//! \brief L and Ndot per species through each spherical grid: L<species>_<r> in code
//!        luminosity, N<species>_<r> in code luminosity per MeV (radiation_nu_hist.cpp)

void SphereNuHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &grids = pm->pgen->spherical_grids;
  const int nradii = grids.size();
  const int nsp = pmbp->prad->nspecies;
  pdata->nhist = 2*nsp*nradii;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DvceArray5D<Real> mom;
  radiation::NuFluxMoments(pmbp, mom);
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    std::vector<Real> lum(nsp), ndot(nsp);
    radiation::NuSphereLuminosities(grids[g].get(), mom, nsp,
                                    pmbp->pcoord->coord_data.bh_spin,
                                    lum.data(), ndot.data());
    for (int isp=0; isp<nsp; ++isp) {
      std::string tag = radiation::NuHistSpeciesTag(isp, nsp);
      pdata->label[2*nsp*g+isp] = "L" + tag + "_" + rad_str;
      pdata->label[2*nsp*g+nsp+isp] = "N" + tag + "_" + rad_str;
      pdata->hdata[2*nsp*g+isp] = lum[isp];
      pdata->hdata[2*nsp*g+nsp+isp] = ndot[isp];
    }
  }
}

template <class EOSPolicy, class ErrorPolicy>
void SphereImpl(Mesh *pmesh, ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmesh->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_nu_sphere requires <mhd> with DynGRMHD and <adm>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->prad == nullptr || !pmbp->prad->use_nurates || !pmbp->prad->multi_freq) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_nu_sphere requires multifrequency <radiation> with use_nurates"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real radius = pin->GetReal("problem", "radius");
  const Real xc = pin->GetOrAddReal("problem", "x0", 0.0);
  const Real yc = pin->GetOrAddReal("problem", "y0", 0.0);
  const Real zc = pin->GetOrAddReal("problem", "z0", 0.0);
  const Real rho_in = pin->GetReal("problem", "rho_in");
  const Real temp_in = pin->GetReal("problem", "temp_in");
  const Real ye_in = pin->GetReal("problem", "ye_in");
  const Real rho_out = pin->GetReal("problem", "rho_out");
  const Real temp_out = pin->GetReal("problem", "temp_out");
  const Real ye_out = pin->GetOrAddReal("problem", "ye_out", ye_in);
  const Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  const Real vy = pin->GetOrAddReal("problem", "vy", 0.0);
  const Real vz = pin->GetOrAddReal("problem", "vz", 0.0);
  const Real wlor = 1.0/std::sqrt(1.0 - vx*vx - vy*vy - vz*vz);

  auto &eos = static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
                  ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();

  auto &indcs = pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int ng = indcs.ng;
  const int n1 = nx1 + 2*ng;
  const int n2 = (nx2 > 1) ? (nx2 + 2*ng) : 1;
  const int n3 = (nx3 > 1) ? (nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;

  // Matter: the sphere where the cell centre is inside, atmosphere elsewhere
  auto &w0 = pmbp->pmhd->w0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_nu_sphere_mhd", DevExeSpace(), 0, nmb1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real r2 = SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
    bool in = (r2 < SQR(radius));
    Real rho = in ? rho_in : rho_out;
    Real temp = in ? temp_in : temp_out;
    Real ye = in ? ye_in : ye_out;
    Real w = in ? wlor : 1.0;
    w0(m, IDN, k, j, i) = rho;
    w0(m, IVX, k, j, i) = in ? w*vx : 0.0;
    w0(m, IVY, k, j, i) = in ? w*vy : 0.0;
    w0(m, IVZ, k, j, i) = in ? w*vz : 0.0;
    w0(m, IPR, k, j, i) = eos.GetPressure(rho/mb, temp, &ye);
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
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(0, n1 - 1, 0, n2 - 1, 0, n3 - 1);

  // No neutrinos at the start
  Kokkos::deep_copy(pmbp->prad->i0, 0.0);

  // Effective radius of the staircase sphere
  Real nin = 0.0;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  Kokkos::parallel_reduce("pgen_nu_sphere_count",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1 + 1)*nkji),
  KOKKOS_LAMBDA(const int &idx, Real &l_nin) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = idx - m*nkji - k*nji - j*nx1;
    Real x = CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real dv = ((size.d_view(m).x1max - size.d_view(m).x1min)/nx1)*
              ((size.d_view(m).x2max - size.d_view(m).x2min)/nx2)*
              ((size.d_view(m).x3max - size.d_view(m).x3min)/nx3);
    if (SQR(x - xc) + SQR(y - yc) + SQR(z - zc) < SQR(radius)) l_nin += dv;
  }, Kokkos::Sum<Real>(nin));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nin, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  const Real r_eff = std::cbrt(3.0*nin/(4.0*M_PI));

#if ENABLE_NURATES
  // Per species and bin: kappa and B = eta/kappa at the sphere's state, exactly as
  // CalcOpacityNurates evaluates them (radiation_nurates.cpp), with the predictor off.
  auto *prad = pmbp->prad;
  const radiation::NuratesParams params = prad->nurates_params;
  const int nfreq = prad->nfreq;
  const int nspecies = prad->nspecies;
  const int freq_scale = prad->flag_fscale;
  auto freq_grid = prad->freq_grid;
  auto code_units = eos.GetCodeUnitSystem();
  auto eos_units = eos.GetEOSUnitSystem();
  DvceArray3D<Real> tab("nu_sphere_tab", nspecies, nfreq, 4);
  DvceArray1D<Real> state("nu_sphere_state", 6);
  Kokkos::parallel_for("pgen_nu_sphere_rates", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int) {
    Real nb = rho_in/mb;
    Real ye = ye_in;
    Real p = eos.GetPressure(nb, temp_in, &ye);
    Real T = eos.GetTemperatureFromP(nb, p, &ye);
    Real yp = eos.GetProtonFraction(nb, T, &ye);
    Real yn = eos.GetNeutronFraction(nb, T, &ye);
    Real mu_b = eos.GetBaryonChemicalPotential(nb, T, &ye);
    Real mu_q = eos.GetChargeChemicalPotential(nb, T, &ye);
    Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &ye);
    Real mu_n = mu_b, mu_p = mu_b + mu_q, mu_e = mu_le - mu_q;
    state(0) = T; state(1) = yp; state(2) = yn;
    state(3) = mu_n; state(4) = mu_p; state(5) = mu_e;
    auto cu = code_units;
    auto eu = eos_units;
    radiation::NuratesGroupOccupancy spec = {};
    radiation::NuratesCellState cell = radiation::NuratesSpectralCellState(
        nb, T, yp, yn, mu_n, mu_p, mu_e, spec, params, cu, eu);
    for (int ifr = 0; ifr < nfreq; ++ifr) {
      Real e_lo = 0.0, e_hi = 0.0;
      radiation::FreqBinEdgesMeV(freq_grid, ifr, nfreq, freq_scale, e_lo, e_hi);
      Real eta_0[4], eta_1[4], abs_0[4], abs_1[4], scat_1[4];
      radiation::bns_nurates_spectral_bin(cell, e_lo, e_hi, freq_scale, eta_0, eta_1,
                                          abs_0, abs_1, scat_1, params);
      for (int isp = 0; isp < nspecies; ++isp) {
        tab(isp, ifr, 0) = e_lo;
        tab(isp, ifr, 1) = e_hi;
        tab(isp, ifr, 2) = abs_1[isp];
        tab(isp, ifr, 3) = (abs_1[isp] > 0.0) ? eta_1[isp]/abs_1[isp] : 0.0;
      }
    }
  });
  auto tab_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), tab);
  auto state_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), state);

  if (global_variable::my_rank == 0) {
    if (params.use_partial_equilibrium || params.use_iso || params.use_pair ||
        params.use_brem) {
      std::cout << "### rad_nu_sphere: WARNING the analytic luminosity assumes "
                << "use_partial_equilibrium, use_iso, use_pair and use_brem all false"
                << std::endl;
    }
    std::cout << "### rad_nu_sphere: R = " << radius << ", R_eff = " << r_eff
              << ", T = " << state_h(0) << " MeV, Yp = " << state_h(1) << ", Yn = "
              << state_h(2) << ", mu_n, mu_p, mu_e = " << state_h(3) << ", "
              << state_h(4) << ", " << state_h(5) << " MeV" << std::endl;
    FILE *fp = std::fopen("sphere_analytic.txt", "w");
    std::fprintf(fp, "# homogeneous sphere, R_eff = %.8e (code units)\n", r_eff);
    std::fprintf(fp, "# species bin e_lo[MeV] e_hi[MeV] kappa[1/code] B[code edens] "
                     "tau=kappa*R_eff L_exact[code]\n");
    Real ltot = 0.0;
    for (int isp = 0; isp < nspecies; ++isp) {
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        Real kap = tab_h(isp, ifr, 2), bb = tab_h(isp, ifr, 3);
        Real tau = kap*r_eff;
        Real g = (tau > 1.0e-4) ?
                 1.0 - (1.0 - (1.0 + 2.0*tau)*std::exp(-2.0*tau))/(2.0*tau*tau) :
                 4.0*tau/3.0;
        Real lum = M_PI*r_eff*r_eff*bb*g;
        ltot += lum;
        std::fprintf(fp, "%d %d %.8e %.8e %.8e %.8e %.8e %.8e\n", isp, ifr,
                     tab_h(isp, ifr, 0), tab_h(isp, ifr, 1), kap, bb, tau, lum);
      }
    }
    std::fclose(fp);
    std::cout << "### rad_nu_sphere: exact total luminosity " << ltot
              << " (code units), per bin in sphere_analytic.txt" << std::endl;
  }

  // The continuum solution, for a sphere whose matter moves with uniform velocity v (and
  // for v = 0, where it differs from the table above only by the binning).  The region is
  // fixed in the lab, so along a lab ray of direction n the coefficients are uniform:
  // with D = 1/(W (1 - v.n)), kappa_lab(E) = kappa_0(E/D)/D and I_lab(E) = D^3 I_0(E/D),
  // where I_0 = u_eq/(4 pi) is the comoving equilibrium intensity per unit energy.  Every
  // ray then carries I_lab (1 - exp(-kappa_lab s)) out of the chord s, and the sum over
  // impact parameters is pi R^2 g(kappa_lab R), as in the static case, so
  //   L_bin = int_bin dE int dOmega pi R^2 I_lab(E, n) g(kappa_lab(E, n) R).
  // kappa_0 and u_eq are tabulated on a fine log grid with the solver's own functions.
  constexpr int kNE = 3000;
  const Real e_min = 0.02, e_max = 5000.0;
  DvceArray2D<Real> fine("nu_sphere_fine", 4, kNE);
  Kokkos::parallel_for("pgen_nu_sphere_fine", Kokkos::RangePolicy<>(DevExeSpace(), 0, kNE),
  KOKKOS_LAMBDA(const int ie) {
    Real e = e_min*exp(log(e_max/e_min)*ie/(kNE - 1.0));
    Real nb = rho_in/mb;
    Real ye = ye_in;
    Real p = eos.GetPressure(nb, temp_in, &ye);
    Real T = eos.GetTemperatureFromP(nb, p, &ye);
    Real yp = eos.GetProtonFraction(nb, T, &ye);
    Real yn = eos.GetNeutronFraction(nb, T, &ye);
    Real mu_b = eos.GetBaryonChemicalPotential(nb, T, &ye);
    Real mu_q = eos.GetChargeChemicalPotential(nb, T, &ye);
    Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &ye);
    Real mu_n = mu_b, mu_p = mu_b + mu_q, mu_e = mu_le - mu_q;
    auto cu = code_units;
    auto eu = eos_units;
    radiation::NuratesGroupOccupancy spec = {};
    radiation::NuratesCellState cell = radiation::NuratesSpectralCellState(
        nb, T, yp, yn, mu_n, mu_p, mu_e, spec, params, cu, eu);
    bns_nurates::MyQuadrature quad = params.quadrature;
    bns_nurates::SpectralOpacities op =
        radiation::RadiationComputeSpectralOpacitiesStimulatedAbs(e, &quad, cell.spec,
                                                                  &cell.grey_op_params);
    Real n_eq[4], J_eq[4];
    const Real de = 1.0e-3*e;
    radiation::NuratesEqSpectrumBin(e - 0.5*de, e + 0.5*de, T, mu_n, mu_p, mu_e,
                                    params.quadrature, cell.unit_num_dens,
                                    cell.unit_ene_dens, n_eq, J_eq, params);
    for (int isp = 0; isp < 2; ++isp) {
      fine(2*isp, ie) = op.kappa[isp]*cell.unit_length;   // kappa_0, code units
      fine(2*isp + 1, ie) = J_eq[isp]/de;                 // u_eq per MeV, code units
    }
  });
  auto fine_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), fine);

  if (global_variable::my_rank == 0) {
    const Real vmag = std::sqrt(vx*vx + vy*vy + vz*vz);
    const Real dlg = std::log(e_max/e_min)/(kNE - 1.0);
    // log-log interpolation on the fine grid; zero outside it
    auto interp = [&](int row, Real e) -> Real {
      Real x = std::log(e/e_min)/dlg;
      if (x < 0.0 || x >= kNE - 1.0) return 0.0;
      int i = static_cast<int>(x);
      Real a = fine_h(row, i), b = fine_h(row, i + 1);
      if (a <= 0.0 || b <= 0.0) return (1.0 - (x - i))*a + (x - i)*b;
      return std::exp((1.0 - (x - i))*std::log(a) + (x - i)*std::log(b));
    };
    auto gfun = [](Real tau) -> Real {
      return (tau > 1.0e-4) ?
             1.0 - (1.0 - (1.0 + 2.0*tau)*std::exp(-2.0*tau))/(2.0*tau*tau) :
             4.0*tau/3.0;
    };
    constexpr int kNMu = 400, kNSub = 32;
    FILE *fp = std::fopen("sphere_exact_continuum.txt", "w");
    std::fprintf(fp, "# continuum solution, |v| = %.6f, R_eff = %.8e\n", vmag, r_eff);
    std::fprintf(fp, "# species bin e_lo[MeV] e_hi[MeV] L_exact[code]\n");
    // number luminosity int_bin L_E/E dE, in a file of its own (the one above keeps its
    // five columns)
    FILE *fpn = std::fopen("sphere_exact_continuum_number.txt", "w");
    std::fprintf(fpn, "# continuum solution, |v| = %.6f, R_eff = %.8e\n", vmag, r_eff);
    std::fprintf(fpn, "# species bin e_lo[MeV] e_hi[MeV] Ndot_exact[code lum/MeV]\n");
    Real ltot = 0.0;
    for (int isp = 0; isp < 2; ++isp) {
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        Real e_lo = tab_h(isp, ifr, 0), e_hi = tab_h(isp, ifr, 1);
        Real lnr = std::log(e_hi/e_lo);
        Real lum = 0.0, num = 0.0;
        for (int is = 0; is < kNSub; ++is) {
          Real e = e_lo*std::exp(lnr*(is + 0.5)/kNSub);
          Real de = e*lnr/kNSub;
          for (int im = 0; im < kNMu; ++im) {
            Real mu = -1.0 + 2.0*(im + 0.5)/kNMu;       // cosine to v
            Real dd = 1.0/(wlor*(1.0 - vmag*mu));
            Real e0 = e/dd;
            Real kap = interp(2*isp, e0)/dd;
            Real ilab = dd*dd*dd*interp(2*isp + 1, e0)/(4.0*M_PI);
            Real dl = (2.0*M_PI*2.0/kNMu)*M_PI*r_eff*r_eff*ilab*gfun(kap*r_eff)*de;
            lum += dl;
            num += dl/e;
          }
        }
        ltot += lum;
        std::fprintf(fp, "%d %d %.8e %.8e %.8e\n", isp, ifr, e_lo, e_hi, lum);
        std::fprintf(fpn, "%d %d %.8e %.8e %.8e\n", isp, ifr, e_lo, e_hi, num);
      }
    }
    std::fclose(fp);
    std::fclose(fpn);
    std::cout << "### rad_nu_sphere: continuum solution at |v| = " << vmag
              << ", total luminosity " << ltot
              << " (code units), per bin in sphere_exact_continuum.txt" << std::endl;
  }
#endif  // ENABLE_NURATES
}

} // namespace
