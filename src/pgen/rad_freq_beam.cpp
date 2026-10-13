//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_freq_beam.cpp
//  \brief Light bending and redshift of a multifrequency beam in Schwarzschild.
//
//  A beam whose spectrum is Gaussian in ln E (width problem/sigma_ln) is held fixed in a
//  disc of proper diameter problem/width around (problem/x0, 0, 0) on the equatorial
//  plane of Cartesian Kerr-Schild (a = 0, M = 1).  The rays are those within
//  problem/spread degrees (full cone) of the null direction whose coordinate spatial part
//  is +y.  Every ray carries the same conserved energy E_inf = nu |n_t| (n_t the covariant
//  time component of its unit null vector), chosen so that the central ray at
//  (x0, 0, 0) peaks at problem/e_peak in the tetrad frame; each ordinate's tetrad
//  spectrum therefore peaks at E_inf/|n_t|.  With problem/nearest_ordinate = true only the
//  ordinate closest to that direction at (x0, 0, 0) is filled (for coarse angular grids,
//  where the cone may hold none); its coordinate direction is printed and written to the
//  diagnostic header, so the geodesic it actually starts on can be drawn.  The pgen also
//  runs grey (multi_freq = false): there the invariant is I |n_t|^4, and each ordinate is
//  held at i_beam (|n_t|_central/|n_t|)^4.  At y = 0 that
//  direction has k^r = 0: it is tangential for the static observer, so the beam from
//  x0 = 3 follows the photon orbit, x0 < 3 falls in and x0 > 3 escapes.  The radiation
//  tetrad's time leg is the Kerr-Schild normal observer, which falls inward, so in the
//  tetrad frame the same rays point slightly outward.  No matter coupling.
//
//  The final function writes, for species 0 and every cell, the tetrad-frame energy
//  density per group, E_g = sum_n I_g(n) dOmega_n, to problem/diag_file, together with
//  L = sum_g,n I_g(n) dOmega_n ln|n_t(n)|.  The energy-weighted mean of ln E_inf in a
//  cell, sum_g E_g ln nu_g/E + L/E, must then equal ln E_inf of the source however the
//  rays are mixed in angle; what it drifts by is discretization error.  The coordinate-frame
//  R^tt = sum I dOmega (n^t)^2 is written too, for maps like Fig. 16 of Stone et al. (2024).  The beam's
//  path and the shift of its spectrum are compared with the geodesic by
//  vis/python/plot_nu_beam_deflection.py.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "pgen/pgen.hpp"

// Zero-intensity user boundaries, defined in pgen/tests/rad_beam.cpp
void ZeroIntensity(Mesh *pm);

namespace {

void BeamInject(Mesh *pm, const Real bdt);
void BeamDiagnostics(ParameterInput *pin, Mesh *pm);

struct BeamParams {
  Real x0, width, spread, e_peak, sig, i_beam;
  Real e_inf;   // conserved energy of the spectral peak, e_peak*|n_t| of the central ray
  Real ntc;     // |n_t| of the central ray at (x0, 0, 0)
  int iang;     // the one ordinate filled (nearest_ordinate), or -1 for the whole cone
  Real kc[3];   // coordinate direction k^i/k^t of the ordinate iang at (x0, 0, 0)
};
BeamParams beam;

// Fraction of a Gaussian in ln E (centre lnep, width sig) inside [e_lo, e_hi].
KOKKOS_INLINE_FUNCTION
Real BinGaussLn(Real e_lo, Real e_hi, Real lnep, Real sig) {
  Real a = (e_lo > 0.0) ? erf((log(e_lo) - lnep)/(sqrt(2.0)*sig)) : -1.0;
  Real b = erf((log(e_hi) - lnep)/(sqrt(2.0)*sig));
  return 0.5*(b - a);
}

// Bin edges of the stored frequency grid; the top bin continues the spacing.
template <class View>
KOKKOS_INLINE_FUNCTION
void BinEdges(const View &grid, int ifr, int nfreq, Real &lo, Real &hi) {
  lo = grid(ifr);
  hi = (ifr < nfreq - 1) ? grid(ifr + 1) : SQR(grid(ifr))/grid(ifr - 1);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  radiation::Radiation *prad = pmbp->prad;
  if (prad == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_freq_beam needs a <radiation> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!user_srcs || !user_bcs) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_freq_beam needs <problem>/user_srcs = true and user boundaries"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_srcs_func = BeamInject;
  user_bcs_func = ZeroIntensity;
  pgen_final_func = BeamDiagnostics;

  beam.x0 = pin->GetReal("problem", "x0");
  beam.width = pin->GetOrAddReal("problem", "width", 0.6);
  beam.spread = pin->GetOrAddReal("problem", "spread", 20.0);
  beam.e_peak = pin->GetOrAddReal("problem", "e_peak", 1.0);
  beam.sig = pin->GetOrAddReal("problem", "sigma_ln", 0.3);
  beam.i_beam = pin->GetOrAddReal("problem", "i_beam", 1.0);

  // |n_t| = |k_t|/k^(0) of the central ray at (x0, 0, 0), k^i = (0, 1, 0)
  auto &coord = pmbp->pcoord->coord_data;
  Real g[4][4], gi[4][4], dgx[4][4], dgy[4][4], dgz[4][4];
  Real e[4][4], ecov[4][4], om[4][4][4];
  ComputeMetricAndInverse(beam.x0, 0.0, 0.0, coord.is_minkowski, coord.bh_spin, g, gi);
  ComputeMetricDerivatives(beam.x0, 0.0, 0.0, coord.is_minkowski, coord.bh_spin,
                           dgx, dgy, dgz);
  ComputeTetrad(beam.x0, 0.0, 0.0, coord.is_minkowski, coord.bh_spin, g, gi, dgx, dgy,
                dgz, e, ecov, om);
  Real d0 = (-2.0*g[0][2] - sqrt(SQR(2.0*g[0][2]) - 4.0*g[0][0]*g[2][2]))/(2.0*g[0][0]);
  Real dc[4];
  for (int mu = 0; mu < 4; ++mu) dc[mu] = g[mu][0]*d0 + g[mu][2];
  Real kt[4];
  for (int a = 0; a < 4; ++a) {
    kt[a] = 0.0;
    for (int mu = 0; mu < 4; ++mu) kt[a] += e[a][mu]*dc[mu];
  }
  beam.ntc = std::fabs(dc[0]/kt[0]);
  beam.e_inf = beam.e_peak*beam.ntc;

  // ordinates of the cone at (x0, 0, 0), and the nearest one
  auto &nh = prad->nh_c;
  Real mu_min = std::cos(0.5*beam.spread*M_PI/180.0);
  int ncone = 0, ibest = 0;
  Real mubest = -2.0;
  for (int n = 0; n < prad->prgeo->nangles; ++n) {
    Real mu = 0.0;
    for (int a = 1; a < 4; ++a) mu += nh.h_view(n, a)*kt[a]/(-kt[0]);
    if (mu > mu_min) ++ncone;
    if (mu > mubest) {mubest = mu; ibest = n;}
  }
  bool nearest = pin->GetOrAddBoolean("problem", "nearest_ordinate", false);
  beam.iang = nearest ? ibest : -1;
  Real kup[4];
  for (int mu = 0; mu < 4; ++mu) {
    kup[mu] = 0.0;
    for (int a = 0; a < 4; ++a) kup[mu] += e[a][mu]*nh.h_view(ibest, a);
  }
  for (int d = 0; d < 3; ++d) beam.kc[d] = kup[d + 1]/kup[0];
  std::cout << "### rad_freq_beam: " << ncone << " ordinates within the "
            << beam.spread << " deg cone; nearest " << ibest << " at "
            << std::acos(std::fmin(mubest, 1.0))*180.0/M_PI
            << " deg, coordinate direction (" << beam.kc[0] << ", " << beam.kc[1]
            << ", " << beam.kc[2] << ")" << (nearest ? "  [filled alone]" : "")
            << std::endl;

  if (restart) return;
  Kokkos::deep_copy(prad->i0, 0.0);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void BeamInject
//  \brief Holds the beam rays in the source disc at I_g = i_beam*(Gaussian bin fraction).

void BeamInject(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  radiation::Radiation *prad = pmbp->prad;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  int nang = prad->prgeo->nangles;
  int nfreq = prad->nfreq;
  int nsp = prad->nspecies;

  auto &size = pmbp->pmb->mb_size;
  auto &flat = pmbp->pcoord->coord_data.is_minkowski;
  auto &spin = pmbp->pcoord->coord_data.bh_spin;
  auto &i0 = prad->i0;
  auto &tet = prad->tet_c;
  auto &tcv = prad->tetcov_c;
  auto &nh = prad->nh_c;
  auto &grid = prad->freq_grid;

  Real x0 = beam.x0;
  Real rad2 = SQR(0.5*beam.width);
  Real mu_min = cos(0.5*beam.spread*M_PI/180.0);
  Real lnei = log(beam.e_inf);
  Real ntc = beam.ntc;
  int ibeam = beam.iang;
  bool multi = prad->multi_freq;
  Real sig = beam.sig;
  Real i_beam = beam.i_beam;

  par_for("rad_freq_beam", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    Real g[4][4], gi[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, g, gi);

    // proper distance to the beam origin (x0, 0, 0)
    Real dx1 = x1v - x0, dx2 = x2v, dx3 = x3v;
    Real dx_sq = g[1][1]*dx1*dx1 + 2.0*g[1][2]*dx1*dx2 + 2.0*g[1][3]*dx1*dx3
               + g[2][2]*dx2*dx2 + 2.0*g[2][3]*dx2*dx3 + g[3][3]*dx3*dx3;
    if (dx_sq >= rad2) return;

    // future null vector with coordinate spatial part (0, 1, 0), lowered
    Real ta = g[0][0], tb = 2.0*g[0][2], tc = g[2][2];
    Real d0 = (-tb - sqrt(SQR(tb) - 4.0*ta*tc))/(2.0*ta);
    Real dc[4];
    for (int mu = 0; mu < 4; ++mu) dc[mu] = g[mu][0]*d0 + g[mu][2];

    // unit spatial direction in the tetrad frame
    Real kt[4];
    for (int a = 0; a < 4; ++a) {
      kt[a] = 0.0;
      for (int mu = 0; mu < 4; ++mu) kt[a] += tet(m, a, mu, k, j, i)*dc[mu];
    }
    Real d1 = kt[1]/(-kt[0]), d2 = kt[2]/(-kt[0]), d3 = kt[3]/(-kt[0]);

    Real n0 = tet(m, 0, 0, k, j, i);
    for (int iang = 0; iang < nang; ++iang) {
      if (ibeam >= 0) {
        if (iang != ibeam) continue;
      } else {
        Real mu = nh.d_view(iang, 1)*d1 + nh.d_view(iang, 2)*d2 + nh.d_view(iang, 3)*d3;
        if (mu <= mu_min) continue;
      }
      Real n_0 = 0.0;
      for (int d = 0; d < 4; ++d) n_0 += tcv(m, d, 0, k, j, i)*nh.d_view(iang, d);
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        Real w = SQR(SQR(ntc/fabs(n_0)));            // grey: I |n_t|^4 uniform
        if (multi) {
          Real lo, hi;
          BinEdges(grid, ifr, nfreq, lo, hi);
          w = BinGaussLn(lo, hi, lnei - log(fabs(n_0)), sig);
        }
        Real val = n0*n_0*i_beam*w;
        for (int isp = 0; isp < nsp; ++isp) {
          i0(m, (isp*nfreq + ifr)*nang + iang, k, j, i) = val;
        }
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void BeamDiagnostics
//  \brief Tetrad-frame energy density per group, species 0, every cell.

void BeamDiagnostics(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  radiation::Radiation *prad = pmbp->prad;
  const int nang = prad->prgeo->nangles;
  const int nfreq = prad->nfreq;
  auto &nh = prad->nh_c;
  auto &dom = prad->prgeo->solid_angles;
  auto &indcs = pm->mb_indcs;

  auto i0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->i0);
  auto tet = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->tet_c);
  auto tcv = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->tetcov_c);
  auto grid = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->freq_grid);
  auto &size = pmbp->pmb->mb_size;

  std::string fname = pin->GetOrAddString("problem", "diag_file", "beam_spec.txt");
  FILE *fp = std::fopen(fname.c_str(), "w");
  std::fprintf(fp, "# x0 = %.6e  e_peak = %.6e  e_inf = %.6e  sigma_ln = %.6e  "
               "width = %.6e  spread = %.6e  time = %.6e\n", beam.x0, beam.e_peak,
               beam.e_inf, beam.sig, beam.width, beam.spread, pm->time);
  std::fprintf(fp, "# ordinate %d  kx %.8e  ky %.8e  kz %.8e\n", beam.iang, beam.kc[0],
               beam.kc[1], beam.kc[2]);
  if (prad->multi_freq) {
    std::fprintf(fp, "# bin_lo");
    for (int ifr = 0; ifr < nfreq; ++ifr) {
      Real lo, hi;
      BinEdges(grid, ifr, nfreq, lo, hi);
      std::fprintf(fp, " %.8e", lo);
    }
    std::fprintf(fp, "\n# bin_hi");
    for (int ifr = 0; ifr < nfreq; ++ifr) {
      Real lo, hi;
      BinEdges(grid, ifr, nfreq, lo, hi);
      std::fprintf(fp, " %.8e", hi);
    }
  } else {
    std::fprintf(fp, "# grey");
  }
  std::fprintf(fp, "\n# x   y   L   R^tt   E_g (tetrad-frame energy density, species 0) "
               "for g = 0..nfreq-1;  L = sum_g,n I dOmega ln|n_t|;  R^tt = sum_g,n I "
               "dOmega (n^t)^2 (coordinate frame)\n");
  int k = indcs.ks;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int j = indcs.js; j <= indcs.je; ++j) {
      Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.h_view(m).x2min,
                             size.h_view(m).x2max);
      for (int i = indcs.is; i <= indcs.ie; ++i) {
        Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.h_view(m).x1min,
                               size.h_view(m).x1max);
        Real n0 = tet(m, 0, 0, k, j, i);
        std::vector<Real> eg(nfreq, 0.0);
        Real lsum = 0.0, rtt = 0.0;
        for (int n = 0; n < nang; ++n) {
          Real n_0 = 0.0, nt = 0.0;
          for (int d = 0; d < 4; ++d) {
            n_0 += tcv(m, d, 0, k, j, i)*nh.h_view(n, d);
            nt += tet(m, d, 0, k, j, i)*nh.h_view(n, d);
          }
          for (int ifr = 0; ifr < nfreq; ++ifr) {
            Real e = i0(m, ifr*nang + n, k, j, i)/(n0*n_0)*dom.h_view(n);
            eg[ifr] += e;
            rtt += e*nt*nt;
            if (e > 0.0) lsum += e*std::log(std::fabs(n_0));
          }
        }
        std::fprintf(fp, "%.6e %.6e %.8e %.6e", x1v, x2v, lsum, rtt);
        for (int ifr = 0; ifr < nfreq; ++ifr) std::fprintf(fp, " %.6e", eg[ifr]);
        std::fprintf(fp, "\n");
      }
    }
  }
  std::fclose(fp);
  std::cout << "### rad_freq_beam: wrote " << fname << std::endl;
}

}  // namespace
