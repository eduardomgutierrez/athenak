//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_freq_redshift.cpp
//  \brief Free streaming of a multifrequency spectrum out of a black-hole potential.
//
//  A spectrum Gaussian in ln E is injected on every outgoing ray at the inner x1 face and
//  streams without matter coupling along the x axis of Cartesian Kerr-Schild.  Along a
//  radial ray the energy measured by the Kerr-Schild normal observer, which is the time
//  leg of the radiation tetrad, is E_inf*sqrt(1+f)/(1-f) with f = 2M/r (M = 1), so the
//  spectrum must shift rigidly in ln E by the log of that ratio.  Without frequency
//  advection it does not shift at all.
//
//  The final function writes, for the most radial outgoing ray and species 0, the lab
//  (tetrad-frame) bin intensity I_g = i0/(n^0 n_0) in every cell to problem/diag_file.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

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

namespace {

void RedshiftDiagnostics(ParameterInput *pin, Mesh *pm);

// Fraction of a Gaussian in ln E (centre lnep, width sig) inside [e_lo, e_hi].
Real BinGaussLn(Real e_lo, Real e_hi, Real lnep, Real sig) {
  Real a = (e_lo > 0.0) ? std::erf((std::log(e_lo) - lnep)/(std::sqrt(2.0)*sig)) : -1.0;
  Real b = std::erf((std::log(e_hi) - lnep)/(std::sqrt(2.0)*sig));
  return 0.5*(b - a);
}

// Bin edges of the stored frequency grid; the top bin continues the spacing.
template <class View>
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
  if (prad == nullptr || !prad->multi_freq) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_freq_redshift needs <radiation> with multi_freq = true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pgen_final_func = RedshiftDiagnostics;

  const Real e_peak = pin->GetReal("problem", "e_peak");
  const Real sig = pin->GetOrAddReal("problem", "sigma_ln", 0.3);
  const int nang = prad->prgeo->nangles;
  const int nfreq = prad->nfreq;
  const int nsp = prad->nspecies;
  auto &coord = pmbp->pcoord->coord_data;

  // Tetrad at the centre of the first inner ghost cell (1D: y = z = 0).
  auto &ms = pmy_mesh_->mesh_size;
  Real dx = (ms.x1max - ms.x1min)/static_cast<Real>(pmy_mesh_->mesh_indcs.nx1);
  Real xg = ms.x1min - 0.5*dx;
  Real g[4][4], gi[4][4], dgx[4][4], dgy[4][4], dgz[4][4];
  Real e[4][4], ecov[4][4], om[4][4][4];
  ComputeMetricAndInverse(xg, 0.0, 0.0, coord.is_minkowski, coord.bh_spin, g, gi);
  ComputeMetricDerivatives(xg, 0.0, 0.0, coord.is_minkowski, coord.bh_spin, dgx, dgy, dgz);
  ComputeTetrad(xg, 0.0, 0.0, coord.is_minkowski, coord.bh_spin, g, gi, dgx, dgy, dgz,
                e, ecov, om);

  auto grid = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->freq_grid);
  auto &nh = prad->nh_c;
  auto &i_in = prad->pbval_i->i_in;
  for (int isp = 0; isp < nsp; ++isp) {
    for (int ifr = 0; ifr < nfreq; ++ifr) {
      Real lo, hi;
      BinEdges(grid, ifr, nfreq, lo, hi);
      Real frac = BinGaussLn(lo, hi, std::log(e_peak), sig);
      for (int n = 0; n < nang; ++n) {
        int idx = (isp*nfreq + ifr)*nang + n;
        Real val = 0.0;
        if (nh.h_view(n, 1) > 0.0) {                 // outgoing rays only
          Real n_0 = 0.0;
          for (int d = 0; d < 4; ++d) n_0 += ecov[d][0]*nh.h_view(n, d);
          val = e[0][0]*n_0*frac;                      // i0 = n^0 n_0 I
        }
        i_in.h_view(idx, BoundaryFace::inner_x1) = val;
      }
    }
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevExeSpace>();

  if (restart) return;
  Kokkos::deep_copy(prad->i0, 0.0);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void RedshiftDiagnostics
//  \brief Lab bin intensities of the most radial outgoing ray, species 0, every cell.

void RedshiftDiagnostics(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  radiation::Radiation *prad = pmbp->prad;
  const int nang = prad->prgeo->nangles;
  const int nfreq = prad->nfreq;
  auto &nh = prad->nh_c;
  auto &indcs = pm->mb_indcs;

  int ibest = 0;
  for (int n = 1; n < nang; ++n) {
    if (nh.h_view(n, 1) > nh.h_view(ibest, 1)) ibest = n;
  }
  auto i0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->i0);
  auto tet = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->tet_c);
  auto tcv = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->tetcov_c);
  auto grid = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prad->freq_grid);
  auto &size = pmbp->pmb->mb_size;

  std::string fname = pin->GetOrAddString("problem", "diag_file", "redshift_ray.txt");
  FILE *fp = std::fopen(fname.c_str(), "w");
  std::fprintf(fp, "# ray %d  n_tet = (%.6f, %.6f, %.6f)  time = %.6e\n", ibest,
               nh.h_view(ibest, 1), nh.h_view(ibest, 2), nh.h_view(ibest, 3), pm->time);
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
  std::fprintf(fp, "\n# x   I_g (lab bin intensity, species 0) for g = 0..nfreq-1\n");
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.h_view(m).x1min,
                             size.h_view(m).x1max);
      Real n0 = tet(m, 0, 0, 0, 0, i);
      Real n_0 = 0.0;
      for (int d = 0; d < 4; ++d) n_0 += tcv(m, d, 0, 0, 0, i)*nh.h_view(ibest, d);
      std::fprintf(fp, "%.8e", x1v);
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        std::fprintf(fp, " %.8e", i0(m, ifr*nang + ibest, 0, 0, i)/(n0*n_0));
      }
      std::fprintf(fp, "\n");
    }
  }
  std::fclose(fp);
  std::cout << "### rad_freq_redshift: wrote " << fname << " (ray " << ibest
            << ", n_x = " << nh.h_view(ibest, 1) << ")" << std::endl;
}

}  // namespace
