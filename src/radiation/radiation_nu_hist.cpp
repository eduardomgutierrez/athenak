//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_nu_hist.cpp
//! \brief Neutrino energy and number luminosities through spheres, for user histories.
//!
//! ## Energy and number currents from i0
//!
//! i0 = n^0 n_0 I_g, with I_g the intensity of ray n integrated over the energy bin g
//! (radnu_coord in outputs/derived_variables.cpp divides by n0*n_0 the same way).  The bins
//! are *lab* bins: g is an interval of the energy eps measured by the tetrad observer
//! (radiation_nurates_remap.hpp), and a neutrino of ray n in bin g has the 4-momentum
//! p^mu = eps n^mu, with n^mu = e_(d)^mu nh^d (nh^0 = 1) the same for every group.  With
//! the distribution f and d^3p/eps = eps deps dOmega in the tetrad frame,
//!
//!   R^{mu nu} = int f p^mu p^nu d^3p/eps = sum_n sum_g I_g n^mu n^nu dOmega_n,
//!   N^mu      = int f p^mu      d^3p/eps = sum_n sum_g (I_g/eps_g) n^mu dOmega_n,
//!
//! since I_eps = eps^3 f.  Dividing by eps_g is done before n^mu projects the ray onto
//! the coordinate basis, i.e. in the frame where the bin is defined, so N^mu needs no
//! fluid velocity and no frame transformation.  The one approximation is
//! int_g (I_eps/eps) deps = I_g/eps_g, exact for a flat dI/deps inside the bin with
//!   eps_g = (e_hi - e_lo)/ln(e_hi/e_lo),
//! and good to < 1% for a smooth spectrum on bins of x1.4.  (The source term counts
//! lepton number at the geometric mean, FreqBinMidMeV; the two differ by 0.5% at x1.4.)
//! eps_g is in MeV, so N^mu is in code luminosity per MeV:
//!   Ndot [1/s] = value * (c^5/G [erg/s]) / (1 MeV [erg]),
//! the same for any mass unit.
//!
//! Energy at infinity: R^i_t = R^{i nu} g_{nu t} = sum I_g n^i n_t dOmega, and
//! I_g n_t = i0/n^0, so R^i_t = sum i0 n^i/n^0 dOmega -- the contraction of radnu_coord's
//! R^{i nu} with g_{nu t}, without the metric.

#include "radiation_nu_hist.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "config.hpp"
#include "mesh/mesh.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "radiation.hpp"
#include "radiation_multi_freq.hpp"
#if ENABLE_NURATES
#include "radiation_nurates.hpp"
#endif

namespace radiation {

namespace {
constexpr int kNuHistMaxSpecies = 4;
} // namespace

//----------------------------------------------------------------------------------------
//! \fn bool NuHistEnabled

bool NuHistEnabled(MeshBlockPack *pmbp) {
#if ENABLE_NURATES
  return (pmbp->prad != nullptr && pmbp->prad->use_nurates && pmbp->prad->multi_freq);
#else
  return false;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn std::string NuHistSpeciesTag

std::string NuHistSpeciesTag(int isp, int nspecies) {
  if (nspecies == 3) {
    const char *tags[3] = {"nue", "anue", "nux"};  // heavy leptons merged in slot 2
    return tags[isp];
  }
  return "s" + std::to_string(isp);
}

//----------------------------------------------------------------------------------------
//! \fn void NuFluxMoments
//! \brief One device pass over i0: per cell and species, R^i_t and N^i (see file header)

void NuFluxMoments(MeshBlockPack *pmbp, DvceArray5D<Real> &mom) {
#if ENABLE_NURATES
  Radiation *prad = pmbp->prad;
  const int nsp = prad->nspecies;
  const int nfreq = prad->nfreq;
  const int nang = prad->prgeo->nangles;
  if (nsp > kNuHistMaxSpecies) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "neutrino history supports at most " << kNuHistMaxSpecies
              << " species" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // cells including ghost zones: the sphere interpolation stencil reaches into them
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb = pmbp->nmb_thispack;
  const int nvar = kNuHistVarsPerSpecies*nsp;
  if (static_cast<int>(mom.extent(0)) != nmb || static_cast<int>(mom.extent(1)) != nvar ||
      static_cast<int>(mom.extent(2)) != n3 || static_cast<int>(mom.extent(3)) != n2 ||
      static_cast<int>(mom.extent(4)) != n1) {
    Kokkos::realloc(mom, nmb, nvar, n3, n2, n1);
  }

  // 1/eps_g [1/MeV], eps_g the bin's mean energy for a flat dI/deps
  DvceArray1D<Real> inv_eps("nu_hist_inv_eps", nfreq);
  auto freq_grid_ = prad->freq_grid;
  const int freq_scale = prad->flag_fscale;
  Kokkos::parallel_for("nu_hist_eps", Kokkos::RangePolicy<>(DevExeSpace(), 0, nfreq),
  KOKKOS_LAMBDA(const int ifr) {
    Real e_lo, e_hi;
    FreqBinEdgesMeV(freq_grid_, ifr, nfreq, freq_scale, e_lo, e_hi);
    Real eps = (e_lo > 0.0 && e_hi > e_lo) ? (e_hi - e_lo)/log(e_hi/e_lo) : 0.5*e_hi;
    inv_eps(ifr) = 1.0/eps;
  });

  auto i0_ = prad->i0;
  auto nh_c_ = prad->nh_c;
  auto tet_c_ = prad->tet_c;
  auto tetcov_c_ = prad->tetcov_c;
  auto solid_angles_ = prad->prgeo->solid_angles;
  auto mom_ = mom;
  par_for("nu_hist_moments", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real acc[kNuHistVarsPerSpecies*kNuHistMaxSpecies];
    for (int v=0; v<kNuHistVarsPerSpecies*kNuHistMaxSpecies; ++v) acc[v] = 0.0;
    const Real n0 = tet_c_(m,0,0,k,j,i);
    for (int n=0; n<nang; ++n) {
      // coordinate components n^i of the ray and its covariant n_0
      Real nx = 0.0, ny = 0.0, nz = 0.0, n_0 = 0.0;
      for (int d=0; d<4; ++d) {
        Real nhd = nh_c_.d_view(n,d);
        nx  += tet_c_(m,d,1,k,j,i)*nhd;
        ny  += tet_c_(m,d,2,k,j,i)*nhd;
        nz  += tet_c_(m,d,3,k,j,i)*nhd;
        n_0 += tetcov_c_(m,d,0,k,j,i)*nhd;
      }
      const Real we = solid_angles_.d_view(n)/n0;   // i0*we*n^i -> R^i_t
      const Real wn = we/n_0;                       // i0*wn*n^i -> I n^i dOmega
      for (int isp=0; isp<nsp; ++isp) {
        Real se = 0.0, sn = 0.0;
        for (int ifr=0; ifr<nfreq; ++ifr) {
          Real ii = i0_(m,isp*nfreq*nang + getFreqAngIndex(ifr,n,nang),k,j,i);
          se += ii;
          sn += ii*inv_eps(ifr);
        }
        Real *a = &acc[kNuHistVarsPerSpecies*isp];
        a[0] += we*se*nx;  a[1] += we*se*ny;  a[2] += we*se*nz;
        a[3] += wn*sn*nx;  a[4] += wn*sn*ny;  a[5] += wn*sn*nz;
      }
    }
    for (int v=0; v<nvar; ++v) mom_(m,v,k,j,i) = acc[v];
  });
#else
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "neutrino history requires a build with bns_nurates" << std::endl;
  std::exit(EXIT_FAILURE);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void NuSphereLuminosities
//! \brief Radial contraction on the sphere, with the dr/dx^i and sqrt(-g) of spherical KS
//!        used by TorusFluxes (src/pgen/gr_torus_compose.cpp)

void NuSphereLuminosities(SphericalGrid *grid, DvceArray5D<Real> &mom, int nspecies,
                          Real spin, Real *lum, Real *ndot) {
  for (int isp=0; isp<nspecies; ++isp) {
    lum[isp] = 0.0;
    ndot[isp] = 0.0;
  }
  grid->InterpolateToSphere(kNuHistVarsPerSpecies*nspecies, mom);
  const Real r = grid->radius;
  const Real r2 = SQR(r);
  const Real a2 = SQR(spin);
  for (int n=0; n<grid->nangles; ++n) {
    Real theta = grid->polar_pos.h_view(n,0);
    Real x1 = grid->interp_coord.h_view(n,0);
    Real x2 = grid->interp_coord.h_view(n,1);
    Real x3 = grid->interp_coord.h_view(n,2);
    Real rad2 = SQR(x1) + SQR(x2) + SQR(x3);
    Real drdx = r*x1/(2.0*r2 - rad2 + a2);
    Real drdy = r*x2/(2.0*r2 - rad2 + a2);
    Real drdz = (r*x3 + a2*x3/r)/(2.0*r2 - rad2 + a2);
    Real dw = grid->solid_angles.h_view(n)*(r2 + SQR(spin*cos(theta)));
    for (int isp=0; isp<nspecies; ++isp) {
      const int v = kNuHistVarsPerSpecies*isp;
      auto &iv = grid->interp_vals;
      Real rrt = drdx*iv.h_view(n,v  ) + drdy*iv.h_view(n,v+1) + drdz*iv.h_view(n,v+2);
      Real nr  = drdx*iv.h_view(n,v+3) + drdy*iv.h_view(n,v+4) + drdz*iv.h_view(n,v+5);
      lum[isp]  += -rrt*dw;
      ndot[isp] +=   nr*dw;
    }
  }
}

} // namespace radiation
