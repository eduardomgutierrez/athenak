#ifndef RADIATION_RADIATION_NU_HIST_HPP_
#define RADIATION_RADIATION_NU_HIST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_nu_hist.hpp
//! \brief Neutrino energy and number luminosities through spheres, for user histories.
//!
//! Two steps, so that one device pass over i0 serves every sphere:
//!  1. NuFluxMoments builds, per cell and species, the radial-flux ingredients from i0
//!     (6 numbers: R^i_t and N^i, i = x, y, z, Cartesian coordinate components);
//!  2. NuSphereLuminosities interpolates them to a SphericalGrid and contracts them with
//!     dr/dx^i, sqrt(-g) and the solid angle, as TorusFluxes does for the mass flux.
//! Multifrequency neutrino radiation with bns_nurates only (NuHistEnabled).

#include <string>

#include "athena.hpp"

class MeshBlockPack;
class SphericalGrid;

namespace radiation {

// Number of per-cell variables NuFluxMoments writes per species
constexpr int kNuHistVarsPerSpecies = 6;

// true if the radiation is multifrequency with bns_nurates (energy groups in MeV)
bool NuHistEnabled(MeshBlockPack *pmbp);

// Column tag of species isp: nue, anue, nux for nspecies = 3, s<isp> otherwise
std::string NuHistSpeciesTag(int isp, int nspecies);

// Per-cell moments, mom(m, 6*isp + c, k, j, i), ghost zones included:
//   c = 0..2  R^i_t = R^{i nu} g_{nu t}       [code energy flux]
//   c = 3..5  N^i, summed over groups of (energy-current part)/eps_g[MeV]
// mom is (re)allocated here if its shape does not match.
void NuFluxMoments(MeshBlockPack *pmbp, DvceArray5D<Real> &mom);

// Per-species luminosities through the (Kerr-Schild) sphere of grid, from mom:
//   lum[isp]  = - oint R^r_t sqrt(-g) dtheta dphi    (energy at infinity, code units)
//   ndot[isp] = + oint N^r   sqrt(-g) dtheta dphi    (code luminosity per MeV)
// Each rank sums the angles it owns; the history output sums over ranks.
void NuSphereLuminosities(SphericalGrid *grid, DvceArray5D<Real> &mom, int nspecies,
                          Real spin, Real *lum, Real *ndot);

} // namespace radiation

#endif // RADIATION_RADIATION_NU_HIST_HPP_
