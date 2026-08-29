//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.cpp
//! \brief implementation of Radiation class constructor and assorted other functions

#include <float.h>

#include <iostream>
#include <algorithm> // max
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "units/units.hpp"
#include "radiation/radiation.hpp"
#include "config.hpp"
#if ENABLE_NURATES
#include "geodesic-grid/gauss_legendre.hpp"
#include "radiation/radiation_nurates_remap.hpp"
#endif

namespace radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
    freq_grid("freq_grid",1),
    nnu_coeff("nnu_coeff",1,1,1,1,1),
    nh_c("nh_c",1,1),
    nh_f("nh_f",1,1,1),
    tet_c("tet_c",1,1,1,1,1,1),
    tetcov_c("tetcov_c",1,1,1,1,1,1),
    tet_d1_x1f("tet_d1_x1f",1,1,1,1,1),
    tet_d2_x2f("tet_d2_x2f",1,1,1,1,1),
    tet_d3_x3f("tet_d3_x3f",1,1,1,1,1),
    na("na",1,1,1,1,1,1),
    norm_to_tet("norm_to_tet",1,1,1,1,1,1),
    i0("i0",1,1,1,1,1),
    i1("i1",1,1,1,1,1),
    iflx("iflx",1,1,1,1,1),
    divfa("divfa",1,1,1,1,1),
    beam_mask("beam_mask",1,1,1,1,1),
    pmy_pack(ppack) {
  // Check for general relativity
  if (!(pmy_pack->pcoord->is_general_relativistic) &&
      !(pmy_pack->pcoord->is_dynamical_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation requires general relativity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydrodynamics, mhd, and units
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  are_units_enabled = pin->DoesBlockExist("units");

#if ENABLE_NURATES
  use_nurates = pin->GetOrAddBoolean("radiation", "use_nurates", false);
  // Read here rather than with the other nurates keys: SetFrequencyGrid() runs
  // before those and needs to know whether to look for EOS units.
  nurates_toy_scattering =
      pin->GetOrAddReal("radiation", "nurates_toy_scattering", -1.0);
  nurates_toy_scat_p = pin->GetOrAddReal("radiation", "nurates_toy_scat_p", 0.0);
  nurates_toy_scat_eref =
      pin->GetOrAddReal("radiation", "nurates_toy_scat_eref", 1.0);
#endif

  // Check flags and parameters for ad hoc fixes
  correct_radsrc_velocity = pin->GetOrAddBoolean("radiation","correct_radsrc_velocity",false);
  correct_radsrc_opacity  = pin->GetOrAddBoolean("radiation","correct_radsrc_opacity",false);
  dfloor_opacity = pin->GetOrAddReal("radiation","dfloor_opacity",1e-100);
  dens_trunc_max = pin->GetOrAddReal("radiation","dens_trunc_max",1e100);
  tau_truncation = pin->GetOrAddReal("radiation","tau_truncation",1e-100);
  sigmoid_residual = pin->GetOrAddReal("radiation","sigmoid_residual",1e-2);
  sigmoid_residual = fmin(sigmoid_residual, 1./3); // sigmoid residual must be less than 1./3
  // Print solver information
  if (pmy_pack->pdyngr != nullptr) {
    std::cout << "### Radiation: coupled to Valencia (DynGRMHD) solver" << std::endl;
  } else if (is_mhd_enabled) {
    std::cout << "### Radiation: coupled to HARM (MHD) solver" << std::endl;
  } else if (is_hydro_enabled) {
    std::cout << "### Radiation: coupled to HARM (Hydro) solver" << std::endl;
  } else {
    std::cout << "### Radiation: no fluid coupling (transport only)" << std::endl;
  }

  // Check for radiation particle type (photon or neutrino)
  std::string rad_type = pin->GetOrAddString("radiation","radiation_type","photon");
  if (rad_type.compare("neutrino") == 0) {
    is_neutrino = true;
    nspecies = pin->GetOrAddInteger("radiation","nspecies",3);
    if (nspecies < 1 || nspecies > 4) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "nspecies must be between 1 and 4 for neutrino transport"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::cout << "### Radiation: neutrino transport with " << nspecies
              << " species" << std::endl;
  } else {
    is_neutrino = false;
    nspecies = 1;
  }
#if ENABLE_NURATES
  if (use_nurates && !is_neutrino) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "use_nurates=true requires <radiation>/radiation_type = neutrino"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  // Check for multi-frequency radiation
  nfreq = 1;
  freq_fluxes = false;
  multi_freq = pin->GetOrAddBoolean("radiation","multi_freq",false);
  if (multi_freq) {
    // number of frequency groups
    nfreq = pin->GetOrAddInteger("radiation","nfreq",3);

    if (nfreq < 3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Number of frequency groups must be >= 3 for multi-frequency radiation" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // frequency grid
    nu_min = pin->GetReal("radiation", "nu_min");
    nu_max = pin->GetReal("radiation", "nu_max");
    std::string freq_scale = pin->GetOrAddString("radiation", "freq_scale", "log");
    if (freq_scale.compare("linear") == 0)
      flag_fscale = 0;
    else if (freq_scale.compare("customize") == 0)
      flag_fscale = 2;
    else // log frequency grid
      flag_fscale = 1;

    if (nu_max <= nu_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Frequency maximum must be larger than frequency minumum for multi-frequency radiation" << std::endl;
      std::exit(EXIT_FAILURE);
    }

#if ENABLE_NURATES
    // The nurates source term solves on the fixed comoving grid, and it gets there by
    // rebinning each ray by the rigid displacement ln(n0_cm)/dlnnu.  That displacement is
    // one number per ray only on a log grid; on a linear grid it grows with the bin
    // energy -- 9.5 bins for the top group of a [10, 2000] MeV, nfreq = 24 grid at
    // v = 0.5 -- so the rebin is no longer a translation and the round trip stops
    // closing.  Refuse it rather than return silent garbage.
    if (use_nurates && flag_fscale != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "use_nurates with multi_freq requires <radiation>/freq_scale = log"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Every bin is in the geometric family on the neutrino grid, so the rebin acts on
    // bins 0 .. nfreq-1.  The four-point stencil of RemapBinValue needs four of them.
    if (use_nurates && nfreq < 4) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "use_nurates with multi_freq requires <radiation>/nfreq >= 4"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // A neutrino grid is geometric over the whole of [nu_min, nu_max], so nu_min is a
    // real bin edge rather than the top of a [0, nu_min] catch-all.  It must be positive:
    // ln(nu_min) sets the grid spacing and the rigid Doppler displacement derived from
    // it, and at nu_min = 0 both are -inf.
    if (use_nurates && nu_min <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "use_nurates with multi_freq requires <radiation>/nu_min > 0"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#endif

    // auxiliary quantities
    Kokkos::realloc(freq_grid,nfreq);
    SetFrequencyGrid();

    // flag for frequency fluxes
    freq_fluxes = pin->GetOrAddBoolean("radiation","freq_fluxes",true);
#if ENABLE_NURATES
    // Frequency advection closes the bottom of the grid by assuming no flux through
    // freq_grid(0) -- exact for photons, where that edge is nu = 0, but on the neutrino
    // grid it is nu_min, so the assumption turns the lowest bin into a reservoir.  The
    // default here is true, so a run that simply omits the key would get that silently.
    if (use_nurates && freq_fluxes) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "use_nurates with multi_freq requires <radiation>/freq_fluxes = false:"
                << std::endl
                << "the nu = 0 boundary it assumes does not exist on a neutrino grid"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#endif
  } // endif (multi_freq)

  // Enable radiation source term (radiation+(M)HD) by default if hydro or mhd enabled
  // Otherwise, disable radiation source term.  The former can be overriden by
  // specification in the input file.
  if (is_hydro_enabled || is_mhd_enabled) {
    rad_source = pin->GetOrAddBoolean("radiation","rad_source",true);
  } else {
    rad_source = false;
  }

  // Set radiation coupling parameters including scattering and absorption opacities,
  // radiation constant, and source term behavior.
  if (rad_source) {
    // kappa_s/kappa_a/kappa_p are unused when use_nurates=true; default to 0.
    kappa_s = pin->GetOrAddReal("radiation","kappa_s",0.0);
    power_opacity = pin->GetOrAddBoolean("radiation","power_opacity",false);
    if (!(power_opacity)) {
      kappa_a = pin->GetOrAddReal("radiation","kappa_a",0.0);
      kappa_p = pin->GetOrAddReal("radiation","kappa_p",0.0);
    }
    is_compton_enabled = pin->GetOrAddBoolean("radiation","compton",false);
    // if (is_compton_enabled && !(are_units_enabled)) {
    //   std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
    //     << std::endl << "Compton requires enabling units" << std::endl;
    //   std::exit(EXIT_FAILURE);
    // }
    if (are_units_enabled) {
      arad = (pmy_pack->punit->rad_constant_cgs*
              SQR(SQR(pmy_pack->punit->temperature_cgs()))/
              pmy_pack->punit->pressure_cgs());
    } else {
      arad = pin->GetOrAddReal("radiation","arad",1.0);
    }
    affect_fluid = pin->GetOrAddBoolean("radiation","affect_fluid",true);
    evolve_ye = pin->GetOrAddBoolean("radiation","evolve_ye",true);
#if ENABLE_NURATES
    // The nurates source terms couple to Valencia (DynGRMHD) only.  Real opacities need
    // the tabulated CompOSE EOS that only DynGRMHDPS carries, and the fluid coupling is
    // written for Valencia's sqrt(gamma)*(E-D) and sqrt(gamma)*S_k, which differ from
    // HARM's T^t_t + D and T^t_k by a sign and a densitisation.  Prescribed-opacity
    // transport that never touches the fluid needs neither, and is the only exception:
    // that is what the diffusion tests run, on ideal-gas hydro.
    if (use_nurates && pmy_pack->pdyngr == nullptr) {
      bool transport_only = (nurates_toy_scattering >= 0.0) && !affect_fluid;
      if (!transport_only) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "use_nurates=true requires DynGRMHD (set <mhd>/dyn_eos), unless "
                  << "the run is prescribed-opacity transport "
                  << "(nurates_toy_scattering >= 0 and affect_fluid = false)."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
#endif
    // The multifrequency path forms the number density change bin by bin from the
    // radiation field itself, so the mean-energy approximation that separates the two
    // grey routes does not arise there and there is nothing for 'opacity' to select.
    // Default to 'moment' and refuse 'opacity' there, rather than accepting a key the
    // path silently ignores.
    bool multifreq_number_moment = multi_freq && use_nurates;
    std::string ye_source_model_str =
        pin->GetOrAddString("radiation", "ye_source_model",
                            multifreq_number_moment ? "moment" : "opacity");
    if (ye_source_model_str == "opacity" ||
        ye_source_model_str == "number_opacity") {
      ye_source_model = 0;
    } else if (ye_source_model_str == "moment" ||
               ye_source_model_str == "number_moment") {
      ye_source_model = 1;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Unknown radiation/ye_source_model='" << ye_source_model_str
                << "'. Use 'opacity' or 'moment'." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (multifreq_number_moment && ye_source_model != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "radiation/ye_source_model='" << ye_source_model_str
                << "' is not implemented on the multifrequency nurates path, which "
                << "always takes the per-bin number moment. Use 'moment'."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Chiral magnetic effect.  Names deliberately match <radiation_m1> so that
    // the same physics reads the same way in both solvers' input files.
    backreact_chiral = pin->GetOrAddBoolean("radiation", "backreact_chiral", false);
    chiral_gamma_m = pin->GetOrAddBoolean("radiation", "chiral_gamma_m", true);

    if (backreact_chiral) {
      // Y5 lives in the second passive scalar slot, IYF+1, and the Gamma_m /
      // E.B kernel needs the ADM metric and a magnetic field.
      // MHD is constructed before Radiation in MeshBlockPack::AddPhysics, so
      // pmhd is already available here.
      if (pmy_pack->pmhd == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "radiation/backreact_chiral requires MHD." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (pmy_pack->pmhd->nscalars < 2) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "radiation/backreact_chiral requires mhd/nscalars >= 2 "
                  << "(slot 0 is Ye, slot 1 is Y5)." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (!pin->DoesBlockExist("adm") && !pin->DoesBlockExist("z4c")) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "radiation/backreact_chiral requires dynamical GRMHD "
                  << "(<adm> or <z4c> block)." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // evolve_ye is NOT required: the URCA source for Y5 rides along with the
      // Ye update and so is inactive without it, but the Gamma_m sink and the
      // E.B anomaly source are independent.  Running with evolve_ye = false and
      // chiral_gamma_m = true isolates them, which is how the chiral unit tests
      // in runs/chiral_check*.athinput work.  Only the combination that does
      // nothing at all is worth complaining about.
      if (!evolve_ye && !chiral_gamma_m) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "radiation/backreact_chiral has no effect with "
                  << "evolve_ye = false and chiral_gamma_m = false: the URCA "
                  << "source needs evolve_ye and the Gamma_m / E.B sources need "
                  << "chiral_gamma_m." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

  // multi-frequency radiation
    if (multi_freq) {
      // flags for fluid update
      if (affect_fluid) {
        update_fluid_energy = pin->GetOrAddBoolean("radiation","update_fluid_energy",true);
        update_fluid_moment = pin->GetOrAddBoolean("radiation","update_fluid_moment",true);
      } else {
        update_fluid_energy = false;
        update_fluid_moment = false;
      }

      // parameters for intensity mapping
      order_multifreq = pin->GetOrAddInteger("radiation","order_multifreq",2);
      limiter_multifreq = pin->GetOrAddInteger("radiation","limiter_multifreq",2);

      // flags and parameters used in compton
      if (is_compton_enabled) {
        num_iter_compton     = pin->GetOrAddInteger("radiation","num_iter_compton",5);
        tol_rel_tgas_compton = pin->GetOrAddReal("radiation","tol_rel_tgas_compton",1e-6);
        est_tgas_4th_compton = pin->GetOrAddBoolean("radiation","est_tgas_4th_compton",true);
        est_tgas_5th_compton = pin->GetOrAddBoolean("radiation","est_tgas_5th_compton",false);
        test_only_compton_therm = pin->GetOrAddBoolean("radiation","test_only_compton_therm",false);
      } // endif is_compton_enabled
    } // endif (multi_freq)

    // if (multi_freq && !are_units_enabled) {
    //   std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
    //     << std::endl << "Units must be specified for multi-frequency radiation" << std::endl;
    //   std::exit(EXIT_FAILURE);
    // }

  } // endif rad_source

  // Check for fluid evolution
  fixed_fluid = pin->GetOrAddBoolean("radiation","fixed_fluid",false);

  // Source terms (if needed)
  beam_source = pin->GetOrAddBoolean("radiation","beam_source",false);
  if (pin->DoesBlockExist("rad_srcterms")) {
    psrc = new SourceTerms("rad_srcterms", ppack, pin);
  }

  // Setup angular mesh and radiation geometry data
  int nlevel = pin->GetInteger("radiation", "nlevel");
  rotate_geo = pin->GetOrAddBoolean("radiation","rotate_geo",true);
  angular_fluxes = pin->GetOrAddBoolean("radiation","angular_fluxes",true);
  n_0_floor = pin->GetOrAddReal("radiation","n_0_floor",0.1);
  prgeo = new GeodesicGrid(nlevel, rotate_geo, angular_fluxes);

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(nh_c,prgeo->nangles,4);
  Kokkos::realloc(nh_f,prgeo->nangles,6,4);
  Kokkos::realloc(tet_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tetcov_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tet_d1_x1f,nmb,4,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(tet_d2_x2f,nmb,4,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(tet_d3_x3f,nmb,4,ncells3+1,ncells2,ncells1);
  if (angular_fluxes) {Kokkos::realloc(na,nmb,prgeo->nangles,ncells3,ncells2,ncells1,6);}
  if (freq_fluxes) {Kokkos::realloc(nnu_coeff,nmb,prgeo->nangles,ncells3,ncells2,ncells1);}
  if (is_hydro_enabled || is_mhd_enabled) {
    Kokkos::realloc(norm_to_tet,nmb,4,4,ncells3,ncells2,ncells1);
  }
  }
  SetOrthonormalTetrad();

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for intensities
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(i0,nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_i0,nmb,nspecies*nfreq*prgeo->nangles,nccells3,nccells2,nccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_i = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_i->InitializeBuffers(nspecies*nfreq*prgeo->nangles);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("radiation","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("teno") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      } else if (xorder.compare("teno") == 0) {
        recon_method = ReconstructionMethod::teno;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<radiation> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    }

    // allocate second registers, fluxes, masks
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(i1,      nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x1f,nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x2f,nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x3f,nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
    if (angular_fluxes || freq_fluxes) {
      Kokkos::realloc(divfa,nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
    }
  }

#if ENABLE_NURATES
  // Initialize bns_nurates library (optional)
  if (use_nurates) {
    nurates_debug_opacity =
        pin->GetOrAddBoolean("bns_nurates","debug_opacity",false);

    // reaction flags
    nurates_params.use_abs_em          = pin->GetOrAddBoolean("bns_nurates","use_abs_em",true);
    nurates_params.use_pair            = pin->GetOrAddBoolean("bns_nurates","use_pair",true);
    nurates_params.use_brem            = pin->GetOrAddBoolean("bns_nurates","use_brem",true);
    nurates_params.use_iso             = pin->GetOrAddBoolean("bns_nurates","use_iso",true);
    // Defaults below are kept identical to the M1 parser in radiation_m1.cpp on
    // purpose.  They used to differ -- these six correction flags defaulted false
    // here and true there -- so an input that named only a few keys silently gave
    // the two solvers different microphysics, with nothing in the output to say
    // so.  Any change here must be mirrored in radiation_m1.cpp.
    nurates_params.use_inelastic_scatt = pin->GetOrAddBoolean("bns_nurates","use_inelastic_scatt",true);
    // correction flags
    nurates_params.use_WM_ab           = pin->GetOrAddBoolean("bns_nurates","use_WM_ab",true);
    nurates_params.use_WM_sc           = pin->GetOrAddBoolean("bns_nurates","use_WM_sc",true);
    nurates_params.use_dU              = pin->GetOrAddBoolean("bns_nurates","use_dU",true);
    nurates_params.use_dm_eff          = pin->GetOrAddBoolean("bns_nurates","use_dm_eff",true);
    nurates_params.use_NN_medium_corr  = pin->GetOrAddBoolean("bns_nurates","use_NN_medium_corr",true);
    nurates_params.neglect_blocking    = pin->GetOrAddBoolean("bns_nurates","neglect_blocking",false);
    nurates_params.use_decay           = pin->GetOrAddBoolean("bns_nurates","use_decay",false);
    nurates_params.use_BRT_brem        = pin->GetOrAddBoolean("bns_nurates","use_BRT_brem",false);
    nurates_params.use_equilibrium_distribution =
        pin->GetOrAddBoolean("bns_nurates","use_equilibrium_distribution",true);
    nurates_params.use_kirchhoff_law =
        pin->GetOrAddBoolean("bns_nurates", "use_kirchhoff_law", true);

    // Partially-equilibrated (T*, Ye*) emissivity predictor.  On by default: it is the
    // scheme, not an option on top of one.  Off, Kirchhoff's law gets the local
    // blackbody at (T^n, Ye^n) -- the predictor's own dt -> 0 limit -- which is a real
    // fallback but runs the emission at the start-of-step rate for the whole step, and
    // so puts a ceiling on dt that has nothing to do with transport.
    // Key names and defaults are identical to the M1 parser in radiation_m1.cpp on
    // purpose, so one <bns_nurates> block drives both solvers; any change here must be
    // mirrored there.  GetOrAddBoolean writes the default into the input, so the
    // "asked for by name?" test has to be read before it, not after.
    const bool peq_requested =
        pin->DoesParameterExist("bns_nurates", "use_partial_equilibrium");
    nurates_params.use_partial_equilibrium =
        pin->GetOrAddBoolean("bns_nurates", "use_partial_equilibrium", true);
    nurates_params.peq_w_floor =
        pin->GetOrAddReal("bns_nurates", "peq_w_floor", 1e-3);
    nurates_params.peq_dlnT_tol =
        pin->GetOrAddReal("bns_nurates", "peq_dlnT_tol", 1e-4);
    nurates_params.peq_dYe_tol =
        pin->GetOrAddReal("bns_nurates", "peq_dYe_tol", 1e-4);

    // The predictor's only output is an equilibrium distribution, and in this module
    // Kirchhoff's law is the only thing that consumes one -- unlike M1, where
    // use_equilibrium_distribution also gives it a second consumer.  So the predictor
    // needs use_kirchhoff_law, and after this block "use_partial_equilibrium" implies
    // "use_kirchhoff_law", which every gate below relies on.
    // Asking for it by name without Kirchhoff is an error; inheriting the default there
    // is not, since "no equilibrium closure at all" is a legitimate thing to configure
    // and it would be rude to fail at startup over a line the user never wrote.  Turn
    // it off instead, and say so.
    if (nurates_params.use_partial_equilibrium &&
        !nurates_params.use_kirchhoff_law) {
      if (peq_requested) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "<bns_nurates>/use_partial_equilibrium = true requires "
                     "use_kirchhoff_law = true; the predicted (T*, Ye*) blackbody is "
                     "what Kirchhoff's law multiplies, and with it off nothing reads "
                     "the prediction." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nurates_params.use_partial_equilibrium = false;
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING: <bns_nurates>/use_partial_equilibrium defaults to "
                     "true but use_kirchhoff_law is false, so nothing would read the "
                     "predicted (T*, Ye*). Turning it off; the emissivity is "
                     "bns_nurates' own, evaluated at the start of the step."
                  << std::endl;
      }
    }

    if (!multi_freq && !nurates_params.use_equilibrium_distribution) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Gray Boltzmann nurates requires "
                << "<bns_nurates>/use_equilibrium_distribution = true. "
                << "The gray Boltzmann variables do not carry an independent "
                << "neutrino number density needed to reconstruct a non-equilibrium "
                << "distribution." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // floors -- key names and defaults match radiation_m1.cpp.  nb_min is in
    // fm^-3, i.e. the EOS number-density unit, and is compared against nb before
    // the conversion to bns_nurates units; it used to be compared after, so the
    // same number meant two things 1e18 apart in the two solvers.
    nurates_params.nb_min      = pin->GetOrAddReal("bns_nurates","nb_min_fm-3",0.0);
    nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates","temp_min_mev",0.0);

    // 1d Gauss-Legendre quadrature
    nurates_params.quad_nx = pin->GetOrAddInteger("bns_nurates","nurates_quad_nx",6);
    nurates_params.quadrature.nx   = nurates_params.quad_nx;
    nurates_params.quadrature.dim  = 1;
    nurates_params.quadrature.type = kGauleg;
    nurates_params.quadrature.x1   = 0.;
    nurates_params.quadrature.x2   = 1.;
    GaussLegendre(&nurates_params.quadrature);

    // 2d quadrature (same number of points as 1d unless overridden)
    nurates_params.quad_nx_2 = pin->GetOrAddInteger("bns_nurates","nurates_quad_nx_2",-1);
    if (nurates_params.quad_nx_2 < 0) {
      nurates_params.quad_nx_2 = nurates_params.quad_nx;
    }
    nurates_params.quadrature_2.nx   = nurates_params.quad_nx_2;
    nurates_params.quadrature_2.dim  = 1;
    nurates_params.quadrature_2.type = kGauleg;
    nurates_params.quadrature_2.x1   = 0.;
    nurates_params.quadrature_2.x2   = 1.;
    GaussLegendre(&nurates_params.quadrature_2);

    // allocate per-species opacity arrays [nmb, nspecies, nk, nj, ni]
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(nurates_eta_0,  nmb, nspecies, ncells3, ncells2, ncells1);
    Kokkos::realloc(nurates_eta_1,  nmb, nspecies, ncells3, ncells2, ncells1);
    Kokkos::realloc(nurates_abs_0,  nmb, nspecies, ncells3, ncells2, ncells1);
    Kokkos::realloc(nurates_abs_1,  nmb, nspecies, ncells3, ncells2, ncells1);
    Kokkos::realloc(nurates_scat_1, nmb, nspecies, ncells3, ncells2, ncells1);
    if (multi_freq) {
      Kokkos::realloc(nurates_eta_0_freq,  nmb, nspecies, nfreq, ncells3, ncells2, ncells1);
      Kokkos::realloc(nurates_eta_1_freq,  nmb, nspecies, nfreq, ncells3, ncells2, ncells1);
      Kokkos::realloc(nurates_abs_0_freq,  nmb, nspecies, nfreq, ncells3, ncells2, ncells1);
      Kokkos::realloc(nurates_abs_1_freq,  nmb, nspecies, nfreq, ncells3, ncells2, ncells1);
      Kokkos::realloc(nurates_scat_1_freq, nmb, nspecies, nfreq, ncells3, ncells2, ncells1);
      int nwork = NURATES_MF_WORK_SIZE(nfreq, prgeo->nangles);
      Kokkos::realloc(nurates_mf_work, nmb, nwork, ncells3, ncells2, ncells1);
    }
  }
#endif
  if (beam_source) {
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(beam_mask,nmb,nspecies*nfreq*prgeo->nangles,ncells3,ncells2,ncells1);
  }
}

//----------------------------------------------------------------------------------------
// destructor

Radiation::~Radiation() {
  delete pbval_i;
  delete prgeo;
  if (psrc != nullptr) {delete psrc;}
}

} // namespace radiation
