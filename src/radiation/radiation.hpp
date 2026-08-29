#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "config.hpp"
#if ENABLE_NURATES
#include "radiation_nurates.hpp"
#endif

// forward declarations
class EquationOfState;
class Coordinates;
class SourceTerms;
class GeodesicGrid;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation tasks

struct RadiationTaskIDs {
  TaskID rad_irecv;
  TaskID mhd_irecv;
  TaskID hyd_irecv;
  TaskID copyu;
  TaskID rad_flux;
  TaskID mhd_flux;
  TaskID hyd_flux;
  TaskID rad_sendf;
  TaskID mhd_sendf;
  TaskID hyd_sendf;
  TaskID rad_recvf;
  TaskID mhd_recvf;
  TaskID hyd_recvf;
  TaskID rad_rkupdt;
  TaskID mhd_rkupdt;
  TaskID hyd_rkupdt;
  TaskID rad_src;
  TaskID mhd_src;
  TaskID hyd_src;
  TaskID rad_calcop;   // nurates opacity calculation
  TaskID rad_coupl;
  TaskID rad_chiral;   // chiral Gamma_m sink + E.B anomaly source
  TaskID rad_resti;
  TaskID hyd_restu;
  TaskID mhd_restu;
  TaskID rad_sendi;
  TaskID mhd_sendu;
  TaskID hyd_sendu;
  TaskID rad_recvi;
  TaskID mhd_recvu;
  TaskID hyd_recvu;
  TaskID mhd_efld;
  TaskID mhd_sende;
  TaskID mhd_recve;
  TaskID mhd_ct;
  TaskID mhd_restb;
  TaskID mhd_sendb;
  TaskID mhd_recvb;
  TaskID bcs;
  TaskID rad_prol;
  TaskID mhd_prol;
  TaskID hyd_prol;
  TaskID mhd_c2p;
  TaskID hyd_c2p;
  TaskID rad_csend;
  TaskID mhd_csend;
  TaskID hyd_csend;
  TaskID rad_crecv;
  TaskID mhd_crecv;
  TaskID hyd_crecv;
};

namespace radiation {

//----------------------------------------------------------------------------------------
//! \class Radiation

class Radiation {
 public:
  Radiation(MeshBlockPack *ppack, ParameterInput *pin);
  ~Radiation();

  // flags to denote hydro/mhd is enabled or units enabled
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool are_units_enabled;

  // bns_nurates opacity library
  bool use_nurates = false;
  // >= 0 replaces the library opacities with elastic scattering, so the nurates
  // source terms can be run against an analytic solution.
  Real nurates_toy_scattering = -1.0;
  // sigma_s = nurates_toy_scattering*(e_mid/nurates_toy_scat_eref)^nurates_toy_scat_p,
  // with e_mid the comoving bin midpoint.  p = 0 is the constant opacity the grey
  // diffusion test uses, and keeps the answer separable: elastic scattering cannot move
  // energy between comoving groups, so every group shows the grey relative error.  p != 0
  // gives each comoving group its own diffusion coefficient, so nothing but a correct
  // per-ray comoving energy can reproduce it -- which is the whole frame question, and
  // the only configuration that measures it.  The grey slot keeps the p = 0 value.
  Real nurates_toy_scat_p = 0.0;
  Real nurates_toy_scat_eref = 1.0;
#if ENABLE_NURATES
  NuratesParams nurates_params;
  bool nurates_debug_opacity = false;
  Real nurates_baryon_mass = 1.0;    // baryon mass in code units (for Ye update)
  // code energy density -> EOS energy density (MeV/fm^3).  Dividing by a neutrino
  // energy in MeV then turns a code-unit radiation energy density into fm^-3.
  Real nurates_code_edens_to_eos = 1.0;
  // Neutrino number densities (nurates_eta_0*, and the N reconstructed from them in
  // the source terms) are in the EOS number-density unit, fm^-3, as in radiation_m1/;
  // everything else below is in code units.
  DvceArray5D<Real> nurates_eta_0;   // number emissivity    [nspecies, nk, nj, ni]
  DvceArray5D<Real> nurates_eta_1;   // energy emissivity
  DvceArray5D<Real> nurates_abs_0;   // number absorption opacity
  DvceArray5D<Real> nurates_abs_1;   // energy absorption opacity
  DvceArray5D<Real> nurates_scat_1;  // energy scattering opacity
  DvceArray6D<Real> nurates_eta_0_freq;   // [nmb, nspecies, nfreq, nk, nj, ni]
  DvceArray6D<Real> nurates_eta_1_freq;   // [nmb, nspecies, nfreq, nk, nj, ni]
  DvceArray6D<Real> nurates_abs_0_freq;
  DvceArray6D<Real> nurates_abs_1_freq;
  DvceArray6D<Real> nurates_scat_1_freq;
  // Per-cell workspace for the multi-frequency source term, [nmb, nwork, nk, nj, ni].
  // It lives here rather than in team scratch because the kernel is dispatched with a
  // flat par_for: the body is serial over the whole cell, so it must own its cell
  // outright, and a team-scratch allocation shared by several team members would be
  // written by all of them.  See MultiFreqRadFluidCouplingNurates.
  DvceArray5D<Real> nurates_mf_work;
#endif

  // Radiation source term parameters
  bool rad_source;          // flag to enable/disable radiation source term
  bool fixed_fluid;         // flag to enable/disable fluid integration
  bool affect_fluid;        // flag to enable/disable feedback of rad field on fluid
  bool evolve_ye;           // update fluid electron fraction from neutrino sources
  // 0 = 'opacity' (integrate the number source), 1 = 'moment' (energy density
  // change over a mean energy).  Grey nurates only: the multifrequency path takes
  // the per-bin number moment and rejects anything but 'moment'.
  int ye_source_model;
  bool backreact_chiral;    // source the chiral imbalance Y5 from the weak reactions
  bool chiral_gamma_m;      // apply the chirality-flip sink and the E.B anomaly source
  Real arad;                // radiation constant
  Real kappa_a;             // constant Rosseland mean absorption coefficient
  Real kappa_s;             // constant scattering coefficient
  Real kappa_p;             // Planck - Rosseland mean coefficient
  bool power_opacity;       // flag to enable Kramer's law opacity for kappa_a
  bool is_compton_enabled;  // flag to enable/disable compton
  bool correct_radsrc_velocity;
  bool correct_radsrc_opacity;
  Real dfloor_opacity;
  Real dens_trunc_max;
  Real tau_truncation;
  Real sigmoid_residual;

  // Extra physics (i.e., other srcterms)
  bool beam_source;        // flag to enable/disable user beam source masks
  SourceTerms *psrc = nullptr;

  // Angular mesh
  bool rotate_geo;                    // rotate geodesic mesh
  bool angular_fluxes;                // flag to enable/disable angular fluxes
  Real n_0_floor;                     // floor on n_0
  GeodesicGrid *prgeo = nullptr;      // pointer to radiation angular mesh

  // Radiation particle type
  bool is_neutrino = false;    // true for neutrino transport, false for photon (default)

  // Multi-frequency, multi-species radiation
  int nspecies = 1;            // number of radiation species (1 for photons, 3-4 for neutrinos)
  bool multi_freq = false;
  int nfreq = 1;               // for multi-frequency, nfreq >= 3
  // Number of (species, frequency) slots in the multi-frequency output arrays.
  // i0 is indexed nspecies*nfreq*nangles, so the radnu_* moments carry one slot
  // per species per group, ordered species-major: slot = isp*nfreq + ifr.
  int NFreqOut() const { return nspecies*nfreq; }
  int flag_fscale;             // 0: linear, 1: log, 2: customize
  Real nu_max, nu_min;         // minimum and maximum frequency (excluding zero and infinity)
  bool freq_fluxes;            // flag to enable/disable frequency fluxes
  // Bin lower edges.  Photons: code frequency units.  Neutrinos with nurates: MeV,
  // since a code-unit neutrino energy underflows single precision.
  DvceArray1D<Real> freq_grid;
  // Multiplies a freq_grid entry to give a code energy/frequency.  1 except on the
  // nurates neutrino path; only the blackbody tail in the frequency fluxes needs it.
  Real nu_grid_to_code = 1.0;
  // nu_min/nu_max expressed in the units freq_grid actually holds, i.e. divided by
  // the same nu_unit SetFrequencyGrid applied.  Any bin spacing must be derived from
  // these: a log grid is a ratio and does not care, but a linear grid built from the
  // raw nu_min/nu_max is off by nu_unit wherever that is not 1.
  Real grid_nu_min = 0.0, grid_nu_max = 0.0;
  DvceArray5D<Real> nnu_coeff; // n^a n^b omega^0_{ab} for computing frequency fluxes
  Real tol_rel_tgas_compton;
  int num_iter_compton;
  int order_multifreq;    // reconstruction order used in intensity mapping; option: 0, 1, 2 (default)
  int limiter_multifreq;  // reconstruction limiter used in intensity mapping; 0: no limiter, 1: minmod, 2: van Leer (default)

  // Flags used in multi-frequency radiation
  bool update_fluid_energy;
  bool update_fluid_moment;
  bool test_only_compton_therm;
  bool est_tgas_4th_compton;
  bool est_tgas_5th_compton;

  // DvceArray2D<Real> matrix_imap;
  Real kappa_r_multi_freq; // constant Rosseland mean absoprtion coefficient
  Real kappa_s_multi_freq; // constant scattering coefficient
  Real kappa_p_multi_freq; // Planck mean coefficient
  void SetFrequencyGrid();

  // Tetrad arrays and functions
  DualArray2D<Real> nh_c;             // normal vector computed at face center
  DualArray3D<Real> nh_f;             // normal vector computed at face edges
  DvceArray6D<Real> tet_c;            // tetrad components at cell centers
  DvceArray6D<Real> tetcov_c;         // covariant tetrad components at cell centers
  DvceArray5D<Real> tet_d1_x1f;       // tetrad components (subset) at x1f
  DvceArray5D<Real> tet_d2_x2f;       // tetrad components (subset) at x2f
  DvceArray5D<Real> tet_d3_x3f;       // tetrad components (subset) at x3f
  DvceArray6D<Real> na;               // n^a
  DvceArray6D<Real> norm_to_tet;      // used in transform b/w normal frame and tet frame
  void SetOrthonormalTetrad();

  // intensity arrays
  DvceArray5D<Real> i0;         // intensities
  DvceArray5D<Real> coarse_i0;  // intensities on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for i
  MeshBoundaryValuesCC *pbval_i;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;         // intensity at intermediate step
  DvceFaceFld5D<Real> iflx;     // spatial fluxes on zone faces
  DvceArray5D<Real> divfa;      // angular flux divergence + frequency flux divergence (if applicable)
  DvceArray5D<bool> beam_mask;  // boolean mask used for beam source term
  Real dtnew;

  // reconstruction method
  ReconstructionMethod recon_method;

  // container to hold names of TaskIDs
  RadiationTaskIDs id;

  // functions...
  void AssembleRadTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalculateFluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus RadSrcTerms(Driver *d, int stage);
  TaskStatus RadFluidCoupling(Driver *d, int stage);
#if ENABLE_NURATES
  TaskStatus RadFluidCouplingNurates(Driver *d, int stage);
  TaskStatus CalcOpacityNurates(Driver *d, int stage);
  TaskStatus CalcOpacityNuratesToy(Driver *d, int stage);
  template <class EOSPolicy, class ErrorPolicy>
  TaskStatus CalcOpacityNurates_(Driver *d, int stage);
#endif
  // chiral magnetic effect: Gamma_m sink + E.B anomaly source for Y5.  Not
  // gated on ENABLE_NURATES -- it needs only the EOS, MHD and ADM.  (The Y5
  // URCA source lives with the Ye update in radiation_source_nurates.cpp.)
  TaskStatus ChiralSources(Driver *d, int stage);
  template <class EOSPolicy, class ErrorPolicy>
  TaskStatus ChiralSources_(Driver *d, int stage);
  TaskStatus RestrictI(Driver *d, int stage);
  TaskStatus SendI(Driver *d, int stage);
  TaskStatus RecvI(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);

  // Multi-frequency radiation
  TaskStatus MultiFreqRadFluidCoupling(Driver *d, int stage);
#if ENABLE_NURATES
  TaskStatus MultiFreqRadFluidCouplingNurates(Driver *d, int stage);
#endif

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Radiation
};

} // namespace radiation
#endif // RADIATION_RADIATION_HPP_
