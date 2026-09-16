//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_nu_merge.cpp
//  \brief Unit tests for the heavy-flavour slot fold: NuratesFoldHeavySum,
//         NuratesFoldHeavyMean, NuratesMirrorHeavy, and the occupancy multiplicity.
//
//  <radiation>/nspecies selects how many slots the heavy flavours occupy: two at
//  nspecies = 4 (nu_x and nubar_x, two states each) or one at nspecies = 3 (all four
//  states).  bns_nurates always resolves four species, so the choice is a fold applied
//  where its wrappers return.
//
//  ## What the fold has to satisfy
//
//   1. **A no-op at nspecies = 4.**  Everything below runs on every existing simulation,
//      so nothing may move a bit in the four-slot case.
//   2. **Emission is conserved.**  The merged slot must radiate exactly what the two
//      slots radiated between them, or the heavy-lepton luminosity depends on a
//      discretization choice.
//   3. **Equal inputs are a fixed point of the mean.**  Every channel but the
//      electron-positron pair kernel gives nu_x and nubar_x identical coefficients, so
//      wherever the two agree the merge must be exact, not merely close.  This is what
//      lets a pair-off run be compared against a four-species run to roundoff.
//   4. **Kirchhoff survives the fold.**  The partial-equilibrium predictor sets
//      eta = kappa*J_eq, and J_eq is equal across the heavy slots identically (both carry
//      eta_t = 0).  Mean-folding kappa against sum-folding J_eq has to reproduce the
//      four-species sum term for term, or the equilibrium the source relaxes towards
//      moves.  This is the property that makes the merged emissivity exact rather than
//      approximate.
//   5. **The occupancy is per state.**  f feeds 1 - f in the pair and bremsstrahlung
//      emissivities, so it counts states, not slots: doubling the slot energy while
//      doubling its multiplicity must leave f alone.
//   6. **Both bns_nurates heavy ids get the merged occupancy.**  The kernels block each
//      heavy species against its partner, so a merged species blocks against itself.
//      Leaving id_anux at zero switches off the whole pair/brem absorption of nu_x.
//
//  These run on host mirrors, so nothing here needs a mesh, an EOS or a time step.
//  Run via inputs/ut_rad_nu_merge.athinput; exits non-zero on failure, which is what
//  tst/test_suite/unit_tests/test_rad_nu_merge_cpu.py checks.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

#if ENABLE_NURATES
#include "radiation/radiation_nurates.hpp"
#endif

namespace {

int nfail = 0;
int nrun = 0;

void CheckExact(const std::string &what, Real got, Real want) {
  ++nrun;
  bool ok = (got == want);
  if (!ok) { ++nfail; }
  std::cout << (ok ? "  [ ok ] " : "  [FAIL] ") << std::left << std::setw(50) << what
            << std::right << std::scientific << std::setprecision(17)
            << " got " << got << "  want " << want << std::endl;
}

void CheckTrue(const std::string &what, bool ok) {
  ++nrun;
  if (!ok) { ++nfail; }
  std::cout << (ok ? "  [ ok ] " : "  [FAIL] ") << what << std::endl;
}

}  // namespace

#if ENABLE_NURATES

namespace {

//! The occupancy conversion in CalcOpacityNurates_, built on the same
//! NuratesInvMultiplicity the kernel calls so that changing one moves both.
Real OccupancyOfSlot(Real j_bin, Real ph_bin, int isp, int nspecies) {
  Real inv_mult = radiation::NuratesInvMultiplicity(nspecies, isp);
  return (ph_bin > 0.0) ? (inv_mult*j_bin)/ph_bin : 0.0;
}

}  // namespace

void ProblemGenerator::RadNuMerge(ParameterInput *pin, const bool restart) {
  if (restart) { return; }

  std::cout << "=== rad_nu_merge: heavy-flavour slot fold ===" << std::endl;

  // Magnitudes from subnormal to near-overflow, and pairs separated by up to 50 decades,
  // so the sum fold is exercised where the two heavy slots are wildly unequal as well as
  // where they agree.
  const Real q2[] = {0.0, 1.0e-300, 3.7e-12, 1.0, 2.5, 6.02e23, 1.0e300};
  const Real q3[] = {0.0, 2.0e-300, 8.1e-12, 1.0, 1.0e-38, 6.02e23, 1.0e250};
  const int nq = static_cast<int>(sizeof(q2)/sizeof(q2[0]));

  // ---------------------------------------------------------------------------------
  std::cout << "-- 1. nspecies = 4 leaves every slot alone" << std::endl;
  {
    bool sum_noop = true, mean_noop = true, mirror_noop = true;
    for (int t = 0; t < nq; ++t) {
      Real ref[4] = {1.5, -2.5, q2[t], q3[t]};
      Real a[4], b[4];
      for (int i = 0; i < 4; ++i) { a[i] = ref[i]; b[i] = ref[i]; }
      radiation::NuratesFoldHeavySum(4, a);
      radiation::NuratesFoldHeavyMean(4, b);
      for (int i = 0; i < 4; ++i) {
        sum_noop = sum_noop && (a[i] == ref[i]);
        mean_noop = mean_noop && (b[i] == ref[i]);
      }
      bns_nurates::BS_REAL f[total_num_species] = {0.25, 0.5, q2[t], q3[t]};
      radiation::NuratesMirrorHeavy(4, f);
      mirror_noop = mirror_noop && (f[id_anux] == static_cast<bns_nurates::BS_REAL>(q3[t]));
    }
    CheckTrue("sum fold is the identity at nspecies = 4", sum_noop);
    CheckTrue("mean fold is the identity at nspecies = 4", mean_noop);
    CheckTrue("mirror is the identity at nspecies = 4", mirror_noop);
    CheckTrue("merge predicate: 3 merges, 1/2/4 do not",
              radiation::NuratesMergeHeavy(3) && !radiation::NuratesMergeHeavy(4) &&
              !radiation::NuratesMergeHeavy(1) && !radiation::NuratesMergeHeavy(2));
  }

  // ---------------------------------------------------------------------------------
  std::cout << "-- 2. the sum fold conserves, over 40 decades of separation" << std::endl;
  {
    bool clears = true;
    for (int t = 0; t < nq; ++t) {
      Real a[4] = {1.5, -2.5, q2[t], q3[t]};
      radiation::NuratesFoldHeavySum(3, a);
      CheckExact("sum fold: slot 2 carries both heavy slots", a[2], q2[t] + q3[t]);
      clears = clears && (a[3] == 0.0) && (a[0] == 1.5) && (a[1] == -2.5);
    }
    CheckTrue("sum fold clears slot 3 and leaves 0/1 alone", clears);
  }

  // ---------------------------------------------------------------------------------
  std::cout << "-- 3. the mean fold is exact where the two heavy slots agree" << std::endl;
  {
    bool idem = true, clears = true;
    for (int t = 0; t < nq; ++t) {
      Real a[4] = {1.5, -2.5, q2[t], q2[t]};
      radiation::NuratesFoldHeavyMean(3, a);
      idem = idem && (a[2] == q2[t]);
      clears = clears && (a[3] == 0.0);
    }
    CheckTrue("mean fold: equal heavy slots are a fixed point", idem);
    CheckTrue("mean fold clears slot 3", clears);
    // And it really is the mean when they differ.
    Real b[4] = {0.0, 0.0, 3.0, 5.0};
    radiation::NuratesFoldHeavyMean(3, b);
    CheckExact("mean fold: unequal heavy slots average", b[2], 4.0);
  }

  // ---------------------------------------------------------------------------------
  std::cout << "-- 4. Kirchhoff survives the fold: kappa_bar*J_eq_sum == sum kappa*J_eq"
            << std::endl;
  {
    // J_eq[2] == J_eq[3] identically -- both heavy species carry eta_t = 0 -- which is
    // what makes this an identity rather than an approximation.
    const Real kap2[] = {0.0, 1.0e-20, 0.125, 7.75, 1.0e18};
    const Real kap3[] = {3.5, 4.0e-20, 0.375, 2.25, 3.0e18};
    const Real jeqs[] = {0.0, 1.0e-30, 0.5, 1.0e12};
    Real worst = 0.0;
    for (int a = 0; a < 5; ++a) {
      for (int b = 0; b < 4; ++b) {
        Real kap[4] = {1.0, 1.0, kap2[a], kap3[a]};
        Real jeq[4] = {1.0, 1.0, jeqs[b], jeqs[b]};
        radiation::NuratesFoldHeavyMean(3, kap);
        radiation::NuratesFoldHeavySum(3, jeq);
        const Real merged = kap[2]*jeq[2];
        const Real split = kap2[a]*jeqs[b] + kap3[a]*jeqs[b];
        const Real rel = (split != 0.0) ? fabs(merged - split)/fabs(split)
                                        : fabs(merged);
        if (rel > worst) {
          worst = rel;
          std::cout << "       kappa = (" << kap2[a] << ", " << kap3[a]
                    << ")  J_eq = " << jeqs[b] << "  merged " << merged
                    << "  split " << split << "  rel " << rel << std::endl;
        }
      }
    }
    CheckTrue("merged emissivity matches the four-species sum to roundoff",
              worst <= 4.0*std::numeric_limits<Real>::epsilon());
  }

  // ---------------------------------------------------------------------------------
  std::cout << "-- 5. the occupancy counts states, not slots" << std::endl;
  {
    const Real ph = 3.25e-4;
    bool same = true;
    for (int t = 0; t < nq; ++t) {
      // Doubling the slot energy and its multiplicity together must leave f alone: this
      // is what makes a merged run reproduce a four-species one on the blocking factor.
      const Real f4 = OccupancyOfSlot(q2[t], ph, 2, 4);
      const Real f3 = OccupancyOfSlot(2.0*q2[t], ph, 2, 3);
      same = same && (f4 == f3);
    }
    CheckTrue("f is invariant under (slot energy, multiplicity) doubling", same);
    CheckExact("electron flavours take no multiplicity",
               OccupancyOfSlot(1.0, 4.0, 0, 3), 0.25);
    CheckExact("heavy slot at nspecies = 4 halves",
               OccupancyOfSlot(1.0, 4.0, 2, 4), 0.125);
    CheckExact("heavy slot at nspecies = 3 quarters",
               OccupancyOfSlot(1.0, 4.0, 2, 3), 0.0625);
  }

  // ---------------------------------------------------------------------------------
  std::cout << "-- 6. NuratesOccupancyAt fills both heavy ids on every branch"
            << std::endl;
  {
    const int nfreq = 6;
    Real fstore[3*nfreq];
    for (int isp = 0; isp < 3; ++isp) {
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        fstore[isp*nfreq + ifr] = 0.05 + 0.1*isp + 0.03*ifr;
      }
    }
    radiation::NuratesGroupOccupancy spec = {};
    spec.f = fstore;
    spec.sp_stride = nfreq;
    spec.fr_stride = 1;
    spec.nspecies = 3;
    spec.nfreq = nfreq;
    spec.log_grid = true;
    spec.x0 = log(10.0);
    spec.inv_dx = 1.0/(log(20.0) - log(10.0));

    bns_nurates::BS_REAL f[total_num_species];

    // Interior: the interpolation branch.
    radiation::NuratesOccupancyAt(spec, 45.0, f);
    CheckExact("interior: id_anux mirrors id_nux", f[id_anux], f[id_nux]);
    CheckTrue("interior: the merged slot is populated", f[id_nux] > 0.0);
    CheckTrue("interior: electron flavours differ from the heavy slot",
              f[id_nue] != f[id_nux] && f[id_anue] != f[id_nux]);

    // Below mid(0): the saturation branch, where a fermion occupation goes to 1.
    radiation::NuratesOccupancyAt(spec, 1.0, f);
    CheckExact("saturated: id_anux mirrors id_nux", f[id_anux], f[id_nux]);
    CheckExact("saturated: the merged slot saturates at 1", f[id_nux], 1.0);

    // Above mid(nfreq-1): Boltzmann-suppressed, so zero for all four.
    radiation::NuratesOccupancyAt(spec, 1.0e6, f);
    CheckExact("suppressed: id_anux mirrors id_nux", f[id_anux], f[id_nux]);
    CheckExact("suppressed: the merged slot vanishes", f[id_nux], 0.0);

    // A non-positive energy takes the zero branch.
    radiation::NuratesOccupancyAt(spec, -1.0, f);
    CheckExact("non-positive energy: id_anux mirrors id_nux", f[id_anux], f[id_nux]);

    // At nspecies = 4 the stored slot 3 is read, not mirrored.
    Real fstore4[4*nfreq];
    for (int isp = 0; isp < 4; ++isp) {
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        fstore4[isp*nfreq + ifr] = 0.05 + 0.1*isp + 0.03*ifr;
      }
    }
    radiation::NuratesGroupOccupancy spec4 = spec;
    spec4.f = fstore4;
    spec4.nspecies = 4;
    radiation::NuratesOccupancyAt(spec4, 45.0, f);
    CheckTrue("nspecies = 4: id_anux keeps its own stored slot", f[id_anux] != f[id_nux]);
  }

  // ---------------------------------------------------------------------------------
  std::cout << "-- 7. the fold reaches a real call site" << std::endl;
  {
    // NuratesEqSpectrumBin is the cheapest of the four wrappers to drive on the host --
    // no cell state, no opacity flags -- and it carries both the mu+tau factor of two and
    // the fold, so it pins the composition of the two.  Checks 1-6 exercise the
    // primitives alone and would pass if a call site folded with the wrong rule.
    bns_nurates::MyQuadrature quad = {};
    quad.nx = 6;
    quad.dim = 1;
    quad.type = bns_nurates::kGauleg;
    quad.x1 = 0.;
    quad.x2 = 1.;
    bns_nurates::GaussLegendre(&quad);

    radiation::NuratesParams np4{}, np3{};
    np4.nspecies = 4;
    np3.nspecies = 3;

    const Real temp = 12.0, mu_n = 950.0, mu_p = 930.0, mu_e = 60.0;
    Real n4[4] = {0.}, j4[4] = {0.}, n3[4] = {0.}, j3[4] = {0.};
    radiation::NuratesEqSpectrumBin(10.0, 30.0, temp, mu_n, mu_p, mu_e, quad,
                                    1.0, 1.0, n4, j4, np4);
    radiation::NuratesEqSpectrumBin(10.0, 30.0, temp, mu_n, mu_p, mu_e, quad,
                                    1.0, 1.0, n3, j3, np3);

    CheckTrue("four-species heavy slots are populated and equal",
              n4[2] > 0.0 && n4[2] == n4[3] && j4[2] == j4[3]);
    CheckExact("merged n_eq[2] is the four-state sum", n3[2], n4[2] + n4[3]);
    CheckExact("merged J_eq[2] is the four-state sum", j3[2], j4[2] + j4[3]);
    CheckTrue("electron flavours untouched by the merge",
              n3[0] == n4[0] && n3[1] == n4[1] && j3[0] == j4[0] && j3[1] == j4[1]);
    CheckExact("merged slot 3 is cleared", n3[3], 0.0);
    // A mean fold here would give n4[2], half the right answer; this is what separates
    // the two rules at the call site rather than in the primitive.
    CheckTrue("the sum rule, not the mean rule, is what ran", n3[2] != n4[2]);

    // The frequency-integrated counterpart, which takes no quadrature.
    Real gn4[4] = {0.}, gj4[4] = {0.}, gn3[4] = {0.}, gj3[4] = {0.};
    radiation::NuratesEqDensities(temp, mu_n, mu_p, mu_e, 1.0, 1.0, gn4, gj4, np4);
    radiation::NuratesEqDensities(temp, mu_n, mu_p, mu_e, 1.0, 1.0, gn3, gj3, np3);
    CheckTrue("NuratesEqDensities: four-species heavy slots equal and populated",
              gn4[2] > 0.0 && gn4[2] == gn4[3] && gj4[2] == gj4[3]);
    CheckExact("NuratesEqDensities: merged n_eq[2] is the sum", gn3[2], gn4[2] + gn4[3]);
    CheckExact("NuratesEqDensities: merged J_eq[2] is the sum", gj3[2], gj4[2] + gj4[3]);
  }

  std::cout << "=== rad_nu_merge: " << (nrun - nfail) << "/" << nrun
            << " checks passed" << std::endl;
  if (nfail > 0) {
    std::cout << "### FATAL ERROR: " << nfail << " rad_nu_merge check(s) failed"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::exit(EXIT_SUCCESS);
}

#else   // !ENABLE_NURATES

void ProblemGenerator::RadNuMerge(ParameterInput *pin, const bool restart) {
  if (restart) { return; }
  std::cout << "=== rad_nu_merge: skipped, built without bns_nurates" << std::endl;
  std::exit(EXIT_SUCCESS);
}

#endif  // ENABLE_NURATES
