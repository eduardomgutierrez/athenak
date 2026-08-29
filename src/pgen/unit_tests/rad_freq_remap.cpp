//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_freq_remap.cpp
//  \brief Unit tests for the frequency remap of the nurates multi-frequency source term:
//         RemapBinValue, and the three passes built on it.
//
//  These run on host mirrors, so nothing here needs a mesh, an EOS or a time step, and
//  the whole suite costs milliseconds.  They are also the only place several of these
//  properties can be seen at all: the invariance tests in the suite either run at rest,
//  where the shift is the identity, or run a grey opacity, where it does not matter.
//
//  ## What the scheme has to satisfy
//
//  MultiFreqRadFluidCouplingNurates builds the scattering target on the fixed comoving
//  grid and then applies the update to the stored lab bins at each ray's own comoving
//  energy.  Four properties decide whether that is right:
//
//   1. **Exactness at rest.**  n0_cm == 1 gives a zero shift, and every lookup must
//      return the tabulated value bit-for-bit, or v = 0 stops reproducing the answer with
//      no frame treatment at all.
//   2. **Exactness on a power law.**  Bins uniform in ln(nu) make ln(q) linear in the bin
//      index for any power law in energy -- point-sampled and bin-integrated alike -- so
//      the log-space cubic must reproduce it, and its shift, to roundoff.  This is what
//      lets the bin-width stretch be carried with no separate Jacobian.
//   3. **Comoving isotropy is a fixed point, ray by ray.**  Isoenergetic scattering acts
//      at fixed comoving energy: it moves intensity between directions and cannot move
//      energy between comoving groups, so on a comoving-isotropic field it must do
//      nothing -- and, since the target is evaluated per ray, nothing *on every ray*, not
//      merely in the angular sum.  A fixed lab bin covers gamma_a*[e_lo, e_hi], different
//      on every ray, so this is a real constraint.  Section 3 drives the whole chain:
//      un-shift every ray, solve the implicit angular system on the fixed grid, read the
//      result back at each ray's own comoving energy, and ask whether it equals what that
//      ray is carrying.
//   4. **Conditioning.**  The previous scheme satisfied 1-3 and still failed, because its
//      target was an amplitude gfac(ifr) times a shifted spectrum, and both of gfac's
//      angular sums divide by that ray's own spectrum.  Once the spectrum spans enough
//      decades every ray but the most redshifted falls below the summation rounding and
//      the "angular mean" becomes one ray's value.  Section 4 asks the question that
//      catches it: are the terms entering the angular sums of comparable size?
//
//  Run via inputs/ut_rad_freq_remap.athinput; exits non-zero on failure, which is what
//  tst/test_suite/unit_tests/test_rad_freq_remap_cpu.py checks.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

#if ENABLE_NURATES
#include "radiation/radiation_nurates.hpp"
#include "radiation/radiation_nurates_remap.hpp"
#endif

namespace {

int nfail = 0;
int nrun = 0;

void Check(const std::string &what, Real got, Real want, Real tol) {
  ++nrun;
  Real err = (want != 0.0) ? fabs(got - want)/fabs(want) : fabs(got);
  bool ok = (err <= tol) && std::isfinite(got);
  if (!ok) { ++nfail; }
  std::cout << (ok ? "  [ ok ] " : "  [FAIL] ") << std::left << std::setw(46) << what
            << std::right << std::scientific << std::setprecision(6)
            << " got " << got << "  want " << want << "  rel " << err
            << "  tol " << tol << std::endl;
}

void CheckExact(const std::string &what, Real got, Real want) {
  ++nrun;
  bool ok = (got == want);
  if (!ok) { ++nfail; }
  std::cout << (ok ? "  [ ok ] " : "  [FAIL] ") << std::left << std::setw(46) << what
            << std::right << std::scientific << std::setprecision(17)
            << " got " << got << "  want " << want << std::endl;
}

}  // namespace

#if ENABLE_NURATES

namespace {

constexpr int kMaxF = 64;
constexpr int kMaxA = 256;

//! Frequency grid, mirroring SetFrequencyGrid then FreqBinEdgesMeV for the top bin.
struct Grid {
  int nfreq;
  Real edges[kMaxF+1];
  Real mid[kMaxF];
  Real dln;

  Grid(int nf, Real numin, Real numax) : nfreq(nf) {
    Real r = pow(numax/numin, 1.0/static_cast<Real>(nf-1));
    for (int f = 0; f < nf; ++f) { edges[f] = numin*pow(r, f); }
    edges[nf] = edges[nf-1]*edges[nf-1]/edges[nf-2];
    for (int f = 0; f < nf; ++f) { mid[f] = sqrt(edges[f]*edges[f+1]); }
    dln = log(numax/numin)/static_cast<Real>(nf-1);
  }
};

//! Bin integral of E^p: a power law in the bin index on a log grid, which is what
//! RemapBinValue has to reproduce exactly.
Real PowerLawBin(Real e_lo, Real e_hi, Real p) {
  return (pow(e_hi, p+1.0) - pow(e_lo, p+1.0))/(p + 1.0);
}

//! Bin integral of a Fermi-Dirac E^3/(1+exp(E/T - eta)), by trapezoid in ln(E).
Real FermiBin(Real e_lo, Real e_hi, Real temp, Real eta) {
  if (e_lo <= 0.0) { e_lo = 1.0e-8*e_hi; }
  const int nq = 2048;
  Real acc = 0.0, h = (log(e_hi) - log(e_lo))/static_cast<Real>(nq-1);
  for (int i = 0; i < nq; ++i) {
    Real e = exp(log(e_lo) + i*h);
    Real x = e/temp - eta;
    if (x > 700.0) { x = 700.0; }
    acc += (e*e*e/(1.0 + exp(x)))*e*h*((i == 0 || i == nq-1) ? 0.5 : 1.0);
  }
  return acc;
}

//! Bin integral of a Gaussian in ln(E) -- the seed the spectral diffusion test uses, and
//! the one spectral shape here that has margin at *both* ends of the grid.  ln of its bin
//! integral is quadratic in the bin index, which the log-space cubic reproduces exactly,
//! and so is its shift; so this is the shape on which the whole chain must close to
//! roundoff.  A power law cannot play that role: it grows without bound in one direction,
//! so it always runs off one end, where the continuation is deliberately clipped.
Real GaussLogBin(Real e_lo, Real e_hi, Real eref, Real sigma) {
  if (e_lo <= 0.0) { e_lo = 1.0e-8*e_hi; }
  const int nq = 2048;
  Real acc = 0.0, h = (log(e_hi) - log(e_lo))/static_cast<Real>(nq-1);
  for (int i = 0; i < nq; ++i) {
    Real y = log(e_lo) + i*h;
    Real z = (y - log(eref))/sigma;
    acc += exp(-0.5*z*z)*exp(y)*h*((i == 0 || i == nq-1) ? 0.5 : 1.0);
  }
  return acc;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadFreqRemap()

void ProblemGenerator::RadFreqRemap(ParameterInput *pin, const bool restart) {
  if (restart) { return; }

  int nfreq = pin->GetOrAddInteger("problem", "nfreq", 24);
  Real numin = pin->GetOrAddReal("problem", "nu_min", 10.0);
  Real numax = pin->GetOrAddReal("problem", "nu_max", 2000.0);
  int nang = pin->GetOrAddInteger("problem", "nang", 92);
  Real vx = pin->GetOrAddReal("problem", "vx", 0.5);
  Real temp = pin->GetOrAddReal("problem", "spec_temp", 12.0);
  Real eta = pin->GetOrAddReal("problem", "spec_eta", 3.0);
  Real anis = pin->GetOrAddReal("problem", "anisotropy", 0.1);
  // A power law is exact algebra for the log-space cubic; only roundoff separates it.
  Real tol_exact = pin->GetOrAddReal("problem", "tol_exact", 1.0e-11);
  // The Fermi-Dirac fixed point does not close exactly: the un-shift/re-read pair is
  // exact only where ln(q) is linear in the bin index.  This bounds the residual on the
  // bins that carry the spectrum; it is a characterisation, not a convergence claim.
  Real tol_fixed_fd = pin->GetOrAddReal("problem", "tol_fixed_fd", 3.0e-2);
  // Accuracy of the Gaussian seed's own quadrature, which floors the exactness check.
  Real tol_seed = pin->GetOrAddReal("problem", "tol_seed", 1.0e-6);
  // Terms in an angular sum must be of comparable size.  The previous closure's spread
  // reached 1e80 on this grid, which is what made its "angular mean" one ray's value.
  Real tol_spread = pin->GetOrAddReal("problem", "tol_spread", 1.0e6);

  if (nfreq > kMaxF || nang > kMaxA || nfreq < 5) {
    std::cout << "### FATAL ERROR: rad_freq_remap needs 5 <= nfreq <= " << kMaxF
              << " and nang <= " << kMaxA << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Grid g(nfreq, numin, numax);
  HostArray1D<Real> q("q", nfreq), lq("lq", nfreq);
  HostArray1D<Real> jr("jr", nfreq), ljr("ljr", nfreq);
  HostArray1D<Real> emid("emid", nfreq);
  for (int f = 0; f < nfreq; ++f) { emid(f) = g.mid[f]; }

  // Ray set: uniform in mu, which is all the shift machinery sees of the angular grid.
  Real wl = 1.0/sqrt(1.0 - vx*vx);
  HostArray1D<Real> n0_cm_a("n0cm", nang), dlt_a("dlt", nang), mu_a("mu", nang);
  HostArray1D<Real> wgt_a("wgt", nang);
  Real wght_sum = 0.0;
  for (int k = 0; k < nang; ++k) {
    mu_a(k) = -1.0 + 2.0*static_cast<Real>(k)/static_cast<Real>(nang-1);
    n0_cm_a(k) = wl*(1.0 - vx*mu_a(k));
    dlt_a(k) = log(n0_cm_a(k))/g.dln;
    wgt_a(k) = 1.0/(n0_cm_a(k)*n0_cm_a(k));   // omega_cm with a uniform solid angle
    wght_sum += wgt_a(k);
  }

  std::cout << "=== rad_freq_remap: nfreq=" << nfreq << " nu=[" << numin << ", "
            << numax << "] dlnnu=" << std::fixed << std::setprecision(4) << g.dln
            << " nang=" << nang << " v=" << vx << " max|delta|="
            << std::setprecision(3) << fmax(fabs(dlt_a(0)), fabs(dlt_a(nang-1)))
            << " bins" << std::endl;

  // ------------------------------------------------------------ 1. identity at rest
  // A ray at rest sits exactly on the grid, and the lookup must read the tabulated value
  // back bit-for-bit rather than exp(log(q)).  This is what makes v = 0 bit-identical.
  std::cout << "-- identity at n0_cm = 1" << std::endl;
  {
    for (int f = 0; f < nfreq; ++f) {
      q(f) = FermiBin(g.edges[f], g.edges[f+1], temp, eta);
    }
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    Real worst = 0.0;
    for (int f = 0; f < nfreq; ++f) {
      Real got = radiation::RemapBinValue(q, lq, 0, 0, nfreq-1, static_cast<Real>(f),
                                          radiation::RemapAsymptote::kOpacity, g.dln);
      worst = fmax(worst, fabs(got - q(f))/q(f));
    }
    // Relative, and it is roundoff rather than zero: the log-space form evaluates
    // exp(log(q)) even on a node.  That is precisely why the kernel short-circuits the
    // whole lookup on |delta| <= 1e-12 instead of relying on this -- so v = 0 is
    // bit-identical to running with no frame treatment, not merely close to it.
    // A few ulp: exp(log(q)) is not q.  The bound is on the *relative* error, and the
    // kernel never pays it -- the short-circuit means at-rest runs skip the lookup.
    Check("rest: lookup on a node returns the node", worst, 0.0, 4.0e-14);
  }

  // ---------------------------------------------------- 2. exactness on a power law
  std::cout << "-- lookup: exact on a power law" << std::endl;
  for (Real p : {0.0, 2.0, 3.0}) {
    for (int f = 0; f < nfreq; ++f) { q(f) = PowerLawBin(g.edges[f], g.edges[f+1], p); }
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    Real worst = 0.0, worst_out = 0.0;
    bool bounded = true;
    for (Real ncm : {wl*(1.0-vx), 1.3, wl*(1.0+vx)}) {
      Real dlt = log(ncm)/g.dln;
      for (int ifr = 0; ifr < nfreq; ++ifr) {
        Real x = static_cast<Real>(ifr) + dlt;
        Real got = radiation::RemapBinValue(q, lq, 0, 0, nfreq-1, x,
                                            radiation::RemapAsymptote::kOpacity, g.dln);
        // The bin the shifted index points at is ncm*[e_lo, e_hi] of bin ifr.
        Real want = PowerLawBin(ncm*g.edges[ifr], ncm*g.edges[ifr+1], p);
        if (x >= 0.0 && x <= nfreq-1) {
          worst = fmax(worst, fabs(got - want)/want);
        } else {
          // Off the tabulated grid the kOpacity continuation is an analytic power law at
          // the measured edge slope, so it is *exact* here, not merely bounded -- which
          // is
          // the point of giving the opacities their own asymptote.  sigma_s is exactly
          // E^2
          // and the beta-process exponentials cancel out of sigma_a, so this is the real
          // asymptotic behaviour and not a convenience.
          if (!(got >= 0.0) || !std::isfinite(got)) { bounded = false; }
          worst_out = fmax(worst_out, fabs(got - want)/want);
        }
      }
    }
    char buf[96];
    snprintf(buf, sizeof(buf), "E^%.0f: shifted bin integral, on grid", p);
    Check(buf, worst, 0.0, tol_exact);
    if (!bounded) { ++nrun; ++nfail;
      std::cout << "  [FAIL] E^" << p << ": off-grid value non-finite or negative"
                << std::endl;
    }
    snprintf(buf, sizeof(buf), "E^%.0f: shifted bin integral, off grid", p);
    Check(buf, worst_out, 0.0, tol_exact);
  }

  // ------------------------------- 3. comoving isotropy is a fixed point, ray by ray
  // The whole chain, with pure elastic scattering and a comoving-isotropic field, whose
  // exact answer is that nothing happens.  Pass 1 un-shifts every ray onto the fixed
  // grid; pass 2 solves the implicit angular system there; pass 3 reads the result back
  // at each ray's own comoving energy.  The residual reported is |target/own - 1|.
  std::cout << "-- comoving isotropy: target == the ray's own bin content" << std::endl;
  {
    HostArray1D<Real> ray("ray", nfreq), lray("lray", nfreq);
    for (int mode = 0; mode < 2; ++mode) {
      const bool power_law = (mode == 0);   // "power_law" now means the exact case
      const Real dtcsigs = 1.0;    // pure scattering; the fixed point cannot depend on it
      // Pass 1 and 2, accumulated over rays.
      for (int f = 0; f < nfreq; ++f) { jr(f) = 0.0; }
      HostArray1D<Real> sum1("s1", nfreq), sum2("s2", nfreq);
      for (int f = 0; f < nfreq; ++f) { sum1(f) = 0.0; sum2(f) = 0.0; }
      Real spread_lo = 1.0e300, spread_hi = 0.0;
      for (int a = 0; a < nang; ++a) {
        Real ncm = n0_cm_a(a);
        for (int f = 0; f < nfreq; ++f) {
          Real lo = ncm*g.edges[f], hi = ncm*g.edges[f+1];
          ray(f) = power_law ? GaussLogBin(lo, hi, g.mid[nfreq/2], 3.0*g.dln)
                             : FermiBin(lo, hi, temp, eta);
        }
        radiation::RemapFillLogs(ray, lray, 0, 0, nfreq-1);
        for (int f = 0; f < nfreq; ++f) {
          Real un = radiation::RemapBinValue(ray, lray, 0, 0, nfreq-1,
                                             static_cast<Real>(f) - dlt_a(a),
                                             radiation::RemapAsymptote::kSpectrum, g.dln);
          // n0 = 1 in flat space with this tetrad; the algebra below mirrors the kernel.
          Real vnc = 1.0/(1.0 + dtcsigs*ncm);
          sum1(f) += wgt_a(a)*ncm*vnc;
          sum2(f) += un*wgt_a(a)*vnc;
          if (f == nfreq/2) {
            spread_lo = fmin(spread_lo, un*wgt_a(a)*vnc);
            spread_hi = fmax(spread_hi, un*wgt_a(a)*vnc);
          }
        }
      }
      for (int f = 0; f < nfreq; ++f) {
        Real s1 = sum1(f)/wght_sum, s2 = sum2(f)/wght_sum;
        jr(f) = s2/(1.0 - s1*dtcsigs);
      }
      radiation::RemapFillLogs(jr, ljr, 0, 0, nfreq-1);

      // Pass 3: read jr back at every ray's own comoving energy.
      // Split the report at the coverage band.  Outside it the shifted index leaves the
      // tabulated range and the lookup is a bounded chord extrapolation -- exact for a
      // power law, but a genuine approximation for anything else.  The band is where
      // every ray stays on the grid; widening [nu_min, nu_max] by ceil(max|delta|) bins
      // is what shrinks the edges to nothing.
      Real dmax = 0.0, dmin = 0.0;
      for (int a = 0; a < nang; ++a) {
        dmax = fmax(dmax, dlt_a(a));
        dmin = fmin(dmin, dlt_a(a));
      }
      // The assertion band is where the target lookup's whole four-node stencil sits on
      // jr entries that were themselves built from on-grid un-shifts; outside it the
      // residual reported is the grid's coverage, not the scheme's accuracy.  Two steps:
      // jr(g) is clean only for g in [ceil(dmax), floor(nfreq-1+dmin)], because building
      // it un-shifts every ray by -dlt; and the read at f+dlt touches nodes
      // floor(f+dlt)-1 .. floor(f+dlt)+2.  Requiring the second inside the first is
      // stricter than the two-bin margin this used to carry -- that margin left the
      // outermost asserted bins reading one extrapolated node apiece, which the previous
      // closure happened to be forgiving about and this one is not.
      int g_lo = static_cast<int>(ceil(dmax));
      int g_hi = static_cast<int>(floor(static_cast<Real>(nfreq-1) + dmin));
      int b_lo = static_cast<int>(ceil(static_cast<Real>(g_lo) + 2.0 - dmin));
      int b_hi = static_cast<int>(floor(static_cast<Real>(g_hi) - 2.0 - dmax));
      Real worst = 0.0; int wray = -1, wbin = -1;
      Real worst_edge = 0.0;
      Real peak = 0.0;
      for (int f = 0; f < nfreq; ++f) {
        Real lo = g.edges[f], hi = g.edges[f+1];
        peak = fmax(peak, power_law ? GaussLogBin(lo, hi, g.mid[nfreq/2], 3.0*g.dln)
                                    : FermiBin(lo, hi, temp, eta));
      }
      for (int a = 0; a < nang; ++a) {
        Real ncm = n0_cm_a(a);
        for (int f = 0; f < nfreq; ++f) {
          Real own = power_law
                       ? GaussLogBin(ncm*g.edges[f], ncm*g.edges[f+1],
                                     g.mid[nfreq/2], 3.0*g.dln)
                       : FermiBin(ncm*g.edges[f], ncm*g.edges[f+1], temp, eta);
          // Only where the spectrum carries something.  A bin many decades below the
          // peak holds no energy, its relative residual is set by whatever the far wing
          // of the angular average happens to contain, and asserting on it measures the
          // test's own seed rather than the scheme.
          if (own < 1.0e-8*peak) { continue; }
          Real tgt = radiation::RemapBinValue(
              jr, ljr, 0, 0, nfreq-1, static_cast<Real>(f) + dlt_a(a),
              radiation::RemapAsymptote::kSpectrum, g.dln);
          Real err = fabs(tgt - own)/own;
          if (f >= b_lo && f <= b_hi) {
            if (err > worst) { worst = err; wray = a; wbin = f; }
          } else {
            worst_edge = fmax(worst_edge, err);
          }
        }
      }
      const std::string tag = power_law ? "Gaussian in lnE" : "Fermi-Dirac";
      std::cout << "     " << tag << ": band = bins " << b_lo << ".." << b_hi
                << " of " << nfreq << "; worst in band at bin " << wbin << ", ray "
                << wray << " (n0_cm = " << std::fixed << std::setprecision(3)
                << n0_cm_a(wray) << "); worst outside = " << std::scientific
                << std::setprecision(3) << worst_edge << std::endl;
      // ln of a Gaussian's bin integral is quadratic in the bin index, which the
      // log-space cubic reproduces exactly -- so the residual there is the accuracy of
      // the *seed*, built by trapezoid quadrature at 2048 points, not of the scheme.
      // Measured 4.8e-8, and it falls with the quadrature, not with anything under test.
      // The Fermi-Dirac case is the real characterisation: its ln q is not polynomial in
      // the index, so the un-shift/re-read pair does not close, and that residual is the
      // scheme's own -- but nothing is written back into i0_, so it does not accumulate.
      Check(tag + ": target == ray content, in band", worst, 0.0,
            power_law ? tol_seed : tol_fixed_fd);
      // Nothing is asserted outside the band: there the lookup leaves the tabulated
      // range and is deliberately clipped to decay outward, so the residual reported
      // above is a statement about the grid's margin, not about the scheme.

      // ------------------------------------------------------------ 4. conditioning
      // The question the previous closure failed.  Its target was an amplitude times a
      // shifted spectrum, and both of the amplitude's angular sums divide by that ray's
      // own spectrum, so once max/min across the ray set passes ~1e16 every ray but one
      // falls below the summation rounding.  Here the sums carry the intensity itself.
      std::cout << "     " << tag << ": angular-sum term spread at the peak bin = "
                << std::scientific << std::setprecision(3)
                << spread_hi/fmax(spread_lo, 1.0e-300) << std::endl;
      Check(tag + ": angular sum is not one-ray dominated",
            spread_hi/fmax(spread_lo, 1.0e-300), 1.0, tol_spread);
    }
  }

  // --------------------------------------------- 5. anisotropy, positivity, sentinels
  // The update must stay finite and non-negative on a field with holes -- which is the
  // normal case, since the update floors at zero -- and an empty ray must stay empty.
  std::cout << "-- anisotropic field, holes and empty rays" << std::endl;
  {
    for (int f = 0; f < nfreq; ++f) {
      q(f) = FermiBin(g.edges[f], g.edges[f+1], temp, eta)*(1.0 + anis);
    }
    q(nfreq/2) = 0.0;
    q(1) = 0.0;
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    bool ok = true;
    Real dlt = log(wl*(1.0+vx))/g.dln;
    for (int f = 0; f < nfreq; ++f) {
      for (Real off : {-dlt, dlt}) {
        Real got = radiation::RemapBinValue(q, lq, 0, 0, nfreq-1,
                                            static_cast<Real>(f) + off,
                                            radiation::RemapAsymptote::kSpectrum, g.dln);
        if (!(got >= 0.0) || !std::isfinite(got)) { ok = false; }
      }
    }
    ++nrun;
    if (!ok) { ++nfail; }
    std::cout << (ok ? "  [ ok ] " : "  [FAIL] ")
              << "spectrum with holes stays finite and >= 0" << std::endl;

    for (int f = 0; f < nfreq; ++f) { q(f) = 0.0; }
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    Real acc = 0.0;
    bool finite = true;
    for (int f = 0; f < nfreq; ++f) {
      Real got = radiation::RemapBinValue(q, lq, 0, 0, nfreq-1,
                                          static_cast<Real>(f) + dlt,
                                          radiation::RemapAsymptote::kSpectrum, g.dln);
      acc += fabs(got);
      if (!std::isfinite(got)) { finite = false; }
    }
    ++nrun;
    if (!finite) { ++nfail; }
    std::cout << (finite ? "  [ ok ] " : "  [FAIL] ") << "empty ray stays finite"
              << std::endl;
    CheckExact("empty ray stays empty", acc, 0.0);
  }

  // ------------------------------------------- 6. no source leaves the field untouched
  // The property the whole design rests on.  The photon path rebins i0_ and writes the
  // result back, and the round trip is not the identity -- measured drift of 3.5e-3 per
  // trip on this grid for a log-space cubic and 5.6e-1 for a conservative exponential
  // rebin, against O(1e4) steps in a production run.  Transforming the *update* instead
  // means a vanishing source cannot move the field at all, at any velocity.
  std::cout << "-- zero source leaves the stored field bit-identical" << std::endl;
  {
    Real worst = 0.0;
    for (int a = 0; a < nang; ++a) {
      Real ncm = n0_cm_a(a);
      for (int f = 0; f < nfreq; ++f) {
        Real i_old = FermiBin(ncm*g.edges[f], ncm*g.edges[f+1], temp, eta);
        // Exactly the kernel's pass 3 with sigma_a = sigma_s = eta = 0.
        Real vnc = 1.0/(1.0 + 0.0*ncm);
        Real di = ((0.0*0.0 + 0.0*0.0 - (0.0 + 0.0)*i_old)*ncm*vnc);
        Real i_new = fmax(i_old + di, 0.0);
        worst = fmax(worst, fabs(i_new - i_old));
      }
    }
    CheckExact("no source: i0_ unchanged, all rays", worst, 0.0);
  }

  std::cout << "=== rad_freq_remap: " << (nrun - nfail) << "/" << nrun
            << " checks passed" << std::endl;
  if (nfail > 0) {
    std::cout << "### FATAL ERROR: " << nfail << " rad_freq_remap check(s) failed"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::exit(EXIT_SUCCESS);
}

#else   // !ENABLE_NURATES

void ProblemGenerator::RadFreqRemap(ParameterInput *pin, const bool restart) {
  if (restart) { return; }
  std::cout << "=== rad_freq_remap: skipped, built without bns_nurates" << std::endl;
  std::exit(EXIT_SUCCESS);
}

#endif  // ENABLE_NURATES
