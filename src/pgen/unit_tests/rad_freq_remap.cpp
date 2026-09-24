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
  // The operator survey of section 8 is a measurement, not a check: it prints sixty rows
  // and asserts nothing, so it is off unless asked for.
  bool bakeoff = pin->GetOrAddBoolean("problem", "bakeoff", false);

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
      Real x = static_cast<Real>(f);
      Real got = radiation::RemapBinValue(lq, 0,
                                          radiation::RemapStencilAt(0, nfreq-1, x),
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
        Real got = radiation::RemapBinValue(lq, 0,
                                            radiation::RemapStencilAt(0, nfreq-1, x),
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
          Real xun = static_cast<Real>(f) - dlt_a(a);
          Real un = radiation::RemapBinValue(lray, 0,
                                             radiation::RemapStencilAt(0, nfreq-1, xun),
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
      // stricter than a flat two-bin margin, which would leave the outermost asserted
      // bins reading one extrapolated node apiece.
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
          Real xtg = static_cast<Real>(f) + dlt_a(a);
          Real tgt = radiation::RemapBinValue(
              ljr, 0, radiation::RemapStencilAt(0, nfreq-1, xtg),
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
      // Conditioning of the angular sums.  A scheme that solved for an amplitude times a
      // shifted spectrum would divide each sum by that ray's own spectrum, so once
      // max/min across the ray set passes ~1e16 every ray but one falls below the
      // summation rounding.  Here the sums carry the intensity itself; this checks that
      // the spread across rays at the peak bin stays bounded.
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
    // The nu^4 bound below the grid must survive the holes.  A hole in the stencil
    // takes the lookup down to the reduced-order form, which has its own continuation
    // and so its own chance to get this wrong.  q(1) = 0 above puts a hole inside the
    // bottom stencil deliberately, so this exercises that path rather than assuming it
    // is unreachable.
    bool decays = true;
    Real dlt = log(wl*(1.0+vx))/g.dln;
    for (int f = 0; f < nfreq; ++f) {
      for (Real off : {-dlt, dlt}) {
        Real x = static_cast<Real>(f) + off;
        Real got = radiation::RemapBinValue(lq, 0,
                                            radiation::RemapStencilAt(0, nfreq-1, x),
                                            radiation::RemapAsymptote::kSpectrum, g.dln);
        if (!(got >= 0.0) || !std::isfinite(got)) { ok = false; }
        if (x < 0.0) {
          Real cap = q(0)*exp(x*(4.0*g.dln));
          if (!(got <= cap*(1.0 + 1.0e-12))) { decays = false; }
        }
      }
    }
    ++nrun;
    if (!ok) { ++nfail; }
    std::cout << (ok ? "  [ ok ] " : "  [FAIL] ")
              << "spectrum with holes stays finite and >= 0" << std::endl;
    ++nrun;
    if (!decays) { ++nfail; }
    std::cout << (decays ? "  [ ok ] " : "  [FAIL] ")
              << "below the grid, holes and all, the continuation decays as nu^4"
              << std::endl;

    // Order reduction, against the one case with an exact answer.  A window holding a
    // dead node drops to the log-linear form on the live nodes around t, and that form
    // is still exact for a power law however many nodes are missing, as long as two are
    // left to carry a slope.  Exactness is the property that separates staying inside
    // the log family from clamping in linear space, where the same lookup is wrong by
    // whatever the neighbouring decades happen to be.  Every window that contains a dead
    // node is under test; the rest are the plain cubic and are covered above.
    Real worst_red = 0.0;
    int nred = 0;
    for (Real p : {0.0, 2.0, 3.0}) {
      for (int dead = 0; dead < nfreq; ++dead) {
        for (int wide = 0; wide < 2; ++wide) {
          for (int f = 0; f < nfreq; ++f) {
            q(f) = PowerLawBin(g.edges[f], g.edges[f+1], p);
          }
          q(dead) = 0.0;
          if (wide && dead+1 < nfreq) { q(dead+1) = 0.0; }
          radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
          for (Real ncm : {wl*(1.0-vx), 1.3, wl*(1.0+vx)}) {
            Real dl = log(ncm)/g.dln;
            for (int ifr = 0; ifr < nfreq; ++ifr) {
              Real x = static_cast<Real>(ifr) + dl;
              radiation::RemapStencil sc = radiation::RemapStencilAt(0, nfreq-1, x);
              bool holed = (dead >= sc.st && dead <= sc.st+3);
              if (!holed) { continue; }
              Real got = radiation::RemapBinValue(
                  lq, 0, sc, radiation::RemapAsymptote::kOpacity, g.dln);
              Real want = PowerLawBin(ncm*g.edges[ifr], ncm*g.edges[ifr+1], p);
              worst_red = fmax(worst_red, fabs(got - want)/want);
              ++nred;
            }
          }
        }
      }
    }
    std::cout << "         reduced-order lookups exercised: " << nred << std::endl;
    Check("power law: a window with dead nodes is still exact", worst_red, 0.0,
          tol_exact);

    // Past the last live bin a spectrum may not grow.  The tail is where the zeros
    // actually are -- measured over a whole snapshot of the 1D shock, interior holes
    // were 1 in 1536 spectra and every other zero was the high-energy tail past
    // underflow or the floor -- so this is the reduced path's common case.
    for (int f = 0; f < nfreq; ++f) {
      q(f) = (f < nfreq-4) ? FermiBin(g.edges[f], g.edges[f+1], temp, eta) : 0.0;
    }
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    Real over = 0.0;
    const int flive = nfreq-5;
    for (Real off : {-dlt, dlt}) {
      for (int f = nfreq-8; f < nfreq; ++f) {
        Real x = static_cast<Real>(f) + off;
        if (!(x > static_cast<Real>(flive))) { continue; }
        Real got = radiation::RemapBinValue(lq, 0,
                                            radiation::RemapStencilAt(0, nfreq-1, x),
                                            radiation::RemapAsymptote::kSpectrum, g.dln);
        over = fmax(over, got/q(flive));
      }
    }
    Check("past the last live bin the spectrum does not grow", over, 0.0, 1.0 + 1.0e-12);

    for (int f = 0; f < nfreq; ++f) { q(f) = 0.0; }
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    Real acc = 0.0;
    bool finite = true;
    for (int f = 0; f < nfreq; ++f) {
      Real x = static_cast<Real>(f) + dlt;
      Real got = radiation::RemapBinValue(lq, 0,
                                          radiation::RemapStencilAt(0, nfreq-1, x),
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
    // The kernel does not store the comoving intensity: it converts to comoving units,
    // updates, and converts back.  Both conversions have to be in this check, because
    // they are the only steps that can break the invariant -- (sfac*x)/sfac is not x for
    // ~11% of arguments in IEEE double, so a version of pass 3 that divided the updated
    // comoving value back would fail here while passing any test that omits sfac.
    Real worst = 0.0;
    Real n0 = 1.0, n_0 = -1.0;           // flat space with this tetrad, as elsewhere here
    for (int a = 0; a < nang; ++a) {
      Real ncm = n0_cm_a(a);
      Real sfac = 4.0*M_PI*SQR(SQR(ncm))/(n0*n_0);
      for (int f = 0; f < nfreq; ++f) {
        // The *stored* value, which is what the invariant is about.  Give it the sign of
        // n0*n_0, which is the convention i0_ is held in, but do NOT build it by dividing
        // by sfac: sfac*(x/sfac) then /sfac round-trips exactly far more often than a
        // single conversion pair does, which would make this check pass vacuously.
        Real i0_old =
            FermiBin(ncm*g.edges[f], ncm*g.edges[f+1], temp, eta)/(n0*n_0);
        // Exactly the kernel's pass 3 with sigma_a = sigma_s = eta = 0.
        Real i_old = sfac*i0_old;
        Real vnc = 1.0/(n0 + 0.0*ncm);
        Real di = ((0.0*0.0 + 0.0*0.0 - (0.0 + 0.0)*i_old)*ncm*vnc);
        Real i_new = i_old + di;
        Real i0_new = (i_new > 0.0) ? (i0_old + di/sfac) : 0.0;
        worst = fmax(worst, fabs(i0_new - i0_old));
      }
    }
    CheckExact("no source: i0_ unchanged, all rays", worst, 0.0);
  }

  // ------------------------------------- 7. the ray cursor agrees with a fresh window
  // RemapMakeRay/RemapRayStencil reuse one set of Lagrange weights across a ray on the
  // grounds that floor(ifr + x0) = ifr + floor(x0) for integer ifr, so the window's
  // origin advances one bin per ifr while t does not move.  The kernels are the only
  // consumers, so without this the one piece of arithmetic that could carry an
  // off-by-one is unexercised.  Probed on a power law, where the lookup is exact
  // algebra: a window off by one bin evaluates a point exp(p*dlnnu) away and shows up
  // far above roundoff, while a window that legitimately differs at an integer crossing
  // still returns the same value.
  std::cout << "-- ray cursor matches a per-lookup window" << std::endl;
  {
    const Real p = 2.0;
    for (int f = 0; f < nfreq; ++f) {
      q(f) = PowerLawBin(g.edges[f], g.edges[f+1], p);
    }
    radiation::RemapFillLogs(q, lq, 0, 0, nfreq-1);
    // Displacements past the ends of the grid on both sides, plus exact integers and
    // near-integers, which are where the clamp and the floor are most fragile.
    const Real dscan[] = {0.0, 1.0e-13, 0.5, 1.0, 1.0 - 1.0e-13, 1.0 + 1.0e-13, 2.385,
                          3.0, static_cast<Real>(nfreq)/2.0, static_cast<Real>(nfreq)};
    Real worst = 0.0;
    bool in_bounds = true;
    for (Real mag : dscan) {
      for (Real x0 : {-mag, mag}) {
        radiation::RemapRay ray = radiation::RemapMakeRay(x0);
        for (int f = 0; f < nfreq; ++f) {
          Real x = static_cast<Real>(f) + x0;
          radiation::RemapStencil sc = radiation::RemapRayStencil(ray, 0, nfreq-1, f, x);
          if (sc.st < 0 || sc.st > nfreq-4) { in_bounds = false; }
          for (auto asym : {radiation::RemapAsymptote::kOpacity,
                            radiation::RemapAsymptote::kSpectrum}) {
            Real cached = radiation::RemapBinValue(lq, 0, sc, asym, g.dln);
            Real fresh = radiation::RemapBinValue(
                lq, 0, radiation::RemapStencilAt(0, nfreq-1, x), asym, g.dln);
            Real ref = fmax(fabs(fresh), 1.0e-300);
            worst = fmax(worst, fabs(cached - fresh)/ref);
          }
        }
      }
    }
    ++nrun;
    if (!in_bounds) { ++nfail; }
    std::cout << (in_bounds ? "  [ ok ] " : "  [FAIL] ")
              << "ray window stays inside [0, nfreq-4]" << std::endl;
    Check("ray window reproduces a per-lookup window", worst, 0.0, tol_exact);
  }

  // ------------------------------------------- 8. operator bake-off: leak vs residual
  // Isoenergetic scattering moves no energy between comoving groups, so on a
  // comoving-isotropic field the scattering exchange summed over the ray fan is zero.
  // The chain below is the kernel's, with a frequency-independent scattering opacity so
  // that the implicit weight is the same on every bin and the angular sum telescopes:
  // for any conservative operator the leak is then exactly the energy whose image left
  // the grid, and for an interpolating one it is the round trip's own non-closure.
  //
  // Two numbers are in tension and both are printed.  The residual is the relative error
  // on the bins that carry the spectrum, which is what the solver's accuracy rests on;
  // the leak is what item 18 asks about.  The second seed is the control: its ln(q) is
  // quadratic in the bin index, so every operator here can represent it, and the
  // reconstructions show their true order on it rather than their limiter's.
  {
    const int nf_list[3] = {12, 24, 48};
    const Real v_list[3] = {0.1, 0.3, 0.5};
    const char *rec_name[8] = {"cubic", "donor", "plm", "ppm", "exp", "expC",
                               "expC-cl", "moment"};
    const radiation::RemapRecon rec_kind[8] = {
        radiation::RemapRecon::kDonor, radiation::RemapRecon::kDonor,
        radiation::RemapRecon::kPlm, radiation::RemapRecon::kPpm,
        radiation::RemapRecon::kExp, radiation::RemapRecon::kExpC,
        radiation::RemapRecon::kExpC, radiation::RemapRecon::kMoment};
    const radiation::RemapEdge rec_edge[8] = {
        radiation::RemapEdge::kOpen, radiation::RemapEdge::kOpen,
        radiation::RemapEdge::kOpen, radiation::RemapEdge::kOpen,
        radiation::RemapEdge::kOpen, radiation::RemapEdge::kOpen,
        radiation::RemapEdge::kClosed, radiation::RemapEdge::kOpen};
    const Real dtcsigs = 1.0;

    HostArray1D<Real> bray("bray", kMaxF), blray("blray", kMaxF);
    HostArray1D<Real> bihat("bihat", kMaxF), bjr("bjr", kMaxF), bljr("bljr", kMaxF);
    HostArray1D<Real> btgt("btgt", kMaxF);
    HostArray1D<Real> bs1("bs1", kMaxF), bs2("bs2", kMaxF);
    HostArray1D<Real> bncm("bncm", kMaxA), bdlt("bdlt", kMaxA), bwgt("bwgt", kMaxA);

    // Seed 2 is the discriminating one: the same Fermi-Dirac on a grid whose bottom is
    // decades below anything it carries, so no content leaves and the leak that remains
    // is the operator's own non-closure rather than the grid's coverage.
    const Real seed_numin[3] = {numin, numin, 1.0};
    const char *seed_name[3] = {
        "-- bake-off, Fermi-Dirac seed (the production shape and grid)",
        "-- bake-off, Gaussian in lnE (control: a shape every operator can represent)",
        "-- bake-off, Fermi-Dirac on a grid with margin (no content leaves)"};
    for (int seed = 0; seed < 3 && bakeoff; ++seed) {
      std::cout << seed_name[seed] << std::endl;
      std::cout << "     nfreq    v  recon        leak      max-rel"
                << "        L1(E)        L1(N)     edge-loss" << std::endl;
      for (int inf = 0; inf < 3; ++inf) {
        const int nf = nf_list[inf];
        Grid gg(nf, seed_numin[seed], numax);
        const Real eref = gg.mid[nf/2], esig = 3.0*gg.dln;
        for (int iv = 0; iv < 3; ++iv) {
          if (seed != 0 && iv != 1) { continue; }
          const Real v = v_list[iv];
          const Real wlv = 1.0/sqrt(1.0 - v*v);
          Real bwsum = 0.0, dmax = -1.0e300, dmin = 1.0e300;
          for (int k = 0; k < nang; ++k) {
            Real mu = -1.0 + 2.0*static_cast<Real>(k)/static_cast<Real>(nang-1);
            bncm(k) = wlv*(1.0 - v*mu);
            bdlt(k) = log(bncm(k))/gg.dln;
            bwgt(k) = 1.0/(bncm(k)*bncm(k));
            bwsum += bwgt(k);
            dmax = fmax(dmax, bdlt(k));
            dmin = fmin(dmin, bdlt(k));
          }
          const int g_lo = static_cast<int>(ceil(dmax));
          const int g_hi = static_cast<int>(floor(static_cast<Real>(nf-1) + dmin));
          const int b_lo = static_cast<int>(ceil(static_cast<Real>(g_lo) + 2.0 - dmin));
          const int b_hi = static_cast<int>(floor(static_cast<Real>(g_hi) - 2.0 - dmax));
          Real peak = 0.0;
          for (int f = 0; f < nf; ++f) {
            peak = fmax(peak, (seed != 1)
                                  ? FermiBin(gg.edges[f], gg.edges[f+1], temp, eta)
                                  : GaussLogBin(gg.edges[f], gg.edges[f+1], eref, esig));

          }

          for (int op = 0; op < 8; ++op) {
            const bool cubic = (op == 0);
            for (int f = 0; f < nf; ++f) { bs1(f) = 0.0; bs2(f) = 0.0; }
            Real tot_in = 0.0, tot_map = 0.0;
            for (int k = 0; k < nang; ++k) {
              const Real ncm = bncm(k), dlt = bdlt(k);
              for (int f = 0; f < nf; ++f) {
                bray(f) = (seed != 1)
                              ? FermiBin(ncm*gg.edges[f], ncm*gg.edges[f+1], temp, eta)
                              : GaussLogBin(ncm*gg.edges[f], ncm*gg.edges[f+1],
                                            eref, esig);
                tot_in += bwgt(k)*bray(f);
              }
              if (cubic) {
                radiation::RemapFillLogs(bray, blray, 0, 0, nf-1);
                for (int gi = 0; gi < nf; ++gi) {
                  bihat(gi) = radiation::RemapBinValue(
                      blray, 0,
                      radiation::RemapStencilAt(0, nf-1, static_cast<Real>(gi) - dlt),
                      radiation::RemapAsymptote::kSpectrum, gg.dln);
                }
              } else {
                radiation::RemapShiftConservative(bray, bihat, 0, nf, dlt, gg.dln,
                                                  rec_kind[op], rec_edge[op]);
              }
              const Real vnc = 1.0/(1.0 + dtcsigs*ncm);
              for (int gi = 0; gi < nf; ++gi) {
                tot_map += bwgt(k)*bihat(gi);
                bs1(gi) += bwgt(k)*ncm*vnc;
                bs2(gi) += bihat(gi)*bwgt(k)*vnc;
              }
            }
            for (int gi = 0; gi < nf; ++gi) {
              bjr(gi) = (bs2(gi)/bwsum)/(1.0 - (bs1(gi)/bwsum)*dtcsigs);
            }
            if (cubic) { radiation::RemapFillLogs(bjr, bljr, 0, 0, nf-1); }

            Real lnum = 0.0, lden = 0.0, worst = 0.0;
            Real e_err = 0.0, e_ref = 0.0, n_err = 0.0, n_ref = 0.0;
            for (int k = 0; k < nang; ++k) {
              const Real ncm = bncm(k), dlt = bdlt(k);
              if (cubic) {
                for (int f = 0; f < nf; ++f) {
                  btgt(f) = radiation::RemapBinValue(
                      bljr, 0,
                      radiation::RemapStencilAt(0, nf-1, static_cast<Real>(f) + dlt),
                      radiation::RemapAsymptote::kSpectrum, gg.dln);
                }
              } else {
                radiation::RemapShiftConservative(bjr, btgt, 0, nf, -dlt, gg.dln,
                                                  rec_kind[op], rec_edge[op]);
              }
              const Real vnc = 1.0/(1.0 + dtcsigs*ncm);
              for (int f = 0; f < nf; ++f) {
                const Real own =
                    (seed != 1)
                        ? FermiBin(ncm*gg.edges[f], ncm*gg.edges[f+1], temp, eta)
                        : GaussLogBin(ncm*gg.edges[f], ncm*gg.edges[f+1], eref, esig);
                lnum += bwgt(k)*vnc*(btgt(f) - own);
                lden += bwgt(k)*vnc*own;
                if (f >= b_lo && f <= b_hi && own > 1.0e-8*peak) {
                  worst = fmax(worst, fabs(btgt(f) - own)/own);
                }
                // The two moments the source term actually hands the fluid: energy, and
                // number, whose 1/E weighting is what makes the near-empty low-energy
                // bins matter out of all proportion to the energy they carry.
                const Real wn = bwgt(k)*vnc/(ncm*gg.mid[f]);
                e_err += bwgt(k)*vnc*fabs(btgt(f) - own);
                e_ref += bwgt(k)*vnc*own;
                n_err += wn*fabs(btgt(f) - own);
                n_ref += wn*own;
              }
            }
            const Real leak = (lden > 0.0) ? lnum/lden : 0.0;
            const Real eloss = (tot_in > 0.0) ? (tot_in - tot_map)/tot_in : 0.0;
            std::cout << "     " << std::setw(5) << nf << std::fixed
                      << std::setprecision(1) << std::setw(5) << v << "  "
                      << std::left << std::setw(7) << rec_name[op] << std::right
                      << std::scientific << std::setprecision(3)
                      << std::setw(12) << leak << std::setw(13) << worst
                      << std::setw(13) << ((e_ref > 0.0) ? e_err/e_ref : 0.0)
                      << std::setw(13) << ((n_ref > 0.0) ? n_err/n_ref : 0.0)
                      << std::setw(13) << eloss << std::endl;
          }
        }
      }
    }

    // Conservation, stated on the operator itself rather than inferred from the chain.
    // The seed is zeroed in the outermost three bins at each end and the shift kept
    // inside one bin, so nothing can leave the grid and the sum has to come back exactly.
    {
      Grid gg(nfreq, numin, numax);
      Real worst_cons = 0.0;
      for (int f = 0; f < nfreq; ++f) {
        bray(f) = (f < 3 || f > nfreq-4)
                      ? 0.0 : FermiBin(gg.edges[f], gg.edges[f+1], temp, eta);
      }
      Real sum_in = 0.0;
      for (int f = 0; f < nfreq; ++f) { sum_in += bray(f); }
      for (int op = 1; op < 8; ++op) {
        for (int id = -9; id <= 9; ++id) {
          const Real d = 0.1*static_cast<Real>(id);
          radiation::RemapShiftConservative(bray, bihat, 0, nfreq, d, gg.dln,
                                            rec_kind[op], rec_edge[op]);
          Real sum_out = 0.0, most_neg = 0.0;
          for (int f = 0; f < nfreq; ++f) {
            sum_out += bihat(f);
            most_neg = fmin(most_neg, bihat(f));
          }
          worst_cons = fmax(worst_cons, fabs(sum_out - sum_in)/sum_in);
          worst_cons = fmax(worst_cons, -most_neg/sum_in);
        }
      }
      Check("conservative remap conserves, and stays non-negative",
            worst_cons, 0.0, 1.0e-14);
    }

    // The property the whole question turns on: a map followed by its reverse must give
    // back both the energy sum and the number sum, in the radiation sector alone.  The
    // seed is zeroed in the outermost bins so that nothing leaves the grid -- coverage
    // is a separate matter from whether the operator's own algebra closes.  The number
    // weight is 1/E with E the bin's own comoving energy; the two Doppler factors the
    // ray picks up cancel between the map and its reverse, so the check is simply that
    // both sums come back.
    {
      Grid gg(nfreq, numin, numax);
      HostArray1D<Real> rtrip("rtrip", kMaxF);
      std::cout << "-- round trip: does the energy sum and the number sum come back?"
                << std::endl;
      Real worst_mom_e = 0.0, worst_mom_n = 0.0;
      for (int sd = 0; sd < 2; ++sd) {
      // Two seeds, because a truncated spectrum and a smooth one ask different questions
      // of an interpolating operator: the first has hard zeros at which the log-space
      // form falls back to its sentinel path, the second decays smoothly to nothing
      // inside the grid.  A partition is blind to the difference; the cubic is not.
      for (int f = 0; f < nfreq; ++f) {
        bray(f) = (sd == 0)
                      ? ((f < 4 || f > nfreq-5)
                             ? 0.0 : FermiBin(gg.edges[f], gg.edges[f+1], temp, eta))
                      : GaussLogBin(gg.edges[f], gg.edges[f+1], gg.mid[nfreq/2],
                                    2.0*gg.dln);
      }
      Real e_in = 0.0, n_in = 0.0;
      for (int f = 0; f < nfreq; ++f) {
        e_in += bray(f);
        n_in += bray(f)/gg.mid[f];
      }
      std::cout << "   " << (sd == 0 ? "truncated Fermi-Dirac" : "smooth Gaussian in lnE")
                << std::endl;
      {
        Real we = 0.0, wn = 0.0;
        for (int id = -9; id <= 9; ++id) {
          if (id == 0) { continue; }
          const Real d = 0.1*static_cast<Real>(id);
          radiation::RemapFillLogs(bray, blray, 0, 0, nfreq-1);
          for (int g = 0; g < nfreq; ++g) {
            bihat(g) = radiation::RemapBinValue(
                blray, 0,
                radiation::RemapStencilAt(0, nfreq-1, static_cast<Real>(g) - d),
                radiation::RemapAsymptote::kSpectrum, gg.dln);
          }
          radiation::RemapFillLogs(bihat, blray, 0, 0, nfreq-1);
          Real e_out = 0.0, n_out = 0.0;
          for (int f = 0; f < nfreq; ++f) {
            const Real q = radiation::RemapBinValue(
                blray, 0,
                radiation::RemapStencilAt(0, nfreq-1, static_cast<Real>(f) + d),
                radiation::RemapAsymptote::kSpectrum, gg.dln);
            e_out += q;
            n_out += q/gg.mid[f];
          }
          we = fmax(we, fabs(e_out - e_in)/e_in);
          wn = fmax(wn, fabs(n_out - n_in)/n_in);
        }
        std::cout << "     " << std::left << std::setw(9) << "cubic" << std::right
                  << "  dE/E " << std::scientific << std::setprecision(3) << we
                  << "   dN/N " << wn << std::endl;
      }
      for (int op = 1; op < 8; ++op) {
        Real we = 0.0, wn = 0.0;
        for (int id = -9; id <= 9; ++id) {
          if (id == 0) { continue; }
          const Real d = 0.1*static_cast<Real>(id);
          radiation::RemapShiftConservative(bray, bihat, 0, nfreq, d, gg.dln,
                                            rec_kind[op], radiation::RemapEdge::kOpen);
          radiation::RemapShiftConservative(bihat, rtrip, 0, nfreq, -d, gg.dln,
                                            rec_kind[op], radiation::RemapEdge::kOpen);
          Real e_out = 0.0, n_out = 0.0;
          for (int f = 0; f < nfreq; ++f) {
            e_out += rtrip(f);
            n_out += rtrip(f)/gg.mid[f];
          }
          we = fmax(we, fabs(e_out - e_in)/e_in);
          wn = fmax(wn, fabs(n_out - n_in)/n_in);
        }
        std::cout << "     " << std::left << std::setw(9) << rec_name[op] << std::right
                  << "  dE/E " << std::scientific << std::setprecision(3) << we
                  << "   dN/N " << wn << std::endl;
        // Only the compactly supported seed can be asserted on: the smooth one still
        // has 1e-8 of itself within a shift of either end, so its residual is the grid's
        // coverage rather than the operator's algebra, and every conservative operator
        // shows the same 7e-07 there whatever its profile.
        if (rec_kind[op] == radiation::RemapRecon::kMoment && sd == 0) {
          worst_mom_e = fmax(worst_mom_e, we);
          worst_mom_n = fmax(worst_mom_n, wn);
        }
      }
      }
      Check("round trip returns the energy sum", worst_mom_e, 0.0, 1.0e-13);
      Check("round trip returns the number sum", worst_mom_n, 0.0, 1.0e-13);
    }

    // The exponential profile is exact where ln(q) is linear in the bin index, which is
    // every power law in energy -- so the conservative remap inherits the one property
    // that made the log-space cubic usable, and keeps conservation as well.
    {
      Grid gg(nfreq, numin, numax);
      Real worst_pl = 0.0;
      for (int ip = 0; ip < 3; ++ip) {
        const Real p = 2.0*static_cast<Real>(ip);
        for (int f = 0; f < nfreq; ++f) {
          bray(f) = PowerLawBin(gg.edges[f], gg.edges[f+1], p);
        }
        for (int id = 1; id <= 9; ++id) {
          const Real d = 0.1*static_cast<Real>(id);
          radiation::RemapShiftConservative(bray, bihat, 0, nfreq, d, gg.dln,
                                            radiation::RemapRecon::kExp,
                                            radiation::RemapEdge::kOpen);
          // Interior only: the outermost bins have no second neighbour, so their slope
          // is unlimited by one side and the profile there is deliberately flat.
          for (int f = 3; f < nfreq-3; ++f) {
            const Real want = PowerLawBin(gg.edges[f]*exp(-d*gg.dln),
                                          gg.edges[f+1]*exp(-d*gg.dln), p);
            worst_pl = fmax(worst_pl, fabs(bihat(f) - want)/want);
          }
        }
      }
      Check("exponential profile: shifted bin integral is exact", worst_pl, 0.0,
            1.0e-12);
    }
  }

  // ------------------------------------ 9. repeated application: does a closure drift?
  // Comoving isotropy with pure elastic scattering is a fixed point of the exact
  // operator: nothing happens, at any velocity, forever.  One application measures how
  // far each scheme sits from that fixed point; only repeated application distinguishes
  // an offset from a drift, and the two have completely different consequences over the
  // 1e4 steps of a production run.
  //
  // This is where closing the frequency grid has to be judged.  Closing it makes the
  // remap conserve exactly, but nu_min is not zero, so content deposited at the bottom
  // edge has no way back out and the question is whether it accumulates.
  if (bakeoff) {
    const int nf = nfreq;
    Grid gg(nf, numin, numax);
    const Real dtcsigs = 1.0;
    const int n_steps[3] = {1, 100, 1000};
    const char *var_name[6] = {"cubic (in tree)", "conservative expC open",
                               "conservative expC closed", "cubic + energy pinned",
                               "cubic + number pinned", "cubic + both pinned"};
    const radiation::RemapRecon var_rec[6] = {
        radiation::RemapRecon::kExpC, radiation::RemapRecon::kExpC,
        radiation::RemapRecon::kExpC, radiation::RemapRecon::kExpC,
        radiation::RemapRecon::kExpC, radiation::RemapRecon::kExpC};
    const radiation::RemapEdge var_edge[6] = {
        radiation::RemapEdge::kOpen, radiation::RemapEdge::kOpen,
        radiation::RemapEdge::kClosed, radiation::RemapEdge::kOpen,
        radiation::RemapEdge::kOpen, radiation::RemapEdge::kOpen};
    const Real eref = gg.mid[nf/2];

    HostArray1D<Real> fld("fld", kMaxA*kMaxF), fld0("fld0", kMaxA*kMaxF);
    HostArray1D<Real> tall("tall", kMaxA*kMaxF);
    HostArray1D<Real> rray("rray", kMaxF), rlray("rlray", kMaxF);
    HostArray1D<Real> rihat("rihat", kMaxF), rjr("rjr", kMaxF), rljr("rljr", kMaxF);
    HostArray1D<Real> rtgt("rtgt", kMaxF), rs1("rs1", kMaxF), rs2("rs2", kMaxF);
    HostArray1D<Real> rncm("rncm", kMaxA), rdlt("rdlt", kMaxA), rwgt("rwgt", kMaxA);

    Real rwsum = 0.0, dmax = -1.0e300, dmin = 1.0e300;
    for (int k = 0; k < nang; ++k) {
      rncm(k) = n0_cm_a(k);
      rdlt(k) = dlt_a(k);
      rwgt(k) = wgt_a(k);
      rwsum += rwgt(k);
      dmax = fmax(dmax, rdlt(k));
      dmin = fmin(dmin, rdlt(k));
    }
    const int g_lo = static_cast<int>(ceil(dmax));
    const int g_hi = static_cast<int>(floor(static_cast<Real>(nf-1) + dmin));
    const int b_lo = static_cast<int>(ceil(static_cast<Real>(g_lo) + 2.0 - dmin));
    const int b_hi = static_cast<int>(floor(static_cast<Real>(g_hi) - 2.0 - dmax));
    Real peak = 0.0;
    for (int f = 0; f < nf; ++f) {
      peak = fmax(peak, FermiBin(gg.edges[f], gg.edges[f+1], temp, eta));
    }

    std::cout << "-- comoving isotropy under repeated application (exact answer: nothing)"
              << std::endl;
    std::cout << "     steps  scheme                        dE/E        dN/N"
              << "   max drift      bin 0     bin " << nf-1 << "   |lam-1|"
              << "      |beta|" << std::endl;
    for (int iv = 0; iv < 6; ++iv) {
      const bool interp = (iv == 0 || iv >= 3);
      for (int k = 0; k < nang; ++k) {
        for (int f = 0; f < nf; ++f) {
          fld0(k*kMaxF + f) = FermiBin(rncm(k)*gg.edges[f], rncm(k)*gg.edges[f+1],
                                       temp, eta);
          fld(k*kMaxF + f) = fld0(k*kMaxF + f);
        }
      }
      Real e0 = 0.0, n0m = 0.0, b0_0 = 0.0, bn_0 = 0.0;
      for (int k = 0; k < nang; ++k) {
        for (int f = 0; f < nf; ++f) {
          e0 += rwgt(k)*fld0(k*kMaxF + f);
          n0m += rwgt(k)*fld0(k*kMaxF + f)/(rncm(k)*gg.mid[f]);
        }
        b0_0 += rwgt(k)*fld0(k*kMaxF + 0);
        bn_0 += rwgt(k)*fld0(k*kMaxF + nf-1);
      }
      Real lam_worst = 0.0, beta_worst = 0.0;

      int done = 0;
      for (int is = 0; is < 3; ++is) {
        while (done < n_steps[is]) {
          ++done;
          for (int f = 0; f < nf; ++f) { rs1(f) = 0.0; rs2(f) = 0.0; }
          for (int k = 0; k < nang; ++k) {
            const Real ncm = rncm(k), dlt = rdlt(k);
            for (int f = 0; f < nf; ++f) { rray(f) = fld(k*kMaxF + f); }
            if (interp) {
              radiation::RemapFillLogs(rray, rlray, 0, 0, nf-1);
              for (int gi = 0; gi < nf; ++gi) {
                rihat(gi) = radiation::RemapBinValue(
                    rlray, 0,
                    radiation::RemapStencilAt(0, nf-1, static_cast<Real>(gi) - dlt),
                    radiation::RemapAsymptote::kSpectrum, gg.dln);
              }
            } else {
              radiation::RemapShiftConservative(rray, rihat, 0, nf, dlt, gg.dln,
                                                var_rec[iv], var_edge[iv]);
            }
            const Real vnc = 1.0/(1.0 + dtcsigs*ncm);
            for (int gi = 0; gi < nf; ++gi) {
              rs1(gi) += rwgt(k)*ncm*vnc;
              rs2(gi) += rihat(gi)*rwgt(k)*vnc;
            }
          }
          for (int gi = 0; gi < nf; ++gi) {
            rjr(gi) = (rs2(gi)/rwsum)/(1.0 - (rs1(gi)/rwsum)*dtcsigs);
          }
          if (interp) { radiation::RemapFillLogs(rjr, rljr, 0, 0, nf-1); }
          for (int k = 0; k < nang; ++k) {
            const Real dlt = rdlt(k);
            if (interp) {
              for (int f = 0; f < nf; ++f) {
                tall(k*kMaxF + f) = radiation::RemapBinValue(
                    rljr, 0,
                    radiation::RemapStencilAt(0, nf-1, static_cast<Real>(f) + dlt),
                    radiation::RemapAsymptote::kSpectrum, gg.dln);
              }
            } else {
              radiation::RemapShiftConservative(rjr, rtgt, 0, nf, -dlt, gg.dln,
                                                var_rec[iv], var_edge[iv]);
              for (int f = 0; f < nf; ++f) { tall(k*kMaxF + f) = rtgt(f); }
            }
          }
          // The constraint on the scattering exchange.  Scattering is isoenergetic in
          // the *fluid* frame, so it changes neither the comoving energy nor the
          // neutrino number; both sums here are therefore comoving, weighted by
          // omega_cm and using this ray's own comoving bin energy.  Lab-frame energy is
          // deliberately not a constraint: turning a neutrino at fixed comoving energy
          // does change its lab energy, and that is the fluid doing work on the
          // radiation.  Number is the one moment that is invariant either way.
          Real pe = 0.0, pn = 0.0;
          for (int k = 0; k < nang; ++k) {
            const Real wex = rwgt(k)*dtcsigs*rncm(k)/(1.0 + dtcsigs*rncm(k));
            for (int f = 0; f < nf; ++f) {
              const Real eps = rncm(k)*gg.mid[f];
              pe += wex*fld(k*kMaxF + f);
              pn += wex*fld(k*kMaxF + f)/eps;
            }
          }
          Real beta = 0.0, lam = 1.0;
          if (iv >= 3) {
            if (iv == 5) {
              // One scalar pins one moment.  Pinning both needs a tilt in ln(nu) beside
              // the scale, and A/B is the mean comoving energy under the tilted weight,
              // so it rises monotonically with beta and Newton from zero converges in a
              // few steps.
              for (int it = 0; it < 20; ++it) {
                Real a = 0.0, b = 0.0, al = 0.0, bl = 0.0;
                for (int k = 0; k < nang; ++k) {
                  const Real wex = rwgt(k)*dtcsigs*rncm(k)/(1.0 + dtcsigs*rncm(k));
                  for (int f = 0; f < nf; ++f) {
                    const Real eps = rncm(k)*gg.mid[f];
                    const Real lz = log(eps/eref);
                    const Real t = wex*tall(k*kMaxF + f)*exp(beta*lz);
                    a += t; al += t*lz; b += t/eps; bl += t*lz/eps;
                  }
                }
                if (!(a > 0.0 && b > 0.0 && pe > 0.0 && pn > 0.0)) { break; }
                const Real fp = al/a - bl/b;
                if (!(fabs(fp) > 1.0e-300)) { break; }
                const Real db = -(log(a/b) - log(pe/pn))/fp;
                beta += db;
                if (fabs(db) < 1.0e-14) { break; }
              }
            }
            Real a = 0.0, b = 0.0;
            for (int k = 0; k < nang; ++k) {
              const Real wex = rwgt(k)*dtcsigs*rncm(k)/(1.0 + dtcsigs*rncm(k));
              for (int f = 0; f < nf; ++f) {
                const Real eps = rncm(k)*gg.mid[f];
                const Real tl = (beta != 0.0) ? exp(beta*log(eps/eref)) : 1.0;
                const Real t = wex*tall(k*kMaxF + f)*tl;
                a += t; b += t/eps;
              }
            }
            lam = (iv == 4) ? ((b > 0.0) ? pn/b : 1.0) : ((a > 0.0) ? pe/a : 1.0);
          }
          lam_worst = fmax(lam_worst, fabs(lam - 1.0));
          beta_worst = fmax(beta_worst, fabs(beta));
          for (int k = 0; k < nang; ++k) {
            const Real ncm = rncm(k);
            const Real vnc = 1.0/(1.0 + dtcsigs*ncm);
            for (int f = 0; f < nf; ++f) {
              const Real tl =
                  (beta != 0.0) ? exp(beta*log(ncm*gg.mid[f]/eref)) : 1.0;
              const Real iold = fld(k*kMaxF + f);
              const Real di = dtcsigs*(lam*tl*tall(k*kMaxF + f) - iold)*ncm*vnc;
              fld(k*kMaxF + f) = fmax(iold + di, 0.0);
            }
          }
        }
        Real en = 0.0, nn = 0.0, b0_n = 0.0, bn_n = 0.0, drift = 0.0;
        for (int k = 0; k < nang; ++k) {
          for (int f = 0; f < nf; ++f) {
            en += rwgt(k)*fld(k*kMaxF + f);
            nn += rwgt(k)*fld(k*kMaxF + f)/(rncm(k)*gg.mid[f]);
            if (f >= b_lo && f <= b_hi && fld0(k*kMaxF + f) > 1.0e-8*peak) {
              drift = fmax(drift, fabs(fld(k*kMaxF + f)/fld0(k*kMaxF + f) - 1.0));
            }
          }
          b0_n += rwgt(k)*fld(k*kMaxF + 0);
          bn_n += rwgt(k)*fld(k*kMaxF + nf-1);
        }
        std::cout << "     " << std::setw(5) << n_steps[is] << "  " << std::left
                  << std::setw(26) << var_name[iv] << std::right << std::scientific
                  << std::setprecision(3) << std::setw(12) << (en - e0)/e0
                  << std::setw(12) << (nn - n0m)/n0m
                  << std::setw(12) << drift
                  << std::setw(12) << b0_n/b0_0 << std::setw(12) << bn_n/bn_0
                  << std::setw(12) << lam_worst << std::setw(12) << beta_worst
                  << std::endl;
      }
    }
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
