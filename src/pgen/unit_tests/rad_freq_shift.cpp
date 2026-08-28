//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_freq_shift.cpp
//  \brief Unit tests for the frame-shift helpers of the nurates multi-frequency
//         source term: ShiftedBinValue and ShiftedSpectrum.
//
//  These are the checks that no invariance test can make, and they are cheap: the
//  helpers run on host mirrors, so nothing here needs a mesh, an EOS or a time step.
//
//  What the scattering operator has to satisfy.  Isoenergetic scattering acts at fixed
//  comoving energy: it moves intensity between directions and cannot move energy between
//  comoving energy groups.  So for a comoving-isotropic field the operator must be a
//  no-op -- and, since the target is evaluated per ray, a no-op *ray by ray*, not merely
//  in the angular sum.  A fixed lab bin covers the comoving interval
//  gamma_a*[e_lo, e_hi], different on every ray, so this is a real constraint on the
//  shift machinery and on the un-shift that reconstructs the isotropic spectrum:
//
//      qbar        = < un-shift of I_a >_omega_cm         (built by the kernel's pass 1)
//      target_a    = qbar(ifr + delta_a)                  (ShiftedSpectrum)
//      requirement:  target_a == I_a(ifr)  for a comoving-isotropic field.
//
//  Section 4 below drives that whole round trip. For a spectrum that is a power law in
//  energy -- a power law in the bin index on a log grid, which the log-space cubic
//  reproduces exactly -- it closes to roundoff. For a Fermi-Dirac it does not, and the
//  residual is reported rather than asserted: it is the O(1e-4) un-shift/re-shift error
//  of section 2.4 of the algorithm note, and nothing is written back into i0_, so it
//  does not accumulate.
//
//  History.  The target used to be a shape factor S = qbar(ifr+delta)/qbar(ifr) times an
//  implicitly solved bin mean jr_bin ~ qbar(ifr).  That cancellation is exact in algebra
//  and not in floating point, and jr_bin was an angular average weighted by sigma_s and
//  vncsigma, which vary over the rays in a bin as soon as sigma_s depends on energy.  The
//  consequences, both measured: 47x (v = 0.1) to 137x (v = 0.3) the grey L1 error on
//  inputs/tests/rad_diffusion_spectral_nurates.athinput when S weighted the denominator,
//  and an O(v) loss going non-finite by t ~ 1.5 for sigma_s ~ E^2 at v = 0.3 whichever
//  way S was placed.  See sections 2 and 12 of
//  notes/multifreq-frame-consistent-review.md in the ChiralDynamo superproject.
//
//  Run via inputs/ut_rad_freq_shift.athinput; exits non-zero on failure, which is
//  what tst/test_suite/unit_tests/test_rad_freq_shift_cpu.py checks.

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
constexpr Real kSentinel = -1.0e30;

//! Frequency grid, mirroring SetFrequencyGrid then FreqBinEdgesMeV for the top bin.
struct Grid {
  int nfreq;
  Real edges[kMaxF+1];
  Real mid[kMaxF];
  Real dln;

  Grid(int nf, Real numin, Real numax) : nfreq(nf) {
    Real r = pow(numax/numin, 1.0/static_cast<Real>(nf-2));
    edges[0] = 0.0;
    for (int f = 1; f < nf; ++f) { edges[f] = numin*pow(r, f-1); }
    edges[nf] = edges[nf-1]*edges[nf-1]/edges[nf-2];
    for (int f = 0; f < nf; ++f) {
      mid[f] = (edges[f] > 0.0) ? sqrt(edges[f]*edges[f+1])
                                : 0.5*(edges[f] + edges[f+1]);
    }
    dln = log(numax/numin)/static_cast<Real>(nf-2);
  }
};

//! Fill a HostArray1D pair (q, log q) with the sentinel convention of the helpers.
void FillLogs(HostArray1D<Real> &q, HostArray1D<Real> &lq, int n) {
  for (int f = 0; f < n; ++f) {
    lq(f) = (q(f) > 0.0) ? log(q(f)) : kSentinel;
  }
}

//! Bin integral of A*E^p, which is a power law in the bin index on a log grid and
//! is therefore what ShiftedBinValue must reproduce exactly.
Real PowerLawBin(Real e_lo, Real e_hi, Real p) {
  return (pow(e_hi, p+1.0) - pow(e_lo, p+1.0))/(p + 1.0);
}

//! Bin integral of a Fermi-Dirac E^3/(1+exp(E/T - eta)), by trapezoid.
Real FermiBin(Real e_lo, Real e_hi, Real temp, Real eta) {
  const int nq = 512;
  Real acc = 0.0, h = (e_hi - e_lo)/static_cast<Real>(nq-1);
  for (int i = 0; i < nq; ++i) {
    Real e = e_lo + i*h;
    Real x = e/temp - eta;
    if (x > 700.0) { x = 700.0; }
    acc += (e*e*e/(1.0 + exp(x)))*h*((i == 0 || i == nq-1) ? 0.5 : 1.0);
  }
  return acc;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadFreqShift()

void ProblemGenerator::RadFreqShift(ParameterInput *pin, const bool restart) {
  if (restart) { return; }

  int nfreq = pin->GetOrAddInteger("problem", "nfreq", 24);
  Real numin = pin->GetOrAddReal("problem", "nu_min", 10.0);
  Real numax = pin->GetOrAddReal("problem", "nu_max", 2000.0);
  int nang = pin->GetOrAddInteger("problem", "nang", 92);
  Real vx = pin->GetOrAddReal("problem", "vx", 0.5);
  Real temp = pin->GetOrAddReal("problem", "spec_temp", 12.0);
  Real eta = pin->GetOrAddReal("problem", "spec_eta", 3.0);
  // The power-law round trip is exact algebra; only roundoff separates it from 1.
  Real tol_fixed = pin->GetOrAddReal("problem", "tol_fixed", 1.0e-11);
  // The Fermi-Dirac round trip is not exact.  This bounds it; it does not claim a rate.
  Real tol_fixed_fd = pin->GetOrAddReal("problem", "tol_fixed_fd", 3.0e-2);

  if (nfreq > kMaxF || nang > kMaxA || nfreq < 5) {
    std::cout << "### FATAL ERROR: rad_freq_shift needs 5 <= nfreq <= " << kMaxF
              << " and nang <= " << kMaxA << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Grid g(nfreq, numin, numax);
  HostArray1D<Real> q("q", nfreq), lq("lq", nfreq);
  HostArray1D<Real> emid("emid", nfreq);
  for (int f = 0; f < nfreq; ++f) { emid(f) = g.mid[f]; }

  // Ray set: uniform in mu, which is enough to exercise the algebra.  The Doppler
  // factor is all the helpers see of the angular grid.
  Real wl = 1.0/sqrt(1.0 - vx*vx);
  HostArray1D<Real> n0_cm_a("n0cm", nang), dlt_a("dlt", nang), mu_a("mu", nang);
  for (int k = 0; k < nang; ++k) {
    mu_a(k) = -1.0 + 2.0*static_cast<Real>(k)/static_cast<Real>(nang-1);
    n0_cm_a(k) = wl*(1.0 - vx*mu_a(k));
    dlt_a(k) = log(n0_cm_a(k))/g.dln;
  }
  HostArray1D<Real> dlt0("dlt0", 1);
  dlt0(0) = 0.0;

  std::cout << "=== rad_freq_shift: nfreq=" << nfreq << " nu=[" << numin << ", "
            << numax << "] dlnnu=" << g.dln << " nang=" << nang << " v=" << vx
            << " max|delta|=" << std::fixed << std::setprecision(3)
            << fmax(fabs(dlt_a(0)), fabs(dlt_a(nang-1))) << std::endl;

  // ---------------------------------------------------- 1. ShiftedBinValue exactness
  // A power law in energy is a power law in the bin index, so the log-space cubic must
  // reproduce it to roundoff.  This is the property section 2.3 relies on for the
  // "no separate Jacobian" claim, and what makes section 4 below exact.
  std::cout << "-- ShiftedBinValue: exactness on a power law" << std::endl;
  for (Real p : {0.0, 2.0, 3.0}) {
    for (int f = 0; f < nfreq; ++f) { q(f) = PowerLawBin(g.edges[f], g.edges[f+1], p); }
    FillLogs(q, lq, nfreq);
    Real ncm = 1.3;
    Real dlt = log(ncm)/g.dln;
    for (int ifr : {2, nfreq/2, nfreq-2}) {
      Real got = radiation::ShiftedBinValue(q, lq, 0, 1, nfreq-1,
                                            static_cast<Real>(ifr) + dlt);
      Real want = PowerLawBin(ncm*g.edges[ifr], ncm*g.edges[ifr+1], p);
      Check("E^" + std::to_string(static_cast<int>(p)) + ", bin "
            + std::to_string(ifr), got, want, 1.0e-12);
    }
  }

  // A zero bin trips the sentinel, and the linear-space fallback must stay finite and
  // inside the tabulated range rather than returning exp(-1e30) garbage or a NaN.
  std::cout << "-- ShiftedBinValue: sentinel path on an empty bin" << std::endl;
  for (int f = 0; f < nfreq; ++f) { q(f) = PowerLawBin(g.edges[f], g.edges[f+1], 2.0); }
  q(nfreq/2) = 0.0;
  FillLogs(q, lq, nfreq);
  {
    Real got = radiation::ShiftedBinValue(q, lq, 0, 1, nfreq-1,
                                          static_cast<Real>(nfreq/2) + 0.4);
    Real hi = fmax(fmax(q(nfreq/2-1), q(nfreq/2)), fmax(q(nfreq/2+1), q(nfreq/2+2)));
    ++nrun;
    bool ok = std::isfinite(got) && got >= 0.0 && got <= hi;
    if (!ok) { ++nfail; }
    std::cout << (ok ? "  [ ok ] " : "  [FAIL] ")
              << "finite, non-negative and bounded across a zero bin: " << got
              << " (bound " << hi << ")" << std::endl;
  }

  // -------------------------------------------------- 2. ShiftedSpectrum, at rest etc.
  // Exactness at rest is what makes a static fluid inert: the tabulated value has to
  // come back bit for bit, not as exp(log(q)).
  std::cout << "-- ShiftedSpectrum: identity on the degenerate paths" << std::endl;
  for (int f = 0; f < nfreq; ++f) { q(f) = PowerLawBin(g.edges[f], g.edges[f+1], 3.0); }
  FillLogs(q, lq, nfreq);
  for (int ifr : {2, nfreq/2, nfreq-2}) {
    CheckExact("at rest, bin " + std::to_string(ifr),
               radiation::ShiftedSpectrum(q, lq, 0, ifr, 0, nfreq, true, 1,
                                          dlt0, 1.0, emid, g.dln, 1, nfreq-1),
               q(ifr));
  }
  CheckExact("bin 0 (outside the log family)",
             radiation::ShiftedSpectrum(q, lq, 0, 0, 0, nfreq, true, 1, dlt_a,
                                        n0_cm_a(0), emid, g.dln, 1, nfreq-1), q(0));
  CheckExact("shift disabled",
             radiation::ShiftedSpectrum(q, lq, 0, 3, 0, nfreq, false, 1, dlt_a,
                                        n0_cm_a(0), emid, g.dln, 1, nfreq-1), q(3));

  // ------------------------------------------ 3. ShiftedSpectrum is the bin integral
  // The target for a ray is the spectrum integrated over *that ray's* comoving bin.
  std::cout << "-- ShiftedSpectrum: equals the ray's own comoving bin integral"
            << std::endl;
  std::cout << "   (rays whose comoving bin runs off the grid are clamped, and skipped"
            << " here)" << std::endl;
  for (Real p : {2.0, 3.0}) {
    for (int f = 0; f < nfreq; ++f) { q(f) = PowerLawBin(g.edges[f], g.edges[f+1], p); }
    FillLogs(q, lq, nfreq);
    Real worst = 0.0;
    int wbin = -1, wang = -1;
    for (int ifr = 1; ifr < nfreq; ++ifr) {
      for (int k = 0; k < nang; ++k) {
        Real xp = static_cast<Real>(ifr) + dlt_a(k);
        if (xp < 1.0 || xp > static_cast<Real>(nfreq-1)) { continue; }
        Real got = radiation::ShiftedSpectrum(q, lq, 0, ifr, k, nfreq, true, 1,
                                              dlt_a, n0_cm_a(k), emid, g.dln,
                                              1, nfreq-1);
        Real want = PowerLawBin(n0_cm_a(k)*g.edges[ifr],
                                n0_cm_a(k)*g.edges[ifr+1], p);
        Real err = fabs(got - want)/want;
        if (err > worst) { worst = err; wbin = ifr; wang = k; }
      }
    }
    std::cout << "  worst at bin " << wbin << ", ray " << wang << " (n0_cm = "
              << std::fixed << std::setprecision(4)
              << n0_cm_a((wang > 0) ? wang : 0) << ")" << std::endl;
    Check("E^" + std::to_string(static_cast<int>(p)) + ", all bins and rays",
          worst, 0.0, 1.0e-11);
  }

  // ------------------------------------------------- 4. The fixed point, round trip
  // The whole scheme in miniature.  Seed a genuinely comoving-isotropic field -- ray a
  // gets the spectrum integrated over its own comoving bin -- run the kernel's un-shift
  // and angular average to rebuild qbar, then ask ShiftedSpectrum for each ray's target
  // and compare with what that ray is actually carrying.  Elastic scattering must be a
  // no-op here, ray by ray.
  //
  // Bin 0 is excluded: it spans [0, nu_min], sits outside the log family and is never
  // shifted, so it is not a fixed point and never was.  That is a known limitation of
  // the lab-bin representation, not of these helpers.
  std::cout << "-- Round trip: comoving isotropy is a fixed point, ray by ray"
            << std::endl;
  Real dmin = 0.0, dmax_ = 0.0;
  for (int k = 0; k < nang; ++k) {
    dmin = fmin(dmin, dlt_a(k));
    dmax_ = fmax(dmax_, dlt_a(k));
  }
  for (int which = 0; which < 2; ++which) {
    const bool power_law = (which == 0);
    // The exact comoving spectrum, as a function of a real-valued bin position.
    auto qtrue = [&](Real x) -> Real {
      Real lo = numin*exp((x - 1.0)*g.dln);
      Real hi = numin*exp(x*g.dln);
      return power_law ? PowerLawBin(lo, hi, 3.0) : FermiBin(lo, hi, temp, eta);
    };

    // Ray a's bin content for a comoving-isotropic field.
    HostArray1D<Real> ii("ii", nang*nfreq);
    for (int k = 0; k < nang; ++k) {
      for (int f = 0; f < nfreq; ++f) {
        Real lo = n0_cm_a(k)*g.edges[f], hi = n0_cm_a(k)*g.edges[f+1];
        ii(k*nfreq + f) = power_law ? PowerLawBin(lo, hi, 3.0)
                                    : FermiBin(lo, hi, temp, eta);
      }
    }

    // The kernel's per-bin shift gate: [u_lo, u_hi] is where every ray's un-shifted
    // position stays on the grid, so where qbar means the comoving spectrum rather than
    // the plain lab-bin mean.  Outside it the bin falls back to the lab-bin treatment,
    // as bin 0 always does, and every qbar lookup is restricted to the band.
    int u_lo = static_cast<int>(ceil(1.0 + dmax_));
    int u_hi = static_cast<int>(floor(static_cast<Real>(nfreq-1) + dmin));
    // Inside [s_lo, s_hi] the target lookup also stays inside [u_lo, u_hi], so the
    // fixed point is exact there.  Between s and u the lookup is clamped, and outside u
    // the bin falls back to lab bins; both are reported, not asserted.
    int s_lo = static_cast<int>(ceil(static_cast<Real>(u_lo) - dmin));
    int s_hi = static_cast<int>(floor(static_cast<Real>(u_hi) - dmax_));
    HostArray1D<int> shift_on("shift_on", nfreq);
    for (int f = 0; f < nfreq; ++f) {
      shift_on(f) = (f >= u_lo && f <= u_hi) ? 1 : 0;
    }

    // The kernel's first pass: un-shift each ray onto the fixed grid and average with
    // the comoving solid-angle measure.
    HostArray1D<Real> qbar("qbar", nfreq), lqbar("lqbar", nfreq);
    HostArray1D<Real> rtmp("rtmp", nfreq), lrtmp("lrtmp", nfreq);
    for (int f = 0; f < nfreq; ++f) { qbar(f) = 0.0; }
    Real wsum = 0.0;
    for (int k = 0; k < nang; ++k) {
      Real om_cm = 1.0/SQR(n0_cm_a(k));
      wsum += om_cm;
      for (int f = 0; f < nfreq; ++f) { rtmp(f) = ii(k*nfreq + f); }
      FillLogs(rtmp, lrtmp, nfreq);
      for (int f = 0; f < nfreq; ++f) {
        Real val = rtmp(f);
        if (shift_on(f) >= 1) {
          Real xpos = static_cast<Real>(f) - dlt_a(k);
          if (fabs(xpos - static_cast<Real>(f)) > 1.0e-12) {
            val = radiation::ShiftedBinValue(rtmp, lrtmp, 0, 1, nfreq-1, xpos);
          }
        }
        qbar(f) += om_cm*val;
      }
    }
    for (int f = 0; f < nfreq; ++f) { qbar(f) /= wsum; }
    FillLogs(qbar, lqbar, nfreq);

    // Split interior from edge.  Within ceil(|delta|max) bins of either end the shift
    // reaches comoving energies the grid does not tabulate, so the fixed point cannot
    // hold there and no amount of care in the interpolation will make it: the
    // information is absent.  The interior is asserted on; the edge is reported so the
    // limitation stays visible.  Bin 0 is always edge -- it spans [0, nu_min], sits
    // outside the log family and is never shifted.
    int f_lo = s_lo, f_hi = s_hi;
    Real worst_q = 0.0, worst_t = 0.0, worst_edge = 0.0, worst_edge_abs = 0.0;
    Real qpeak = 0.0;
    for (int f = 0; f < nfreq; ++f) { qpeak = fmax(qpeak, qbar(f)); }
    int wbin = -1, wqbin = -1;
    for (int f = 1; f < nfreq; ++f) {
      bool interior = (f >= f_lo && f <= f_hi);
      Real ex = qtrue(static_cast<Real>(f));
      if (interior && ex > 0.0) {
        Real e = fabs(qbar(f) - ex)/ex;
        if (e > worst_q) { worst_q = e; wqbin = f; }
      }
      for (int k = 0; k < nang; ++k) {
        Real tgt = radiation::ShiftedSpectrum(qbar, lqbar, 0, f, k, nfreq,
                                              shift_on(f) >= 1, 1,
                                              dlt_a, n0_cm_a(k), emid, g.dln,
                                              u_lo, u_hi);
        Real have = ii(k*nfreq + f);
        if (!(have > 0.0)) { continue; }
        Real err = fabs(tgt - have)/have;
        if (interior) {
          if (err > worst_t) { worst_t = err; wbin = f; }
        } else {
          worst_edge = fmax(worst_edge, err);
          worst_edge_abs = fmax(worst_edge_abs, fabs(tgt - have));
        }
      }
    }
    const std::string tag = power_law ? "E^3" : "Fermi-Dirac";
    std::cout << "  " << tag << ": |delta|max = " << std::fixed
              << std::setprecision(2) << fmax(fabs(dmin), fabs(dmax_))
              << " bins; shift on " << u_lo << ".." << u_hi << ", exact on "
              << s_lo << ".." << s_hi << ", clamped between, lab bins outside"
              << std::endl;
    std::cout << "  " << tag << ": qbar recovered to " << std::scientific
              << std::setprecision(4) << worst_q << " (bin " << wqbin
              << "), worst residual in the exact band " << worst_t << " (bin " << wbin
              << "), outside " << worst_edge << " relative but "
              << worst_edge_abs/fmax(qpeak, 1.0e-300) << " of the peak" << std::endl;
    Check(tag + ": target == ray content, exact band", worst_t, 0.0,
          power_law ? tol_fixed : tol_fixed_fd);
  }

  std::cout << "=== rad_freq_shift: " << (nrun - nfail) << "/" << nrun
            << " checks passed" << std::endl;
  if (nfail > 0) {
    std::cout << "### FATAL ERROR: " << nfail << " rad_freq_shift check(s) failed"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::exit(EXIT_SUCCESS);
}

#else   // !ENABLE_NURATES

void ProblemGenerator::RadFreqShift(ParameterInput *pin, const bool restart) {
  if (restart) { return; }
  std::cout << "=== rad_freq_shift: skipped, built without bns_nurates" << std::endl;
  std::exit(EXIT_SUCCESS);
}

#endif  // ENABLE_NURATES
