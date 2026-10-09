#!/usr/bin/env python3
"""Figures and a summary table for scripts/nu_validation/run_suite.sh.

Usage:  nu_validation_plots.py SUITE_DIR [--out FIGDIR]

Compares the multifrequency Boltzmann solver (mf) with grey M1 (m1) and analytic
solutions.  Writes FIGDIR/fig_*.png and FIGDIR/summary.txt (FIGDIR defaults to SUITE_DIR).
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_nu_singlezone as sz
import radnu_read as rr
from rad_diffusion_compare import analytic as diffusion_analytic
from rad_diffusion_compare import read_tab

plt.rcParams.update({"font.size": 10, "legend.fontsize": 8, "axes.titlesize": 10})
C_MF, C_M1, C_AN = "C0", "C3", "k"
ROWS = []  # (test, quantity, value, tolerance)


def check(test, quantity, value, tol=None):
    """tol=None marks a reference number (reported, not judged)."""
    ROWS.append((test, quantity, value, tol))


def sz_series(run):
    """Fluid-frame single-zone history: time, Ye, T [MeV], W."""
    s = sz.read_tab_series(f"{run}/tab/*.mhd_w.*.tab")
    return s["time"], s["s_00"], s["temperature"], sz.lorentz_factor(
        s["velx"], s["vely"], s["velz"])


def trapped_eq(run):
    """Analytic trapped beta-equilibrium (T, Ye) printed by the pgen."""
    txt = open(f"{run}/log.txt").read()
    m = re.search(r"T_eq = ([0-9.eE+-]+) MeV, Ye_eq = ([0-9.eE+-]+)", txt)
    return float(m.group(1)), float(m.group(2))


# ---------------------------------------------------------------------------------------
def equilibration(d, out):
    states = [("A", r"$10^{13}$ g/cc, 50 MeV, $Y_e$ 0.1"),
              ("D13", r"$10^{13}$ g/cc, 10 MeV, $Y_e$ 0.3"),
              ("D12", r"$10^{12}$ g/cc, 10 MeV, $Y_e$ 0.3"),
              ("N11", r"$10^{11}$ g/cc, 5 MeV, $Y_e$ 0.3")]
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.5), sharex="col")
    for c, (name, label) in enumerate(states):
        teq, yeq = trapped_eq(f"{d}/eq_flat/{name}/mf")
        for solver, col, ls in (("mf", C_MF, "-"), ("m1", C_M1, "--")):
            t, ye, temp, _ = sz_series(f"{d}/eq_flat/{name}/{solver}")
            ax[0, c].plot(t[1:], ye[1:], ls, color=col, label=solver)
            ax[1, c].plot(t[1:], temp[1:], ls, color=col)
            check(f"eq {name}", f"{solver} Ye/Ye_eq - 1", ye[-1]/yeq - 1, 5e-3)
            check(f"eq {name}", f"{solver} T/T_eq - 1", temp[-1]/teq - 1, 5e-3)
        ax[0, c].axhline(yeq, color=C_AN, ls=":", label="analytic")
        ax[1, c].axhline(teq, color=C_AN, ls=":")
        ax[0, c].set_title(label)
        ax[1, c].set_xscale("log")
        ax[1, c].set_xlabel("t [code]")
    ax[0, 0].set_ylabel("$Y_e$")
    ax[1, 0].set_ylabel("T [MeV]")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_equilibration.png", dpi=150)
    plt.close(fig)

    # final spectra of the densest 10 MeV state against a bin-averaged Fermi-Dirac fit
    run = f"{d}/eq_flat/D13/mf"
    r = sz.read_radnu_series(run, sz.find_basename(run))
    edges = rr.freq_bin_edges_mev(r["nu_min"], r["nu_max"], r["nfreq"])
    ec = np.sqrt(edges[:-1]*edges[1:])
    temp = sz_series(run)[2][-1]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s, nm in enumerate(sz.SPECIES_NAMES):
        jg = r["j00"][-1, s]
        a, tf, eta = sz.fit_fermi_dirac(edges, jg, temp)
        model = a*np.array([sz._bin_integral_unit(edges[g], edges[g+1], tf, eta)
                            for g in range(len(ec))])
        ax.loglog(ec, jg, "o", color=f"C{s}", ms=4, label=rf"{nm}: $T_{{fit}}/T$ = {tf/temp:.4f}")
        ax.loglog(ec, model, "-", color=f"C{s}", lw=0.8)
        check("eq D13", f"mf spectrum {nm} T_fit/T - 1", tf/temp - 1, 1e-2)
    ax.set_ylim(1e-8*r["j00"][-1].max(), 3*r["j00"][-1].max())
    ax.set_xlabel("E [MeV]")
    ax.set_ylabel("comoving group energy")
    ax.set_title(r"$10^{13}$ g/cc, final spectra (markers) vs Fermi-Dirac fit (lines)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_spectra.png", dpi=150)
    plt.close(fig)

    # equilibrium Ye error vs frequency resolution
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for name, col in (("A", "C0"), ("D13", "C1")):
        yeq = trapped_eq(f"{d}/eq_flat/{name}/mf")[1]
        nfs = [12, 24, 48]
        runs = [f"{d}/eq_nfreq/{name}/nf12", f"{d}/eq_flat/{name}/mf", f"{d}/eq_nfreq/{name}/nf48"]
        err = np.array([abs(sz_series(r)[1][-1] - yeq) for r in runs])
        ax.loglog(nfs, err, "o-", color=col, label=states[[s[0] for s in states].index(name)][1])
        check(f"eq {name}", "mf |dYe| convergence order - 2 (nf 24->48)",
              math.log2(err[1]/err[2]) - 2, 1.0)
    ax.loglog([12, 48], [3e-3, 3e-3/16], "k:", label=r"$\propto$ nfreq$^{-2}$")
    ax.set_xticks(nfs, [str(n) for n in nfs])
    ax.minorticks_off()
    ax.set_xlabel("nfreq")
    ax.set_ylabel(r"$|Y_e - Y_e^{eq}|$ at the end")
    ax.set_title("Equilibrium error vs frequency resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_eq_convergence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def boost(d, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ref = {}
    for tag, base, ls in (("rest", f"{d}/eq_flat/A", "-"), ("v = 0.3", f"{d}/eq_boost", "--")):
        for solver, col in (("mf", C_MF), ("m1", C_M1)):
            t, ye, temp, w = sz_series(f"{base}/{solver}")
            tau = t/w
            ax[0].plot(tau[1:], ye[1:], ls, color=col, label=f"{solver}, {tag}")
            ax[1].plot(tau[1:], temp[1:], ls, color=col)
            if tag == "rest":
                ref[solver] = (tau, ye, temp)
            else:
                tr, yr, tmr = ref[solver]
                m = tau <= tr[-1]
                check("boost", f"{solver} max|Ye(tau)/Ye_rest - 1|",
                      np.max(np.abs(ye[m]/np.interp(tau[m], tr, yr) - 1)), 1e-2)
                check("boost", f"{solver} max|T(tau)/T_rest - 1|",
                      np.max(np.abs(temp[m]/np.interp(tau[m], tr, tmr) - 1)), 1e-2)
    for a, lab in zip(ax, ("$Y_e$", "T [MeV]")):
        a.set_xscale("log")
        a.set_xlim(5e-3, 3.5)
        a.set_xlabel(r"fluid proper time $\tau = t/W$")
        a.set_ylabel(lab)
    ax[0].legend()
    fig.suptitle(r"Boosted single zone ($10^{13}$ g/cc, 50 MeV): fluid-frame history vs $\tau$")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_boost.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def curved(d, out):
    radii = [20, 10, 6, 4]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    gain = {}
    for solver, col, mk in (("mf", C_MF, "o"), ("m1", C_M1, "s")):
        tf, yf, _, _ = sz_series(f"{d}/eq_curved/flat/{solver}")
        g = []
        for r in radii:
            _, ye, _, _ = sz_series(f"{d}/eq_curved/r{r}/{solver}")
            g.append((ye[-1] - 0.1)/(yf[-1] - 0.1) - 1)
        gain[solver] = np.array(g)
        ax[0].plot(radii, 100*gain[solver], mk + "-", color=col, label=solver)
        for r, ls in ((4, "-"), (10, "--")):
            t, ye, _, _ = sz_series(f"{d}/eq_curved/r{r}/{solver}")
            ax[1].plot(t[1:], 100*(ye[1:]/np.interp(t[1:], tf, yf) - 1), ls, color=col,
                       label=f"{solver}, r = {r}M")
    alpha = np.array([1/math.sqrt(1 + 2/r) for r in radii])
    ax[0].plot(radii, 100*0.5*(1/alpha - 1)**2, "k:", label=r"$\frac{1}{2}(1/\alpha-1)^2$")
    for i, r in enumerate(radii):
        check("curved", f"r={r}M lepton gain mf - m1 [rel. to flat]",
              gain["mf"][i] - gain["m1"][i], 5e-3)
    ax[0].set_xlabel("r / M")
    ax[0].set_ylabel(r"$\Delta Y_e(r)/\Delta Y_e^{flat} - 1$  [%]")
    ax[0].set_title("Lepton gain at t = 1 relative to flat space")
    ax[0].legend()
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$Y_e(r)/Y_e^{flat} - 1$  [%]")
    ax[1].set_title("History relative to flat space")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_curved.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def freestream(d, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for f in sorted(glob.glob(f"{d}/freestream/mf/tab/*.rad_coord.*.tab"))[1:]:
        t, x, c = read_tab(f)
        xa = np.linspace(0, 1, 400)
        ax.plot(xa, np.clip(1 - xa/t, 0, None), color=C_AN, lw=0.8)
        e = c["r00"]
        e0 = e[0]/(1 - x[0]/t)  # wall value E(0+), from the first cell
        ax.plot(x, e/e0, "-", color=C_MF, label=f"mf" if t < 0.3 else None)
        check("freestream", f"mf L1(E/E0 - analytic), t={t:.2f}",
              np.mean(np.abs(e/e0 - np.clip(1 - x/t, 0, None))), 2e-2)
    ax.plot([], [], color=C_AN, lw=0.8, label=r"analytic $1 - x/t$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$E/E(0^+)$")
    ax.set_title("Hot wall into vacuum, t = 0.25, 0.5, 0.75")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_freestream.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def diffusion(d, out, v=0.3, kappa_s=100.0, nu=4.0):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    mf = sorted(glob.glob(f"{d}/diffusion/mf/tab/*.rad_coord_fluid.*.tab"))
    t0, x0, c0 = read_tab(mf[0])
    j0 = diffusion_analytic(x0, t0, v, kappa_s, nu, 0.0)[0]
    norm = c0["r00_ff"].sum()/j0.sum()
    xa = np.linspace(x0[0], x0[-1], 600)
    for f in mf:
        t, x, c = read_tab(f)
        if abs(t - round(2*t)/2) > 1e-6:
            continue
        ja = diffusion_analytic(x, t, v, kappa_s, nu, 0.0)[0]
        ax[0].plot(xa, diffusion_analytic(xa, t, v, kappa_s, nu, 0.0)[0], color=C_AN, lw=0.8)
        ax[0].plot(x, c["r00_ff"]/norm, "-", color=C_MF, label="mf" if t == 0 else None)
        if t > 0:
            check("diffusion", f"mf L1(J - analytic)/max, t={t:.1f}",
                  np.mean(np.abs(c["r00_ff"]/norm - ja))/ja.max(), 2e-2)
    for f in sorted(glob.glob(f"{d}/diffusion/m1/tab/*.rad_m1_J.*.tab")):
        t, x, c = read_tab(f)
        ja = diffusion_analytic(x, t, v, kappa_s, nu, 0.0)[0]
        ax[0].plot(x, c["J:0"], "--", color=C_M1, label="m1" if t == 0 else None)
        if t > 0:
            check("diffusion", f"m1 L1(J - analytic)/max, t={t:.1f}",
                  np.mean(np.abs(c["J:0"] - ja))/ja.max(), 2e-2)
    ax[0].plot([], [], color=C_AN, lw=0.8, label="analytic")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("comoving J (normalised)")
    ax[0].set_title(rf"Diffusion in a medium moving at v = {v}, $\kappa_s$ = {kappa_s:g}")
    ax[0].legend()

    snaps = [rr.read_radnu(f)
             for f in sorted(glob.glob(f"{d}/diffusion/mf/bin/*.radnu_fluid.*.bin"))]
    jg = np.array([s["data"][0, :, 0, 0, 0, :] for s in snaps])
    frac = jg.sum(axis=2)/jg.sum(axis=(1, 2))[:, None]
    edges = rr.freq_bin_edges_mev(snaps[0]["nu_min"], snaps[0]["nu_max"], snaps[0]["nfreq"])
    ec = np.sqrt(edges[:-1]*edges[1:])
    for it in (0, len(snaps) - 1):
        ax[1].loglog(ec, frac[it], "o-" if it == 0 else "x--", ms=4,
                     label=f"t = {snaps[it]['time']:.2f}")
    main = frac[0] > 1e-3*frac[0].max()
    check("diffusion", "mf max group-share drift (groups >1e-3 of peak)",
          np.max(np.abs(frac[-1][main]/frac[0][main] - 1)), 1e-2)
    ax[1].set_xlabel("E [MeV]")
    ax[1].set_ylabel("share of comoving energy")
    ax[1].set_title("Group shares (elastic scattering: must not change)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_diffusion.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def lepton(d, out):
    """Does the Ye source balance the neutrino number the field actually lost?

    Uniform periodic zone, so there is no transport and the budget is exact:
        Delta(rho W Ye)/m_b = -Delta(N_nue,E - N_anue,E)
    with both sides read from the outputs.  The multifrequency number is the lab
    (normal-frame) one, summed as sum_g E_g/E_g,mid; M1 evolves it directly as N.
    """
    txt = open(f"{d}/lepton/v0.0/mf/log.txt").read()
    mb = float(re.search(r"mb_code = ([0-9.eE+-]+)", txt).group(1))
    cc = float(re.search(r"code_edens_to_eos = ([0-9.eE+-]+)", txt).group(1))

    def mf_number(run):
        t, n = [], []
        for f in sorted(glob.glob(f"{run}/bin/*.radnu_coord.*.bin")):
            s = rr.read_radnu(f)
            i00 = list(s["var_names"]).index("rnu00")
            edges = rr.freq_bin_edges_mev(s["nu_min"], s["nu_max"], s["nfreq"],
                                          s["freq_scale"])
            e_g = s["data"][:, :, i00, 0, 0, sz.ICELL]/np.sqrt(edges[:-1]*edges[1:])
            t.append(s["time"])
            n.append((e_g[0].sum() - e_g[1].sum())*cc)   # -> fm^-3, as M1's N
        o = np.argsort(t)
        return np.array(t)[o], np.array(n)[o]

    def m1_number(run):
        s = sz.read_tab_series(f"{run}/tab/*.rad_m1_N.*.tab")
        cols = sorted(k for k in s if k.startswith("N"))
        return s["time"], s[cols[0]] - s[cols[1]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for v, mk in (("0.0", "o"), ("0.3", "s")):
        for solver, col, num in (("mf", C_MF, mf_number), ("m1", C_M1, m1_number)):
            run = f"{d}/lepton/v{v}/{solver}"
            m = sz.read_tab_series(f"{run}/tab/*.mhd_w.*.tab")
            w = sz.lorentz_factor(m["velx"], m["vely"], m["velz"])
            lep = m["dens"]*w*m["s_00"]/mb                 # lab electron number
            tn, n = num(run)
            gain = np.interp(tn, m["time"], lep) - lep[0]  # matter gained
            loss = -(n - n[0])                             # neutrinos lost
            good = np.abs(loss) > 1e-12*np.abs(loss).max()
            err = gain[good]/loss[good] - 1.0
            ax[0].plot(tn[good], gain[good], mk + "-", color=col, ms=3,
                       label=f"{solver}, v = {v}: matter")
            ax[0].plot(tn[good], loss[good], mk + ":", color=col, ms=3,
                       label=f"{solver}, v = {v}: neutrinos")
            ax[1].semilogy(tn[good], np.abs(err), mk + "-", color=col, ms=3,
                           label=f"{solver}, v = {v}")
            check("lepton", f"{solver} v={v} max|matter gain/neutrino loss - 1|",
                  np.max(np.abs(err)), 1e-3)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"lepton number exchanged [fm$^{-3}$]")
    ax[0].set_title("Matter gain (solid) vs neutrino loss (dotted)")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("|budget error|")
    ax[1].set_title("Lepton-number budget of the $Y_e$ source")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_lepton.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def redshift(d, out):
    fname = f"{d}/redshift/mf/redshift_ray.txt"
    lines = open(fname).readlines()
    lo = np.array([float(v) for v in lines[1].split()[2:]])
    hi = np.array([float(v) for v in lines[2].split()[2:]])
    a = np.loadtxt(fname)
    a = a[np.argsort(a[:, 0])]
    x, ig = a[:, 0], np.clip(a[:, 1:], 0, None)
    # The photon grid's bin 0 is [0, nu_min]: centre it half a log width below its top edge.
    lnc = np.where(lo > 0, 0.5*(np.log(np.where(lo > 0, lo, 1)) + np.log(hi)),
                   np.log(hi) - 0.5*np.log(hi[1]/lo[1]))
    mean_ln = (ig*lnc).sum(1)/ig.sum(1)

    def ana(r):
        f = 2/r
        return np.log(np.sqrt(1 + f)/(1 - f))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(x, mean_ln - mean_ln[0], "o", color=C_MF, ms=3, label="mf")
    ax[0].plot(x, ana(x) - ana(x[0]), "-", color=C_AN,
               label=r"analytic $\ln[\sqrt{1+2M/r}/(1-2M/r)]$")
    ax[0].set_xlabel("r / M")
    ax[0].set_ylabel(r"$\langle\ln E\rangle - \langle\ln E\rangle_{r_{in}}$")
    ax[0].set_title("Energy along the most radial outgoing ray")
    ax[0].legend()
    ec = np.exp(lnc)
    for i, lab in ((0, "inner"), (-1, "outer")):
        ax[1].semilogx(ec, ig[i]/ig[i].max(), "o-", ms=3, label=f"r = {x[i]:.1f} M ({lab})")
    ax[1].set_xlabel("E (normal observer) [code]")
    ax[1].set_ylabel("bin intensity (normalised)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig_redshift.png", dpi=150)
    plt.close(fig)
    ratio = (mean_ln[-1] - mean_ln[0])/(ana(x[-1]) - ana(x[0]))
    check("redshift", "mf measured/analytic shift - 1", ratio - 1, 5e-2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite")
    p.add_argument("--out")
    a = p.parse_args()
    out = a.out or a.suite
    os.makedirs(out, exist_ok=True)
    for f in (equilibration, boost, curved, freestream, diffusion, redshift,
              lepton):
        try:
            f(a.suite, out)
        except Exception as e:  # keep going; a missing run should not hide the others
            print(f"{f.__name__}: FAILED to analyse ({e!r})")
    with open(f"{out}/summary.txt", "w") as fh:
        for test, q, val, tol in ROWS:
            verdict = ("reference" if tol is None else
                       f"tol {tol:.0e}  {'PASS' if abs(val) <= tol else 'FAIL'}")
            line = f"{test:12s} {q:48s} {val:+.3e}  {verdict}"
            print(line)
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
