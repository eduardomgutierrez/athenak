#!/usr/bin/env python3
"""Plot the single-zone neutrino relaxation test (matter, radiation, spectra).

CLI: plot_nu_singlezone.py RUN_DIR[:LABEL] ... --out PREFIX [--vs-proper-time] [--title TEXT]

Reads RUN_DIR/tab/*.mhd_w.*.tab, RUN_DIR/tab/*.rad_fluid.*.tab and
RUN_DIR/bin/*.radnu_fluid.*.bin (basename auto-detected from whatever is present).
No scipy is used: the Fermi-Dirac spectral fit uses numpy.polynomial.legendre.leggauss
for the bin integrals and a from-scratch Nelder-Mead for the (T, eta) search (the
amplitude A has a closed form at fixed T, eta, see fit_fermi_dirac()).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radnu_read as rr

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                      "axes.titlesize": 14})
DPI = 170

SPECIES_NAMES = ["nu_e", "anti-nu_e", "nu_x", "anti-nu_x"]
SPECIES_TEX = [r"$\nu_e$", r"$\bar\nu_e$", r"$\nu_x$", r"$\bar\nu_x$"]
ICELL = 2  # middle of 5 cells (0-based physical index, x1v = 0)


# ---------------------------------------------------------------------------
# Tab file reading (robust to the "j x2v k x3v" columns being named in the
# header but absent from 1D data rows, as documented in bin_convert.py's peers).
# ---------------------------------------------------------------------------
def read_tab(fname):
    """Return (time, {colname: array over rows}) for one .tab snapshot file."""
    with open(fname) as f:
        lines = f.readlines()
    time = float(lines[0].split("time=")[1].split()[0])
    header = lines[1].lstrip("#").split()
    rows = [line.split() for line in lines[2:] if line.strip()]
    ncols = len(rows[0])
    if ncols == len(header):
        names = header
    else:
        drop = {"j", "x2v", "k", "x3v"}
        names = [n for n in header if n not in drop]
        if len(names) != ncols:
            raise ValueError(f"{fname}: cannot reconcile header {header} with "
                              f"{ncols}-column data row")
    cols = {n: np.array([float(r[i]) for r in rows]) for i, n in enumerate(names)}
    return time, cols


def read_tab_series(pattern, icell=ICELL):
    """Read every snapshot matching a glob pattern, sorted by time.
    Returns dict of 1D arrays (over time) for every column, plus 'time'.
    Asserts all cells agree to 1e-10 relative (warns if not) and keeps cell `icell`."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    series = None
    times = []
    for fn in files:
        t, cols = read_tab(fn)
        for name, vals in cols.items():
            if name in ("gid", "i", "x1v", "x2v", "x3v"):
                continue
            spread = np.max(np.abs(vals - vals[icell])) / max(np.abs(vals[icell]), 1e-300)
            if spread > 1e-10:
                print(f"WARNING: {fn} column {name} not uniform across cells "
                      f"(max relative spread {spread:.3e})", file=sys.stderr)
        times.append(t)
        if series is None:
            series = {name: [] for name in cols if name not in ("gid", "i")}
            # keep x1v out of the per-time series build below too (it's a coordinate,
            # not a physical quantity to track vs time) -- drop it explicitly.
            series.pop("x1v", None)
            series.pop("x2v", None)
            series.pop("x3v", None)
        for name in series:
            series[name].append(cols[name][icell])
    out = {"time": np.array(times)}
    for name in series:
        out[name] = np.array(series[name])
    order = np.argsort(out["time"])
    for k in out:
        out[k] = out[k][order]
    return out


def find_basename(run_dir):
    tabs = glob.glob(os.path.join(run_dir, "tab", "*.mhd_w.*.tab"))
    if not tabs:
        raise FileNotFoundError(f"no *.mhd_w.*.tab found under {run_dir}/tab")
    m = re.match(r"(.+)\.mhd_w\.\d+\.tab$", os.path.basename(tabs[0]))
    return m.group(1)


def read_trapped_eq(run_dir):
    """(T_eq [MeV], Ye_eq) printed by the rad_neutrino_singlezone pgen, or None.

    Looks in every *.txt / *.log file of the run directory for the line
    '### rad_neutrino_singlezone: trapped equilibrium T_eq = ... MeV, Ye_eq = ...'.
    """
    pat = re.compile(r"trapped equilibrium T_eq = ([0-9.eE+-]+) MeV, Ye_eq = ([0-9.eE+-]+)")
    for fname in sorted(glob.glob(os.path.join(run_dir, "*.txt")) +
                        glob.glob(os.path.join(run_dir, "*.log"))):
        try:
            with open(fname, errors="replace") as f:
                for line in f:
                    if "FAILED" in line:
                        continue
                    m = pat.search(line)
                    if m:
                        return float(m.group(1)), float(m.group(2))
        except OSError:
            continue
    return None


def lorentz_factor(vx, vy, vz):
    """W = 1/sqrt(1-v^2) from primitive velx=W*vx etc. (u^i = W v^i, flat space);
    see src/pgen/rad_neutrino_singlezone.cpp:131,148-150. Since
    velx^2+vely^2+velz^2 = W^2 v^2 = W^2 - 1, W = sqrt(1 + velx^2+vely^2+velz^2)."""
    return np.sqrt(1.0 + vx**2 + vy**2 + vz**2)


# ---------------------------------------------------------------------------
# Radiation (per-species, per-group) series
# ---------------------------------------------------------------------------
def read_radnu_series(run_dir, basename, icell=ICELL):
    files = sorted(glob.glob(os.path.join(run_dir, "bin", f"{basename}.radnu_fluid.*.bin")))
    if not files:
        return None
    snaps = [rr.read_radnu(f) for f in files]
    snaps.sort(key=lambda d: d["time"])
    times = np.array([d["time"] for d in snaps])
    nspecies = snaps[0]["nspecies"]
    # J00 per species per group per time: (ntime, nspecies, nfreq)
    j00 = np.array([d["data"][:, :, 0, 0, 0, icell] for d in snaps])
    return {"time": times, "j00": j00, "nspecies": nspecies,
            "nfreq": snaps[0]["nfreq"], "nu_min": snaps[0]["nu_min"],
            "nu_max": snaps[0]["nu_max"], "freq_scale": snaps[0]["freq_scale"],
            "snaps": snaps}


# ---------------------------------------------------------------------------
# Fermi-Dirac spectral fit (no scipy)
# ---------------------------------------------------------------------------
_GL_N, _GL_W = np.polynomial.legendre.leggauss(32)


def _bin_integral_unit(e_lo, e_hi, T, eta):
    """integral_{e_lo}^{e_hi} E^3/(exp(E/T-eta)+1) dE via 32-pt Gauss-Legendre in ln E."""
    u_lo, u_hi = np.log(e_lo), np.log(e_hi)
    jac = 0.5 * (u_hi - u_lo)
    u = jac * _GL_N + 0.5 * (u_hi + u_lo)
    E = np.exp(u)
    integrand = E**4 / (np.exp(np.clip(E / T - eta, -500, 500)) + 1.0)
    return jac * np.sum(_GL_W * integrand)


def _model_and_amplitude(edges, data_j, T, eta):
    """Unit-amplitude (A=1) group energies, and the closed-form best-fit log A
    (least squares in log space, i.e. geometric mean of the ratios) given T, eta."""
    unit = np.array([_bin_integral_unit(edges[g], edges[g + 1], T, eta)
                      for g in range(len(edges) - 1)])
    good = (data_j > 0) & (unit > 0) & np.isfinite(unit)
    if not np.any(good):
        return unit, 1.0, 1e300
    log_ratio = np.log(data_j[good]) - np.log(unit[good])
    log_A = np.mean(log_ratio)
    resid = np.log(unit[good]) + log_A - np.log(data_j[good])
    cost = np.sum(resid**2)
    return unit, np.exp(log_A), cost


def _cost(params, edges, data_j):
    T, eta = params
    if T <= 0:
        return 1e300
    _, _, cost = _model_and_amplitude(edges, data_j, T, eta)
    return cost


def _nelder_mead(f, x0, args=(), step=None, maxiter=300, tol=1e-10):
    """Minimal Nelder-Mead for a 2-parameter cost function (no scipy available)."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    if step is None:
        step = 0.1 * np.abs(x0) + 0.1
    simplex = [x0]
    for i in range(n):
        p = x0.copy()
        p[i] += step[i]
        simplex.append(p)
    simplex = np.array(simplex)
    fvals = np.array([f(p, *args) for p in simplex])
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(maxiter):
        order = np.argsort(fvals)
        simplex, fvals = simplex[order], fvals[order]
        if np.max(np.abs(fvals - fvals[0])) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr, *args)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe, *args)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc, *args)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i], *args)
    order = np.argsort(fvals)
    return simplex[order][0], fvals[order][0]


def fit_fermi_dirac(edges, data_j, T_guess):
    """Fit J_g = A * int_bin E^3/(exp(E/T-eta)+1) dE by least squares in log space.
    Returns (A, T, eta). A coarse (T, eta) grid seeds a Nelder-Mead polish."""
    T_grid = T_guess * np.geomspace(0.2, 5.0, 24)
    eta_grid = np.linspace(-6.0, 6.0, 25)
    best = (1e300, T_guess, 0.0)
    for T in T_grid:
        for eta in eta_grid:
            c = _cost((T, eta), edges, data_j)
            if c < best[0]:
                best = (c, T, eta)
    x0 = np.array([best[1], best[2]])
    xopt, _ = _nelder_mead(_cost, x0, args=(edges, data_j),
                            step=np.array([0.1 * best[1], 0.5]))
    T, eta = xopt
    _, A, _ = _model_and_amplitude(edges, data_j, T, eta)
    return A, T, eta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="RUN_DIR[:LABEL]")
    ap.add_argument("--out", required=True, help="output file prefix")
    ap.add_argument("--vs-proper-time", action="store_true",
                     help="use t/W (fluid proper time) as the x-axis")
    ap.add_argument("--title", default=None)
    ap.add_argument("--log-time", action="store_true",
                    help="logarithmic time axis (t = 0 is dropped)")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if ":" in spec:
            d, label = spec.split(":", 1)
        else:
            d, label = spec, os.path.basename(os.path.normpath(spec))
        runs.append((d, label))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 2)))

    loaded = []
    for (d, label), color in zip(runs, colors):
        basename = find_basename(d)
        mhd = read_tab_series(os.path.join(d, "tab", f"{basename}.mhd_w.*.tab"))
        radf = read_tab_series(os.path.join(d, "tab", f"{basename}.rad_fluid.*.tab"))
        radnu = read_radnu_series(d, basename)
        loaded.append({"dir": d, "label": label, "color": color, "basename": basename,
                        "mhd": mhd, "radf": radf, "radnu": radnu})

    # ---------------- Figure 1: matter ----------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    xlabel = "time t [code units]"
    for L in loaded:
        mhd = L["mhd"]
        if mhd is None:
            continue
        if args.vs_proper_time:
            W = lorentz_factor(mhd["velx"], mhd["vely"], mhd["velz"])
            x = mhd["time"] / W
            xlabel = "proper time t/W [code units]"
        else:
            x = mhd["time"]
            xlabel = "time t [code units]"
        Ye_f, T_f = mhd["s_00"][-1], mhd["temperature"][-1]
        lbl = f"{L['label']} (Ye_f={Ye_f:.4f}, T_f={T_f:.2f} MeV)"
        sel = x > 0 if args.log_time else slice(None)
        axes[0].plot(x[sel], mhd["s_00"][sel], color=L["color"], label=lbl)
        axes[1].plot(x[sel], mhd["temperature"][sel], color=L["color"], label=lbl)
        axes[2].plot(x[sel], mhd["press"][sel], color=L["color"], label=lbl)
    # Analytic end state, if the pgen printed it (it does when the zone starts empty).
    target = next((t for t in (read_trapped_eq(L["dir"]) for L in loaded) if t), None)
    if target is not None:
        axes[0].axhline(target[1], color="k", ls=":", lw=1.6,
                        label=rf"trapped $\beta$-equilibrium, $Y_e$ = {target[1]:.4f}")
        axes[1].axhline(target[0], color="k", ls=":", lw=1.6,
                        label=f"trapped equilibrium, T = {target[0]:.2f} MeV")
        axes[1].legend(loc="best")
    axes[0].set_ylabel(r"electron fraction $Y_e$")
    axes[1].set_ylabel("matter temperature T [MeV]")
    axes[2].set_ylabel("pressure P [code units]")
    axes[2].set_xlabel(xlabel)
    if args.log_time:
        axes[2].set_xscale("log")
    axes[0].legend(loc="best")
    if args.title:
        axes[0].set_title(args.title)
    fig.tight_layout()
    fig.savefig(f"{args.out}_matter.png", dpi=DPI)
    plt.close(fig)

    # ---------------- Figure 2: radiation energy density ----------------
    # Colour identifies the run, line style the species; one legend entry each, so the
    # legend stays readable with many runs.
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = ["-", "--", ":", "-."]
    nsp_max = 0
    for L in loaded:
        if L["radnu"] is None:
            continue
        rn = L["radnu"]
        nsp_max = max(nsp_max, rn["nspecies"])
        per_species = rn["j00"].sum(axis=2)  # (ntime, nspecies)
        sel = rn["time"] > 0 if args.log_time else slice(None)
        for isp in range(rn["nspecies"]):
            ax.semilogy(rn["time"][sel], per_species[sel, isp], color=L["color"],
                        linestyle=styles[isp % 4], alpha=0.85)
    run_handles = [plt.Line2D([], [], color=L["color"], label=L["label"]) for L in loaded
                   if L["radnu"] is not None]
    species_handles = [plt.Line2D([], [], color="k", linestyle=styles[isp % 4],
                                  label=SPECIES_TEX[isp]) for isp in range(nsp_max)]
    leg = ax.legend(handles=run_handles, fontsize=11, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), title="run")
    ax.add_artist(leg)
    ax.legend(handles=species_handles, fontsize=11, loc="lower left",
              bbox_to_anchor=(1.01, 0.0), title="species")
    if args.log_time:
        ax.set_xscale("log")
    ax.set_xlabel("time t [code units]")
    ax.set_ylabel(r"comoving radiation energy density $J$ [code units]")
    if args.title:
        ax.set_title(args.title)
    fig.subplots_adjust(left=0.13, right=0.74, top=0.92, bottom=0.11)
    fig.savefig(f"{args.out}_radiation.png", dpi=DPI)
    plt.close(fig)

    # ---------------- Figure 3: spectra (first run only) ----------------
    L0 = loaded[0]
    rn = L0["radnu"]
    if rn is not None and rn["nfreq"] > 1:
        edges = rr.freq_bin_edges_mev(rn["nu_min"], rn["nu_max"], rn["nfreq"],
                                       rn["freq_scale"])
        widths = np.diff(edges)
        centers = np.sqrt(edges[:-1] * edges[1:])

        ntime = len(rn["time"])
        pick = np.unique(np.linspace(0, ntime - 1, min(5, ntime)).astype(int))
        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=rn["time"][pick[0]], vmax=rn["time"][pick[-1]])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        T_fluid_final = L0["mhd"]["temperature"][-1] if L0["mhd"] is not None else None
        print("\nFermi-Dirac fit (last time), species / T_nu[MeV] / eta / T_fluid[MeV] / T_nu/T_fluid-1")
        for isp in range(3):
            ax = axes[isp]
            for it in pick:
                Jg = rn["j00"][it, isp, :]
                spec_dens = Jg / widths
                mask = spec_dens > 0
                ax.loglog(centers[mask], spec_dens[mask], "o-", color=cmap(norm(rn["time"][it])),
                          markersize=4, linewidth=1)
            # Fermi-Dirac fit at the last time
            Jg_last = rn["j00"][-1, isp, :]
            T_guess = T_fluid_final if T_fluid_final else 10.0
            A, T_nu, eta = fit_fermi_dirac(edges, Jg_last, T_guess)
            # spectral density is the (unbinned) FD integrand E^3/(exp(E/T-eta)+1),
            # i.e. the limit of J_g/dE_g as the bin narrows -- consistent with what
            # fit_fermi_dirac() integrated over each finite bin.
            E_fine = np.geomspace(edges[0], edges[-1], 200)
            model_dens = A * E_fine**3 / (np.exp(np.clip(E_fine / T_nu - eta, -500, 500)) + 1.0)
            ax.loglog(E_fine, model_dens, "k--", linewidth=1.0, alpha=0.6,
                      label=f"FD fit (pointwise): T={T_nu:.2f} MeV, eta={eta:.2f}")
            # On a steep tail a wide bin's average sits far above the pointwise value at
            # its centre, so compare the data with the fit averaged over the same bins.
            model_bin = np.array([A*_bin_integral_unit(edges[g], edges[g + 1], T_nu, eta)
                                  for g in range(len(widths))]) / widths
            ax.loglog(centers, model_bin, "kx", markersize=7, mew=1.5,
                      label="FD fit, averaged over each bin")
            ax.set_xlabel("E [MeV]")
            ax.set_ylabel(r"$J_g/\Delta E_g$ [code units / MeV]" if isp == 0 else "")
            ax.set_title(SPECIES_TEX[isp])
            ax.legend(fontsize=11)
            ratio = T_nu / T_fluid_final - 1.0 if T_fluid_final else float("nan")
            print(f"{SPECIES_NAMES[isp]:10s}  T_nu={T_nu:8.3f}  eta={eta:7.3f}  "
                  f"T_fluid={T_fluid_final:8.3f}  T_nu/T_fluid-1={ratio:+.4f}")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label="time t [code units]", fraction=0.02, pad=0.02)
        if args.title:
            fig.suptitle(args.title)
        fig.savefig(f"{args.out}_spectra.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        print("No multi-group radnu data (or nfreq==1) for the first run: skipping spectra plot.")

    print(f"\nWrote {args.out}_matter.png, {args.out}_radiation.png"
          + (f", {args.out}_spectra.png" if rn is not None and rn['nfreq'] > 1 else ""))


if __name__ == "__main__":
    main()
