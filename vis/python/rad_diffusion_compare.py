#!/usr/bin/env python3
"""Compare grey M1, discrete-ordinates transport and the semi-analytic solution for the
1D diffusion test in a moving medium (pgen rad_m1_diffusiontest, initial_data=diffusion).

The reference is the exact solution of the comoving diffusion equation

    dJ/dt' = D d^2J/dx'^2,   H^x' = -D dJ/dx',   D = 1/(3 kappa_s),

for the Gaussian J(x', -t0) = exp(-nu^2 x'^2), pulled onto the lab slice through
x' = W(x - v t), t' = W(t - v x).  Lab moments follow from the Eddington closure:

    E = W^2 [ J (1 + v^2/3) + 2 v H ],   F^x = W^2 [ (4/3) v J + (1 + v^2) H ].

Usage:
    rad_diffusion_compare.py --m1 <dir> --sn <dir> [--v 0.1] [--kappa-s 100]
                             [--nu 4.0] [--t0 0.0] [--plot out.png] [--numeric]
"""

import argparse
import glob
import os
import re

import numpy as np


def read_tab(path):
    """Return (time, x1v, {varname: array}) for an AthenaK 1D .tab file."""
    with open(path) as f:
        lines = f.readlines()
    time = float(re.search(r"time=([0-9eE.+-]+)", lines[0]).group(1))
    # header is "# gid i x1v j x2v k x3v <var> ..."; the j/x2v/k/x3v columns are
    # dropped from the data in 1D, so key off the number of variables instead.
    names = lines[1].lstrip("#").split()[7:]
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    ncoord = data.shape[1] - len(names)
    x = data[:, 2]
    cols = {n: data[:, ncoord + i] for i, n in enumerate(names)}
    order = np.argsort(x)
    return time, x[order], {k: v[order] for k, v in cols.items()}


def analytic(x, t, v, kappa_s, nu, t0):
    """Comoving (J, H^x) and lab (E, F^x) of the advection-diffusion solution."""
    w = 1.0 / np.sqrt(1.0 - v * v)
    dd = 1.0 / (3.0 * kappa_s)
    xp = w * (x - v * t)
    ss = 1.0 + 4.0 * dd * nu**2 * (t0 + w * (t - v * x))
    jj = np.exp(-(nu**2) * xp**2 / ss) / np.sqrt(ss)
    hh = 2.0 * dd * nu**2 * xp * jj / ss
    ee = w**2 * (jj * (1.0 + v * v / 3.0) + 2.0 * v * hh)
    ff = w**2 * ((4.0 / 3.0) * v * jj + (1.0 + v * v) * hh)
    return jj, hh, ee, ff


def numeric(x, t, v, kappa_s, nu, t0, xmax=40.0, nk=4096):
    """Independent spectral solve of dJ/dt' = D d^2 J/dx'^2 from J(x', -t0).

    Propagates the t' = -t0 profile in Fourier space and evaluates it at the (x', t')
    of each lab point, so it also exercises the slicing that `analytic` assumes.
    """
    w = 1.0 / np.sqrt(1.0 - v * v)
    dd = 1.0 / (3.0 * kappa_s)
    xp = w * (x - v * t)
    tp = t0 + w * (t - v * x)          # elapsed comoving time since the initial profile
    # exp(-nu^2 y^2) has transform sqrt(pi)/nu exp(-k^2/(4 nu^2)); build it on a grid
    k = np.linspace(-nk / xmax * np.pi, nk / xmax * np.pi, nk)
    jhat = np.sqrt(np.pi) / nu * np.exp(-(k**2) / (4.0 * nu**2))
    damp = np.exp(-dd * np.outer(tp, k**2))
    phase = np.exp(1j * np.outer(xp, k))
    dk = k[1] - k[0]
    jj = np.real((damp * phase * jhat).sum(axis=1)) * dk / (2.0 * np.pi)
    hh = np.real((damp * phase * (1j * k) * jhat).sum(axis=1)) * dk / (2.0 * np.pi)
    return jj, -dd * hh


def errors(num, ref, weight):
    """Weighted relative L1 and unweighted max error, both normalised by max|ref|."""
    scale = np.max(np.abs(ref))
    l1 = np.sum(weight * np.abs(num - ref)) / (np.sum(weight) * scale)
    return l1, np.max(np.abs(num - ref)) / scale


def collect(directory, kind):
    """Return {time: dict(J=, H=, E=, F=)} for an M1 or discrete-ordinates run."""
    out = {}
    if kind == "m1":
        for path in sorted(glob.glob(os.path.join(directory, "**", "*.rad_m1_J.*.tab"),
                                     recursive=True)):
            t, x, c = read_tab(path)
            out.setdefault(round(t, 9), {})["x"] = x
            out[round(t, 9)]["J"] = c["J:0"]
        for tag, key in (("rad_m1_H", "Hx:0"), ("rad_m1_E", "E:0")):
            for path in sorted(glob.glob(
                    os.path.join(directory, "**", "*.%s.*.tab" % tag), recursive=True)):
                t, x, c = read_tab(path)
                out.setdefault(round(t, 9), {})[key[0]] = c[key]
    else:
        for path in sorted(glob.glob(
                os.path.join(directory, "**", "*.rad_coord_fluid.*.tab"),
                recursive=True)):
            t, x, c = read_tab(path)
            out[round(t, 9)] = {"x": x, "J": c["r00_ff"], "H": c["r01_ff"],
                                "E": c["r00"], "F": c["r01"]}
    return out


COLORS = {"M1": "tab:blue", "Sn": "tab:orange"}
CYCLE = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:olive"]


def color_for(name, order):
    """Fixed colours for the built-in labels, a stable cycle for --run labels."""
    if name in COLORS:
        return COLORS[name]
    return CYCLE[order.index(name) % len(CYCLE)]


def plot_energy(runs, times, args, path, logy):
    """One panel per output time: J(x) for each code against the diffusion solution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extra = [n for n in runs if n not in COLORS]
    ncol = min(3, len(times))
    nrow = int(np.ceil(len(times) / ncol))
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(4.1 * ncol, 3.4 * nrow),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1] * nrow})
    for n, t in enumerate(times):
        row, col = divmod(n, ncol)
        ax, axr = axes[2 * row, col], axes[2 * row + 1, col]
        ref_x = None
        for name, run in runs.items():
            if t not in run:
                continue
            d = run[t]
            ref_x = d["x"]
            ax.plot(d["x"], d["J"], lw=1.3, color=color_for(name, extra), label=name)
        if ref_x is None:
            continue
        jj = analytic(ref_x, t, args.v, args.kappa_s, args.nu, args.t0)[0]
        ax.plot(ref_x, jj, "k--", lw=1.0, label="diffusion")
        for name, run in runs.items():
            if t not in run:
                continue
            axr.plot(ref_x, (run[t]["J"] - jj) / np.max(jj), lw=1.1,
                     color=color_for(name, extra))
        axr.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_title("t = %.2f" % t)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-8, 2.0)
        if col == 0:
            ax.set_ylabel("J")
            axr.set_ylabel(r"$\Delta J\,/\,\max J$")
        if row == nrow - 1:
            axr.set_xlabel("x")
        if args.xlim:
            ax.set_xlim(*args.xlim)
            axr.set_xlim(*args.xlim)
        if n == 0:
            ax.legend(frameon=False, fontsize=9)
    for n in range(len(times), nrow * ncol):
        row, col = divmod(n, ncol)
        axes[2 * row, col].axis("off")
        axes[2 * row + 1, col].axis("off")
    fig.suptitle(args.title or
                 (r"$v = %g$, $\kappa_s = %g$, $D = %.4g$, $\nu = %g$"
                  % (args.v, args.kappa_s, 1.0 / (3.0 * args.kappa_s), args.nu)),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    print("wrote %s" % path)



def plot_overlay(runs, times, args, path):
    """All output times on one axes: comoving J on the left, lab-frame E on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for n, t in enumerate(times):
        color = cmap(n / max(1, len(times) - 1) * 0.85)
        ref_x = None
        for name, run in runs.items():
            if t not in run:
                continue
            d = run[t]
            ref_x = d["x"]
            style = dict(lw=1.6, color=color) if name == "Sn" else dict(
                lw=1.0, color=color, ls=":")
            axes[0].plot(d["x"], d["J"], **style)
            if "E" in d:
                axes[1].plot(d["x"], d["E"], **style)
        if ref_x is None:
            continue
        jj, _, ee, _ = analytic(ref_x, t, args.v, args.kappa_s, args.nu, args.t0)
        axes[0].plot(ref_x, jj, "k--", lw=0.9)
        axes[1].plot(ref_x, ee, "k--", lw=0.9)
    axes[0].set_ylabel(r"comoving $J$")
    axes[1].set_ylabel(r"lab-frame $E$")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_xlim(-0.8, 1.6)
    handles = [plt.Line2D([], [], color="0.35", lw=1.6, label="Sn (162 angles)"),
               plt.Line2D([], [], color="0.35", lw=1.0, ls=":", label="grey M1"),
               plt.Line2D([], [], color="k", lw=0.9, ls="--", label="diffusion")]
    axes[0].legend(handles=handles, frameon=False, fontsize=9)
    tlabels = [plt.Line2D([], [], color=cmap(n / max(1, len(times) - 1) * 0.85), lw=2.0,
                          label="t = %.3g" % t) for n, t in enumerate(times)]
    axes[1].legend(handles=tlabels, frameon=False, fontsize=9, ncol=2)
    fig.suptitle(r"$v = %g$, $\kappa_s = %g$, $D = %.4g$, $\nu = %g$, "
                 r"nx1 = %d" % (args.v, args.kappa_s, 1.0 / (3.0 * args.kappa_s),
                                args.nu, len(ref_x)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    print("wrote %s" % path)



def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m1", help="directory with the grey M1 tab output")
    p.add_argument("--sn", help="directory with the discrete-ordinates tab output")
    p.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                   help="extra discrete-ordinates run to overlay; repeatable")
    p.add_argument("--v", type=float, default=0.1)
    p.add_argument("--kappa-s", type=float, default=100.0)
    p.add_argument("--nu", type=float, default=4.0)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--numeric", action="store_true",
                   help="cross-check the closed form against a spectral solve")
    p.add_argument("--plot", help="write a comparison figure here")
    p.add_argument("--plot-energy",
                   help="write a J(x, t) figure here: one panel per output time")
    p.add_argument("--plot-energy-log",
                   help="write the same figure on a log scale here")
    p.add_argument("--plot-overlay",
                   help="write a single-panel J(x) and E(x) overlay of all times here")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("LO", "HI"),
                   help="restrict the plotted x range")
    p.add_argument("--title", help="figure title, overriding the parameter line")
    args = p.parse_args()

    runs = {}
    if args.m1:
        runs["M1"] = collect(args.m1, "m1")
    if args.sn:
        runs["Sn"] = collect(args.sn, "sn")
    for spec in args.run:
        if "=" not in spec:
            p.error("--run wants LABEL=DIR, got %r" % spec)
        label, directory = spec.split("=", 1)
        runs[label] = collect(directory, "sn")
    if not runs:
        p.error("give at least one of --m1 / --sn")

    times = sorted(set().union(*[set(r) for r in runs.values()]))
    print("v = %g  kappa_s = %g  D = %.6g  nu = %g  t0 = %g"
          % (args.v, args.kappa_s, 1.0 / (3.0 * args.kappa_s), args.nu, args.t0))
    print("%-6s %-4s %11s %11s %11s %11s" %
          ("time", "code", "L1(J)", "max(J)", "L1(H)", "max(H)"))
    for t in times:
        for name, run in runs.items():
            if t not in run:
                continue
            d = run[t]
            jj, hh, _, _ = analytic(d["x"], t, args.v, args.kappa_s, args.nu, args.t0)
            if args.numeric:
                jn, hn = numeric(d["x"], t, args.v, args.kappa_s, args.nu, args.t0)
                print("       %-4s spectral check: max|dJ|/max|J| = %.3e, "
                      "max|dH|/max|H| = %.3e"
                      % ("", np.max(np.abs(jn - jj)) / np.max(np.abs(jj)),
                         np.max(np.abs(hn - hh)) / np.max(np.abs(hh))))
            l1j, mxj = errors(d["J"], jj, jj)
            l1h, mxh = errors(d["H"], hh, jj)
            print("%-6.2f %-4s %11.4e %11.4e %11.4e %11.4e"
                  % (t, name, l1j, mxj, l1h, mxh))

    if args.plot_overlay:
        plot_overlay(runs, times, args, args.plot_overlay)

    if args.plot_energy or args.plot_energy_log:
        for path, logy in ((args.plot_energy, False), (args.plot_energy_log, True)):
            if path:
                plot_energy(runs, times, args, path, logy)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tsel = [times[0], times[len(times) // 2], times[-1]]
        fig, axes = plt.subplots(2, len(tsel), figsize=(4.2 * len(tsel), 6.4),
                                 sharex=True)
        for col, t in enumerate(tsel):
            ref_x = None
            for name, run in runs.items():
                if t not in run:
                    continue
                d = run[t]
                ref_x = d["x"]
                axes[0, col].plot(d["x"], d["J"], lw=1.2, label=name)
                axes[1, col].plot(d["x"], d["H"], lw=1.2, label=name)
            if ref_x is None:
                continue
            jj, hh, _, _ = analytic(ref_x, t, args.v, args.kappa_s, args.nu, args.t0)
            axes[0, col].plot(ref_x, jj, "k--", lw=1.0, label="diffusion")
            axes[1, col].plot(ref_x, hh, "k--", lw=1.0, label="diffusion")
            axes[0, col].set_title("t = %.2f" % t)
            axes[1, col].set_xlabel("x")
        axes[0, 0].set_ylabel("J")
        axes[1, 0].set_ylabel("H^x")
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print("wrote %s" % args.plot)


if __name__ == "__main__":
    main()
