#!/usr/bin/env python3
"""Mass accretion rate of the torus against time, from the torus pgen's user history.

The history (gr_torus_compose, TorusFluxes) writes mdot_<r> = -oint rho u^r sqrt(-g) dOmega
on Kerr-Schild spheres: positive for inflow.  The innermost sphere is the horizon,
r+ = 1 + sqrt(1 - a^2), in fluid-only runs, and ceil(r_excise + 1) = 2 M with radiation.
In code units Mdot [Msun/s] = Mdot / 4.9255e-6 for any black-hole mass; the time in ms
needs M (--msun, or read from --input).

Several history files can be overlaid (e.g. the neutrino run and its fluid-only control).

Examples:
  python vis/python/torus_mdot.py RUN/torus_smr.user.hst --msun 2.8 -o RUN/mdot.png
  python vis/python/torus_mdot.py NU/torus_nu_smr.user.hst FL/torus_smr.user.hst \\
      --label neutrinos --label "fluid only" --radii 2,12 --log --window 200:500
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

MDOT_CODE_TO_MSUN_S = 1.0/4.9255e-6     # (Msun) / (G Msun/c^3), M cancels
T_MSUN_MS = 4.9255e-3                   # ms per G M/c^3 at M = 1 Msun

# Regime thresholds for M = 3 Msun, a ~ 0.8, alpha ~ 0.02 (Hernandez-Morales & Siegel 2025,
# Table 1; De & Siegel 2021), scaled as M^(4/3).
THRESHOLDS = [(1.5e-3, "ignition"), (2e-2, r"$\nu_e$-opaque"), (4.0, "trapping")]


def read_hst(fname):
    """Columns of an Athena++/AthenaK history file, by name."""
    names = None
    with open(fname) as fp:
        for line in fp:
            if line.startswith("#") and "[1]=" in line:
                names = re.findall(r"\[\d+\]=(\S+)", line)
                break
    if names is None:
        sys.exit(f"{fname}: no column header")
    data = np.genfromtxt(fname, comments="#")
    data = np.atleast_2d(data)
    # a restarted run appends; keep the last value written for each time
    _, keep = np.unique(data[::-1, 0], return_index=True)
    data = data[::-1][np.sort(keep)]
    return {n: data[:, i] for i, n in enumerate(names)}


def msun_from_input(path):
    """<mhd>/M_Msun from an athinput file."""
    inblock = False
    for line in open(path):
        line = line.split("#")[0].strip()
        if line.startswith("<"):
            inblock = (line == "<mhd>")
        elif inblock and line.startswith("M_Msun"):
            return float(line.split("=")[1])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hst", nargs="+", help="user history files (*.user.hst)")
    ap.add_argument("--label", action="append", help="one per history file")
    ap.add_argument("--msun", type=float, help="black-hole mass in Msun (for ms)")
    ap.add_argument("--input", help="athinput to read <mhd>/M_Msun from")
    ap.add_argument("--radii", default="inner",
                    help="'inner' (the horizon sphere), 'all', or a comma list of radii")
    ap.add_argument("--log", action="store_true", help="log scale (|Mdot|, outflow dashed)")
    ap.add_argument("--thresholds", action="store_true",
                    help="draw the NDAF regime thresholds (needs --msun/--input)")
    ap.add_argument("--window", help="tmin:tmax [M] for the printed mean Mdot")
    ap.add_argument("-o", "--out", default="mdot.png")
    ap.add_argument("--csv", help="also write t [M], t [ms], Mdot [Msun/s] columns")
    args = ap.parse_args()

    msun = args.msun or (msun_from_input(args.input) if args.input else None)
    labels = args.label or [os.path.basename(f) for f in args.hst]
    if len(labels) != len(args.hst):
        sys.exit("give one --label per history file")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]   # fixed order: run identity
    styles = ["-", "--", ":", "-."]                          # radius
    csv_rows = []
    for irun, (fname, lab) in enumerate(zip(args.hst, labels)):
        h = read_hst(fname)
        cols = sorted((float(k.split("_", 1)[1]), k) for k in h if k.startswith("mdot_"))
        if args.radii == "inner":
            cols = cols[:1]
        elif args.radii != "all":
            want = [float(r) for r in args.radii.split(",")]
            cols = [c for c in cols if any(abs(c[0] - w) < 0.05 for w in want)]
        t = h["time"]
        tx = t*msun*T_MSUN_MS if msun else t
        for irad, (rad, key) in enumerate(cols):
            md = h[key]*MDOT_CODE_TO_MSUN_S
            name = f"{lab}, r = {rad:g} M" if len(cols) > 1 or len(args.hst) > 1 else \
                   f"r = {rad:g} M"
            c, ls = colors[irun % 4], styles[irad % 4]
            if args.log:
                pos = np.where(md > 0, md, np.nan)
                neg = np.where(md < 0, -md, np.nan)
                ax.semilogy(tx, pos, color=c, ls=ls, lw=1.8, label=name)
                if np.any(md < 0):
                    ax.semilogy(tx, neg, color=c, ls=ls, lw=0.8, alpha=0.6)
            else:
                ax.plot(tx, md, color=c, ls=ls, lw=1.8, label=name)
            if args.window:
                lo, hi = (float(v) for v in args.window.split(":"))
                sel = (t >= lo) & (t <= hi)
                if sel.any():
                    print(f"{lab}: <Mdot>(r = {rad:g} M, t = {lo:g}-{hi:g} M) = "
                          f"{np.mean(md[sel]):.4e} Msun/s")
            csv_rows.append((lab, rad, t, tx, md))
    if args.thresholds and msun:
        for v, name in THRESHOLDS:
            v *= (msun/3.0)**(4.0/3.0)
            ax.axhline(v, color="0.45", ls=(0, (1, 2)), lw=1)
            ax.annotate(name, (1.0, v), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                        textcoords="offset points", ha="right", fontsize=8, color="0.3")
    if not args.log:
        ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$t$ [ms]" if msun else r"$t$ [$GM/c^3$]")
    ax.set_ylabel(r"$\dot M$ [$M_\odot$ s$^{-1}$]" + (" (thin: outflow)" if args.log else ""))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)
    if args.csv:
        with open(args.csv, "w") as fp:
            fp.write("# run, r [M], t [M], t [ms or M], Mdot [Msun/s]\n")
            for lab, rad, t, tx, md in csv_rows:
                for a, b, c in zip(t, tx, md):
                    fp.write(f"{lab},{rad:g},{a:.6e},{b:.6e},{c:.6e}\n")


if __name__ == "__main__":
    main()
