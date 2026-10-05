#!/usr/bin/env python3
"""Analyse the free-streaming redshift test (pgen rad_freq_redshift).

Reads the redshift_ray.txt written by the pgen (lab bin intensities of the most radial
outgoing ray, every cell) for one or more runs, and plots:
  - the shift of the intensity-weighted mean ln E against r, with the analytic shift of
    an exactly radial ray measured by the Kerr-Schild normal observer,
    ln[ sqrt(1+f)/(1-f) ] - (same at the first cell),  f = 2/r;
  - the spectrum I_g / d(ln E) at a few radii for each run.

Usage:
    plot_freq_redshift.py RUN_DIR:LABEL [RUN_DIR:LABEL ...] --out PREFIX
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_ray(path):
    with open(path) as f:
        head = [next(f) for _ in range(4)]
    lo = np.array([float(v) for v in head[1].split()[2:]])
    hi = np.array([float(v) for v in head[2].split()[2:]])
    data = np.loadtxt(path)
    x, ig = data[:, 0], data[:, 1:]
    order = np.argsort(x)
    return head[0].strip("# \n"), lo, hi, x[order], ig[order]


def radial_shift(r):
    f = 2.0 / r
    return np.log(np.sqrt(1 + f) / (1 - f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    runs = []
    for spec in a.runs:
        d, label = spec.split(":", 1) if ":" in spec else (spec, spec)
        ray, lo, hi, x, ig = read_ray(f"{d}/redshift_ray.txt")
        lnmid = np.where(lo > 0, 0.5 * (np.log(np.where(lo > 0, lo, 1)) + np.log(hi)),
                         np.log(hi) - 0.5 * np.log(hi[1] / lo[1]))
        w = np.clip(ig, 0, None)
        mean_ln = (w * lnmid).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-300)
        runs.append(dict(label=label, ray=ray, lo=lo, hi=hi, x=x, ig=ig, lnmid=lnmid,
                         mean_ln=mean_ln))
        print(f"{label:28s} {ray}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x0 = runs[0]["x"]
    ax[0].plot(x0, radial_shift(x0) - radial_shift(x0[0]), "k--", lw=2,
               label="analytic, exactly radial ray")
    for R in runs:
        ax[0].plot(R["x"], R["mean_ln"] - R["mean_ln"][0], label=R["label"])
    ax[0].set_xlabel("r / M")
    ax[0].set_ylabel(r"$\langle\ln E\rangle(r) - \langle\ln E\rangle(r_{\rm in})$")
    ax[0].set_title("Energy shift along the most radial outgoing ray")
    ax[0].legend(fontsize=8)

    styles = ["-", "--", ":", "-."]
    for k, R in enumerate(runs):
        dln = np.log(R["hi"]) - np.log(np.where(R["lo"] > 0, R["lo"], R["hi"] / 1e3))
        emid = np.exp(R["lnmid"])
        for ir, col in zip(np.linspace(0, len(R["x"]) - 1, 4).astype(int),
                           ["C0", "C1", "C2", "C3"]):
            spec = R["ig"][ir] / dln
            spec = spec / max(spec.max(), 1e-300)
            ax[1].semilogx(emid, spec, styles[k % 4], color=col,
                           label=f"{R['label']}, r={R['x'][ir]:.1f}" if k < 4 else None)
    ax[1].set_xlabel("E (grid units)")
    ax[1].set_ylabel(r"$I_g/\Delta\ln E$ (normalised)")
    ax[1].set_title("Spectrum of the ray at four radii")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{a.out}.png", dpi=150)
    print(f"wrote {a.out}.png")

    print("\nshift of <ln E> at the outer radius vs analytic radial:")
    xa = runs[0]["x"]
    print(f"  analytic radial: {radial_shift(xa[-1]) - radial_shift(xa[0]):+.4f}")
    for R in runs:
        print(f"  {R['label']:28s}: {R['mean_ln'][-1] - R['mean_ln'][0]:+.4f}")


if __name__ == "__main__":
    main()
