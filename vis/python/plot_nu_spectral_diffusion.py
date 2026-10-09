#!/usr/bin/env python3
"""Per-group view of the spectral diffusion test (pgen rad_m1_diffusiontest,
initial_data=diffusion, multi_freq) with elastic, energy-independent scattering.

Elastic scattering cannot move energy between comoving groups, so each group's share
f_g of the total comoving energy must stay fixed, and each group's profile divided by
its share must follow the grey analytic solution.

Usage:
    plot_nu_spectral_diffusion.py RUN_DIR --out PREFIX [--v 0.3] [--kappa-s 100]
                                  [--nu 4.0] [--t0 0.0]
RUN_DIR needs bin/*.radnu_fluid.*.bin (single MeshBlock).
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                     "axes.titlesize": 14})
import numpy as np

import radnu_read
from rad_diffusion_compare import analytic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p.add_argument("--v", type=float, default=0.3)
    p.add_argument("--kappa-s", type=float, default=100.0)
    p.add_argument("--nu", type=float, default=4.0)
    p.add_argument("--t0", type=float, default=0.0)
    a = p.parse_args()

    snaps = [radnu_read.read_radnu(f)
             for f in sorted(glob.glob(f"{a.run}/bin/*.radnu_fluid.*.bin"))]
    x = snaps[0]["x1v"]
    jg = np.array([s["data"][0, :, 0, 0, 0, :] for s in snaps])   # (time, group, x)
    times = [s["time"] for s in snaps]
    frac = jg.sum(axis=2) / jg.sum(axis=(1, 2))[:, None]            # (time, group)
    groups = np.arange(jg.shape[1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for it, t in enumerate(times):
        ax[0].semilogy(groups, frac[it], "o-", label=f"t = {t:.2f}")
    ax[0].set_xlabel("frequency group")
    ax[0].set_ylabel("share of comoving energy $f_g$")
    ax[0].set_title("Group shares (elastic scattering: must not change)")
    ax[0].legend(fontsize=11)
    drift = np.abs(frac[-1] / frac[0] - 1.0)
    ax[1].semilogy(groups, drift, "ko-")
    ax[1].set_xlabel("frequency group")
    ax[1].set_ylabel(r"$|f_g(t_{end})/f_g(0) - 1|$")
    ax[1].set_title("Drift of each group's share")
    fig.tight_layout()
    fig.savefig(f"{a.out}_shares.png", dpi=170)

    t = times[-1]
    jj, _, _, _ = analytic(x, t, a.v, a.kappa_s, a.nu, a.t0)
    j0, _, _, _ = analytic(x, times[0], a.v, a.kappa_s, a.nu, a.t0)
    norm = jg[0].sum() / j0.sum()          # code normalisation of the seed
    fig, ax = plt.subplots(figsize=(7, 4.5))
    main_groups = np.argsort(frac[0])[-5:]
    for g in sorted(main_groups):
        ax.plot(x, jg[-1, g] / (frac[0, g] * norm), label=f"group {g} / $f_g$")
    ax.plot(x, jj, "k--", linewidth=2, label="grey analytic")
    ax.set_xlabel("x [code units]")
    ax.set_ylabel(r"comoving $J_g/f_g$ (normalised)")
    ax.set_title(f"t = {t:.2f}, v = {a.v}: each group divided by its share")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{a.out}_profiles.png", dpi=170)

    print("group  share(t0)    share(tend)   drift")
    for g in groups:
        print(f"{g:5d}  {frac[0, g]:.4e}  {frac[-1, g]:.4e}  {drift[g]:.2e}")
    print(f"wrote {a.out}_shares.png, {a.out}_profiles.png")


if __name__ == "__main__":
    main()
