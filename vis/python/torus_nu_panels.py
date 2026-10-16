#!/usr/bin/env python3
"""One multipanel x-z frame per time of the neutrino torus, and a GIF (or movie) of them:
matter (rho, p_mag/p_gas, T, Ye), the grey neutrino field (E_nu and the radial flux factor
F^r/E, coordinate frame, all species) and the neutrino-matter exchange (net cooling,
emission, the absorbed fraction and dYe/dt, from radnu_src; see torus_nu_src.py).

The three slice families are written on their own schedules (an output block added on a
restart starts at the restart time), so each frame takes the radnu_src slice and the
nearest matter and radiation slices to it.  Colour limits are fixed: the source maps take
theirs from the --ref frame, the rest are physical ranges that can be overridden with --lim.

Example:
  python vis/python/torus_nu_panels.py RUN/bin -o RUN/panels --extent 40 --stride 2 \\
      --gif RUN/torus_panels.gif -j 6
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LogNorm, Normalize, SymLogNorm  # noqa: E402
from matplotlib.patches import Ellipse                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bin_convert as bc                              # noqa: E402
from torus_slices import header_value, horizon, make_movie   # noqa: E402
from torus_nu_src import (RHO_MSUN, T_MSUN_S, C2, load_slice, panels, default_limits,
                          symlog_ticks)               # noqa: E402

LIMITS = {"dens": (1e6, 3e12), "beta": (1e-4, 10.0), "temperature": (0.1, 20.0),
          "ye": (0.0, 0.5), "enu": (1e24, 1e31), "ff": (-1.0, 1.0)}


def file_time(fname):
    """The time from the preheader, without reading the data."""
    with open(fname, "rb") as fp:
        for _ in range(4):
            line = fp.readline()
            if line.strip().startswith(b"time="):
                return float(line.split(b"=")[1])
    return bc.read_binary(fname)["time"]


def nearest(times, files, t):
    return files[int(np.argmin(np.abs(times - t)))]


def matter_fields(fname):
    d = bc.read_binary_as_athdf(fname)
    hdr = bc.read_binary(fname)["header"]
    msun = float(header_value(hdr, "mhd", "M_Msun", 1.0))
    sq = lambda k: np.asarray(d[k], dtype=np.float64)[:, 0, :]
    f = {"dens": sq("dens")*RHO_MSUN/msun**2, "temperature": sq("temperature"),
         "ye": sq("s_00")}
    if "bcc1" in d:
        b2 = sq("bcc1")**2 + sq("bcc2")**2 + sq("bcc3")**2
        f["beta"] = 0.5*b2/np.maximum(sq("press"), 1e-300)
    return np.asarray(d["x1f"]), np.asarray(d["x3f"]), f


def rad_fields(fname, msun):
    d = bc.read_binary_as_athdf(fname)
    sq = lambda k: np.asarray(d[k], dtype=np.float64)[:, 0, :]
    X, Z = np.meshgrid(np.asarray(d["x1v"]), np.asarray(d["x3v"]))
    R = np.hypot(X, Z)
    E = sq("r00")
    with np.errstate(divide="ignore", invalid="ignore"):
        ff = (sq("r01")*X + sq("r03")*Z)/np.maximum(R, 1e-30)/np.where(E > 0, E, np.nan)
    return np.asarray(d["x1f"]), np.asarray(d["x3f"]), \
        {"enu": E*RHO_MSUN/msun**2*C2, "ff": ff}


def plot_frame(job):
    src, mat, rad, lim, slim, extent, outdir, dpi = job
    s = load_slice(src)
    xm, zm, mf = matter_fields(mat)
    xr, zr, rf = rad_fields(rad, s["msun"])
    p = panels(s, mf["dens"], None)
    xc = 0.5*(xm[1:] + xm[:-1]); zc = 0.5*(zm[1:] + zm[:-1])
    emit = p["E"] > slim["E"][0]
    rows = [
        [("dens", mf["dens"], xm, zm, r"$\rho$ [g cm$^{-3}$]", "viridis", LogNorm(*lim["dens"])),
         ("beta", mf.get("beta"), xm, zm, r"$p_{\rm mag}/p_{\rm gas}$", "magma",
          LogNorm(*lim["beta"])),
         ("temperature", mf["temperature"], xm, zm, r"$T$ [MeV]", "inferno",
          LogNorm(*lim["temperature"])),
         ("ye", mf["ye"], xm, zm, r"$Y_e$", "cividis", Normalize(*lim["ye"])),
         ("enu", rf["enu"], xr, zr, r"$E_\nu$ (all species) [erg cm$^{-3}$]", "viridis",
          LogNorm(*lim["enu"]))],
        [("ff", rf["ff"], xr, zr, r"radial flux factor $F^r/E$", "RdBu_r",
          Normalize(*lim["ff"])),
         ("Q", p["Q"], s["x"], s["z"], r"net cooling $Q^-$ [erg cm$^{-3}$ s$^{-1}$]", "RdBu_r",
          SymLogNorm(slim["Q"][0], vmin=-slim["Q"][1], vmax=slim["Q"][1])),
         ("E", p["E"], s["x"], s["z"], r"emission $\eta$ [erg cm$^{-3}$ s$^{-1}$]", "magma",
          LogNorm(*slim["E"])),
         ("absf", np.ma.masked_where(~emit, p["absf"]), s["x"], s["z"],
          r"absorbed fraction $(\eta - Q^-)/\eta$", "viridis", Normalize(0.0, 1.0)),
         ("dye", np.ma.masked_where(~emit, p["dye"]) if "dye" in p else None, s["x"], s["z"],
          r"$dY_e/dt$ [s$^{-1}$]", "PuOr_r",
          SymLogNorm(slim["dye"][0], vmin=-slim["dye"][1], vmax=slim["dye"][1])
          if "dye" in slim else None)],
    ]
    fig, axes = plt.subplots(2, 5, figsize=(19.0, 8.6), constrained_layout=True)
    w, h = horizon(s["spin"])
    for r, row in enumerate(rows):
        for c, (key, v, x, z, label, cmap, norm) in enumerate(row):
            ax = axes[r][c]
            if v is None or norm is None:
                ax.set_axis_off()
                continue
            im = ax.pcolormesh(x, z, np.ma.masked_invalid(v), cmap=cmap, norm=norm,
                               shading="flat", rasterized=True)
            if key in ("Q", "E", "absf", "dye"):
                ax.contour(xc, zc, np.log10(np.maximum(mf["dens"], 1e-30)),
                           levels=[10, 11, 12], colors="k", linewidths=0.4, alpha=0.5)
            ax.add_patch(Ellipse((0, 0), w, h, color="black", zorder=3))
            ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            if r == 1:
                ax.set_xlabel(r"$x$ [$M$]", fontsize=9)
            if c == 0:
                ax.set_ylabel(r"$z$ [$M$]", fontsize=9)
            cb = fig.colorbar(im, ax=ax, shrink=0.85, location="top")
            cb.set_label(label, fontsize=9)
            cb.ax.tick_params(labelsize=7)
            if isinstance(norm, SymLogNorm):
                cb.set_ticks(symlog_ticks(norm.linthresh, norm.vmax))
    fig.suptitle(f"t = {s['t']:.0f} M = {s['t']*s['msun']*T_MSUN_S*1e3:.2f} ms"
                 r"    (source maps: contours $\log_{10}\rho$ = 10, 11, 12)", fontsize=12)
    out = os.path.join(outdir, os.path.basename(src).replace(".bin", "_panels.png"))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def make_gif(frames, gif, fps):
    from PIL import Image
    imgs = [Image.open(f).convert("RGB") for f in frames]
    pal = imgs[-1].quantize(colors=255, method=Image.MEDIANCUT)    # one shared palette
    imgs = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000/fps),
                 loop=0, optimize=True)
    return gif


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bindir", help="directory with the x-z slice .bin files")
    ap.add_argument("-o", "--outdir", default="panels")
    ap.add_argument("--src", default="*.radnu_src_xz.*.bin")
    ap.add_argument("--matter", default="*.mhd_w_bcc_xz.*.bin")
    ap.add_argument("--rad", default="*.rad_coord_xz.*.bin")
    ap.add_argument("--extent", type=float, default=40.0)
    ap.add_argument("--stride", type=int, default=1, help="every n-th radnu_src slice")
    ap.add_argument("--skip-first", action="store_true",
                    help="drop the first radnu_src slice (all zeros after a restart)")
    ap.add_argument("--ref", type=int, default=-1, help="frame setting the source limits")
    ap.add_argument("--lim", action="append", default=[],
                    help="override a fixed range, e.g. dens:1e6:1e12 (repeatable)")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--gif", help="output GIF")
    ap.add_argument("--movie", help="output movie (.mp4, else .avi)")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args()

    # the newest file of a running job may be incomplete
    g = lambda pat: sorted(glob.glob(os.path.join(args.bindir, pat)))[:-1]
    src, mat, rad = g(args.src), g(args.matter), g(args.rad)
    if args.skip_first:
        src = src[1:]
    src = src[::args.stride]
    if not (src and mat and rad):
        sys.exit("missing slice files")
    tm = np.array([file_time(f) for f in mat]); tr = np.array([file_time(f) for f in rad])
    pairs = [(f, nearest(tm, mat, t), nearest(tr, rad, t)) for f in src
             for t in [file_time(f)]]
    lim = dict(LIMITS)
    for spec in args.lim:
        n, lo, hi = spec.split(":")
        lim[n] = (float(lo), float(hi))
    ref = pairs[args.ref % len(pairs)]
    slim = default_limits(load_slice(ref[0]), matter_fields(ref[1])[2]["dens"], None)
    os.makedirs(args.outdir, exist_ok=True)
    jobs = [(s, m, r, lim, slim, args.extent, args.outdir, args.dpi) for s, m, r in pairs]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            frames = pool.map(plot_frame, jobs, chunksize=4)
    else:
        frames = [plot_frame(j) for j in jobs]
    print(f"{len(frames)} frames in {args.outdir}")
    if args.gif:
        print("gif:", make_gif(frames, args.gif, args.fps))
    if args.movie:
        print("movie:", make_movie(frames, args.movie, args.fps))


if __name__ == "__main__":
    main()
