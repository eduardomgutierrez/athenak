#!/usr/bin/env python3
"""x-z slices of the tabulated-EOS torus (density, temperature, Ye, ...), one PNG per
output, and optionally a movie of them.

Reads the mhd_w_bcc slice outputs (file_type = bin, slice_x2 = 0), including SMR runs:
bin_convert.read_binary_as_athdf resamples every level onto the finest grid.  Units are
taken from the file's own input header: <mhd>/M_Msun for g/cm^3 and ms, <coord>/a for
the horizon.  Colour limits are fixed across frames (from the first file, or --lim), so
a movie does not flicker.

Examples:
  python vis/python/torus_slices.py RUN/bin/torus_smr.mhd_w_bcc_xz.*.bin -o RUN/slices
  python vis/python/torus_slices.py 'RUN/bin/*xz*.bin' -o RUN/slices --extent 32 \\
      --vars dens,temperature,ye,beta --lim dens:1e6:1e12 --movie RUN/torus.mp4 -j 8
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LogNorm, Normalize      # noqa: E402
from matplotlib.patches import Ellipse                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bin_convert as bc                              # noqa: E402

RHO_MSUN = 6.1764e17     # g/cm^3 per code density unit at M = 1 Msun (scales as M^-2)
T_MSUN_MS = 4.9255e-3    # ms per code time unit at M = 1 Msun (scales as M)

# name: (label, colormap, log scale, how to get it from the file's variables)
VARS = {
    "dens": (r"$\rho$ [g cm$^{-3}$]", "viridis", True, None),
    "temperature": (r"$T$ [MeV]", "inferno", True, None),
    "ye": (r"$Y_e$", "cividis", False, None),
    "press": (r"$p$ [code]", "magma", True, None),
    "vr": (r"$v^r$ (normal frame, $c$)", "RdBu_r", False, None),
    "beta": (r"$p_{\rm mag}/p_{\rm gas}$", "magma", True, None),
}


def header_value(header, block, key, default=None):
    """A parameter from the input file echoed in the binary header."""
    inblock = False
    for line in header:
        line = line.split("#")[0].strip()
        if line.startswith("<"):
            inblock = (line == f"<{block}>")
        elif inblock and "=" in line:
            k, v = [s.strip() for s in line.split("=", 1)]
            if k == key:
                return v
    return default


def load(fname):
    """One slice: coordinates, time and the plotted fields in physical units."""
    d = bc.read_binary_as_athdf(fname)
    hdr = bc.read_binary(fname)["header"]
    msun = float(header_value(hdr, "mhd", "M_Msun", 1.0))
    spin = float(header_value(hdr, "coord", "a", 0.0))
    sq = lambda k: np.asarray(d[k], dtype=np.float64)[:, 0, :]      # (nz, nx), y = 0
    x, z = np.asarray(d["x1f"]), np.asarray(d["x3f"])
    xc, zc = np.asarray(d["x1v"]), np.asarray(d["x3v"])
    out = {"dens": sq("dens")*RHO_MSUN/msun**2}
    if "temperature" in d:
        out["temperature"] = sq("temperature")
    if "s_00" in d:
        out["ye"] = sq("s_00")
    out["press"] = sq("press")
    X, Z = np.meshgrid(xc, zc)
    R = np.sqrt(X*X + Z*Z)
    out["vr"] = (sq("velx")*X + sq("velz")*Z)/np.maximum(R, 1e-30)
    if "bcc1" in d:
        b2 = sq("bcc1")**2 + sq("bcc2")**2 + sq("bcc3")**2
        out["beta"] = 0.5*b2/np.maximum(sq("press"), 1e-300)
    return dict(t=float(d["Time"]), msun=msun, spin=spin, x=x, z=z, f=out)


def default_limits(s, names):
    """Colour limits from the first frame."""
    lim = {}
    for n in names:
        a = s["f"][n]
        a = a[np.isfinite(a)]
        if n == "dens":
            lim[n] = (a.max()*1e-7, a.max())
        elif n == "temperature":
            lim[n] = (0.1, max(1.0, 3.0*a.max()))
        elif n == "ye":
            lim[n] = (0.0, 0.5)
        elif n == "vr":
            lim[n] = (-0.5, 0.5)
        elif VARS[n][2]:
            pos = a[a > 0]
            lim[n] = (np.percentile(pos, 1), pos.max()) if pos.size else (1e-30, 1.0)
        else:
            lim[n] = (a.min(), a.max())
    return lim


def horizon(spin):
    """The horizon in the x-z plane of Cartesian Kerr-Schild: x^2/(r+^2+a^2) + z^2/r+^2 = 1."""
    rp = 1.0 + np.sqrt(1.0 - spin*spin)
    return 2.0*np.sqrt(rp*rp + spin*spin), 2.0*rp


def plot_one(job):
    fname, names, lim, extent, outdir = job
    s = load(fname)
    names = [n for n in names if n in s["f"]]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6*len(names) + 0.6, 4.0),
                             squeeze=False, constrained_layout=True)
    w, h = horizon(s["spin"])
    for ax, n in zip(axes[0], names):
        label, cmap, logs, _ = VARS[n]
        lo, hi = lim[n]
        norm = LogNorm(lo, hi) if logs else Normalize(lo, hi)
        im = ax.pcolormesh(s["x"], s["z"], np.ma.masked_invalid(s["f"][n]), cmap=cmap,
                           norm=norm, shading="flat", rasterized=True)
        ax.add_patch(Ellipse((0, 0), w, h, color="black", zorder=3))
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$M$]")
        if ax is axes[0][0]:
            ax.set_ylabel(r"$z$ [$M$]")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label, location="top")
    fig.suptitle(f"t = {s['t']:.0f} M = {s['t']*s['msun']*T_MSUN_MS:.2f} ms", fontsize=11)
    out = os.path.join(outdir, os.path.basename(fname).replace(".bin", ".png"))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_movie(frames, movie, fps):
    """mp4 through ffmpeg when it exists (module load ffmpeg on clusters), else a GIF."""
    if movie.endswith(".mp4") and shutil.which("ffmpeg"):
        lst = movie + ".frames.txt"
        with open(lst, "w") as fp:
            for f in frames:
                fp.write(f"file '{os.path.abspath(f)}'\nduration {1.0/fps}\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", lst, "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-pix_fmt", "yuv420p", "-vcodec", "libx264", movie], check=True)
        os.remove(lst)
        return movie
    from PIL import Image
    gif = os.path.splitext(movie)[0] + ".gif"
    imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)
    if movie.endswith(".mp4"):
        print("ffmpeg not found: wrote a GIF instead")
    return gif


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="slice .bin files or quoted glob patterns")
    ap.add_argument("-o", "--outdir", default="slices")
    ap.add_argument("--vars", default="dens,temperature,ye",
                    help="comma list of: " + ", ".join(VARS))
    ap.add_argument("--extent", type=float, default=32.0, help="half-width shown [M]")
    ap.add_argument("--lim", action="append", default=[],
                    help="fixed colour limits, e.g. dens:1e6:1e12 (repeatable)")
    ap.add_argument("--movie", help="output movie (.mp4 with ffmpeg, else .gif)")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel frames")
    args = ap.parse_args()

    files = sorted({f for p in args.files for f in glob.glob(p)})
    if not files:
        sys.exit("no files match")
    names = [n.strip() for n in args.vars.split(",") if n.strip()]
    bad = [n for n in names if n not in VARS]
    if bad:
        sys.exit(f"unknown variable(s): {bad}")
    os.makedirs(args.outdir, exist_ok=True)

    lim = default_limits(load(files[0]), [n for n in names if n in load(files[0])["f"]])
    for spec in args.lim:
        n, lo, hi = spec.split(":")
        lim[n] = (float(lo), float(hi))

    jobs = [(f, names, lim, args.extent, args.outdir) for f in files]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            frames = pool.map(plot_one, jobs)
    else:
        frames = [plot_one(j) for j in jobs]
    print(f"{len(frames)} frames in {args.outdir}")
    if args.movie:
        print("movie:", make_movie(frames, args.movie, args.fps))


if __name__ == "__main__":
    main()
