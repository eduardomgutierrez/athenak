#!/usr/bin/env python3
"""Plot a rad_beam run in the style of Fig. 16 of Stone et al. (2024, AthenaK).

CLI: plot_beam_fig16.py BIN_FILE --out PNG [--x0 3] [--width 0.7] [--spread 10]
                                 [--peak 50] [--frame static|tetrad]

Reads the coordinate-frame R^tt (variable r00 of a rad_coord bin output, uniform grid)
and draws log R^tt on inferno over [1, 100] in the window [-0.5, 3.5] x [-0.5, 4], with
the event horizon (white), the source disc and the analytic geodesics of the beam's inner
and outer boundaries (lime).  R^tt is rescaled so that its maximum is --peak, the level of
the paper's source region; the absolute scale of the injection is arbitrary.

The boundary geodesics start at the disc's edges, r0 -+ (width/2)/sqrt(1 + 2/r0) on the
x axis, tilted by -+ spread/2 from the k^r = 0 direction.  --frame says in which frame
that tilt is measured: 'static' (the paper's lines) or 'tetrad', the Kerr-Schild normal
observer in which the source applies its cone.  Small tilts differ by the aberration
factor sqrt(1 - v^2), v = 2/r0 the normal observer's speed relative to the static one.
"""
import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bin_convert
import plot_nu_beam_deflection as pb


def read_uniform(fname, var):
    """Assemble variable `var` of a 2D uniform-grid bin file into (x, y, F[ny, nx])."""
    fd = bin_convert.read_binary(fname)
    nx, ny = fd["Nx1"], fd["Nx2"]
    mx, my = fd["nx1_out_mb"], fd["nx2_out_mb"]
    F = np.zeros((ny, nx))
    for b in range(fd["n_mbs"]):
        lx, ly = fd["mb_logical"][b][0], fd["mb_logical"][b][1]
        F[ly*my:(ly + 1)*my, lx*mx:(lx + 1)*mx] = np.asarray(fd["mb_data"][var][b])[0]
    dx = (fd["x1max"] - fd["x1min"])/nx
    dy = (fd["x2max"] - fd["x2min"])/ny
    x = fd["x1min"] + (np.arange(nx) + 0.5)*dx
    y = fd["x2min"] + (np.arange(ny) + 0.5)*dy
    return x, y, F, fd["time"]


def static_to_tetrad(theta, r0):
    """Normal-frame tilt of a ray tilted by theta from tangential in the static frame."""
    v = 2.0/r0
    # static-frame unit vector (sin theta, cos theta) in (r, phi); boost by v inward
    sr, sp = math.sin(theta), math.cos(theta)
    gam = 1.0/math.sqrt(1.0 - v*v)
    nr = (sr + v)/(1.0 + v*sr)
    np_ = sp/(gam*(1.0 + v*sr))
    return math.atan2(nr, np_) - math.asin(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--x0", type=float, default=3.0)
    ap.add_argument("--width", type=float, default=0.7)
    ap.add_argument("--spread", type=float, default=10.0, help="full cone, degrees")
    ap.add_argument("--peak", type=float, default=50.0)
    ap.add_argument("--frame", choices=["static", "tetrad"], default="static")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    x, y, Rtt, time = read_uniform(args.file, "r00")
    Rtt = Rtt*args.peak/Rtt.max()
    dx, dy = x[1] - x[0], y[1] - y[0]
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_under("black")
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(np.where(Rtt > 0, Rtt, np.nan), origin="lower", cmap=cmap,
                   norm=LogNorm(vmin=1.0, vmax=100.0), interpolation="nearest",
                   extent=[x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2])
    ax.set_facecolor("black")
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(2.0*np.cos(th), 2.0*np.sin(th), color="white", lw=1.5)

    lime = "#7fff00"
    r0 = args.x0
    dxl = 0.5*args.width/math.sqrt(1.0 + 2.0/r0)
    ax.add_patch(Ellipse((r0, 0.0), 2*dxl, args.width, fill=False, ec=lime, lw=3))
    half = math.radians(0.5*args.spread)
    for xl, t in [(r0 - dxl, -half), (r0 + dxl, half)]:
        tt = static_to_tetrad(t, xl) if args.frame == "static" else t
        phi, r, _ = pb.geodesic(xl, 1e3, du0=pb.launch_du0(xl, tt))
        ax.plot(r*np.cos(phi), r*np.sin(phi), color=lime, lw=3)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 4.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/r_g$")
    ax.set_ylabel(r"$y/r_g$")
    if args.title:
        ax.set_title(args.title, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.set_label(r"$R^{tt}$")
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}  (t = {time:g})")


if __name__ == "__main__":
    main()
