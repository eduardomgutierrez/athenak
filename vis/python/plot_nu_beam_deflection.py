#!/usr/bin/env python3
"""Plot the rad_freq_beam runs: beam bending and spectral shift against the geodesic.

CLI: plot_nu_beam_deflection.py DIAG_FILE ... --out PREFIX [--thresh 1e-3] [--fig16]

Each DIAG_FILE is the problem/diag_file of one rad_freq_beam run (x, y and the
tetrad-frame energy density per group, species 0).  One column per run:

  top     log10 of the total energy density, with the null geodesic launched from
          (x0, 0) with k^r = 0 (Schwarzschild, M = 1);
  bottom  the spectrum's energy-weighted mean ln E, sampled along the geodesic,
          as E / E(launch), against the analytic energy for the Kerr-Schild normal
          observer (the solver's tetrad frame) and, for reference, for the static
          observer, sqrt(f0/f) with f = 1 - 2/r.  The tetrad energy of a ray depends
          on its direction, so a cell holding rays spread in angle need not match the
          single geodesic; the source gives every ray the same conserved
          E_inf = nu |n_t| (header e_inf), and the cell mean of that (green) must stay
          at E_inf whatever the angular mix.

The geodesic is integrated from the Binet equation u'' = 3u^2 - u (u = 1/r) with RK4;
no scipy.  On it k^r/E = -b u', b = r0/sqrt(1 - 2/r0), and with H = 2/r the
Kerr-Schild normal observer measures
  E_n/E_inf = (1+H)^(-1/2) (1 + H + H^2 b^2/r^2)/(1 - H k^r/E),
which is (1+H)^(-1/2) (1 + H k^r/E)/(1 - H) rewritten to be regular at r = 2.
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                     "axes.titlesize": 14})
DPI = 170
C_SIM = "#2a78d6"
C_GEO = "#eb6834"
C_REF = "#52514e"
C_INF = "#1baf7a"


def read_diag(fname):
    """Return (header dict, bin_lo, bin_hi, x, y, L[ny, nx], E[ny, nx, nfreq]).

    hdr["Rtt"] holds the coordinate-frame R^tt[ny, nx]."""
    hdr, lo, hi = {}, None, None
    with open(fname) as f:
        for line in f:
            if not line.startswith("#"):
                break
            tok = line[1:].split()
            if tok[0] == "bin_lo":
                lo = np.array(tok[1:], dtype=float)
            elif tok[0] == "bin_hi":
                hi = np.array(tok[1:], dtype=float)
            elif tok[0] == "ordinate":
                hdr["ordinate"] = int(tok[1])
                hdr["kc"] = (float(tok[3]), float(tok[5]), float(tok[7]))
            elif tok[0] == "x0":
                for i, t in enumerate(tok):
                    if t == "=":
                        hdr[tok[i - 1]] = float(tok[i + 1])
    d = np.loadtxt(fname)
    xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
    L = d[:, 2].reshape(len(ys), len(xs))
    hdr["Rtt"] = d[:, 3].reshape(len(ys), len(xs))
    E = d[:, 4:].reshape(len(ys), len(xs), -1)
    return hdr, lo, hi, xs, ys, L, E


def geodesic(r0, rmax, dphi=1e-3, phimax=2*np.pi, du0=0.0):
    """Equatorial null geodesic from (r0, phi = 0) with u' = du0 (0: k^r = 0), k^phi > 0."""
    def rhs(u, du):
        return du, 3.0*u*u - u
    u, du, phi = 1.0/r0, du0, 0.0
    out = [(phi, u, du)]
    while phi < phimax and 2.0 < 1.0/u < rmax:
        k1 = rhs(u, du)
        k2 = rhs(u + 0.5*dphi*k1[0], du + 0.5*dphi*k1[1])
        k3 = rhs(u + 0.5*dphi*k2[0], du + 0.5*dphi*k2[1])
        k4 = rhs(u + dphi*k3[0], du + dphi*k3[1])
        u += dphi/6.0*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        du += dphi/6.0*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        phi += dphi
        out.append((phi, u, du))
    phi, u, du = np.array(out).T
    keep = u < 0.5
    return phi[keep], 1.0/u[keep], du[keep]


def launch_du0(xl, theta):
    """u'(0) for a ray launched at (xl, 0, 0), tilted by theta (rad, + = outward) in the
    Kerr-Schild normal frame from the direction with k^r = 0.  With H = 2/xl the k^r = 0
    ray points at psi = asin(H) from e_y towards e_x in that frame, and
    k^x/nu = (sin psi' - H)/sqrt(1+H), k^y/nu = cos psi' for psi' = psi + theta."""
    H = 2.0/xl
    psi = math.asin(H) + theta
    kx = (math.sin(psi) - H)/math.sqrt(1.0 + H)
    ky = math.cos(psi)
    return -kx/(xl*ky)


def energies(r, du, r0):
    """Normal- and static-observer energies over E_inf along the geodesic."""
    b = r0/math.sqrt(1.0 - 2.0/r0)
    kr = -b*du
    H = 2.0/r
    e_n = (1.0 + H + H*H*b*b/(r*r))/(1.0 - H*kr)/np.sqrt(1.0 + H)
    e_s = 1.0/np.sqrt(1.0 - H)
    return e_n, e_s


def bilinear(xs, ys, F, px, py):
    """Bilinear interpolation of F[ny, nx, ...] at points (px, py); NaN outside."""
    fx = np.interp(px, xs, np.arange(len(xs)), left=np.nan, right=np.nan)
    fy = np.interp(py, ys, np.arange(len(ys)), left=np.nan, right=np.nan)
    ok = np.isfinite(fx) & np.isfinite(fy)
    out = np.full((len(px),) + F.shape[2:], np.nan)
    i0 = np.clip(np.floor(fx[ok]).astype(int), 0, len(xs) - 2)
    j0 = np.clip(np.floor(fy[ok]).astype(int), 0, len(ys) - 2)
    tx = (fx[ok] - i0)[:, None]
    ty = (fy[ok] - j0)[:, None]
    out[ok] = ((1 - tx)*(1 - ty)*F[j0, i0] + tx*(1 - ty)*F[j0, i0 + 1]
               + (1 - tx)*ty*F[j0 + 1, i0] + tx*ty*F[j0 + 1, i0 + 1])
    return out


def bin_centres(lo, hi):
    """Geometric bin centres (a [0, nu_min] photon bin gets hi/2)."""
    return np.where(lo > 0.0, np.sqrt(np.maximum(lo, 1e-300)*hi), 0.5*hi)


def plot_fig16(hdr, xs, ys, out, peak=50.0):
    """One panel in the style of Fig. 16 of Stone et al. (2024): log R^tt on inferno over
    [1, 100], window [-0.5, 3.5] x [-0.5, 4], white horizon, lime source disc and the two
    outer beam-boundary geodesics.  R^tt is rescaled so its maximum (in the source disc)
    is `peak`, the level of the paper's source region; the absolute scale is arbitrary."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Ellipse
    r0 = hdr["x0"]
    Rtt = hdr["Rtt"]*peak/Rtt_max(hdr)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_under("black")
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(np.where(Rtt > 0, Rtt, np.nan), origin="lower", cmap=cmap,
                   norm=LogNorm(vmin=1.0, vmax=100.0), interpolation="nearest",
                   extent=[xs[0] - dx/2, xs[-1] + dx/2, ys[0] - dy/2, ys[-1] + dy/2])
    ax.set_facecolor("black")
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(2.0*np.cos(th), 2.0*np.sin(th), color="white", lw=1.5)
    lime = "#7fff00"
    dxl = 0.5*hdr["width"]/math.sqrt(1.0 + 2.0/r0)
    ax.add_patch(Ellipse((r0, 0.0), 2*dxl, hdr["width"], fill=False, ec=lime, lw=3))
    half = math.radians(0.5*hdr["spread"])
    rmax = 1e3
    for xl, t in [(r0 - dxl, -half), (r0 + dxl, half)]:
        pb, rb, _ = geodesic(xl, rmax, du0=launch_du0(xl, t))
        ax.plot(rb*np.cos(pb), rb*np.sin(pb), color=lime, lw=3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 4.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/r_g$")
    ax.set_ylabel(r"$y/r_g$")
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.set_label(r"$R^{tt}$")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


def Rtt_max(hdr):
    return float(np.max(hdr["Rtt"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix (writes PREFIX.png)")
    ap.add_argument("--fig16", action="store_true",
                    help="also write PREFIX_fig16.png per run, in the style of Fig. 16 of "
                         "Stone et al. (2024)")
    ap.add_argument("--thresh", type=float, default=1e-3,
                    help="sample the spectrum only where E_tot > thresh*E_tot(launch)")
    args = ap.parse_args()

    runs = [read_diag(f) for f in args.files]
    runs.sort(key=lambda r: r[0]["x0"])
    if args.fig16:
        for hdr, lo, hi, xs, ys, L, E in runs:
            plot_fig16(hdr, xs, ys, f"{args.out}_fig16_x{hdr['x0']:g}.png")
    n = len(runs)
    grey = runs[0][1] is None
    if grey:
        fig, axes = plt.subplots(1, n, figsize=(5.2*n, 5.0), squeeze=False)
    else:
        fig, axes = plt.subplots(2, n, figsize=(5.2*n, 9.4),
                                 gridspec_kw={"height_ratios": [1.0, 0.62]}, squeeze=False)
    summary = []
    for c, (hdr, lo, hi, xs, ys, L, E) in enumerate(runs):
        r0 = hdr["x0"]
        Etot = E.sum(axis=2)
        Rtt = hdr["Rtt"]
        rmax = min(abs(xs[0]), xs[-1], abs(ys[0]), ys[-1])*math.sqrt(2.0)
        phi, r, du = geodesic(r0, rmax)
        gx, gy = r*np.cos(phi), r*np.sin(phi)

        # top: beam over the geodesic
        ax = axes[0, c]
        top = np.log10(max(Rtt.max(), 1e-300))
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        im = ax.imshow(np.log10(np.maximum(Rtt, 1e-300)), origin="lower", cmap="Blues",
                       vmin=top - 4.0, vmax=top, interpolation="nearest",
                       extent=[xs[0] - dx/2, xs[-1] + dx/2, ys[0] - dy/2, ys[-1] + dy/2])
        ax.add_patch(Circle((0, 0), 2.0, color="black", zorder=3))
        ax.add_patch(Circle((0, 0), 3.0, fill=False, ls=":", lw=1.0, color=C_REF,
                            zorder=3))
        ax.plot(gx, gy, color=C_GEO, lw=2.0, ls="--", zorder=4,
                label=r"geodesic, $k^r = 0$")
        if hdr.get("ordinate", -1) < 0:
            # beam boundaries: the disc's two edges x0 -+ w/2 (proper), each with the two
            # cone edges -+ spread/2, as in Fig. 16 of Stone et al. (2024)
            dxl = 0.5*hdr["width"]/math.sqrt(1.0 + 2.0/r0)
            half = math.radians(0.5*hdr["spread"])
            for k, (xl, th) in enumerate([(r0 - dxl, -half), (r0 - dxl, half),
                                          (r0 + dxl, -half), (r0 + dxl, half)]):
                pb, rb, _ = geodesic(xl, rmax, du0=launch_du0(xl, th))
                ax.plot(rb*np.cos(pb), rb*np.sin(pb), color=C_INF, lw=1.2, zorder=4,
                        label="beam-boundary geodesics" if k == 0 else None)
        if hdr.get("ordinate", -1) >= 0:
            kx, ky, kz = hdr["kc"]
            p2, r2, _ = geodesic(r0, rmax, du0=-kx/(r0*ky))
            ax.plot(r2*np.cos(p2), r2*np.sin(p2), color=C_INF, lw=2.0, zorder=4,
                    label=rf"geodesic of launched ordinate ($k^z/k^y = {kz/ky:.2f}$)")
        ax.plot([r0], [0.0], marker="o", ms=8, mfc=C_GEO, mec="white", mew=2, zorder=5)
        ax.set_xlim(xs[0] - dx/2, xs[-1] + dx/2)
        ax.set_ylim(ys[0] - dy/2, ys[-1] + dy/2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x\,[M]$")
        if c == 0:
            ax.set_ylabel(r"$y\,[M]$")
        if c == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=9)
        ax.set_title(rf"$x_0 = {r0:g}\,M$  ($t = {hdr.get('time', np.nan):.0f}\,M$)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\log_{10} R^{tt}$ (coordinate frame)")
        if grey:
            continue

        # bottom: spectral shift along the geodesic
        ax = axes[1, c]
        e_n, e_s = energies(r, du, r0)
        lnc = np.log(bin_centres(lo, hi))
        Eg = bilinear(xs, ys, E, gx, gy)
        Et = Eg.sum(axis=1)
        E_launch = bilinear(xs, ys, Etot[:, :, None], np.array([r0]), np.array([0.0]))[0, 0]
        good = np.isfinite(Et) & (Et > args.thresh*E_launch)
        mean_ln = np.full(len(phi), np.nan)
        mean_ln[good] = (Eg[good]*lnc).sum(axis=1)/Et[good]
        ratio_sim = np.exp(mean_ln - mean_ln[0])
        # conserved energy E_inf = nu |n_t|, per ray; its cell mean must stay at the source's
        Lg = bilinear(xs, ys, L[:, :, None], gx, gy)[:, 0]
        mean_inf = np.full(len(phi), np.nan)
        mean_inf[good] = mean_ln[good] + Lg[good]/Et[good]
        ratio_inf = np.exp(mean_inf - math.log(hdr["e_inf"]))
        deg = np.degrees(phi)
        ax.plot(deg, e_s/e_s[0], color=C_REF, lw=1.0, ls="--",
                label=r"static observer $\sqrt{f_0/f}$")
        ax.plot(deg, e_n/e_n[0], color=C_GEO, lw=2.0, label="normal observer, geodesic")
        ax.plot(deg, ratio_sim, color=C_SIM, lw=0, marker="o", ms=3.5, markevery=25,
                label=r"simulation, $\exp\langle\ln E\rangle$")
        ax.plot(deg, ratio_inf, color=C_INF, lw=0, marker="s", ms=3.5, markevery=25,
                label=r"simulation, $E_\infty = \exp\langle\ln(\nu|n_t|)\rangle$")
        ax.set_xlabel(r"$\varphi$ along the geodesic [deg]")
        if c == 0:
            ax.set_ylabel(r"$E\,/\,E_{\rm launch}$")
            ax.legend(loc="best", frameon=False)
        lo_y = np.nanmin([np.nanmin(ratio_sim), np.nanmin(ratio_inf), (e_n/e_n[0]).min()])
        hi_y = np.nanmax([np.nanmax(ratio_sim), np.nanmax(ratio_inf), (e_n/e_n[0]).max()])
        pad = 0.08*max(hi_y - lo_y, 0.05)
        ax.set_ylim(lo_y - pad, max(hi_y, 1.0) + pad)
        ax.grid(alpha=0.25)

        err = ratio_sim/(e_n/e_n[0]) - 1.0
        summary.append((r0, np.nanmax(np.abs(err[good])) if good.any() else np.nan,
                        np.nanmax(np.abs(ratio_inf[good] - 1.0)) if good.any() else np.nan,
                        deg[good].max() if good.any() else np.nan))

    fig.tight_layout()
    out = args.out + ".png"
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")
    if grey:
        return
    for r0, emax, imax, dmax in summary:
        print(f"x0 = {r0:g}: max |E_sim/E_geo - 1| = {emax:.3e}, "
              f"max |E_inf/E_inf(launch) - 1| = {imax:.3e}, over phi <= {dmax:.0f} deg")


if __name__ == "__main__":
    main()
