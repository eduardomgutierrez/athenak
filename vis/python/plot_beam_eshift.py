#!/usr/bin/env python3
"""Energy shift along the Schwarzschild photon-orbit beam (bh_beam_schw_multifreq).

CLI: plot_beam_eshift.py RAD_COORD_BIN RADNU_COORD_BIN --out PNG
                         [--x0 3] [--width 0.7] [--spread 10] [--e-peak 1] [--sigma-ln 0.3]
                         [--thresh 0.02]

The frequency bins are tetrad-frame (Kerr-Schild normal observer) energies, so that is the
frame of the shift.  In each cell the beam's energy is exp<ln nu>, the mean of ln nu over
the groups weighted by each group's coordinate-frame R^tt (rnu00), divided by the same mean
of the injected spectrum on the same bins (the source gives every ray the same tetrad
spectrum, a Gaussian in ln nu).

Left: map of E/E_source where R^tt > thresh*max, with the central geodesic (k^r = 0, the
photon orbit) and the two boundary geodesics of plot_beam_fig16.py (disc edges, -+ spread/2
in the static frame).  Middle: the same map for exact transport, ray-traced (see
raytrace_reference).  Right: E/E_source sampled along those three geodesics, simulation
(markers) and ray-traced reference (dotted), against the single ray's analytic
E_n(P)/E_n(launch) (lines).  The drawn lines are not the envelope of the injected rays:
every source cell aims its cone along coordinate +y, which below the axis points inward
and above it outward, so cells near a line hold rays with larger shifts than the line's.  On a geodesic with impact
parameter b, k^r/E_inf = -b u' and
  E_n/E_inf = (1 + H + H^2 b^2/r^2)/((1 - H k^r/E_inf) sqrt(1 + H)),  H = 2/r.
"""
import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radnu_read
import plot_nu_beam_deflection as pb
import plot_beam_fig16 as pf

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10})
COLORS = {"inner": "#2a78d6", "central": "#eb6834", "outer": "#1baf7a"}


def photon_edges(nu_min, nu_max, nfreq):
    """Photon log grid: [0, nu_min], nfreq-2 log groups, [nu_max, inf)."""
    inner = np.exp(np.linspace(np.log(nu_min), np.log(nu_max), nfreq - 1))
    return np.concatenate([[0.0], inner, [np.inf]])


def neutrino_edges(nu_min, nu_max, nfreq):
    """Neutrino (nurates) grid: nfreq geometric lower edges over [nu_min, nu_max]; the top
    group is [nu_max, nu_max e^dln] for its centre, and runs to infinity for the source."""
    lo = np.exp(np.linspace(np.log(nu_min), np.log(nu_max), nfreq))
    return np.concatenate([lo, [np.inf]])


def centres(edges):
    lo, hi = edges[:-1], edges[1:]
    c = np.sqrt(np.where(lo > 0, lo, 1.0)*np.where(np.isfinite(hi), hi, 1.0))
    if lo[0] == 0.0:
        c[0] = 0.5*hi[0]
    c[-1] = lo[-1]*math.sqrt(edges[-2]/edges[-3])
    return c


def injected_mean_ln(edges, e_peak, sig):
    def cdf(e):
        if e <= 0:
            return 0.0
        if not np.isfinite(e):
            return 1.0
        return 0.5*(1.0 + math.erf((math.log(e) - math.log(e_peak))/(math.sqrt(2)*sig)))
    frac = np.array([cdf(h) - cdf(l) for l, h in zip(edges[:-1], edges[1:])])
    return np.sum(np.log(centres(edges))*frac)/np.sum(frac)


def en_ratio(r, du, u0, du0):
    """E_n(P)/E_n(launch) along a geodesic, b from the launch point."""
    b = 1.0/math.sqrt(du0**2 + u0**2*(1.0 - 2.0*u0))

    def en(r, du):
        H = 2.0/r
        kr = -b*du
        return (1.0 + H + H*H*b*b/(r*r))/((1.0 - H*kr)*np.sqrt(1.0 + H))
    return en(r, du)/en(1.0/u0, du0)


def launch(x, y, theta):
    """In-plane null ray at (x, y) whose coordinate spatial part is +y, tilted by theta
    (rad, + = outward) in the KS normal frame.  Returns (r0, phi0, u0, u'0)."""
    r = math.hypot(x, y)
    ph0 = math.atan2(y, x)
    H = 2.0/r
    kr_c, kT_c = math.sin(ph0), math.cos(ph0)
    a, b, c = -(1 - H), 2*H*kr_c, (1 + H)*kr_c**2 + kT_c**2
    kt = (-b - math.sqrt(b*b - 4*a*c))/(2*a)
    al, br = 1/math.sqrt(1 + H), H/(1 + H)
    nu = al*kt
    psi = math.atan2((kr_c/nu + br/al)*math.sqrt(1 + H), kT_c/nu) + theta
    kr = math.sin(psi)/math.sqrt(1 + H) - br/al
    kT = math.cos(psi)
    return r, ph0, 1.0/r, -kr/(r*kT)


def raytrace_reference(x, y, x0, width, spread, dl=0.02, ntilt=21, dphi=2e-3):
    """Exact-transport map of exp<ln(E/E_source)> on the cell centres (x, y).

    In-plane rays are launched from a lattice (spacing dl) over the source disc
    (KS proper distance < width/2 from (x0, 0)), each aimed along coordinate +y tilted by
    theta in [-spread/2, spread/2] (KS normal frame) with weight sqrt(1 - (2 theta/spread)^2),
    the projection of a uniform cone onto the plane.  Each ray deposits, per unit
    coordinate path length, weight w = (E/E_launch)^4 (frequency-integrated I, since
    I_nu/nu^3 is invariant) and w ln(E/E_launch).  Rays stop at r = 2 or outside the grid."""
    pts = []
    for px in np.arange(x0 - width, x0 + width, dl):
        for py in np.arange(-width, width, dl):
            r = math.hypot(px, py)
            H = 2.0/r
            l1, l2 = px/r, py/r
            d1, d2 = px - x0, py
            dsq = d1*d1 + d2*d2 + H*(l1*d1 + l2*d2)**2
            if dsq < (0.5*width)**2:
                pts.append((px, py))
    half = math.radians(0.5*spread)
    tilts = np.linspace(-half, half, ntilt)
    wt = np.sqrt(np.clip(1.0 - (tilts/half)**2, 0.0, None)) + 1e-12
    r0, ph0, u0, du0, wr = [], [], [], [], []
    for px, py in pts:
        for t, w in zip(tilts, wt):
            a, b, c, d = launch(px, py, t)
            r0.append(a); ph0.append(b); u0.append(c); du0.append(d); wr.append(w)
    r0, ph0, u0, du0, wr = map(np.array, (r0, ph0, u0, du0, wr))
    b = 1.0/np.sqrt(du0**2 + u0**2*(1.0 - 2.0*u0))

    def en(u, du):
        H = 2.0*u
        kr = -b*du
        return (1.0 + H + H*H*b*b*u*u)/((1.0 - H*kr)*np.sqrt(1.0 + H))
    e_launch = en(u0, du0)

    dx, dy = x[1] - x[0], y[1] - y[0]
    W = np.zeros((len(y), len(x)))
    L = np.zeros_like(W)
    u, du = u0.copy(), du0.copy()
    phi = np.zeros_like(u)
    alive = np.ones(len(u), dtype=bool)
    xo, yo = r0*np.cos(ph0), r0*np.sin(ph0)
    for _ in range(int(2*np.pi/dphi)):
        def f(u, du):
            return du, 3.0*u*u - u
        k1 = f(u, du)
        k2 = f(u + 0.5*dphi*k1[0], du + 0.5*dphi*k1[1])
        k3 = f(u + 0.5*dphi*k2[0], du + 0.5*dphi*k2[1])
        k4 = f(u + dphi*k3[0], du + dphi*k3[1])
        u = u + dphi/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        du = du + dphi/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        phi += dphi
        rr = 1.0/u
        xn, yn = rr*np.cos(phi + ph0), rr*np.sin(phi + ph0)
        inside = ((u < 0.5) & (xn >= x[0] - dx/2) & (xn <= x[-1] + dx/2)
                  & (yn >= y[0] - dy/2) & (yn <= y[-1] + dy/2))
        alive &= inside
        if not alive.any():
            break
        ds = np.hypot(xn - xo, yn - yo)
        ratio = en(u, du)/e_launch
        w = wr*ratio**4*ds
        i = np.clip(((xn - x[0] + dx/2)/dx).astype(int), 0, len(x) - 1)
        j = np.clip(((yn - y[0] + dy/2)/dy).astype(int), 0, len(y) - 1)
        np.add.at(W, (j[alive], i[alive]), w[alive])
        np.add.at(L, (j[alive], i[alive]), (w*np.log(ratio))[alive])
        xo, yo = xn, yn
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.exp(L/W)
    return np.where(W > 0, ref, np.nan), W, L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rad_coord")
    ap.add_argument("radnu_coord")
    ap.add_argument("--out", required=True)
    ap.add_argument("--x0", type=float, default=3.0)
    ap.add_argument("--width", type=float, default=0.7)
    ap.add_argument("--spread", type=float, default=10.0)
    ap.add_argument("--e-peak", type=float, default=None,
                    help="default: <rad_srcterms>/e_peak from the file header")
    ap.add_argument("--sigma-ln", type=float, default=None)
    ap.add_argument("--species", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.02,
                    help="show/sample only where R^tt > thresh*max (paper's 1/50)")
    args = ap.parse_args()

    x, y, Rtt, time = pf.read_uniform(args.rad_coord, "r00")
    d = radnu_read.read_radnu(args.radnu_coord)
    Eg = d["data"][args.species, :, 0, 0]             # (nfreq, ny, nx): rnu00 per group
    hget = lambda b, k, dflt: (radnu_read._get_from_header(d["header"], b, k)
                               if any(l.strip().startswith(k) for l in d["header"]) else dflt)
    neutrino = hget("<radiation>", "radiation_type", "photon").strip() == "neutrino"
    edges = (neutrino_edges if neutrino else photon_edges)(d["nu_min"], d["nu_max"],
                                                           d["nfreq"])
    e_peak = args.e_peak if args.e_peak is not None else \
        float(hget("<rad_srcterms>", "e_peak", 1.0))
    sigma_ln = args.sigma_ln if args.sigma_ln is not None else \
        float(hget("<rad_srcterms>", "sigma_ln", 0.3))
    lnc = np.log(centres(edges))
    ln_src = injected_mean_ln(edges, e_peak, sigma_ln)
    print(f"{'neutrino' if neutrino else 'photon'} grid, species {args.species}, "
          f"e_peak {e_peak:g}, sigma_ln {sigma_ln:g}")

    tot = Eg.sum(axis=0)
    good = Rtt > args.thresh*Rtt.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.exp((Eg*lnc[:, None, None]).sum(axis=0)/tot - ln_src)
    ratio_map = np.where(good, ratio, np.nan)

    # geodesics: central (k^r = 0) and the two boundaries (static-frame -+ spread/2)
    r0 = args.x0
    dxl = 0.5*args.width/math.sqrt(1.0 + 2.0/r0)
    half = math.radians(0.5*args.spread)
    geos = {"central": (r0, 0.0)}
    for name, xl, t in [("inner", r0 - dxl, -half), ("outer", r0 + dxl, half)]:
        geos[name] = (xl, pb.launch_du0(xl, pf.static_to_tetrad(t, xl)))

    ref, Wref, Lref = raytrace_reference(x, y, args.x0, args.width, args.spread)
    ref = np.where(Wref > args.thresh*Wref.max(), ref, np.nan)

    fig, (ax, cx, bx) = plt.subplots(1, 3, figsize=(20, 6.2),
                                     gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    dx, dy = x[1] - x[0], y[1] - y[0]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#1a1a19")
    lim = 0.25
    im = ax.imshow(ratio_map, origin="lower", cmap=cmap,
                   norm=TwoSlopeNorm(vcenter=1.0, vmin=1.0 - lim, vmax=1.0 + lim),
                   interpolation="nearest",
                   extent=[x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2])
    cx.imshow(ref, origin="lower", cmap=cmap,
              norm=TwoSlopeNorm(vcenter=1.0, vmin=1.0 - lim, vmax=1.0 + lim),
              interpolation="nearest",
              extent=[x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2])
    th = np.linspace(0, 2*np.pi, 400)
    for a in (ax, cx):
        a.plot(2.0*np.cos(th), 2.0*np.sin(th), color="white", lw=1.2)
    for name, (xl, du0) in geos.items():
        phi, r, du = pb.geodesic(xl, 1e3, du0=du0)
        for a in (ax, cx):
            a.plot(r*np.cos(phi), r*np.sin(phi), color=COLORS[name], lw=2.0,
                   ls="--" if name == "central" else "-")

        # sample along the geodesic inside the window
        gx, gy = r*np.cos(phi), r*np.sin(phi)
        inwin = (gx >= x[0]) & (gx <= x[-1]) & (gy >= y[0]) & (gy <= y[-1])
        Es = pb.bilinear(x, y, np.moveaxis(Eg, 0, -1), gx[inwin], gy[inwin])
        Rs = pb.bilinear(x, y, Rtt[:, :, None], gx[inwin], gy[inwin])[:, 0]
        ok = Rs > args.thresh*Rtt.max()
        sim = np.full(inwin.sum(), np.nan)
        sim[ok] = np.exp((Es[ok]*lnc).sum(axis=1)/Es[ok].sum(axis=1) - ln_src)
        ana = en_ratio(r[inwin], du[inwin], 1.0/xl, du0)
        deg = np.degrees(phi[inwin])
        bx.plot(deg, ana, color=COLORS[name], lw=2.0,
                ls="--" if name == "central" else "-", label=f"{name}, geodesic")
        bx.plot(deg, sim, color=COLORS[name], lw=0, marker="o", ms=4, markevery=40,
                label=f"{name}, simulation")
        # interpolate the sums, not the ratio, so empty cells do not drag it to zero
        WL = pb.bilinear(x, y, np.stack([Wref, Lref], axis=-1), gx[inwin], gy[inwin])
        with np.errstate(invalid="ignore", divide="ignore"):
            rs = np.exp(WL[:, 1]/WL[:, 0])
        rs = np.where(ok & (WL[:, 0] > 0), rs, np.nan)
        bx.plot(deg, rs, color=COLORS[name], lw=1.5, ls=":",
                label=f"{name}, exact transport")
        print(f"{name:8s}: phi <= {deg[-1]:5.0f} deg  single ray {ana.min():.3f}..{ana.max():.3f}"
              f"  sim {np.nanmin(sim):.3f}..{np.nanmax(sim):.3f}"
              f"  exact transport {np.nanmin(rs):.3f}..{np.nanmax(rs):.3f}")

    for a in (ax, cx):
        a.set_xlim(-0.5, 3.5)
        a.set_ylim(-0.5, 4.0)
        a.set_aspect("equal")
        a.set_facecolor("#1a1a19")
        a.set_xlabel(r"$x/r_g$")
    ax.set_ylabel(r"$y/r_g$")
    ax.set_title(rf"simulation, $E/E_{{\rm source}}$ ($t = {time:g}\,M$)")
    cx.set_title(r"exact transport (ray-traced, in-plane)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"$\exp\langle\ln\nu\rangle / E_{\rm source}$")

    bx.axhline(1.0, color="#52514e", lw=0.8)
    bx.set_xlabel(r"$\varphi$ along the geodesic [deg]")
    bx.set_ylabel(r"$E/E_{\rm launch}$ (KS normal observer)")
    bx.grid(alpha=0.25)
    bx.legend(frameon=False, ncol=3, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
