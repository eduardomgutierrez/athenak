#!/usr/bin/env python3
"""Where the torus emits and absorbs neutrinos: x-z maps of the radnu_src output (net
cooling, emission, the absorbed fraction and dYe/dt), and the volume-integrated rates.

radnu_src is the exchange of the multifrequency source per species s, averaged over each
step with the integrator's stage weights, per unit proper volume and proper time:
  Qnet:s  net cooling u^mu G_mu, > 0 where the fluid loses energy  [code energy dens/time]
  etaE:s  comoving energy emissivity, summed over groups              [same]
  Rnet:s  net number emission                                          [fm^-3 / code time]
  etaN:s  comoving number emissivity                                   [same]
Absorption is emission minus net, so (etaE - Qnet)/etaE is the fraction of the emitted
energy that is absorbed again in the same cell (0: transparent, 1: equilibrium, > 1: net
heating), and dYe/dt = m_b (Rnet:1 - Rnet:0)/rho.

Slices (radnu_src with slice_x2 = 0) take their density contours from the matter slice
written at the same time (--matter).  3D radnu_src files go to --integrate, which sums
each species over the meshblocks: int Q dV, the fluid-frame cooling luminosity.

Examples:
  python vis/python/torus_nu_src.py 'RUN/bin/*radnu_src_xz*.bin' \\
      --matter 'RUN/bin/*mhd_w_bcc_xz*.bin' -o RUN/src --extent 40 --movie RUN/src.mp4 -j 8
  python vis/python/torus_nu_src.py --integrate 'RUN/bin/*.radnu_src.*.bin' -o RUN/src
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
from torus_slices import header_value, horizon, make_movie  # noqa: E402

RHO_MSUN = 6.1764e17     # g/cm^3 per code density unit at M = 1 Msun (scales as M^-2)
T_MSUN_S = 4.9255e-6     # s per code time unit at M = 1 Msun (scales as M)
L_MSUN_CM = 1.4766e5     # cm per code length unit at M = 1 Msun (scales as M)
C2 = 8.98755e20          # c^2 in cm^2/s^2
M_B = 1.674927e-24       # baryon mass in g (the neutron mass, as in the CompOSE tables)
SPECIES = [r"$\nu_e$", r"$\bar\nu_e$", r"$\nu_x$", r"$\bar\nu_x$"]


def units(msun):
    """Code -> cgs for an energy rate density, a number rate density and a time."""
    t_s = T_MSUN_S*msun
    erate = RHO_MSUN/msun**2*C2/t_s           # erg cm^-3 s^-1
    nrate = 1.0e39/t_s                        # cm^-3 s^-1 (from fm^-3 per code time)
    return erate, nrate, t_s


def nspecies(names):
    return sum(1 for n in names if n.startswith("Qnet:"))


def load_slice(fname):
    d = bc.read_binary_as_athdf(fname)
    hdr = bc.read_binary(fname)["header"]
    msun = float(header_value(hdr, "mhd", "M_Msun", 1.0))
    spin = float(header_value(hdr, "coord", "a", 0.0))
    sq = lambda k: np.asarray(d[k], dtype=np.float64)[:, 0, :]      # (nz, nx), y = 0
    names = [k for k in d if ":" in k]
    erate, nrate, _ = units(msun)
    nsp = nspecies(names)
    f = {}
    for s in range(nsp):
        f[f"Q{s}"] = sq(f"Qnet:{s}")*erate
        f[f"E{s}"] = sq(f"etaE:{s}")*erate
        f[f"R{s}"] = sq(f"Rnet:{s}")*nrate
    return dict(t=float(d["Time"]), msun=msun, spin=spin, nsp=nsp,
                x=np.asarray(d["x1f"]), z=np.asarray(d["x3f"]), f=f)


def load_rho(fname):
    d = bc.read_binary_as_athdf(fname)
    hdr = bc.read_binary(fname)["header"]
    msun = float(header_value(hdr, "mhd", "M_Msun", 1.0))
    return float(d["Time"]), np.asarray(d["dens"], dtype=np.float64)[:, 0, :]*RHO_MSUN/msun**2


def panels(s, rho, species):
    """The four maps, for one species or (species = None) summed over all of them."""
    sel = range(s["nsp"]) if species is None else [species]
    Q = sum(s["f"][f"Q{i}"] for i in sel)
    E = sum(s["f"][f"E{i}"] for i in sel)
    with np.errstate(divide="ignore", invalid="ignore"):
        absf = np.where(E > 0, (E - Q)/E, np.nan)
    out = {"Q": Q, "E": E, "absf": absf}
    if rho is not None and s["nsp"] >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["dye"] = M_B*(s["f"]["R1"] - s["f"]["R0"])/rho
    return out


def symlog_ticks(lin, vmax):
    """At most two decades either side of zero, so the labels do not overlap."""
    hi = int(np.floor(np.log10(vmax))); lo = int(np.ceil(np.log10(lin)))
    step = max(1, (hi - lo)//2)
    dec = [10.0**k for k in range(hi, lo - 1, -step)][:2]
    return sorted([-d for d in dec] + [0.0] + dec)


def plot_one(job):
    fname, matter, lim, extent, outdir, species = job
    s = load_slice(fname)
    rho = load_rho(matter)[1] if matter else None
    p = panels(s, rho, species)
    spec = r"all species" if species is None else SPECIES[species]
    spec_q = {"Q": (rf"net cooling $Q^-$ ({spec}) [erg cm$^{{-3}}$ s$^{{-1}}$]", "RdBu_r",
                    SymLogNorm(lim["Q"][0], vmin=-lim["Q"][1], vmax=lim["Q"][1])),
              "E": (rf"emission $\eta$ ({spec}) [erg cm$^{{-3}}$ s$^{{-1}}$]", "magma",
                    LogNorm(*lim["E"])),
              "absf": (r"absorbed fraction $(\eta - Q^-)/\eta$", "viridis",
                       Normalize(0.0, 1.0)),
              "dye": (r"$dY_e/dt$ [s$^{-1}$]", "PuOr_r",
                      SymLogNorm(lim["dye"][0], vmin=-lim["dye"][1], vmax=lim["dye"][1]))}
    names = [n for n in ("Q", "E", "absf", "dye") if n in p]
    fig, axes = plt.subplots(1, len(names), figsize=(3.7*len(names) + 0.6, 4.3),
                             squeeze=False, constrained_layout=True)
    w, h = horizon(s["spin"])
    xc = 0.5*(s["x"][1:] + s["x"][:-1]); zc = 0.5*(s["z"][1:] + s["z"][:-1])
    for ax, n in zip(axes[0], names):
        label, cmap, norm = spec_q[n]
        a = np.ma.masked_invalid(p[n])
        if n != "Q" and n != "E":
            a = np.ma.masked_where(~(p["E"] > lim["E"][0]), a)     # only where it emits
        im = ax.pcolormesh(s["x"], s["z"], a, cmap=cmap, norm=norm, shading="flat",
                           rasterized=True)
        if rho is not None:
            ax.contour(xc, zc, np.log10(np.maximum(rho, 1e-30)), levels=[9, 10, 11, 12],
                       colors="k", linewidths=0.5, alpha=0.6)
        ax.add_patch(Ellipse((0, 0), w, h, color="black", zorder=3))
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$M$]")
        if ax is axes[0][0]:
            ax.set_ylabel(r"$z$ [$M$]")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, label=label, location="top")
        if isinstance(norm, SymLogNorm):
            cb.set_ticks(symlog_ticks(lim[n][0], lim[n][1]))
    cont = r"; contours $\log_{10}\rho$ = 9, 10, 11, 12" if rho is not None else ""
    fig.suptitle(f"t = {s['t']:.0f} M = {s['t']*s['msun']*T_MSUN_S*1e3:.2f} ms{cont}",
                 fontsize=11)
    tag = "" if species is None else f"_s{species}"
    out = os.path.join(outdir, os.path.basename(fname).replace(".bin", f"{tag}.png"))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def default_limits(s, rho, species):
    """Colour limits from one frame, so a movie does not flicker."""
    p = panels(s, rho, species)
    lim = {}
    qmax = np.nanmax(np.abs(p["Q"])) if np.any(p["Q"]) else 1.0
    emax = np.nanmax(p["E"]) if np.any(p["E"] > 0) else 1.0
    lim["Q"] = (qmax*1e-6, qmax)
    lim["E"] = (emax*1e-6, emax)
    if "dye" in p:
        d = np.abs(p["dye"][np.isfinite(p["dye"])])
        dmax = d.max() if d.size and d.max() > 0 else 1.0
        lim["dye"] = (dmax*1e-4, dmax)
    return lim


def match_matter(files, matter):
    """For each slice, the matter slice written at the same time (or None)."""
    if not matter:
        return [None]*len(files)
    times = {}
    for m in matter:
        times[round(bc.read_binary(m)["time"], 6)] = m
    keys = np.array(sorted(times))
    out = []
    for f in files:
        t = bc.read_binary(f)["time"]
        k = keys[np.argmin(np.abs(keys - t))]
        out.append(times[k] if abs(k - t) < 1e-6*max(1.0, abs(t)) else None)
    return out


def integrate(files, outdir):
    """int Q dV and int eta dV per species over the whole mesh, against time.  The
    meshblocks of a bin output are the leaves of the tree, so they tile the domain."""
    rows = []
    for f in files:
        fd = bc.read_binary(f)
        hdr = fd["header"]
        msun = float(header_value(hdr, "mhd", "M_Msun", 1.0))
        erate, nrate, t_s = units(msun)
        vol = (L_MSUN_CM*msun)**3
        nsp = nspecies(fd["var_names"])
        tot = np.zeros((4, nsp))
        for b in range(fd["n_mbs"]):
            g = fd["mb_geometry"][b]
            nz, ny, nx = fd["mb_data"]["Qnet:0"][b].shape
            dv = (g[1]-g[0])/nx*(g[3]-g[2])/ny*(g[5]-g[4])/nz
            for s in range(nsp):
                for q, key in enumerate(("Qnet", "etaE", "Rnet", "etaN")):
                    tot[q, s] += fd["mb_data"][f"{key}:{s}"][b].sum()*dv
        tot[0:2] *= erate*vol; tot[2:4] *= nrate*vol
        rows.append((fd["time"], fd["time"]*t_s*1e3, tot))
        print(f"t = {fd['time']:8.1f} M  int Q dV = {tot[0].sum():.4e} erg/s  "
              f"int eta dV = {tot[1].sum():.4e} erg/s  "
              f"lepton rate = {tot[2,1]-tot[2,0]:.4e} /s")
    rows.sort(key=lambda r: r[0])
    nsp = rows[0][2].shape[1]
    t_ms = np.array([r[1] for r in rows])
    Q = np.array([r[2][0] for r in rows]); E = np.array([r[2][1] for r in rows])
    R = np.array([r[2][2] for r in rows])
    table = os.path.join(outdir, "nu_src_integrals.txt")
    with open(table, "w") as fp:
        fp.write("# t[M] t[ms] " + " ".join(f"Q{s}[erg/s]" for s in range(nsp)) + " " +
                 " ".join(f"eta{s}[erg/s]" for s in range(nsp)) + " " +
                 " ".join(f"R{s}[1/s]" for s in range(nsp)) + "\n")
        for r, q, e, n in zip(rows, Q, E, R):
            fp.write(f"{r[0]:.4f} {r[1]:.5f} " + " ".join(f"{v:.6e}" for v in
                                                         np.concatenate([q, e, n])) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for s in range(nsp):
        c = f"C{s}"
        ax[0].semilogy(t_ms, np.abs(Q[:, s]), c, label=SPECIES[s] + r" net $\int Q^- dV$")
        ax[0].semilogy(t_ms, E[:, s], c + "--", label=SPECIES[s] + r" $\int\eta\, dV$")
    ax[0].semilogy(t_ms, np.abs(Q.sum(1)), "k", lw=2, label="net, all")
    ax[0].set_xlabel("t [ms]"); ax[0].set_ylabel("erg s$^{-1}$"); ax[0].legend(fontsize=7)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].plot(t_ms, 1.0 - Q.sum(1)/E.sum(1), "k", lw=2, label="all")
        for s in range(nsp):
            ax[1].plot(t_ms, 1.0 - Q[:, s]/E[:, s], f"C{s}", label=SPECIES[s])
    ax[1].set_xlabel("t [ms]"); ax[1].set_ylabel(r"absorbed fraction $1 - \int Q^-/\int\eta$")
    ax[1].legend(fontsize=8)
    out = os.path.join(outdir, "nu_src_integrals.png")
    fig.savefig(out, dpi=130)
    print("wrote", table, "and", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="radnu_src x-z slice files or quoted globs")
    ap.add_argument("--matter", action="append", default=[],
                    help="matter (mhd_w_bcc) x-z slices for the density and dYe/dt")
    ap.add_argument("--integrate", action="append", default=[],
                    help="3D radnu_src files: volume integrals against time")
    ap.add_argument("-o", "--outdir", default="nu_src")
    ap.add_argument("--extent", type=float, default=32.0, help="half-width shown [M]")
    ap.add_argument("--species", type=int, default=None,
                    help="one species (0 nu_e, 1 anti-nu_e, 2 nu_x) instead of their sum")
    ap.add_argument("--ref", type=int, default=-1,
                    help="frame that sets the colour limits (default: the last)")
    ap.add_argument("--movie", help="output movie: .mp4 (H.264), else .avi, else .gif")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel frames")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.integrate:
        files3d = sorted({f for p in args.integrate for f in glob.glob(p)})
        integrate(files3d, args.outdir)
    files = sorted({f for p in args.files for f in glob.glob(p)})
    if not files:
        if not args.integrate:
            sys.exit("no files match")
        return
    matter = sorted({f for p in args.matter for f in glob.glob(p)})
    mfiles = match_matter(files, matter)
    ref = args.ref % len(files)
    lim = default_limits(load_slice(files[ref]),
                         load_rho(mfiles[ref])[1] if mfiles[ref] else None, args.species)
    jobs = [(f, m, lim, args.extent, args.outdir, args.species)
            for f, m in zip(files, mfiles)]
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
