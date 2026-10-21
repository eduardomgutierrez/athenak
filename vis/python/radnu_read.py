"""
Reader for AthenaK's per-species/per-frequency-group ("6D") multifrequency radiation
binary outputs (radnu_coord, radnu_fluid, radnu_coord_fluid).

bin_convert.read_binary() cannot parse these: it assumes one scalar per cell per
variable, but each of these "variables" (rnu00_ff, ..., rnu33_ff) actually holds
nspecies*nfreq values per cell (one per species+frequency-group slot).

On-disk layout (per MeshBlock, after the 10 index/geometry ints/reals -- see
src/outputs/binary.cpp WriteOutputFile, ~lines 140-245):
    for n_component in 0..9:       # rnu00_ff, rnu01_ff, ..., rnu33_ff (labels, in file order)
      for isf in 0..NFreqOut()-1:  # combined species*frequency slot
        for k, j, i:                # fastest-varying is i

NFreqOut() = nspecies*nfreq (src/radiation/radiation.hpp:189), and the slot index is
species-major, frequency-minor:
    isf = isp*nfreq + ifr   =>   isp = isf // nfreq,  ifr = isf % nfreq
(src/outputs/derived_variables.cpp ~1090-1100: "isp = isf/nfreq_; ifr = isf - isp*nfreq_",
also src/radiation/radiation.hpp:187-189 comment "ordered species-major: slot = isp*nfreq+ifr").

Confirmed against src/outputs/binary.cpp ~148-243: n_freq_ = NFreqOut(); the innermost
write loop for the 6D case is `for f in 0..n_freq_: for k: for j: for i`, and this whole
(f,k,j,i) block is written once per output "variable" n (i.e. once per tensor component).
So the slowest index is the component, then isf, then k, j, i.
"""
import numpy as np
import bin_convert as bc


def _get_from_header(header, blockname, keyname):
    """Same lookup bin_convert.read_binary uses internally, exposed standalone."""
    blockname = blockname.strip()
    keyname = keyname.strip()
    if not blockname.startswith("<"):
        blockname = "<" + blockname
    if blockname[-1] != ">":
        blockname += ">"
    block = "<none>"
    for line in header:
        if line.startswith("<"):
            block = line
            continue
        key, value = line.split("=", 1)
        if block == blockname and key.strip() == keyname:
            return value.strip()
    raise KeyError(f"no parameter called {blockname}/{keyname}")


def read_radnu(filename):
    """Read a radnu_* .bin file into a dict.

    Returns a dict with (at least):
      time, cycle, x1v, x2v, x3v (cell centres, 1D arrays for the root grid; only
        correct for a single, unrefined MeshBlock covering the whole domain -- this
        is the single-zone smoke test's layout, not a general multi-MeshBlock reader),
      nspecies, nfreq, nu_min, nu_max, freq_scale (0=linear,1=log,2=customize),
      var_names (10 component labels, e.g. rnu00_ff..rnu33_ff),
      data: ndarray shaped (nspecies, nfreq, ncomp, nx3, nx2, nx1).
    """
    fp = open(filename, "rb")
    fp.seek(0, 2)
    filesize = fp.tell()
    fp.seek(0, 0)

    code_header = fp.readline().split()
    if code_header[0] != b"Athena":
        raise TypeError("bad file format (should be Athena binary)")
    version = code_header[-1].split(b"=")[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version}")

    pheader_count = int(fp.readline().split(b"=")[-1])
    pheader = {}
    for _ in range(pheader_count - 1):
        key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=", 1)]
        pheader[key] = val
    time = float(pheader["time"])
    cycle = int(pheader["cycle"])
    locsizebytes = int(pheader["size of location"])
    varsizebytes = int(pheader["size of variable"])

    nvars_file = int(fp.readline().split(b"=")[-1])  # = ncomp (10), not nspecies*nfreq*ncomp
    var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b"=")[-1])
    header = [
        line.decode("utf-8").split("#")[0].strip()
        for line in fp.read(header_size).split(b"\n")
    ]
    header = [line for line in header if len(line) > 0]

    locfmt = "d" if locsizebytes == 8 else "f"
    varfmt = "d" if varsizebytes == 8 else "f"

    Nx1 = int(_get_from_header(header, "<mesh>", "nx1"))
    Nx2 = int(_get_from_header(header, "<mesh>", "nx2"))
    Nx3 = int(_get_from_header(header, "<mesh>", "nx3"))
    x1min = float(_get_from_header(header, "<mesh>", "x1min"))
    x1max = float(_get_from_header(header, "<mesh>", "x1max"))
    x2min = float(_get_from_header(header, "<mesh>", "x2min"))
    x2max = float(_get_from_header(header, "<mesh>", "x2max"))
    x3min = float(_get_from_header(header, "<mesh>", "x3min"))
    x3max = float(_get_from_header(header, "<mesh>", "x3max"))

    try:
        nspecies = int(_get_from_header(header, "<radiation>", "nspecies"))
    except KeyError:
        nspecies = 1   # the photon path never reads <radiation>/nspecies
    nfreq = int(_get_from_header(header, "<radiation>", "nfreq"))
    nu_min = float(_get_from_header(header, "<radiation>", "nu_min"))
    nu_max = float(_get_from_header(header, "<radiation>", "nu_max"))
    freq_scale_str = _get_from_header(header, "<radiation>", "freq_scale")
    freq_scale = {"linear": 0, "customize": 2}.get(freq_scale_str, 1)

    nghost = int(_get_from_header(header, "<mesh>", "nghost"))

    nsf = nspecies * nfreq
    ncomp = nvars_file
    if ncomp != len(var_list):
        raise ValueError("component count mismatch between header and var list")

    # Assemble onto the root grid (assumes a single MeshBlock spanning the whole
    # domain, as in the single-zone smoke test; a general reader would need to
    # stitch MeshBlocks using mb_index/mb_logical as bin_convert.read_binary does).
    out = np.full((nspecies, nfreq, ncomp, Nx3, Nx2, Nx1), np.nan,
                  dtype=np.float64 if varfmt == "d" else np.float32)

    while fp.tell() < filesize:
        mb_idx = np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64)
        ois, oie, ojs, oje, oks, oke = mb_idx
        # Ghost zones exist only along dimensions with more than one active cell
        # (matches bin_convert.read_binary's "- nghost", which is only correct there
        # because that function never places data by absolute index; here we do).
        if Nx1 > 1:
            ois -= nghost; oie -= nghost
        if Nx2 > 1:
            ojs -= nghost; oje -= nghost
        if Nx3 > 1:
            oks -= nghost; oke -= nghost
        nx1_out = oie - ois + 1
        nx2_out = oje - ojs + 1
        nx3_out = oke - oks + 1
        fp.read(16)  # logical location + level
        fp.read(6 * locsizebytes)  # x1min,x1max,x2min,x2max,x3min,x3max

        count = nx1_out * nx2_out * nx3_out * nsf * ncomp
        data = np.fromfile(fp, dtype=np.float64 if varfmt == "d" else np.float32,
                            count=count)
        # component (slowest) -> isf -> k -> j -> i (fastest); see module docstring.
        data = data.reshape(ncomp, nsf, nx3_out, nx2_out, nx1_out)
        # split isf = isp*nfreq + ifr (species-major, frequency-minor)
        data = data.reshape(ncomp, nspecies, nfreq, nx3_out, nx2_out, nx1_out)
        # -> (nspecies, nfreq, ncomp, k, j, i)
        data = np.transpose(data, (1, 2, 0, 3, 4, 5))
        out[:, :, :, oks:oks + nx3_out, ojs:ojs + nx2_out, ois:ois + nx1_out] = data

    fp.close()

    dx1 = (x1max - x1min) / Nx1
    dx2 = (x2max - x2min) / Nx2
    dx3 = (x3max - x3min) / Nx3
    x1v = x1min + (np.arange(Nx1) + 0.5) * dx1
    x2v = x2min + (np.arange(Nx2) + 0.5) * dx2
    x3v = x3min + (np.arange(Nx3) + 0.5) * dx3

    return {
        "time": time, "cycle": cycle,
        "x1v": x1v, "x2v": x2v, "x3v": x3v,
        "nspecies": nspecies, "nfreq": nfreq,
        "nu_min": nu_min, "nu_max": nu_max, "freq_scale": freq_scale,
        "var_names": var_list,
        "data": out,
        "header": header,
    }


def freq_bin_edges_mev(nu_min, nu_max, nfreq, freq_scale=1):
    """Lower edges nu_f (f=0..nfreq-1) plus the extrapolated top edge (nfreq+1 total),
    matching src/radiation/radiation_multi_freq.cpp SetFrequencyGrid (~109-131) for the
    grid points, and src/radiation/radiation_nurates.hpp FreqBinEdgesMeV (~85-104) for
    how the top bin's upper edge continues the geometric spacing."""
    if freq_scale != 1:
        raise NotImplementedError("only the log grid (freq_scale=1) is implemented")
    f = np.arange(nfreq)
    lo = nu_min * (nu_max / nu_min) ** (f / (nfreq - 1))
    top = lo[-1] ** 2 / lo[-2]
    edges = np.append(lo, top)
    return edges


if __name__ == "__main__":
    import sys
    d = read_radnu(sys.argv[1])
    print("time", d["time"], "cycle", d["cycle"])
    print("nspecies", d["nspecies"], "nfreq", d["nfreq"], "nu_min", d["nu_min"],
          "nu_max", d["nu_max"], "freq_scale", d["freq_scale"])
    print("var_names", d["var_names"])
    print("data.shape", d["data"].shape)
