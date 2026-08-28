"""
Closure of the nurates multi-frequency source term, against an absolute answer

A comoving spectrum diffusing through a purely scattering medium: elastic scattering
cannot move energy between comoving groups, so the exact result is the grey result times
the spectral weight of each bin.  This is the only test in the suite that has an absolute
answer while the shape factor is not identically 1, so it is the only one that can tell
the candidate closure weights apart.  The pgen exits non-zero when the L1 error against
the analytic solution exceeds problem/tol_l1, and skips the check when the build has no
bns_nurates.
"""

# Modules
import test_suite.testutils as testutils


def test_rad_diffusion_spectral():
    input_file = "inputs/rad_diffusion_spectral_nurates.athinput"
    assert testutils.run(input_file)
