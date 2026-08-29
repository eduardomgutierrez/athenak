"""
Unit tests for the conservative log-space rebin used by the nurates
multi-frequency source term
"""

# Modules
import test_suite.testutils as testutils


def test_rad_freq_remap():
    input_file = "inputs/ut_rad_freq_remap.athinput"
    assert testutils.run(input_file)
