"""
Unit tests for the nurates multi-frequency frame-shift helpers
"""

# Modules
import test_suite.testutils as testutils


def test_rad_freq_shift():
    input_file = "inputs/ut_rad_freq_shift.athinput"
    assert testutils.run(input_file)
