"""
Unit tests for the heavy-flavour slot fold used by the nurates opacity path
"""

# Modules
import test_suite.testutils as testutils


def test_rad_nu_merge():
    input_file = "inputs/ut_rad_nu_merge.athinput"
    assert testutils.run(input_file)
