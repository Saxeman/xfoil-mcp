"""Parser tests against fixtures captured from real XFOIL runs.

No XFOIL needed: these run on the host in milliseconds.
"""

from pathlib import Path

import pytest

from xfoil_mcp.wrapper import BoundaryLayer, _parse_bl_file

FIXTURES = Path(__file__).parent / "fixtures"
BL_A4 = FIXTURES / "bl_2412_re1e6_a4.txt"


@pytest.fixture
def layer() -> BoundaryLayer:
    parsed = _parse_bl_file(BL_A4, alpha=4.0)
    assert parsed is not None
    return parsed


def test_surface_and_wake_are_split_by_column_count(layer):
    assert layer.n_surface == 160
    assert len(layer.x) == 183          # 160 surface + 23 wake


def test_surface_runs_upper_te_to_lower_te(layer):
    assert layer.x[0] == 1.0 and layer.y[0] > 0
    last = layer.n_surface - 1
    assert layer.x[last] == 1.0 and layer.y[last] < 0


def test_leading_edge_is_minimum_x(layer):
    i = layer.le_index
    assert layer.x[i] == pytest.approx(0.0, abs=1e-4)
    assert layer.s[i] == pytest.approx(1.0272, abs=1e-4)


def test_wake_has_no_skin_friction(layer):
    assert all(cf == 0.0 for cf in layer.cf[layer.n_surface:])


def test_unexpected_column_count_rejects_the_file(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("# header\n 0.0 1.0 0.0\n")
    assert _parse_bl_file(bad, alpha=0.0) is None


def test_missing_file_returns_none(tmp_path):
    assert _parse_bl_file(tmp_path / "nope.txt", alpha=0.0) is None
