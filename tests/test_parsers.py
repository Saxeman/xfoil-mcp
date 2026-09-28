"""Parser tests against fixtures captured from real XFOIL runs.

No XFOIL needed: these run on the host in milliseconds.
"""

from pathlib import Path

import pytest
import shutil

from xfoil_mcp.wrapper import BoundaryLayer, PolarPoint, _collect_bl, _parse_bl_file

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

# --- gating: which dump files get read --------------------------------------

def pt(alpha: float) -> PolarPoint:
    return PolarPoint(alpha=alpha, cl=0.7, cd=0.007, cdp=0.001, cm=-0.05,
                      top_xtr=0.4, bot_xtr=1.0)


def test_dump_for_converged_alpha_is_attached(tmp_path):
    shutil.copy(BL_A4, tmp_path / "bl_000.txt")
    bl, warnings = _collect_bl(tmp_path, [4.0], {4.0: pt(4.0)})
    assert set(bl) == {4.0}
    assert warnings == []


def test_dump_for_failed_alpha_is_ignored(tmp_path):
    """A perfectly good file, for an alpha that is not in the polar."""
    shutil.copy(BL_A4, tmp_path / "bl_000.txt")
    bl, warnings = _collect_bl(tmp_path, [4.0], {})
    assert bl == {}
    assert warnings == []


def test_missing_dump_for_converged_alpha_is_reported(tmp_path):
    bl, warnings = _collect_bl(tmp_path, [4.0], {4.0: pt(4.0)})
    assert bl == {}
    assert len(warnings) == 1 and "4.0" in warnings[0]


def test_files_are_matched_to_alphas_by_position(tmp_path):
    """bl_001 belongs to the second requested alpha, whatever its value."""
    shutil.copy(BL_A4, tmp_path / "bl_001.txt")
    bl, warnings = _collect_bl(tmp_path, [2.0, 4.0], {2.0: pt(2.0), 4.0: pt(4.0)})
    assert set(bl) == {4.0}
    assert "2.0" in warnings[0]          # bl_000 is missing
