"""Parser tests against fixtures captured from real XFOIL runs.

No XFOIL needed: these run on the host in milliseconds.
"""

from pathlib import Path

import pytest
import shutil

from xfoil_mcp.wrapper import (
    BoundaryLayer, CpDistribution, PolarPoint, _collect_fields, _parse_bl_file, _parse_cp_file, _drop_non_finite, _parse_coords_file
)

FIXTURES = Path(__file__).parent / "fixtures"
BL_A4 = FIXTURES / "bl_2412_re1e6_a4.txt"
CP_A4 = FIXTURES / "cp_2412_re1e6_a4.txt"

COORDS_FLAP = FIXTURES / "naca2412_flap_10.dat"




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
    bl, warnings = _collect_fields(tmp_path, [4.0], {4.0: pt(4.0)}, "bl", _parse_bl_file)
    assert set(bl) == {4.0}
    assert warnings == []


def test_dump_for_failed_alpha_is_ignored(tmp_path):
    """A perfectly good file, for an alpha that is not in the polar."""
    shutil.copy(BL_A4, tmp_path / "bl_000.txt")
    bl, warnings = _collect_fields(tmp_path, [4.0], {}, "bl", _parse_bl_file)
    assert bl == {}
    assert warnings == []


def test_missing_dump_for_converged_alpha_is_reported(tmp_path):
    bl, warnings = _collect_fields(tmp_path, [4.0], {4.0: pt(4.0)}, "bl", _parse_bl_file)
    assert bl == {}
    assert len(warnings) == 1 and "4.0" in warnings[0]


def test_files_are_matched_to_alphas_by_position(tmp_path):
    """bl_001 belongs to the second requested alpha, whatever its value."""
    shutil.copy(BL_A4, tmp_path / "bl_001.txt")
    bl, warnings = _collect_fields(tmp_path, [2.0, 4.0], {2.0: pt(2.0), 4.0: pt(4.0)}, "bl", _parse_bl_file)
    assert set(bl) == {4.0}
    assert "2.0" in warnings[0]          # bl_000 is missing

def test_cp_dump_is_gated_the_same_way(tmp_path):
    shutil.copy(CP_A4, tmp_path / "cp_000.txt")
    cp, _ = _collect_fields(tmp_path, [4.0], {4.0: pt(4.0)}, "cp", _parse_cp_file)
    assert set(cp) == {4.0}
    cp, _ = _collect_fields(tmp_path, [4.0], {}, "cp", _parse_cp_file)
    assert cp == {}

# --- pressure distribution --------------------------------------------------

@pytest.fixture
def cp() -> CpDistribution:
    parsed = _parse_cp_file(CP_A4, alpha=4.0)
    assert parsed is not None
    return parsed


def test_cp_has_one_row_per_surface_node(cp, layer):
    assert len(cp.x) == layer.n_surface == 160


def test_cp_nodes_are_the_bl_surface_nodes(cp, layer):
    """Same nodes, same order. That's what lets position tell upper from
    lower when the Cp file has no y column."""
    assert cp.x == pytest.approx(layer.x[: layer.n_surface], abs=1e-5)


def test_cp_is_one_minus_ue_squared(cp, layer):
    """Bernoulli at Mach 0: Cp = 1 - (Ue/V_inf)^2. The two files are two
    views of one solution; if this fails, one of the parsers is wrong."""
    for c, ue in zip(cp.cp, layer.ue[: layer.n_surface]):
        assert c == pytest.approx(1 - ue**2, abs=1e-4)


def test_peak_cp_is_the_node_nearest_stagnation(cp):
    i = max(range(len(cp.cp)), key=cp.cp.__getitem__)
    assert cp.x[i] == pytest.approx(0.00334, abs=1e-5)
    assert cp.cp[i] < 1.0        # the true stagnation point falls between nodes


def test_cp_rejects_a_row_with_the_wrong_column_count(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("#  x  Cp\n 1.0 0.17 0.5\n")
    assert _parse_cp_file(bad, alpha=0.0) is None


def test_cp_missing_file_returns_none(tmp_path):
    assert _parse_cp_file(tmp_path / "nope.txt", alpha=0.0) is None

# --- non-finite values ------------------------------------------------------

def test_bl_file_with_a_nan_is_rejected(tmp_path):
    text = BL_A4.read_text().replace("0.001183", "NaN", 1)
    (tmp_path / "bl.txt").write_text(text)
    assert _parse_bl_file(tmp_path / "bl.txt", alpha=4.0) is None


def test_cp_file_with_a_nan_is_rejected(tmp_path):
    lines = CP_A4.read_text().splitlines()
    lines[5] = "     0.95000        NaN"
    (tmp_path / "cp.txt").write_text("\n".join(lines))
    assert _parse_cp_file(tmp_path / "cp.txt", alpha=4.0) is None


def test_polar_rows_with_non_finite_values_are_dropped():
    good = PolarPoint(alpha=0.0, cl=0.24, cd=0.0056, cdp=0.0005, cm=-0.05, top_xtr=0.65, bot_xtr=0.68)
    bad = PolarPoint(alpha=1.0, cl=float("nan"), cd=0.0055, cdp=0.0006, cm=-0.05, top_xtr=0.59, bot_xtr=0.86)
    kept, dropped = _drop_non_finite([good, bad])
    assert kept == [good]
    assert dropped == [1.0]

# --- geometry ----------------------------------------------------------------

def test_coords_file_parses_to_160_points():
    pts = _parse_coords_file(COORDS_FLAP)
    assert len(pts) == 160
    assert pts[0] == pytest.approx((0.9956611, -0.0508536))       # upper trailing edge
    assert min(x for x, _ in pts) == pytest.approx(0.0, abs=1e-4)  # the nose


def test_coords_file_with_a_bad_row_is_rejected(tmp_path):
    bad = tmp_path / "bad.dat"
    bad.write_text("1.0 0.001\n0.5\n")
    assert _parse_coords_file(bad) is None


def test_missing_coords_file_returns_none(tmp_path):
    assert _parse_coords_file(tmp_path / "nope.dat") is None
