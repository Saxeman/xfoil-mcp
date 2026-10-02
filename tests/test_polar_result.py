"""The polar file parser and the result it feeds, against text captured from
a real XFOIL run. Host-side; no solver needed.

Fixtures (NACA 2412, Re 1e6):
  polar_2412_re1e6_a0_4.txt            alpha 0, 2, 4, all converged
  polar_2412_re1e6_none_converged.txt  a sweep where nothing converged: header only
  stdout_2412_re1e6_cascade.txt        the stdout of that second sweep (ITER 5, alpha 0..3)

They were captured with the command script the wrapper builds, so they have
to be captured again if _build_commands changes what it sends.
"""

from pathlib import Path

import pytest

from xfoil_mcp.wrapper import PolarPoint, PolarResult, _parse_polar_file, _scan_stdout

FIXTURES = Path(__file__).parent / "fixtures"
POLAR = FIXTURES / "polar_2412_re1e6_a0_4.txt"
POLAR_EMPTY = FIXTURES / "polar_2412_re1e6_none_converged.txt"
STDOUT_CASCADE = FIXTURES / "stdout_2412_re1e6_cascade.txt"


def point(alpha, cl=0.5, cd=0.01) -> PolarPoint:
    return PolarPoint(alpha=alpha, cl=cl, cd=cd, cdp=0.005, cm=-0.05, top_xtr=0.4, bot_xtr=1.0)


def result(requested, points, failed=()) -> PolarResult:
    return PolarResult(airfoil="2412", reynolds=1e6, mach=0.0, n_crit=9.0, max_iter=100,
                       requested_alphas=list(requested), points=list(points),
                       failed_alphas=list(failed))


# --- the polar file -----------------------------------------------------------

def test_a_real_polar_file_parses_to_its_rows():
    points = _parse_polar_file(POLAR)
    assert [p.alpha for p in points] == [0.0, 2.0, 4.0]
    assert points[2] == PolarPoint(alpha=4.0, cl=0.7146, cd=0.00693, cdp=0.00107,
                                   cm=-0.0573, top_xtr=0.398, bot_xtr=1.0)


def test_the_two_trailing_columns_are_ignored():
    """XFOIL 6.99 adds Top_Itr and Bot_Itr after the seven columns we keep."""
    assert len(POLAR.read_text().splitlines()[-1].split()) == 9
    assert len(_parse_polar_file(POLAR)) == 3


def test_a_polar_with_a_header_and_no_rows_is_empty():
    """What XFOIL leaves behind when no alpha converged."""
    assert "------" in POLAR_EMPTY.read_text()
    assert _parse_polar_file(POLAR_EMPTY) == []


def test_a_missing_polar_file_is_empty(tmp_path):
    assert _parse_polar_file(tmp_path / "nope.txt") == []


def test_a_file_without_the_dashed_rule_is_empty(tmp_path):
    """The rule is the anchor. Without it nothing below can be trusted to be data."""
    path = tmp_path / "polar.txt"
    path.write_text("   0.000   0.2371   0.00564   0.00049  -0.0520   0.6517   0.6795\n")
    assert _parse_polar_file(path) == []


def test_the_header_length_does_not_matter(tmp_path):
    """The parser finds the rule; it does not count header lines."""
    lines = POLAR.read_text().splitlines()
    rule = next(i for i, l in enumerate(lines) if "------" in l)
    path = tmp_path / "polar.txt"
    path.write_text("\n".join(["an extra header line"] * 5 + lines[rule - 1:]) + "\n")
    assert [p.alpha for p in _parse_polar_file(path)] == [0.0, 2.0, 4.0]


def test_short_and_non_numeric_rows_are_skipped(tmp_path):
    path = tmp_path / "polar.txt"
    path.write_text(POLAR.read_text()
                    + "   6.000   0.9000\n"                                           # too short
                    + "   8.000   ******   0.00900   0.00100  -0.0500   0.3000   1.0000\n"   # Fortran overflow
                    + "\n")
    assert [p.alpha for p in _parse_polar_file(path)] == [0.0, 2.0, 4.0]


# --- failure runs -------------------------------------------------------------

def test_no_failures_means_no_runs():
    assert result([0.0, 1.0], [point(0.0), point(1.0)]).failure_runs == []


def test_failures_next_to_each_other_form_one_run():
    """A run suggests a warm-start cascade, not independent hard points."""
    r = result([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [point(0.0), point(3.0)], failed=[1.0, 2.0, 4.0, 5.0])
    assert r.failure_runs == [[1.0, 2.0], [4.0, 5.0]]


def test_runs_follow_position_in_the_sweep_not_the_alpha_values():
    """With a 0.25 degree step, 0.25 and 0.5 are neighbours although they differ by less than 1."""
    r = result([0.0, 0.25, 0.5, 0.75], [point(0.0)], failed=[0.25, 0.5, 0.75])
    assert r.failure_runs == [[0.25, 0.5, 0.75]]


# --- status and summary -------------------------------------------------------

@pytest.mark.parametrize("points, failed, status", [
    ([point(0.0), point(1.0)], [], "ok"),
    ([point(0.0)], [1.0], "partial"),
    ([], [0.0, 1.0], "empty"),
])
def test_status_follows_the_points_and_the_failures(points, failed, status):
    assert result([0.0, 1.0], points, failed).status == status


def test_summary_reports_the_best_lift_to_drag_and_the_peak_lift_with_their_alphas():
    r = result([0.0, 2.0, 4.0], [point(0.0, cl=0.24, cd=0.006), point(2.0, cl=0.45, cd=0.005),
                                 point(4.0, cl=0.71, cd=0.010)])
    summary = r.summary()
    assert (summary["best_ld"], summary["best_ld_alpha"]) == (90.0, 2.0)
    assert (summary["max_cl"], summary["max_cl_alpha"]) == (0.71, 4.0)
    assert (summary["requested_points"], summary["converged_points"]) == (3, 3)
    assert summary["status"] == "ok"


def test_summary_of_an_empty_result_has_no_numbers_to_misread():
    summary = result([0.0, 1.0], [], failed=[0.0, 1.0]).summary()
    assert summary["status"] == "empty"
    assert [summary[k] for k in ("best_ld", "best_ld_alpha", "max_cl", "max_cl_alpha")] == [None] * 4
    assert summary["failure_runs"] == [[0.0, 1.0]]


def test_summary_leaves_out_the_point_table_and_the_stdout_log():
    summary = result([0.0], [point(0.0)]).summary()
    assert "points" not in summary and "stdout" not in summary


def test_polar_point_lift_to_drag_is_nan_at_zero_drag():
    assert point(0.0, cl=0.5, cd=0.01).ld == pytest.approx(50.0)
    assert point(0.0, cl=0.5, cd=0.0).ld != point(0.0, cl=0.5, cd=0.0).ld      # NaN


# --- stdout captured from a real failing sweep ----------------------------------

def test_real_failing_stdout_reports_each_failed_point_and_the_march_failures():
    """ITER 5 at alpha 0..3: all four points fail, and two marches fail on the way."""
    warnings, march_failures = _scan_stdout(STDOUT_CASCADE.read_text(), requested_max_iter=5)
    assert march_failures == 2
    assert warnings == [
        "4 point(s) reported VISCAL convergence failure",
        "2 boundary-layer march failure(s) during solves (some points may have converged from a poor path)",
    ]


def test_real_xfoil_echoes_the_iteration_limit_once_and_only_when_asked():
    """XFOIL 6.99 prints nothing for `ITER 5`. It prints "Current iteration
    limit" when a bare ITER makes it prompt, which is why the script sends
    one. The captured run shows exactly one echo, of the value that was set."""
    stdout = STDOUT_CASCADE.read_text()
    echoes = [line for line in stdout.splitlines() if "Current iteration limit" in line]
    assert len(echoes) == 1 and echoes[0].split()[-1] == "5"


def test_a_limit_other_than_the_one_requested_is_reported_from_real_output():
    """The same capture, read as if 100 iterations had been asked for."""
    warnings, _ = _scan_stdout(STDOUT_CASCADE.read_text(), requested_max_iter=100)
    assert "iteration limit reads 5, requested 100" in warnings
