"""Tests for the XFOIL wrapper.

Run inside the container:  pytest

Two kinds of assertion here:

  Determinism - XFOIL is deterministic given identical input and a fresh
  process. Two identical calls must agree exactly. A difference means state
  is leaking between runs (a reused working directory, a surviving polar
  file), not that the solver is noisy.

  Physics - checks that hold independently of XFOIL being right. These catch
  bugs determinism cannot, above all a parser reading the wrong columns:
  swap CD and CDp and two runs still agree perfectly while every number is
  wrong.

Each run is a subprocess, so fixtures are module-scoped to keep the suite
fast.
"""

from __future__ import annotations

import pytest

from xfoil_mcp.wrapper import run_polar

CLEAN = dict(airfoil="2412", reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=1)
CASCADE = dict(airfoil="2412", reynolds=1e6, alpha_start=0, alpha_end=20,
               alpha_step=1, max_iter=5)


@pytest.fixture(scope="module")
def baseline():
    result = run_polar(**CLEAN)
    assert result.status == "ok", f"baseline run was {result.status}: {result.warnings}"
    return result


@pytest.fixture(scope="module")
def by_alpha(baseline):
    return {p.alpha: p for p in baseline.points}


# --- determinism -----------------------------------------------------------

def test_identical_calls_give_identical_points():
    assert run_polar(**CLEAN).points == run_polar(**CLEAN).points


def test_failure_set_is_reproducible():
    """The interesting half: failures must be deterministic too.

    If the same call failed at different alphas on different runs, something
    is leaking between processes.
    """
    first, second = run_polar(**CASCADE), run_polar(**CASCADE)
    assert first.failed_alphas == second.failed_alphas
    assert first.points == second.points


# --- result structure ------------------------------------------------------

def test_every_requested_alpha_is_returned(baseline):
    returned = [p.alpha for p in baseline.points]
    assert returned == baseline.requested_alphas
    assert baseline.failed_alphas == []


def test_non_convergence_is_reported_not_raised():
    """A partial polar is a result, not an error.

    The caller needs to see WHICH alphas are missing to judge whether the
    data supports what it is about to conclude.
    """
    result = run_polar(**CASCADE)
    assert result.status == "partial"
    assert result.failed_alphas == [1.0, 2.0, 13.0, 14.0]
    assert result.converged_count == 17
    assert result.requested_count == 21


def test_consecutive_failures_are_grouped():
    """Scattered failures are hard points; a run of them is a warm-start
    cascade. The structure preserves that distinction."""
    result = run_polar(**CASCADE)
    assert [1.0, 2.0] in result.failure_runs
    assert [13.0, 14.0] in result.failure_runs


def test_summary_carries_enough_to_judge_the_run():
    summary = run_polar(**CASCADE).summary()
    assert summary["status"] == "partial"
    assert summary["converged_points"] < summary["requested_points"]
    assert summary["failed_alphas"]
    assert summary["best_ld"] is not None


# --- physics ---------------------------------------------------------------

def test_cambered_section_lifts_at_zero_incidence(by_alpha):
    """2% camber on a NACA 2412 means positive lift at alpha = 0."""
    assert by_alpha[0.0].cl == pytest.approx(0.237, abs=0.01)


def test_symmetric_section_has_no_lift_at_zero_incidence():
    result = run_polar("0012", 1e6, 0, 4, 1)
    assert result.points[0].cl == pytest.approx(0.0, abs=0.001)


def test_lift_curve_slope_is_near_thin_airfoil_theory(by_alpha):
    """Theory gives 2*pi per radian (~0.11/deg); viscous effects pull it
    slightly below."""
    slope = (by_alpha[5.0].cl - by_alpha[0.0].cl) / 5.0
    assert 0.09 < slope < 0.12


def test_drag_rises_well_above_its_minimum_by_high_alpha(baseline, by_alpha):
    """CD is NOT monotonic - a cambered section has a drag bucket at low
    alpha, where the lower-surface transition moves aft and skin friction
    falls faster than pressure drag rises. What must hold is that drag at
    high alpha greatly exceeds the minimum.
    """
    cd_min = min(p.cd for p in baseline.points)
    assert by_alpha[10.0].cd > 2 * cd_min


def test_upper_surface_transition_moves_forward_with_alpha(by_alpha):
    """The favourable pressure gradient holding laminar flow disappears as
    alpha increases, so transition marches toward the leading edge."""
    assert by_alpha[10.0].top_xtr < by_alpha[0.0].top_xtr
    assert by_alpha[10.0].top_xtr < 0.1


def test_lower_surface_stays_laminar_at_moderate_alpha(by_alpha):
    """At positive alpha the lower surface sees a gentler gradient and stays
    laminar to the trailing edge."""
    assert by_alpha[6.0].bot_xtr == pytest.approx(1.0)
