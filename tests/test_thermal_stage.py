"""Thermal stage tests: an aero CaseResult in, a thermal summary out.

The aero result is built from the step 1 fixture in the same shape the
worker returns, so these run on the host without XFOIL.
"""

import pytest

pytest.importorskip("skfem")     # the thermal extra

from xfoil_mcp.thermal_stage import evaluate_thermal

from builders import THERMAL, aero

def test_converged_alpha_gets_a_thermal_result():
    out = evaluate_thermal(aero())
    assert out["status"] == "ok"
    assert out["coverage"] == 1.0
    assert out["points"][0]["balance_error"] < 0.01
    assert out["worst_case"]["min_heated_temperature_c"] == pytest.approx(24.8, abs=0.5)


def test_coverage_counts_requested_alphas_not_converged_ones():
    """The survivorship guard: a dropped alpha lowers coverage and says why."""
    out = evaluate_thermal(aero(requested=(2.0, 4.0), converged=(4.0,), status="partial"))
    assert out["status"] == "partial"
    assert out["coverage"] == 0.5
    assert out["excluded"] == [{"alpha": 2.0, "reason": "aero did not converge"}]


def test_failed_aero_stage_is_refused():
    out = evaluate_thermal(aero(converged=(), status="empty"))
    assert out["status"] == "error"
    assert out["failure_kind"] == "input"
    assert "upstream" in out["errors"][0]


def test_implied_high_speed_is_refused():
    """Re 1e6 at a 1 cm chord means over 1,000 m/s. XFOIL solved it as slow air."""
    out = evaluate_thermal(aero(thermal={**THERMAL, "chord_m": 0.01}))
    assert out["failure_kind"] == "input"
    assert "Mach" in out["errors"][0]


def test_missing_stagnation_point_excludes_the_alpha_instead_of_crashing():
    result = aero()
    ue = result.data["bl"]["4.0"]["ue"]
    result.data["bl"]["4.0"]["ue"] = [abs(u) for u in ue]    # never crosses zero now
    out = evaluate_thermal(result)
    assert out["status"] == "empty"
    assert "stagnation" in out["excluded"][0]["reason"]
