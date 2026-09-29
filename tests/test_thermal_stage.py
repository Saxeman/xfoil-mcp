"""Thermal stage tests: an aero CaseResult in, a thermal summary out.

The aero result is built from the step 1 fixture in the same shape the
worker returns, so these run on the host without XFOIL.
"""

import dataclasses
from pathlib import Path

import pytest

from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry
from xfoil_mcp.thermal_stage import evaluate_thermal
from xfoil_mcp.wrapper import _parse_bl_file

BL_A4 = Path(__file__).parent / "fixtures" / "bl_2412_re1e6_a4.txt"
THERMAL = dict(
    chord_m=0.5, air_temperature_k=263.15, heater_width=0.1,
    heater_power_w_per_m=500.0, skin_thickness_m=0.001, skin_conductivity_w_mk=200.0,
)


def aero(requested=(4.0,), converged=(4.0,), status="ok", thermal=None) -> CaseResult:
    """An aero result shaped like the worker's, using the fixture's boundary
    layer for every converged alpha."""
    case = Case(
        geometry=Geometry(naca="2412"),
        conditions=Conditions(reynolds=1e6, alpha_start=min(requested),
                              alpha_end=max(requested), alpha_step=2),
        outputs=("forces", "bl"),
        thermal=thermal or THERMAL,
    )
    layer = dataclasses.asdict(_parse_bl_file(BL_A4, alpha=4.0))
    points = [dict(alpha=a, cl=0.7146, cd=0.00693, cdp=0.00107, cm=-0.0573,
                   top_xtr=0.398, bot_xtr=1.0) for a in converged]
    data = {
        "requested_alphas": list(requested),
        "points": points,
        "failed_alphas": [a for a in requested if a not in converged],
        "bl": {str(a): layer for a in converged},
    }
    return CaseResult(
        case=case, status=status,
        failure_kind=None if status == "ok" else "numerical",
        data=None if status == "empty" else data,
    )


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
