"""Thermal stage tests: an aero CaseResult in, a thermal summary out.

The aero result is built from the step 1 fixture in the same shape the
worker returns, so these run on the host without XFOIL.
"""

import copy
import math

import pytest

pytest.importorskip("skfem")     # the thermal extra

from xfoil_mcp import thermal_stage
from xfoil_mcp.thermal import SkinResult
from xfoil_mcp.thermal_stage import _mm, evaluate_thermal

from builders import THERMAL, aero

def test_converged_alpha_gets_a_thermal_result():
    out = evaluate_thermal(aero())
    assert out.status == "ok"
    assert out.coverage == 1.0
    assert out.points[0]["balance_error"] < 0.01
    assert out.worst_case["min_heated_temperature_c"] == pytest.approx(24.8, abs=0.5)
    assert out.content_hash == aero().case.content_hash()       # it names the case it answers


def test_coverage_counts_requested_alphas_not_converged_ones():
    """The survivorship guard: a dropped alpha lowers coverage and says why."""
    out = evaluate_thermal(aero(requested=(2.0, 4.0), converged=(4.0,), status="partial"))
    assert out.status == "partial"
    assert out.coverage == 0.5
    assert out.excluded == [{"alpha": 2.0, "reason": "aero did not converge"}]


def test_failed_aero_stage_is_refused():
    out = evaluate_thermal(aero(converged=(), status="empty"))
    assert out.status == "error"
    assert out.failure_kind == "input"
    assert "upstream" in out.errors[0]


def test_implied_high_speed_is_refused():
    """Re 1e6 at a 1 cm chord means over 1,000 m/s. XFOIL solved it as slow air."""
    out = evaluate_thermal(aero(thermal={**THERMAL, "chord_m": 0.01}))
    assert out.failure_kind == "input"
    assert "Mach" in out.errors[0]


def test_missing_stagnation_point_excludes_the_alpha_instead_of_crashing():
    result = aero()
    ue = result.data["bl"]["4.0"]["ue"]
    result.data["bl"]["4.0"]["ue"] = [abs(u) for u in ue]    # never crosses zero now
    out = evaluate_thermal(result)
    assert out.status == "empty"
    assert "stagnation" in out.excluded[0]["reason"]


def test_millimetres_are_rounded_and_never_negative_zero():
    """The upper ice-free extent is a negated coordinate, so zero arrives as
    -0.0. It must not reach the model as "-0.0"."""
    assert _mm(0.03084) == 30.8
    assert _mm(None) is None
    assert math.copysign(1.0, _mm(-0.0)) == 1.0
    assert str(_mm(-0.00001)) == "0.0"


# --- the gate ---------------------------------------------------------------

def solve_returning(monkeypatch, heat_out):
    """Replace the skin solve with one that reports `heat_out` W/m for 500 W/m in."""
    def fake(*args, **kwargs):
        return SkinResult(z_m=[0.0], temperature_k=[290.0], heat_in_w_per_m=500.0,
                          heat_out_w_per_m=heat_out, min_heated_temperature_k=290.0,
                          ice_free_upper_m=0.01, ice_free_lower_m=0.01)
    monkeypatch.setattr(thermal_stage, "skin_temperature", fake)


def test_a_solve_whose_heat_does_not_balance_is_excluded(monkeypatch):
    """The gate's own branch. The real solver cannot reach it (see
    test_thermal), so it is driven here with a solve that is 10% off."""
    solve_returning(monkeypatch, heat_out=450.0)
    out = evaluate_thermal(aero())
    assert (out.status, out.coverage, out.points) == ("empty", 0.0, [])
    assert out.excluded == [{"alpha": 4.0, "reason": "energy balance off by 10.0%"}]


def test_a_solve_inside_the_tolerance_is_kept(monkeypatch):
    solve_returning(monkeypatch, heat_out=497.0)           # 0.6% off
    out = evaluate_thermal(aero())
    assert out.status == "ok" and len(out.points) == 1


def test_a_solve_that_produced_nan_is_excluded(monkeypatch):
    solve_returning(monkeypatch, heat_out=float("nan"))
    out = evaluate_thermal(aero())
    assert out.points == [] and out.status == "empty"


# --- fan-in across different alphas ------------------------------------------

def two_distinct_alphas():
    """builders.aero gives every alpha the same boundary layer. Here alpha 2
    gets one with 50% more edge speed, so it loses heat faster and runs colder."""
    result = aero(requested=(2.0, 4.0), converged=(2.0, 4.0))
    faster = copy.deepcopy(result.data["bl"]["2.0"])
    faster["ue"] = [1.5 * u for u in faster["ue"]]
    result.data["bl"]["2.0"] = faster
    return result


def test_worst_case_takes_the_coldest_alpha_and_the_shortest_extents():
    out = evaluate_thermal(two_distinct_alphas())
    by_alpha = {p["alpha"]: p for p in out.points}
    assert by_alpha[2.0]["min_heated_temperature_c"] < by_alpha[4.0]["min_heated_temperature_c"]
    assert out.worst_case["at_alpha"] == 2.0
    assert out.worst_case["min_heated_temperature_c"] == by_alpha[2.0]["min_heated_temperature_c"]
    for side in ("ice_free_upper_mm", "ice_free_lower_mm"):
        assert out.worst_case[side] == min(p[side] for p in out.points)


def test_an_excluded_alpha_does_not_stop_the_others():
    """One alpha failing never fails the case."""
    result = two_distinct_alphas()
    layer = copy.deepcopy(result.data["bl"]["4.0"])
    layer["ue"] = [abs(u) for u in layer["ue"]]                 # no stagnation point at alpha 4
    result.data["bl"]["4.0"] = layer
    out = evaluate_thermal(result)
    assert (out.status, out.coverage) == ("partial", 0.5)
    assert [p["alpha"] for p in out.points] == [2.0]
    assert out.excluded[0]["alpha"] == 4.0 and "stagnation" in out.excluded[0]["reason"]


def test_a_converged_polar_row_without_its_boundary_layer_is_excluded():
    result = two_distinct_alphas()
    del result.data["bl"]["4.0"]
    out = evaluate_thermal(result)
    assert [p["alpha"] for p in out.points] == [2.0]
    assert out.excluded[0]["alpha"] == 4.0


def test_a_zero_speed_node_at_one_alpha_does_not_lose_the_others():
    result = two_distinct_alphas()
    layer = copy.deepcopy(result.data["bl"]["4.0"])
    layer["ue"][87] = 0.0                                       # the node just past the stagnation point
    result.data["bl"]["4.0"] = layer
    out = evaluate_thermal(result)
    assert 2.0 in [p["alpha"] for p in out.points]


def test_an_aero_solve_at_high_mach_is_refused():
    """Heat transfer at an implied Mach 0.08 on a boundary layer XFOIL solved
    at Mach 0.6 describes no flight condition."""
    result = aero()
    fast = result.case.model_copy(update={
        "conditions": result.case.conditions.model_copy(update={"mach": 0.6})})
    out = evaluate_thermal(result.model_copy(update={"case": fast}))
    assert (out.status, out.failure_kind) == ("error", "input")


def test_a_result_without_a_thermal_block_is_a_value_error():
    result = aero()
    bare = result.model_copy(update={"case": result.case.model_copy(update={"thermal": None})})
    with pytest.raises(ValueError, match="no thermal block"):
        evaluate_thermal(bare)


def test_worst_case_can_take_each_field_from_a_different_alpha(monkeypatch):
    """Coldest at one alpha, shortest upper extent at another: the worst case
    reports the worst of each, and says which alpha was coldest."""
    solves = iter([
        SkinResult(z_m=[0.0], temperature_k=[280.0], heat_in_w_per_m=500.0, heat_out_w_per_m=500.0,
                   min_heated_temperature_k=280.0, ice_free_upper_m=0.050, ice_free_lower_m=0.020),
        SkinResult(z_m=[0.0], temperature_k=[285.0], heat_in_w_per_m=500.0, heat_out_w_per_m=500.0,
                   min_heated_temperature_k=285.0, ice_free_upper_m=0.010, ice_free_lower_m=0.030),
    ])
    monkeypatch.setattr(thermal_stage, "skin_temperature", lambda *a, **k: next(solves))
    out = evaluate_thermal(aero(requested=(2.0, 4.0), converged=(2.0, 4.0)))
    assert out.worst_case == {"min_heated_temperature_c": 6.85, "at_alpha": 2.0,
                              "ice_free_upper_mm": 10.0, "ice_free_lower_mm": 20.0}


def test_a_frozen_stagnation_point_counts_as_no_ice_free_extent(monkeypatch):
    """None means the skin never got above freezing. In the worst case that is zero millimetres."""
    frozen = SkinResult(z_m=[0.0], temperature_k=[270.0], heat_in_w_per_m=500.0, heat_out_w_per_m=500.0,
                        min_heated_temperature_k=270.0, ice_free_upper_m=None, ice_free_lower_m=None)
    monkeypatch.setattr(thermal_stage, "skin_temperature", lambda *a, **k: frozen)
    out = evaluate_thermal(aero())
    assert out.points[0]["ice_free_upper_mm"] is None
    assert (out.worst_case["ice_free_upper_mm"], out.worst_case["ice_free_lower_mm"]) == (0.0, 0.0)


def test_a_solve_that_raises_at_one_alpha_excludes_that_alpha_only(monkeypatch):
    """Whatever goes wrong inside the heat-transfer or skin solve for one
    alpha, the others are still evaluated."""
    real = thermal_stage.heat_transfer
    calls = []

    def fails_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ZeroDivisionError("float division by zero")
        return real(*args, **kwargs)
    monkeypatch.setattr(thermal_stage, "heat_transfer", fails_once)
    out = evaluate_thermal(aero(requested=(2.0, 4.0), converged=(2.0, 4.0)))
    assert [p["alpha"] for p in out.points] == [4.0]
    assert out.excluded == [{"alpha": 2.0, "reason": "thermal solve failed: float division by zero"}]
