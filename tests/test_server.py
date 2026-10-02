"""Server tests. Harness and dispatch are mocked; no Docker needed.

Tools are invoked through FastMCP's in-memory client rather than by calling
the Python functions directly, so what is tested is what a real MCP client
would see: the tool is registered, the arguments validate, and the payload
has the shape the docstrings promise.

The in-memory client is not the stdio transport. It caught nothing when an
exception escaped run_polar and wedged a real session, because it propagates
exceptions as Python exceptions rather than serializing them. That is why
tools now never raise for expected failures, and why the envelope tests
below assert on return values, never on raises.

Organized by what can go wrong between the harness and the model:
  1. registration    - tools exist; descriptions say what the model needs
  2. envelope        - every response has the same top-level shape
  3. input handling  - bad arguments become data, all at once, before dispatch
  4. pass-through    - harness outcomes reach the model unaltered; data stays out
  5. session state   - _last_batch across success, refusal, and lookup
"""

from __future__ import annotations

import ast
import asyncio
import json
import socket
import subprocess
import textwrap
import threading
import time
from pathlib import Path

import pytest
from fastmcp import Client

from builders import FakeDocker
from xfoil_mcp import harness, server
from xfoil_mcp.schema import Case, CaseResult, Conditions, Flap, Geometry

ENVELOPE = {"status", "failure_kind", "retry_could_help", "errors"}

TOOLS = {"run_polar", "dry_run_campaign", "run_campaign", "get_case_detail",
         "evaluate_design", "request_print", "start_print"}

# --- plumbing --------------------------------------------------------------

def call(tool: str, **args) -> dict:
    async def _go():
        async with Client(server.mcp) as client:
            result = await client.call_tool(tool, args)
            if getattr(result, "data", None) is not None:
                return result.data
            return json.loads(result.content[0].text)
    return asyncio.run(_go())


def list_tools() -> dict[str, str]:
    """name -> description"""
    async def _go():
        async with Client(server.mcp) as client:
            return {t.name: (t.description or "") for t in await client.list_tools()}
    return asyncio.run(_go())


def make_case(naca="2412", reynolds=1e6, label=None) -> Case:
    return Case(
        geometry=Geometry(naca=naca),
        conditions=Conditions(reynolds=reynolds, alpha_start=0, alpha_end=10, alpha_step=1),
        label=label or f"case {naca}",
    )


def ok_result(case: Case) -> CaseResult:
    return CaseResult(
        case=case, status="ok",
        summary={"converged_points": 11, "requested_points": 11, "failed_alphas": [],
                 "best_ld": 104.4, "best_ld_alpha": 5.0},
        data={"points": ["large"]},
    )


def partial_result(case: Case) -> CaseResult:
    return CaseResult(
        case=case, status="partial", failure_kind="numerical",
        summary={"converged_points": 8, "requested_points": 11,
                 "failed_alphas": [8.0, 9.0, 10.0], "failure_runs": [[8.0, 9.0, 10.0]]},
        data={"points": ["large"]},
    )


def infra_result(case: Case) -> CaseResult:
    return CaseResult(
        case=case, status="error", failure_kind="infrastructure",
        summary={"error": "worker exceeded 180s and was killed"},
    )


def report(cases, rejected=(), errors=(), killed=False, approval_hash="abc123",
           infrastructure=False) -> harness.DryRunReport:
    points = sum(c.conditions.point_count() for c in cases)
    return harness.DryRunReport(
        cases=list(cases), rejected=list(rejected), errors=list(errors), killed=killed,
        infrastructure=infrastructure,
        estimate=harness.Estimate(case_count=len(cases), alpha_points=points,
                                  expected_seconds=round(len(cases) * 1.5 + points * 0.05, 1)),
        approval_hash=approval_hash,
    )


def batch(*results: CaseResult, approval_hash="abc123") -> harness.BatchResult:
    return harness.BatchResult(results=list(results), approval_hash=approval_hash)


@pytest.fixture(autouse=True)
def reset_batch():
    server._last_batch = None
    yield
    server._last_batch = None


@pytest.fixture
def no_dispatch(monkeypatch):
    """Fail the test if anything reaches dispatch."""
    monkeypatch.setattr(harness.dispatch, "run_case",
                        lambda c, timeout=180.0: pytest.fail("dispatch was called"))


# ==========================================================================
# 1. registration
# ==========================================================================

def test_expected_tools_are_registered():
    assert set(list_tools()) == TOOLS


@pytest.mark.parametrize("tool, must_mention", [
    # The descriptions are prompts. If one of these phrases disappears, the
    # model loses a piece of guidance it was relying on. These are cheap
    # regression tests for prompt engineering.
    ("run_polar", ["Reynolds", "n_crit", "partial", "Retrying identically will not help"]),
    ("dry_run_campaign", ["Runs no solver", "approval_hash", "scipy.stats.qmc", "seed"]),
    ("run_campaign", ["approval_hash", "complete", "counts", "numerical"]),
    ("get_case_detail", ["Large", "content_hash"]),
])
def test_tool_descriptions_carry_the_guidance_the_model_needs(tool, must_mention):
    description = list_tools()[tool]
    for phrase in must_mention:
        assert phrase in description, f"{tool} description lost the phrase {phrase!r}"


def test_run_polar_advertises_its_parameters():
    """The argument schema is generated from the signature; a renamed
    parameter would silently change what the model can pass."""
    async def _go():
        async with Client(server.mcp) as client:
            tools = {t.name: t for t in await client.list_tools()}
            return set(tools["run_polar"].input_schema["properties"])
    assert asyncio.run(_go()) == {
        "airfoil", "reynolds", "alpha_start", "alpha_end", "alpha_step", "n_crit", "max_iter",
        "flap_deflection_deg", "flap_hinge",
    }


# ==========================================================================
# 2. envelope
# ==========================================================================

def _every_response(monkeypatch) -> list[tuple[str, dict]]:
    """One ok response and one error response from each tool."""
    case = make_case()
    out: list[tuple[str, dict]] = []

    monkeypatch.setattr(harness.dispatch, "run_case", lambda c, timeout=180.0: ok_result(c))
    out.append(("run_polar ok", call("run_polar", airfoil="2412", reynolds=1e6)))
    out.append(("run_polar error", call("run_polar", airfoil="24a2", reynolds=1e6)))

    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report([case]))
    out.append(("dry_run ok", call("dry_run_campaign", source="src")))
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report([], errors=["boom"]))
    out.append(("dry_run error", call("dry_run_campaign", source="src")))

    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(ok_result(case)))
    out.append(("run_campaign ok", call("run_campaign", source="src", approval_hash="abc123")))

    def refuse(s, h, case_timeout=180.0):
        raise harness.ApprovalMismatch("nope")
    monkeypatch.setattr(server.harness, "submit", refuse)
    out.append(("run_campaign error", call("run_campaign", source="src", approval_hash="x")))

    server._last_batch = batch(ok_result(case))
    out.append(("detail ok", call("get_case_detail", content_hash=case.content_hash())))
    out.append(("detail error", call("get_case_detail", content_hash="missing")))
    return out


def test_every_response_carries_the_envelope(monkeypatch):
    for name, response in _every_response(monkeypatch):
        missing = ENVELOPE - set(response)
        assert not missing, f"{name} is missing envelope fields {missing}"
        assert response["status"] in ("ok", "partial", "empty", "error"), name
        assert isinstance(response["errors"], list), name
        assert isinstance(response["retry_could_help"], bool), name


def test_ok_responses_have_no_failure_kind_and_no_errors(monkeypatch):
    for name, response in _every_response(monkeypatch):
        if response["status"] == "ok":
            assert response["failure_kind"] is None, name
            assert response["errors"] == [], name
            assert response["retry_could_help"] is False, name


def test_error_responses_name_a_kind_and_at_least_one_error(monkeypatch):
    for name, response in _every_response(monkeypatch):
        if response["status"] == "error":
            assert response["failure_kind"] in ("input", "infrastructure", "numerical"), name
            assert response["errors"], f"{name} reported an error with no message"


def test_retry_could_help_is_true_only_for_infrastructure(monkeypatch):
    for name, response in _every_response(monkeypatch):
        expected = response["failure_kind"] == "infrastructure"
        assert response["retry_could_help"] is expected, name


# ==========================================================================
# 3. input handling
# ==========================================================================

def test_invalid_airfoil_is_returned_as_data_not_raised(no_dispatch):
    out = call("run_polar", airfoil="24a2", reynolds=1e6)
    assert out["status"] == "error"
    assert out["failure_kind"] == "input"
    assert out["retry_could_help"] is False
    assert any("naca" in e for e in out["errors"])
    assert out["content_hash"] is None


def test_all_input_errors_are_reported_at_once(no_dispatch):
    """One round trip, not one per bad field. Case.model_validate on nested
    dicts lets pydantic collect from every branch before raising."""
    out = call("run_polar", airfoil="24a2", reynolds=-1, n_crit=50)
    assert out["failure_kind"] == "input"
    joined = "\n".join(out["errors"])
    assert "geometry.naca" in joined
    assert "conditions.reynolds" in joined
    assert "conditions.n_crit" in joined


def test_sweep_over_the_point_limit_is_rejected_before_dispatch(no_dispatch):
    out = call("run_polar", airfoil="2412", reynolds=1e6, alpha_start=0, alpha_end=30, alpha_step=0.1)
    assert out["failure_kind"] == "input"
    assert any("points" in e for e in out["errors"])


def test_backwards_sweep_is_rejected_before_dispatch(no_dispatch):
    out = call("run_polar", airfoil="2412", reynolds=1e6, alpha_start=10, alpha_end=0)
    assert out["failure_kind"] == "input"


def test_error_message_locations_are_nested_paths(no_dispatch):
    """The model should see where the problem is, not just what."""
    out = call("run_polar", airfoil="2412", reynolds=0)
    assert out["errors"][0].startswith("conditions.reynolds:")


# ==========================================================================
# 4. pass-through fidelity
# ==========================================================================

def test_run_polar_builds_the_case_the_arguments_describe(monkeypatch):
    seen = {}

    def capture(case, timeout=180.0):
        seen["case"] = case
        return ok_result(case)
    monkeypatch.setattr(harness.dispatch, "run_case", capture)

    out = call("run_polar", airfoil="4412", reynolds=5e5, alpha_end=12, n_crit=4)
    c = seen["case"]
    assert c.geometry.naca == "4412"
    assert c.conditions.reynolds == 5e5
    assert c.conditions.alpha_end == 12
    assert c.conditions.n_crit == 4
    assert out["content_hash"] == c.content_hash()


def test_run_polar_always_asks_for_the_outline(monkeypatch):
    """Without "geometry" in outputs the result could never be printed."""
    seen = {}
    monkeypatch.setattr(harness.dispatch, "run_case",
                        lambda c, timeout=180.0: seen.setdefault("case", c) and ok_result(c))
    call("run_polar", airfoil="2412", reynolds=1e6)
    assert seen["case"].outputs == ("forces", "geometry")
    assert seen["case"].geometry.flap is None


def test_run_polar_flap_arguments_become_the_flap(monkeypatch):
    seen = {}
    monkeypatch.setattr(harness.dispatch, "run_case",
                        lambda c, timeout=180.0: seen.setdefault("case", c) and ok_result(c))
    call("run_polar", airfoil="2412", reynolds=1e6, flap_deflection_deg=10, flap_hinge=0.75)
    flap = seen["case"].geometry.flap
    assert (flap.x_hinge, flap.deflection) == (0.75, 10)


def test_run_polar_flap_out_of_range_is_an_input_error(no_dispatch):
    out = call("run_polar", airfoil="2412", reynolds=1e6, flap_deflection_deg=80)
    assert out["failure_kind"] == "input"
    assert any("flap" in e for e in out["errors"])


def test_run_polar_goes_through_the_harness(monkeypatch):
    """Every tool reaches the worker by way of the harness, so stage logic
    added there applies to run_polar too."""
    seen = []

    def fake(case, timeout=180.0):
        seen.append(case)
        return ok_result(case)
    monkeypatch.setattr(server.harness, "evaluate", fake)
    monkeypatch.setattr(harness.dispatch, "run_case",
                        lambda c, timeout=180.0: pytest.fail("run_polar bypassed the harness"))

    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert out["status"] == "ok"
    assert [c.geometry.naca for c in seen] == ["2412"]
    assert seen[0].thermal is None


def test_a_summary_cannot_override_the_envelope(monkeypatch):
    """A summary is a container's own dict. Keys it shares with the envelope
    must not replace the verdict the host validated."""
    def lying(case, timeout=180.0):
        return CaseResult(case=case, status="ok", summary={
            "status": "partial", "failure_kind": "numerical", "retry_could_help": True,
            "errors": ["from the container"], "best_ld": 104.4,
        })
    monkeypatch.setattr(harness.dispatch, "run_case", lying)
    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert (out["status"], out["failure_kind"], out["retry_could_help"], out["errors"]) == ("ok", None, False, [])
    assert out["best_ld"] == 104.4                  # the rest of the summary still arrives


def test_run_polar_summary_reaches_the_model_without_data(monkeypatch):
    monkeypatch.setattr(harness.dispatch, "run_case", lambda c, timeout=180.0: ok_result(c))
    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert out["status"] == "ok"
    assert out["best_ld"] == 104.4
    assert out["converged_points"] == 11
    assert "data" not in out and "points" not in out


def test_run_polar_partial_result_passes_through_with_failed_alphas(monkeypatch):
    monkeypatch.setattr(harness.dispatch, "run_case", lambda c, timeout=180.0: partial_result(c))
    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert out["status"] == "partial"
    assert out["failure_kind"] == "numerical"
    assert out["retry_could_help"] is False
    assert out["failed_alphas"] == [8.0, 9.0, 10.0]
    assert out["failure_runs"] == [[8.0, 9.0, 10.0]]


def test_run_polar_infrastructure_failure_is_retryable(monkeypatch):
    monkeypatch.setattr(harness.dispatch, "run_case", lambda c, timeout=180.0: infra_result(c))
    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert out["status"] == "error"
    assert out["failure_kind"] == "infrastructure"
    assert out["retry_could_help"] is True
    assert "180s" in out["errors"][0]


def test_dry_run_ok_returns_hash_estimate_and_case_views(monkeypatch):
    cases = [make_case("2412"), make_case("0012")]
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report(cases, approval_hash="h1"))
    out = call("dry_run_campaign", source="src")

    assert out["status"] == "ok"
    assert out["approval_hash"] == "h1"
    assert out["estimate"]["case_count"] == 2
    assert out["estimate"]["alpha_points"] == 22
    assert [c["naca"] for c in out["cases"]] == ["2412", "0012"]
    assert set(out["cases"][0]) == {"content_hash", "label", "naca", "flap", "reynolds", "mach", "n_crit",
                                    "max_iter", "alphas", "outputs", "thermal"}
    assert (out["cases"][0]["flap"], out["cases"][0]["thermal"]) == ("none", None)


def test_dry_run_with_rejections_keeps_valid_cases_but_no_hash(monkeypatch):
    """On failure, return everything learned; withhold only what would let
    the caller proceed."""
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report(
        [make_case("2412")], rejected=["case 1: thickness must be non-zero"]))
    out = call("dry_run_campaign", source="src")

    assert out["status"] == "error"
    assert out["failure_kind"] == "input"
    assert "thickness" in out["errors"][0]
    assert [c["naca"] for c in out["cases"]] == ["2412"]
    assert out["approval_hash"] is None
    assert out["estimate"] is None


def test_dry_run_killed_campaign_is_an_input_failure(monkeypatch):
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report(
        [], errors=["campaign exceeded 30.0s and was killed"], killed=True))
    out = call("dry_run_campaign", source="while True: pass")
    assert out["failure_kind"] == "input"
    assert out["retry_could_help"] is False
    assert any("looping" in e for e in out["errors"])


def test_dry_run_docker_failure_is_infrastructure(monkeypatch):
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report(
        [], errors=["docker not found on host"], infrastructure=True))
    out = call("dry_run_campaign", source="src")
    assert out["failure_kind"] == "infrastructure"
    assert out["retry_could_help"] is True


def test_dry_run_empty_campaign_says_so(monkeypatch):
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report([]))
    out = call("dry_run_campaign", source="def campaign(): return []")
    assert out["status"] == "error"
    assert any("no cases" in e for e in out["errors"])


def test_run_campaign_returns_counts_and_summaries_only(monkeypatch):
    c1, c2 = make_case("2412"), make_case("0012")
    monkeypatch.setattr(server.harness, "submit",
                        lambda s, h, case_timeout=180.0: batch(ok_result(c1), partial_result(c2)))
    out = call("run_campaign", source="src", approval_hash="abc123")

    assert (out["status"], out["failure_kind"], out["errors"]) == ("partial", "numerical", [])    # one of two is incomplete
    assert out["complete"] is True
    assert out["submitted"] == 2
    assert out["counts"] == {"ok": 1, "partial": 1, "empty": 0, "error": 0}
    assert all("data" not in r and "points" not in r for r in out["results"])
    assert out["results"][1]["status"] == "partial"
    assert out["results"][1]["retry_could_help"] is False


def test_run_campaign_passes_the_hash_it_was_given(monkeypatch):
    seen = {}

    def capture(s, h, case_timeout=180.0):
        seen["hash"] = h
        return batch()
    monkeypatch.setattr(server.harness, "submit", capture)
    call("run_campaign", source="src", approval_hash="the-hash")
    assert seen["hash"] == "the-hash"


def test_run_campaign_refusal_is_an_input_failure_with_empty_results(monkeypatch):
    def refuse(s, h, case_timeout=180.0):
        raise harness.ApprovalMismatch("approval hash x does not match current campaign y")
    monkeypatch.setattr(server.harness, "submit", refuse)
    out = call("run_campaign", source="src", approval_hash="x")

    assert out["status"] == "error"
    assert out["failure_kind"] == "input"
    assert "does not match" in out["errors"][0]
    assert out["complete"] is False
    assert out["results"] == []


# ==========================================================================
# 5. session state
# ==========================================================================

def test_detail_before_any_campaign_is_an_input_error():
    out = call("get_case_detail", content_hash="anything")
    assert out["status"] == "error"
    assert out["failure_kind"] == "input"
    assert out["result"] is None


def test_detail_returns_full_data_after_a_campaign(monkeypatch):
    case = make_case()
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(ok_result(case)))
    call("run_campaign", source="src", approval_hash="abc123")

    out = call("get_case_detail", content_hash=case.content_hash())
    assert out["status"] == "ok"
    assert out["result"]["status"] == "ok"
    assert out["result"]["data"] == {"points": ["large"]}
    assert out["result"]["case"]["geometry"]["naca"] == "2412"


def test_detail_for_unknown_hash_is_an_input_error(monkeypatch):
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(ok_result(make_case())))
    call("run_campaign", source="src", approval_hash="abc123")
    out = call("get_case_detail", content_hash="nope")
    assert out["status"] == "error"
    assert "nope" in out["errors"][0]


def test_refused_campaign_does_not_clobber_the_previous_batch(monkeypatch):
    """A failed submit must not erase results the model may still want."""
    case = make_case()
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(ok_result(case)))
    call("run_campaign", source="src", approval_hash="abc123")

    def refuse(s, h, case_timeout=180.0):
        raise harness.ApprovalMismatch("stale")
    monkeypatch.setattr(server.harness, "submit", refuse)
    call("run_campaign", source="src", approval_hash="stale")

    out = call("get_case_detail", content_hash=case.content_hash())
    assert out["status"] == "ok"


def test_new_campaign_replaces_the_previous_batch(monkeypatch):
    old, new = make_case("2412"), make_case("0012")
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(ok_result(old)))
    call("run_campaign", source="src", approval_hash="abc123")
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(ok_result(new)))
    call("run_campaign", source="src2", approval_hash="abc123")

    assert call("get_case_detail", content_hash=new.content_hash())["status"] == "ok"
    assert call("get_case_detail", content_hash=old.content_hash())["status"] == "error"


# -- Misc Tests

@pytest.mark.parametrize("module", ["server", "harness", "sandbox", "dispatch", "containers"])
def test_host_modules_never_print_to_stdout(module):
    """Over stdio, stdout is the protocol channel. One stray print corrupts
    the stream and every call after it fails."""
    src = (Path(server.__file__).parent / f"{module}.py").read_text()
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("print(") and "file=sys.stderr" not in stripped:
            pytest.fail(f"{module}.py:{lineno} prints to stdout: {stripped}")

# ==========================================================================
# 6. evaluate_design
# ==========================================================================

DESIGN = dict(airfoil="2412", reynolds=1e6, chord_m=0.5, air_temperature_c=-10.0,
              heater_power_w_per_m=500.0)


def design_result(case, aero_status="ok", thermal=None):
    """A harness.evaluate result shaped like the real one."""
    kind = None if aero_status == "ok" else (
        "numerical" if aero_status in ("partial", "empty") else "infrastructure")
    thermal = thermal or {
        "status": "ok", "failure_kind": None, "errors": [], "coverage": 1.0,
        "worst_case": {"min_heated_temperature_c": 19.5, "at_alpha": 10.0},
        "points": [{"alpha": 0.0, "min_heated_temperature_c": 25.7,
                    "ice_free_upper_mm": 116.6, "ice_free_lower_mm": 125.4}],
        "excluded": [],
    }
    data = None if aero_status in ("error", "empty") else {
        "points": [{"alpha": 0.0, "cl": 0.2371, "cd": 0.00564, "top_xtr": 0.652}],
        "thermal": thermal,
    }
    return CaseResult(
        case=case, status=aero_status, failure_kind=kind,
        summary={"converged_points": 1, "requested_points": 1, "best_ld": 42.0,
                 "error": None if aero_status != "error" else "worker exceeded 180s"},
        data=data,
    )


def capture_evaluate(monkeypatch, **kw):
    """Replace harness.evaluate, keeping the Case the tool built."""
    seen = {}

    def fake(case, timeout=180.0):
        seen["case"] = case
        return design_result(case, **kw)

    monkeypatch.setattr(server.harness, "evaluate", fake)
    return seen


def test_design_arguments_become_the_case_in_schema_units(monkeypatch):
    seen = capture_evaluate(monkeypatch)
    call("evaluate_design", **DESIGN, skin_thickness_mm=2.0, flap_deflection_deg=10.0)
    case = seen["case"]
    assert case.thermal.air_temperature_k == pytest.approx(263.15)
    assert case.thermal.skin_thickness_m == pytest.approx(0.002)
    assert case.geometry.flap.deflection == 10.0 and case.geometry.flap.x_hinge == 0.7
    assert "bl" in case.outputs


def test_no_flap_argument_means_no_flap(monkeypatch):
    seen = capture_evaluate(monkeypatch)
    call("evaluate_design", **DESIGN)
    assert seen["case"].geometry.flap is None


def test_design_errors_name_the_arguments_the_caller_used(monkeypatch):
    monkeypatch.setattr(server.harness, "evaluate",
                        lambda c, timeout=180.0: pytest.fail("evaluated an invalid design"))
    out = call("evaluate_design", **{**DESIGN, "airfoil": "24a2",
                                     "air_temperature_c": -150.0, "heater_power_w_per_m": -5.0})
    assert out["failure_kind"] == "input"
    joined = "\n".join(out["errors"])
    assert "airfoil:" in joined
    assert "air_temperature_c" in joined
    assert "heater_power_w_per_m:" in joined


def test_design_returns_a_stage_trace_and_per_alpha_table(monkeypatch):
    capture_evaluate(monkeypatch)
    out = call("evaluate_design", **DESIGN)
    assert out["status"] == "ok"
    assert [s["stage"] for s in out["stages"]] == ["aero", "thermal"]
    assert out["stages"][1]["worst_case"]["at_alpha"] == 10.0
    row = out["per_alpha"][0]
    assert row["cl"] == 0.2371 and row["ice_free_upper_mm"] == 116.6


def test_a_summary_cannot_override_the_aero_stage(monkeypatch):
    def lying(case, timeout=180.0):
        result = design_result(case)
        return result.model_copy(update={"summary": {**result.summary, "stage": "thermal", "status": "error"}})
    monkeypatch.setattr(server.harness, "evaluate", lying)
    out = call("evaluate_design", **DESIGN)
    aero = out["stages"][0]
    assert (aero["stage"], aero["status"]) == ("aero", "ok")
    assert out["status"] == "ok"


def test_failed_aero_means_thermal_is_not_run(monkeypatch):
    capture_evaluate(monkeypatch, aero_status="error")
    out = call("evaluate_design", **DESIGN)
    assert out["status"] == "error"
    assert out["failure_kind"] == "infrastructure"
    assert out["retry_could_help"] is True
    assert out["stages"][1]["status"] == "not_run"


def test_partial_thermal_coverage_makes_the_design_partial(monkeypatch):
    capture_evaluate(monkeypatch, thermal={
        "status": "partial", "failure_kind": "numerical", "errors": [], "coverage": 0.5,
        "worst_case": None, "points": [],
        "excluded": [{"alpha": 2.0, "reason": "aero did not converge"}],
    })
    out = call("evaluate_design", **DESIGN)
    assert out["status"] == "partial"
    assert out["stages"][1]["coverage"] == 0.5

# ==========================================================================
# 7. printing
# ==========================================================================

OUTLINE = [[float(v) for v in line.split()]
           for line in (Path(__file__).parent / "fixtures" / "naca2412_flap_10.dat").read_text().splitlines()
           if line.strip()]


@pytest.fixture
def printing(monkeypatch, tmp_path):
    """A fresh print queue per test, writing to a temp outbox, on a free port."""
    pytest.importorskip("cadquery")
    monkeypatch.setenv("XFOIL_PRINT_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.setenv("XFOIL_PRINT_PORT", "0")
    monkeypatch.setattr(server, "_designs", {})
    monkeypatch.setattr(server, "_print_queue", None)
    monkeypatch.setattr(server, "_print_site", None)
    yield tmp_path / "outbox"
    if server._print_site is not None:
        server._print_site.shutdown()
        server._print_site.server_close()


def evaluated_design(monkeypatch, geometry=OUTLINE) -> str:
    """Run evaluate_design against a fake harness whose result carries an outline."""
    def fake(case, timeout=180.0):
        result = design_result(case)
        data = dict(result.data)
        if geometry is not None:
            data["geometry"] = geometry
        return result.model_copy(update={"data": data})

    monkeypatch.setattr(server.harness, "evaluate", fake)
    return call("evaluate_design", **DESIGN)["content_hash"]


def test_print_request_for_an_unknown_design_is_refused(printing):
    out = call("request_print", content_hash="nope")
    assert out["failure_kind"] == "input"


def test_print_request_returns_a_local_review_url(printing, monkeypatch):
    out = call("request_print", content_hash=evaluated_design(monkeypatch))
    assert out["status"] == "ok"
    assert out["print_status"] == "pending"
    assert out["url"].startswith("http://127.0.0.1:")
    assert out["request_id"] in out["url"]


def test_start_print_before_approval_is_refused(printing, monkeypatch):
    request = call("request_print", content_hash=evaluated_design(monkeypatch))
    out = call("start_print", request_id=request["request_id"])
    assert out["status"] == "error"
    assert out["print_status"] == "pending"
    assert "approv" in out["errors"][0]


def test_after_the_person_approves_the_print_is_sent_once(printing, monkeypatch):
    request = call("request_print", content_hash=evaluated_design(monkeypatch))
    queue = server._print_queue
    part = queue.get(request["request_id"]).part
    queue.approve(request["request_id"], part.sha256)       # what the page does when you click

    out = call("start_print", request_id=request["request_id"])
    assert out["status"] == "ok"
    assert Path(out["stl_path"]).read_bytes() == part.stl

    again = call("start_print", request_id=request["request_id"])
    assert again["status"] == "error"
    assert again["print_status"] == "sent"


def polar_run(monkeypatch) -> str:
    """Run run_polar against a fake harness whose result carries an outline."""
    def fake(case, timeout=180.0):
        result = ok_result(case)
        return result.model_copy(update={"data": {**result.data, "geometry": OUTLINE}})

    monkeypatch.setattr(server.harness, "evaluate", fake)
    return call("run_polar", airfoil="2412", reynolds=1e6)["content_hash"]


def test_a_polar_run_can_be_printed(printing, monkeypatch):
    out = call("request_print", content_hash=polar_run(monkeypatch))
    assert out["status"] == "ok"
    assert out["print_status"] == "pending"
    assert out["stats"]["span_mm"] == 40.0


def test_print_size_arguments_reach_the_part(printing, monkeypatch):
    out = call("request_print", content_hash=polar_run(monkeypatch), chord_mm=100, span_mm=20)
    assert out["status"] == "ok"
    assert out["stats"]["span_mm"] == 20.0
    assert out["stats"]["chord_mm"] == pytest.approx(100, abs=1.0)    # the flap shortens the projection


def test_print_size_the_printer_cannot_hold_is_an_input_error(printing, monkeypatch):
    out = call("request_print", content_hash=polar_run(monkeypatch), chord_mm=1000)
    assert out["failure_kind"] == "input"
    assert "chord_mm" in out["errors"][0]


def test_design_without_geometry_says_how_to_get_it(printing, monkeypatch):
    out = call("request_print", content_hash=evaluated_design(monkeypatch, geometry=None))
    assert out["failure_kind"] == "input"
    assert "geometry" in out["errors"][0]
    assert "run_polar" in out["errors"][0]

@pytest.fixture
def busy_port(monkeypatch):
    """A localhost port something else is already listening on, set as the print port."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    monkeypatch.setenv("XFOIL_PRINT_PORT", str(blocker.getsockname()[1]))
    yield blocker
    blocker.close()


def test_a_busy_print_port_is_reported_and_leaves_nothing_half_started(printing, busy_port, monkeypatch):
    content_hash = polar_run(monkeypatch)
    out = call("request_print", content_hash=content_hash)
    assert (out["status"], out["failure_kind"], out["retry_could_help"]) == ("error", "infrastructure", True)
    assert "XFOIL_PRINT_PORT" in out["errors"][0]
    assert server._print_queue is None and server._print_site is None

    busy_port.close()                               # the port frees up; the same call now works
    again = call("request_print", content_hash=content_hash)
    assert again["status"] == "ok"
    assert again["url"].startswith("http://127.0.0.1:")


def test_start_print_reports_a_busy_print_port(printing, busy_port):
    out = call("start_print", request_id="anything")
    assert (out["status"], out["failure_kind"]) == ("error", "infrastructure")
    assert out["request_id"] == "anything"


def test_concurrent_first_print_calls_start_one_queue_and_one_site(printing, monkeypatch):
    """Tools run in worker threads. Two first calls must not each build a queue."""
    from xfoil_mcp import print_site

    real_start, starts = print_site.start_site, []

    def slow_start(queue, port):
        starts.append(queue)
        time.sleep(0.05)                            # hold the window open for the other threads
        return real_start(queue, port)
    monkeypatch.setattr(print_site, "start_site", slow_start)

    results = []
    threads = [threading.Thread(target=lambda: results.append(server._printing())) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(starts) == 1
    assert len({id(queue) for queue, _ in results}) == 1
    assert {base for _, base in results} == {server._print_base_url}


def test_default_outbox_is_at_the_repository_root():
    repo_root = Path(__file__).resolve().parent.parent
    assert server._default_outbox() == repo_root / "outbox"


# ==========================================================================
# 8. gaps in the sections above
# ==========================================================================

def assert_envelope(name: str, response: dict) -> None:
    """The four rules the envelope tests in section 2 apply, for one response."""
    assert ENVELOPE <= set(response), f"{name} is missing {ENVELOPE - set(response)}"
    assert response["status"] in ("ok", "partial", "empty", "error"), name
    assert isinstance(response["errors"], list), name
    assert response["retry_could_help"] is (response["failure_kind"] == "infrastructure"), name
    if response["status"] == "ok":
        assert (response["failure_kind"], response["errors"]) == (None, []), name
    if response["status"] == "error":
        assert response["failure_kind"] in ("input", "infrastructure", "numerical"), name
        assert response["errors"], f"{name} reported an error with no message"


def test_the_design_and_print_tools_carry_the_envelope_too(printing, monkeypatch):
    """Section 2 covers four of the seven tools and has no infrastructure
    response. These are the other three, with one of each kind of outcome."""
    responses = {}
    monkeypatch.setattr(server.harness, "evaluate", lambda c, timeout=180.0: design_result(c))
    responses["evaluate_design ok"] = call("evaluate_design", **DESIGN)
    responses["evaluate_design invalid"] = call("evaluate_design", **{**DESIGN, "chord_m": -1})
    monkeypatch.setattr(server.harness, "evaluate", lambda c, timeout=180.0: design_result(c, aero_status="error"))
    responses["evaluate_design infrastructure"] = call("evaluate_design", **DESIGN)
    monkeypatch.setattr(server.harness, "evaluate", lambda c, timeout=180.0: design_result(c, aero_status="partial"))
    responses["evaluate_design partial"] = call("evaluate_design", **DESIGN)

    responses["request_print unknown"] = call("request_print", content_hash="nope")
    request = call("request_print", content_hash=polar_run(monkeypatch))
    responses["request_print ok"] = request
    responses["request_print oversize"] = call("request_print", content_hash=polar_run(monkeypatch), chord_mm=1000)
    responses["start_print unapproved"] = call("start_print", request_id=request["request_id"])
    responses["start_print unknown"] = call("start_print", request_id="nope")
    part = server._print_queue.get(request["request_id"]).part
    server._print_queue.approve(request["request_id"], part.sha256)
    responses["start_print ok"] = call("start_print", request_id=request["request_id"])

    for name, response in responses.items():
        assert_envelope(name, response)
    assert {name: r["status"] for name, r in responses.items()} == {
        "evaluate_design ok": "ok", "evaluate_design invalid": "error",
        "evaluate_design infrastructure": "error", "evaluate_design partial": "partial",
        "request_print unknown": "error", "request_print ok": "ok", "request_print oversize": "error",
        "start_print unapproved": "error", "start_print unknown": "error", "start_print ok": "ok",
    }
    assert responses["evaluate_design infrastructure"]["retry_could_help"] is True      # the true side
    assert responses["evaluate_design invalid"]["retry_could_help"] is False


def test_an_unknown_print_request_reports_no_status(printing):
    out = call("start_print", request_id="nope")
    assert (out["failure_kind"], out["print_status"], out["request_id"]) == ("input", None, "nope")
    assert "no print request" in out["errors"][0]


def test_detail_cannot_reach_a_design_that_was_not_part_of_a_campaign(monkeypatch):
    """get_case_detail reads the last campaign only. Documented, and pinned here."""
    monkeypatch.setattr(harness.dispatch, "run_case", lambda c, timeout=180.0: ok_result(c))
    content_hash = call("run_polar", airfoil="2412", reynolds=1e6)["content_hash"]
    assert call("get_case_detail", content_hash=content_hash)["failure_kind"] == "input"


def description_example(tool: str) -> str:
    """The indented code block after "Example:" in a tool description."""
    lines = list_tools()[tool].splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Example:") + 1
    block = []
    for line in lines[start:]:
        if line.strip() and not line[0].isspace():      # prose resumes at the left margin
            break
        block.append(line)
    return textwrap.dedent("\n".join(block)).strip() + "\n"


def test_the_campaign_example_in_the_tool_description_runs():
    """This is the only code sample the model is given for campaigns. It is
    run here the way the sandbox would run it."""
    example = description_example("dry_run_campaign")
    assert "def campaign" in example                    # the extraction found the code
    namespace = {}
    exec(compile(example, "<example>", "exec"), namespace)
    cases = namespace["campaign"]()
    assert [c.geometry.naca for c in cases] == ["2412", "4412", "0012"]
    assert all("geometry" in c.outputs for c in cases)


def thermal_failed(case) -> CaseResult:
    """Aero passed, the thermal stage did not: what a missing thermal image produces."""
    failure = {"status": "error", "failure_kind": "infrastructure",
               "errors": ["thermal worker exited 125: Unable to find image"], "coverage": 0.0,
               "worst_case": None, "heater_power_w_per_m": None, "excluded": []}
    return CaseResult(case=case, status="ok", summary={"converged_points": 11, "thermal": failure},
                      data={"thermal": {**failure, "points": []}})


def test_a_campaign_whose_thermal_stage_failed_is_not_reported_as_ok(monkeypatch):
    cases = [make_case("2412"), make_case("4412")]
    monkeypatch.setattr(server.harness, "submit",
                        lambda s, h, case_timeout=180.0: batch(*[thermal_failed(c) for c in cases]))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert out["status"] != "ok"


def test_a_failed_thermal_stage_is_at_least_visible_in_each_case_summary(monkeypatch):
    """Until the aggregate says so (the test above), this is where it shows."""
    case = make_case()
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(thermal_failed(case)))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert out["results"][0]["thermal"]["status"] == "error"
    assert "Unable to find image" in out["results"][0]["thermal"]["errors"][0]


def test_a_campaign_where_every_case_errored_is_not_reported_as_ok(monkeypatch):
    cases = [make_case("2412"), make_case("4412")]
    monkeypatch.setattr(server.harness, "submit",
                        lambda s, h, case_timeout=180.0: batch(*[infra_result(c) for c in cases]))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert out["counts"]["error"] == 2
    assert out["status"] != "ok"


def test_docker_going_away_before_the_run_is_a_retryable_failure(monkeypatch):
    """The campaign was approved; then Docker stopped. That is not the agent's input.
    Driven through the real sandbox module, with docker itself missing."""
    missing = FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", FakeDocker(run_result=missing, rm_result=missing))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert (out["failure_kind"], out["retry_could_help"]) == ("infrastructure", True)
    assert out["errors"] == ["campaign is not runnable: docker not found on host"]
    assert (out["submitted"], out["results"]) == (0, [])


def test_a_campaign_error_that_mentions_docker_is_still_an_input_failure(monkeypatch):
    errors = ["campaign() raised:\nFileNotFoundError: /var/run/docker.sock"]
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report([], errors=errors))
    out = call("dry_run_campaign", source="src")
    assert (out["failure_kind"], out["retry_could_help"]) == ("input", False)


def test_a_campaign_that_ran_out_of_memory_is_not_told_it_is_looping(monkeypatch):
    """Driven through the real sandbox module, with the container exiting 137."""
    monkeypatch.setattr(subprocess, "run", FakeDocker(run_result=FakeDocker.exited(137, stderr="Killed")))
    out = call("dry_run_campaign", source="src")
    assert out["failure_kind"] == "input"
    assert any("memory limit" in e for e in out["errors"])
    assert not any("looping" in e or "timeout" in e for e in out["errors"])


def test_a_campaign_that_ran_out_of_time_is_told_it_is_probably_looping(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeDocker(run_result=subprocess.TimeoutExpired("docker", 30.0)))
    out = call("dry_run_campaign", source="src")
    assert out["failure_kind"] == "input"
    assert any("looping" in e for e in out["errors"])


def test_a_missing_sandbox_image_is_an_infrastructure_failure_not_a_looping_campaign(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        FakeDocker(run_result=FakeDocker.exited(125, stderr="Unable to find image 'xfoil-sandbox'")))
    out = call("dry_run_campaign", source="src")
    assert (out["failure_kind"], out["retry_could_help"]) == ("infrastructure", True)
    assert not any("looping" in e for e in out["errors"])


def test_a_mixed_campaign_is_partial_and_names_the_most_actionable_kind(monkeypatch):
    """One case ok, one whose worker timed out: partial overall, and
    infrastructure because a retry may help that case."""
    c1, c2 = make_case("2412"), make_case("4412")
    monkeypatch.setattr(server.harness, "submit",
                        lambda s, h, case_timeout=180.0: batch(ok_result(c1), infra_result(c2)))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert (out["status"], out["failure_kind"], out["retry_could_help"]) == ("partial", "infrastructure", True)
    assert out["counts"] == {"ok": 1, "partial": 0, "empty": 0, "error": 1}


def test_a_campaign_where_nothing_was_usable_is_an_error_with_the_results_attached(monkeypatch):
    cases = [make_case("2412"), make_case("4412")]
    monkeypatch.setattr(server.harness, "submit",
                        lambda s, h, case_timeout=180.0: batch(*[infra_result(c) for c in cases]))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert (out["status"], out["failure_kind"]) == ("error", "infrastructure")
    assert out["errors"] and len(out["results"]) == 2
    assert_envelope("run_campaign all failed", out)


def test_a_thermal_failure_makes_the_case_an_error_in_counts_and_in_its_row(monkeypatch):
    case = make_case()
    monkeypatch.setattr(server.harness, "submit", lambda s, h, case_timeout=180.0: batch(thermal_failed(case)))
    out = call("run_campaign", source="src", approval_hash="abc123")
    assert out["counts"] == {"ok": 0, "partial": 0, "empty": 0, "error": 1}
    row = out["results"][0]
    assert (row["status"], row["failure_kind"], row["retry_could_help"]) == ("error", "infrastructure", True)
    assert row["converged_points"] == 11                    # the aero numbers are still there


def test_the_case_view_shows_the_heater_a_person_is_asked_to_agree_to(monkeypatch):
    heated = Case(geometry=Geometry(naca="2412"), conditions=make_case().conditions,
                  outputs=("forces", "bl"), thermal=THERMAL_BLOCK)
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report([heated]))
    view = call("dry_run_campaign", source="src")["cases"][0]
    assert view["thermal"]["heater_power_w_per_m"] == 500.0
    assert (view["mach"], view["n_crit"], view["max_iter"]) == (0.0, 9.0, 100)


THERMAL_BLOCK = dict(chord_m=0.5, air_temperature_k=263.15, heater_width=0.1, heater_power_w_per_m=500.0,
                     skin_thickness_m=0.001, skin_conductivity_w_mk=200.0)


def test_cases_that_differ_only_by_a_flap_look_different_at_approval(monkeypatch):
    """What the person is asked to agree to. Without the flap, these two rows
    are told apart only by a label the agent wrote."""
    clean = make_case()
    flapped = Case(geometry=Geometry(naca="2412", flap=Flap(x_hinge=0.7, deflection=10)),
                   conditions=clean.conditions, label=clean.label)
    monkeypatch.setattr(server.harness, "dry_run", lambda s, timeout=30.0: report([clean, flapped]))
    views = call("dry_run_campaign", source="src")["cases"]
    strip = lambda v: {k: v[k] for k in v if k != "content_hash"}
    assert strip(views[0]) != strip(views[1])


HOST_MODULES = ["server", "harness", "sandbox", "dispatch", "containers", "printing", "print_site", "cad", "schema"]


@pytest.mark.parametrize("module", HOST_MODULES)
def test_no_host_module_can_write_to_stdout(module):
    """The scan in section 5 reads four modules and only lines that start
    with print(. This one parses every module the server process imports and
    refuses any print() without file=, and any use of sys.stdout at all."""
    tree = ast.parse((Path(server.__file__).parent / f"{module}.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            assert any(k.arg == "file" for k in node.keywords), f"{module}.py:{node.lineno} prints to stdout"
        if isinstance(node, ast.Attribute) and node.attr == "stdout" and isinstance(node.value, ast.Name) \
                and node.value.id == "sys":
            pytest.fail(f"{module}.py:{node.lineno} uses sys.stdout")
