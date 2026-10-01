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

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client

from xfoil_mcp import harness, server
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry

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


def report(cases, rejected=(), errors=(), killed=False, approval_hash="abc123") -> harness.DryRunReport:
    points = sum(c.conditions.point_count() for c in cases)
    return harness.DryRunReport(
        cases=list(cases), rejected=list(rejected), errors=list(errors), killed=killed,
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
    assert set(out["cases"][0]) == {"content_hash", "label", "naca", "reynolds", "alphas", "outputs"}


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
        [], errors=["docker not found on host"]))
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

    assert out["status"] == "ok"
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

@pytest.mark.parametrize("module", ["server", "harness", "sandbox", "dispatch"])
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

def test_default_outbox_is_at_the_repository_root():
    repo_root = Path(__file__).resolve().parent.parent
    assert server._default_outbox() == repo_root / "outbox"
