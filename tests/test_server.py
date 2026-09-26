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

import pytest
from fastmcp import Client

from xfoil_mcp import harness, server
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry

ENVELOPE = {"status", "failure_kind", "retry_could_help", "errors"}
TOOLS = {"run_polar", "dry_run_campaign", "run_campaign", "get_case_detail"}


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
    monkeypatch.setattr(server.dispatch, "run_case",
                        lambda c, timeout=180.0: pytest.fail("dispatch was called"))


# ==========================================================================
# 1. registration
# ==========================================================================

def test_exactly_four_tools_are_registered():
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
    }


# ==========================================================================
# 2. envelope
# ==========================================================================

def _every_response(monkeypatch) -> list[tuple[str, dict]]:
    """One ok response and one error response from each tool."""
    case = make_case()
    out: list[tuple[str, dict]] = []

    monkeypatch.setattr(server.dispatch, "run_case", lambda c, timeout=180.0: ok_result(c))
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
    monkeypatch.setattr(server.dispatch, "run_case", capture)

    out = call("run_polar", airfoil="4412", reynolds=5e5, alpha_end=12, n_crit=4)
    c = seen["case"]
    assert c.geometry.naca == "4412"
    assert c.conditions.reynolds == 5e5
    assert c.conditions.alpha_end == 12
    assert c.conditions.n_crit == 4
    assert out["content_hash"] == c.content_hash()


def test_run_polar_summary_reaches_the_model_without_data(monkeypatch):
    monkeypatch.setattr(server.dispatch, "run_case", lambda c, timeout=180.0: ok_result(c))
    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert out["status"] == "ok"
    assert out["best_ld"] == 104.4
    assert out["converged_points"] == 11
    assert "data" not in out and "points" not in out


def test_run_polar_partial_result_passes_through_with_failed_alphas(monkeypatch):
    monkeypatch.setattr(server.dispatch, "run_case", lambda c, timeout=180.0: partial_result(c))
    out = call("run_polar", airfoil="2412", reynolds=1e6)
    assert out["status"] == "partial"
    assert out["failure_kind"] == "numerical"
    assert out["retry_could_help"] is False
    assert out["failed_alphas"] == [8.0, 9.0, 10.0]
    assert out["failure_runs"] == [[8.0, 9.0, 10.0]]


def test_run_polar_infrastructure_failure_is_retryable(monkeypatch):
    monkeypatch.setattr(server.dispatch, "run_case", lambda c, timeout=180.0: infra_result(c))
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
# tests/test_server.py, in the registration section
from pathlib import Path

@pytest.mark.parametrize("module", ["server", "harness", "sandbox", "dispatch"])
def test_host_modules_never_print_to_stdout(module):
    """Over stdio, stdout is the protocol channel. One stray print corrupts
    the stream and every call after it fails."""
    src = (Path(server.__file__).parent / f"{module}.py").read_text()
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("print(") and "file=sys.stderr" not in stripped:
            pytest.fail(f"{module}.py:{lineno} prints to stdout: {stripped}")
