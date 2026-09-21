"""Server tests. Harness and dispatch are mocked; no Docker needed.

Tools are invoked through FastMCP's in-memory client rather than by calling
the Python functions directly, so what is tested is what a real MCP client
would see: the tool is registered, the arguments validate, and the payload
has the shape the docstrings promise.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastmcp import Client

from xfoil_mcp import harness, server
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry


def call(tool: str, **args) -> dict:
    async def _go():
        async with Client(server.mcp) as client:
            result = await client.call_tool(tool, args)
            if getattr(result, "data", None) is not None:
                return result.data
            return json.loads(result.content[0].text)
    return asyncio.run(_go())


def tool_names() -> set[str]:
    async def _go():
        async with Client(server.mcp) as client:
            return {t.name for t in await client.list_tools()}
    return asyncio.run(_go())


def make_case(naca="2412") -> Case:
    return Case(
        geometry=Geometry(naca=naca),
        conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=1),
        label=f"case {naca}",
    )


def ok_result(case: Case) -> CaseResult:
    return CaseResult(
        case=case, status="ok",
        summary={"converged_points": 11, "requested_points": 11, "best_ld": 93.0},
        data={"points": ["large"]},
    )


@pytest.fixture(autouse=True)
def reset_batch():
    server._last_batch = None
    yield
    server._last_batch = None


# --- registration ----------------------------------------------------------

def test_four_tools_are_registered():
    assert tool_names() == {"run_polar", "dry_run_campaign", "run_campaign", "get_case_detail"}


# --- run_polar -------------------------------------------------------------

def test_run_polar_builds_a_valid_case_and_returns_summary(monkeypatch):
    seen = {}

    def fake_run_case(case, timeout=180.0):
        seen["case"] = case
        return ok_result(case)

    monkeypatch.setattr(server.dispatch, "run_case", fake_run_case)
    out = call("run_polar", airfoil="2412", reynolds=1e6, alpha_end=10)

    assert seen["case"].geometry.naca == "2412"
    assert seen["case"].conditions.reynolds == 1e6
    assert out["status"] == "ok"
    assert out["retry_could_help"] is False
    assert out["best_ld"] == 93.0
    assert "data" not in out and "points" not in out

def test_run_polar_rejects_invalid_arguments_before_dispatch(monkeypatch):
    called = []
    monkeypatch.setattr(server.dispatch, "run_case", lambda c, timeout=180.0: called.append(c))
    out = call("run_polar", airfoil="24a2", reynolds=1e6)
    assert called == []
    assert out["status"] == "error"
    assert out["failure_kind"] == "input"
    assert any("naca" in e for e in out["errors"])

def test_run_polar_reports_all_input_errors_at_once(monkeypatch):
    monkeypatch.setattr(server.dispatch, "run_case", lambda c, timeout=180.0: pytest.fail("dispatched"))
    out = call("run_polar", airfoil="24a2", reynolds=-1)
    assert out["failure_kind"] == "input"
    assert any("naca" in e for e in out["errors"])
    assert any("reynolds" in e for e in out["errors"])


# --- dry_run_campaign ------------------------------------------------------

def test_dry_run_returns_hash_estimate_and_case_list(monkeypatch):
    cases = [make_case("2412"), make_case("0012")]
    monkeypatch.setattr(server.harness, "dry_run", lambda src, timeout=30.0: harness.DryRunReport(
        cases=cases, rejected=[], errors=[], killed=False,
        estimate=harness.Estimate(case_count=2, alpha_points=22, expected_seconds=4.1),
        approval_hash="abc123",
    ))
    out = call("dry_run_campaign", source="def campaign(): ...")

    assert out["runnable"] is True
    assert out["approval_hash"] == "abc123"
    assert out["estimate"]["case_count"] == 2
    assert [c["naca"] for c in out["cases"]] == ["2412", "0012"]
    assert out["errors"] == [] and out["rejected"] == []


def test_dry_run_surfaces_sandbox_errors(monkeypatch):
    monkeypatch.setattr(server.harness, "dry_run", lambda src, timeout=30.0: harness.DryRunReport(
        cases=[], rejected=[], errors=["campaign() raised: boom"], killed=False,
        estimate=harness.Estimate(0, 0, 0.0), approval_hash="e3b0",
    ))
    out = call("dry_run_campaign", source="bad")
    assert out["runnable"] is False
    assert "boom" in out["errors"][0]


# --- run_campaign ----------------------------------------------------------

def test_run_campaign_refuses_stale_hash(monkeypatch):
    def refuse(src, approval_hash, case_timeout=180.0):
        raise harness.ApprovalMismatch("hash mismatch")
    monkeypatch.setattr(server.harness, "submit", refuse)

    out = call("run_campaign", source="src", approval_hash="stale")
    assert out["complete"] is False
    assert "mismatch" in out["refused"]
    assert out["results"] == []


def test_run_campaign_returns_summaries_with_counts(monkeypatch):
    cases = [make_case("2412"), make_case("0012")]
    results = [
        ok_result(cases[0]),
        CaseResult(case=cases[1], status="partial", failure_kind="numerical",
                   summary={"converged_points": 8}, data={"points": ["large"]}),
    ]
    monkeypatch.setattr(server.harness, "submit",
                        lambda src, h, case_timeout=180.0: harness.BatchResult(results=results, approval_hash=h))

    out = call("run_campaign", source="src", approval_hash="ok")
    assert out["complete"] is True
    assert out["submitted"] == 2
    assert out["counts"] == {"ok": 1, "partial": 1, "empty": 0, "error": 0}
    assert all("data" not in r for r in out["results"])
    assert out["results"][1]["retry_could_help"] is False


# --- get_case_detail -------------------------------------------------------

def test_case_detail_before_any_campaign():
    out = call("get_case_detail", content_hash="anything")
    assert "error" in out


def test_case_detail_returns_full_result(monkeypatch):
    case = make_case()
    monkeypatch.setattr(server.harness, "submit",
                        lambda src, h, case_timeout=180.0: harness.BatchResult(results=[ok_result(case)], approval_hash=h))
    call("run_campaign", source="src", approval_hash="ok")

    out = call("get_case_detail", content_hash=case.content_hash())
    assert out["status"] == "ok"
    assert out["data"] == {"points": ["large"]}

    missing = call("get_case_detail", content_hash="nope")
    assert "error" in missing
