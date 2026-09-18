"""Harness tests. Sandbox and dispatch are mocked; no Docker needed.

What they prove: the approval hash gates submit, dedup works, counts and
summaries are right, and full data stays out of summaries.
"""

from __future__ import annotations

import pytest

from xfoil_mcp import harness, sandbox
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry


def make_case(naca="2412", reynolds=1e6, label=None) -> Case:
    return Case(
        geometry=Geometry(naca=naca),
        conditions=Conditions(reynolds=reynolds, alpha_start=0, alpha_end=10, alpha_step=1),
        label=label,
    )


def fake_sandbox(cases, errors=(), rejected=(), killed=False):
    def _run(source, timeout=30.0):
        return sandbox.SandboxOutcome(
            cases=list(cases), errors=list(errors), rejected=list(rejected), killed=killed,
        )
    return _run


def fake_dispatch(status_for):
    """status_for: dict naca -> (status, failure_kind)"""
    def _run(case, timeout=180.0):
        status, kind = status_for.get(case.geometry.naca, ("ok", None))
        return CaseResult(
            case=case, status=status, failure_kind=kind,
            summary={"converged_points": 11 if status == "ok" else 8},
            data={"points": ["would be large"]},
        )
    return _run


# --- dry run ---------------------------------------------------------------

def test_dry_run_dedups_identical_cases(monkeypatch):
    dup = make_case(label="a")
    same = make_case(label="b")            # different label, same hash
    other = make_case(naca="0012")
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([dup, same, other]))

    report = harness.dry_run("src")
    assert len(report.cases) == 2
    assert report.cases[0].label == "a"     # first occurrence wins
    assert report.runnable


def test_dry_run_estimate_counts_points(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case(), make_case(naca="0012")]))
    report = harness.dry_run("src")
    assert report.estimate.case_count == 2
    assert report.estimate.alpha_points == 22
    assert report.estimate.expected_seconds > 0


def test_dry_run_is_not_runnable_with_errors(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([], errors=["boom"]))
    assert not harness.dry_run("src").runnable


def test_dry_run_is_not_runnable_when_killed(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()], killed=True))
    assert not harness.dry_run("src").runnable


def test_approval_hash_depends_on_case_list(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()]))
    a = harness.dry_run("src").approval_hash
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case(reynolds=2e6)]))
    b = harness.dry_run("src").approval_hash
    assert a != b


# --- submit ----------------------------------------------------------------

def test_submit_refuses_wrong_hash(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()]))
    with pytest.raises(harness.ApprovalMismatch):
        harness.submit("src", approval_hash="not-the-hash")


def test_submit_refuses_unrunnable_campaign(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([], errors=["boom"]))
    with pytest.raises(harness.ApprovalMismatch):
        harness.submit("src", approval_hash="anything")


def test_submit_runs_approved_campaign(monkeypatch):
    cases = [make_case(), make_case(naca="0012"), make_case(naca="4412")]
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox(cases))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({
        "0012": ("partial", "numerical"),
        "4412": ("error", "infrastructure"),
    }))

    report = harness.dry_run("src")
    batch = harness.submit("src", report.approval_hash)

    assert batch.submitted == 3
    assert batch.complete
    assert batch.counts == {"ok": 1, "partial": 1, "empty": 0, "error": 1}


def test_summaries_exclude_full_data(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({}))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)

    s = batch.summaries()[0]
    assert "data" not in s
    assert "points" not in s
    assert s["status"] == "ok"
    assert s["retry_could_help"] is False


def test_summaries_carry_retry_guidance(monkeypatch):
    cases = [make_case(naca="0012"), make_case(naca="4412")]
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox(cases))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({
        "0012": ("partial", "numerical"),
        "4412": ("error", "infrastructure"),
    }))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)
    by_naca = {r.case.geometry.naca: r for r in batch.results}
    assert by_naca["0012"].retry_could_help is False
    assert by_naca["4412"].retry_could_help is True


def test_get_returns_full_result(monkeypatch):
    case = make_case()
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([case]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({}))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)

    full = batch.get(case.content_hash())
    assert full is not None
    assert full.data == {"points": ["would be large"]}
    assert batch.get("nope") is None
