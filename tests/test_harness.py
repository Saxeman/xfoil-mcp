"""Harness tests. Sandbox and dispatch are mocked; no Docker needed.

What they prove: the approval hash gates submit, dedup works, counts and
summaries are right, and full data stays out of summaries.
"""

from __future__ import annotations

import pytest

from xfoil_mcp import harness, sandbox
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry

from builders import THERMAL


def make_case(naca="2412", reynolds=1e6, label=None, thermal=None) -> Case:
    return Case(
        geometry=Geometry(naca=naca),
        conditions=Conditions(reynolds=reynolds, alpha_start=0, alpha_end=10, alpha_step=1),
        outputs=("forces", "bl") if thermal else ("forces",),
        label=label,
        thermal=thermal,
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

# --- thermal chaining -------------------------------------------------------

def fake_thermal(calls):
    """Stand in for dispatch.run_thermal, recording which aero results it was given."""
    def _run(aero, timeout=180.0):
        calls.append(aero)
        return {
            "status": "ok", "failure_kind": None, "errors": [], "coverage": 1.0,
            "worst_case": {"min_heated_temperature_c": 24.8, "at_alpha": 4.0},
            "points": [{"alpha": 4.0}], "excluded": [], "heater_power_w_per_m": 500.0,
            "content_hash": aero.case.content_hash(),
        }
    return _run


def test_case_without_thermal_never_launches_the_thermal_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({}))
    monkeypatch.setattr(harness.dispatch, "run_thermal", fake_thermal(calls))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)
    assert calls == []
    assert "thermal" not in batch.summaries()[0]


def test_thermal_case_runs_thermal_on_its_own_aero_result(monkeypatch):
    calls = []
    case = make_case(thermal=THERMAL)
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([case]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({}))
    monkeypatch.setattr(harness.dispatch, "run_thermal", fake_thermal(calls))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)

    assert len(calls) == 1 and calls[0].case == case
    summary = batch.summaries()[0]
    assert summary["thermal"]["status"] == "ok"
    assert summary["thermal"]["worst_case"]["min_heated_temperature_c"] == 24.8
    assert "points" not in summary["thermal"]                       # verdict only
    assert batch.get(case.content_hash()).data["thermal"]["points"] == [{"alpha": 4.0}]


def test_failed_aero_is_not_sent_to_the_thermal_stage(monkeypatch):
    calls = []
    case = make_case(naca="4412", thermal=THERMAL)
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([case]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({"4412": ("error", "infrastructure")}))
    monkeypatch.setattr(harness.dispatch, "run_thermal", fake_thermal(calls))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)

    assert calls == []
    thermal = batch.summaries()[0]["thermal"]
    assert thermal["status"] == "error"
    assert thermal["failure_kind"] == "input"
    assert "upstream" in thermal["errors"][0]


def test_estimate_prices_the_thermal_stage(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()]))
    plain = harness.dry_run("src").estimate
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case(thermal=THERMAL)]))
    heated = harness.dry_run("src").estimate
    assert heated.thermal_cases == 1
    assert heated.expected_seconds > plain.expected_seconds
