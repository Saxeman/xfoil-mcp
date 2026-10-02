"""Harness tests. Sandbox and dispatch are mocked; no Docker needed.

What they prove: the approval hash gates submit, dedup works, counts and
summaries are right, and full data stays out of summaries.
"""

from __future__ import annotations

import pytest

from xfoil_mcp import harness, sandbox
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry, ThermalResult

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
    with pytest.raises(harness.ApprovalMismatch, match="not runnable: boom"):
        harness.submit("src", approval_hash="anything")


def test_submit_says_why_an_empty_campaign_is_not_runnable(monkeypatch):
    """No errors and no cases used to leave the reason blank."""
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([]))
    with pytest.raises(harness.ApprovalMismatch, match="returned no cases"):
        harness.submit("src", approval_hash="anything")


def test_submit_passes_its_dry_run_timeout_to_the_sandbox(monkeypatch):
    seen = []

    def run(source, timeout=30.0):
        seen.append(timeout)
        return sandbox.SandboxOutcome(cases=[make_case()])
    monkeypatch.setattr(sandbox, "run_campaign_source", run)
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({}))
    approved = harness.dry_run("src").approval_hash
    harness.submit("src", approved)
    harness.submit("src", approved, dry_run_timeout=5.0)
    assert seen == [30.0, 30.0, 5.0]


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


def test_a_summary_cannot_override_the_verdict_in_summaries():
    """The summary dict comes from a container; status and kind come from
    the validated result."""
    case = make_case()
    result = CaseResult(case=case, status="error", failure_kind="infrastructure", summary={
        "status": "ok", "failure_kind": None, "retry_could_help": False, "content_hash": "forged",
        "error": "worker exited 1",
    })
    row = harness.BatchResult(results=[result], approval_hash="h").summaries()[0]
    assert (row["status"], row["failure_kind"], row["retry_could_help"]) == ("error", "infrastructure", True)
    assert row["content_hash"] == case.content_hash()
    assert row["error"] == "worker exited 1"


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
        return ThermalResult(
            status="ok", failure_kind=None, errors=[], coverage=1.0,
            worst_case={"min_heated_temperature_c": 24.8, "at_alpha": 4.0},
            points=[{"alpha": 4.0}], excluded=[], heater_power_w_per_m=500.0,
            content_hash=aero.case.content_hash(),
        )
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
    stored = batch.get(case.content_hash())
    assert stored.data["thermal"]["points"] == [{"alpha": 4.0}]
    assert stored.data["thermal"]["content_hash"] == case.content_hash()
    # What is stored is plain JSON, so the whole result still round-trips for get_case_detail.
    assert CaseResult.model_validate_json(stored.model_dump_json()) == stored


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


# --- gaps in the tests above --------------------------------------------------

def test_dry_run_is_not_runnable_with_errors_even_when_cases_validated(monkeypatch):
    """The errors test above also has zero cases, so it would pass if errors were ignored."""
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([make_case()], errors=["boom"]))
    report = harness.dry_run("src")
    assert len(report.cases) == 1 and not report.runnable


def test_dry_run_is_not_runnable_when_the_host_rejected_a_case(monkeypatch):
    monkeypatch.setattr(sandbox, "run_campaign_source",
                        fake_sandbox([make_case()], rejected=["case 1: reynolds must be > 0"]))
    report = harness.dry_run("src")
    assert report.rejected == ["case 1: reynolds must be > 0"] and not report.runnable
    with pytest.raises(harness.ApprovalMismatch, match="case 1: reynolds"):
        harness.submit("src", report.approval_hash)


def test_the_summaries_the_model_reads_carry_the_retry_guidance(monkeypatch):
    """The test of this name above reads batch.results. This one reads what the model gets."""
    cases = [make_case(naca="0012"), make_case(naca="4412"), make_case(naca="2412")]
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox(cases))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({
        "0012": ("partial", "numerical"), "4412": ("error", "infrastructure")}))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)
    guidance = {s["label"] or s["content_hash"]: (s["status"], s["failure_kind"], s["retry_could_help"])
                for s in batch.summaries()}
    assert sorted(guidance.values(), key=str) == sorted([
        ("partial", "numerical", False), ("error", "infrastructure", True), ("ok", None, False)], key=str)
    assert batch.counts == {"ok": 1, "partial": 1, "empty": 0, "error": 1}


def test_a_partial_aero_result_still_goes_to_the_thermal_stage(monkeypatch):
    """Partial means some alphas converged, and those can be heated."""
    calls = []
    case = make_case(thermal=THERMAL)
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([case]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({"2412": ("partial", "numerical")}))
    monkeypatch.setattr(harness.dispatch, "run_thermal", fake_thermal(calls))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)
    assert len(calls) == 1 and calls[0].status == "partial"
    assert batch.summaries()[0]["thermal"]["status"] == "ok"


def test_an_empty_aero_result_is_not_sent_to_the_thermal_stage(monkeypatch):
    calls = []
    case = make_case(thermal=THERMAL)
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([case]))
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({"2412": ("empty", "numerical")}))
    monkeypatch.setattr(harness.dispatch, "run_thermal", fake_thermal(calls))
    batch = harness.submit("src", harness.dry_run("src").approval_hash)
    assert calls == []
    thermal = batch.summaries()[0]["thermal"]
    assert thermal["status"] == "error" and "status empty" in thermal["errors"][0]


def test_evaluate_runs_one_case_through_both_stages(monkeypatch):
    calls = []
    case = make_case(thermal=THERMAL)
    monkeypatch.setattr(harness.dispatch, "run_case", fake_dispatch({}))
    monkeypatch.setattr(harness.dispatch, "run_thermal", fake_thermal(calls))
    result = harness.evaluate(case, timeout=42.0)
    assert result.status == "ok" and len(calls) == 1
    assert result.data["thermal"]["coverage"] == 1.0


def test_an_unexpected_exception_in_one_case_does_not_lose_the_others(monkeypatch):
    cases = [make_case(naca="2412"), make_case(naca="4412"), make_case(naca="0012")]
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox(cases))
    healthy = fake_dispatch({})

    def run_case(case, timeout=180.0):
        if case.geometry.naca == "4412":
            raise RuntimeError("something nobody anticipated")
        return healthy(case, timeout)
    monkeypatch.setattr(harness.dispatch, "run_case", run_case)
    batch = harness.submit("src", harness.dry_run("src").approval_hash)
    assert batch.counts == {"ok": 2, "partial": 0, "empty": 0, "error": 1}


def result_with(case, aero_status="ok", thermal_status=None, thermal_kind=None) -> CaseResult:
    kind = None if aero_status == "ok" else "numerical" if aero_status in ("partial", "empty") else "infrastructure"
    summary = {} if thermal_status is None else {"thermal": {"status": thermal_status, "failure_kind": thermal_kind}}
    return CaseResult(case=case, status=aero_status, failure_kind=kind, summary=summary)


@pytest.mark.parametrize("aero, thermal, thermal_kind, expected", [
    ("ok", None, None, ("ok", None)),                               # no heater: the aero result is the verdict
    ("partial", None, None, ("partial", "numerical")),
    ("ok", "ok", None, ("ok", None)),
    ("ok", "error", "infrastructure", ("error", "infrastructure")),  # the heater analysis is half the answer
    ("ok", "error", "input", ("error", "input")),
    ("ok", "partial", "numerical", ("partial", "numerical")),
    ("ok", "empty", "numerical", ("partial", "numerical")),
    ("partial", "ok", None, ("partial", "numerical")),
    ("partial", "error", "numerical", ("error", "numerical")),
    ("error", "error", "input", ("error", "infrastructure")),        # aero failed first: its kind, not the gate's
    ("empty", "error", "input", ("empty", "numerical")),
])
def test_a_case_verdict_is_the_first_stage_that_did_not_pass(aero, thermal, thermal_kind, expected):
    assert harness.verdict(result_with(make_case(), aero, thermal, thermal_kind)) == expected


def test_counts_and_the_batch_kind_follow_the_verdicts():
    cases = [make_case(naca=n) for n in ("2412", "4412", "0012", "0015")]
    batch = harness.BatchResult(approval_hash="h", results=[
        result_with(cases[0]),
        result_with(cases[1], thermal_status="error", thermal_kind="input"),
        result_with(cases[2], aero_status="partial"),
        result_with(cases[3], thermal_status="empty", thermal_kind="numerical"),
    ])
    assert batch.counts == {"ok": 1, "partial": 2, "empty": 0, "error": 1}
    assert batch.failure_kind == "numerical"                # no infrastructure failure; numerical outranks input
    assert harness.BatchResult(results=[result_with(cases[0])], approval_hash="h").failure_kind is None


def test_the_contained_case_reports_what_went_wrong(monkeypatch):
    def run_case(case, timeout=180.0):
        raise RuntimeError("something nobody anticipated")
    monkeypatch.setattr(harness.dispatch, "run_case", run_case)
    case = make_case()
    result = harness._evaluate_contained(case, 180.0)
    assert (result.status, result.failure_kind) == ("error", "infrastructure")
    assert result.summary["error"] == "unexpected RuntimeError while evaluating this case: something nobody anticipated"
    assert result.case == case


def test_a_refused_rerun_says_whose_fault_it_was(monkeypatch):
    """submit raises CampaignNotRunnable, which is still an ApprovalMismatch,
    with the kind the server should report."""
    monkeypatch.setattr(sandbox, "run_campaign_source",
                        lambda s, timeout=30.0: sandbox.SandboxOutcome(errors=["docker not found on host"], infrastructure=True))
    with pytest.raises(harness.CampaignNotRunnable) as docker_down:
        harness.submit("src", "anything")
    monkeypatch.setattr(sandbox, "run_campaign_source", fake_sandbox([], errors=["campaign() raised: boom"]))
    with pytest.raises(harness.CampaignNotRunnable) as campaign_broken:
        harness.submit("src", "anything")
    assert (docker_down.value.failure_kind, campaign_broken.value.failure_kind) == ("infrastructure", "input")
    assert isinstance(docker_down.value, harness.ApprovalMismatch)
