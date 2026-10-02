"""The stable API the MCP server calls. Runs on the host.

dry_run  - execute a campaign in the sandbox, return what it would submit
           plus a cost estimate and an approval hash. Runs no solver.
submit   - re-run the dry run, refuse unless the approval hash matches, then
           dispatch each case to a worker and collect the results.

The approval hash makes the human checkpoint structural: submit cannot be
called on a case list nobody has seen, because the hash exists only as the
output of a dry run and changes if the campaign changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from xfoil_mcp import dispatch, sandbox
from xfoil_mcp.schema import Case, CaseResult, ThermalResult

# Rough cost model. Container start dominates for XFOIL; the solve is cheap.
SECONDS_PER_CONTAINER = 1.5
SECONDS_PER_ALPHA_POINT = 0.05

SECONDS_PER_THERMAL_CONTAINER = 1.5
SECONDS_PER_THERMAL_POINT = 0.1
# The part of a thermal result the model reads. Per-alpha points stay in
# data, reachable through get_case_detail.
THERMAL_SUMMARY_KEYS = (
    "status", "failure_kind", "errors", "coverage", "worst_case", "excluded", "heater_power_w_per_m",
)

class ApprovalMismatch(Exception):
    """The campaign changed since it was approved, or was never approved."""


class CampaignNotRunnable(ApprovalMismatch):
    """The campaign could not be dry-run again, so there is no case list to
    hold an approval against. `failure_kind` says whose fault that is:
    "infrastructure" when docker could not run the sandbox, "input" when the
    campaign itself is at fault."""

    def __init__(self, message: str, failure_kind: str):
        super().__init__(message)
        self.failure_kind = failure_kind


@dataclass(frozen=True)
class Estimate:
    case_count: int
    alpha_points: int
    expected_seconds: float
    thermal_cases: int = 0


@dataclass
class DryRunReport:
    cases: list[Case]
    rejected: list[str]
    errors: list[str]
    estimate: Estimate
    approval_hash: str
    killed: bool = False
    infrastructure: bool = False    # docker could not run the sandbox; the campaign is not at fault

    @property
    def runnable(self) -> bool:
        return bool(self.cases) and not self.errors and not self.rejected and not self.killed


def verdict(result: CaseResult) -> tuple[str, str | None]:
    """A case's overall status and failure kind: the first stage that did not pass.

    A CaseResult's own status is the aero stage's. For a case with a heater
    that is half the answer, so the thermal verdict is read as well: a
    thermal error makes the case an error, and incomplete thermal coverage
    makes it partial. Without this a campaign whose every thermal stage
    failed would count as all ok.
    """
    if result.status in ("error", "empty"):
        return result.status, result.failure_kind
    thermal = result.summary.get("thermal")
    if thermal is not None:
        if thermal["status"] == "error":
            return "error", thermal.get("failure_kind") or "infrastructure"
        if thermal["status"] in ("partial", "empty"):
            return "partial", "numerical"
    return result.status, result.failure_kind


@dataclass
class BatchResult:
    results: list[CaseResult]
    approval_hash: str

    @property
    def submitted(self) -> int:
        return len(self.results)

    @property
    def complete(self) -> bool:
        """Every case reached a terminal state. Always true while dispatch is
        sequential, because submit returns only after the last case. It stays
        because the payload contract includes it and a queued implementation
        will need a real answer here."""
        return True

    @property
    def counts(self) -> dict[str, int]:
        """How many cases ended in each overall status (see `verdict`)."""
        out: dict[str, int] = {"ok": 0, "partial": 0, "empty": 0, "error": 0}
        for r in self.results:
            out[verdict(r)[0]] += 1
        return out

    @property
    def failure_kind(self) -> str | None:
        """One kind for the batch: the most actionable among its cases.
        Infrastructure first, because a retry may help; None when all passed."""
        kinds = {verdict(r)[1] for r in self.results}
        return next((k for k in ("infrastructure", "numerical", "input") if k in kinds), None)

    def summaries(self) -> list[dict]:
        """What the model reads. Full data stays out.

        The five keys set here come from the validated result: status and
        kind are the case's overall verdict, across both stages. A summary
        is a container's own dict, so a key it shares with them is dropped
        rather than allowed to override the verdict.
        """
        rows = []
        for r in self.results:
            status, kind = verdict(r)
            row = {
                "content_hash": r.case.content_hash(),
                "label": r.case.label,
                "status": status,
                "failure_kind": kind,
                "retry_could_help": kind == "infrastructure",
            }
            row.update((k, v) for k, v in r.summary.items() if k not in row)
            rows.append(row)
        return rows

    def get(self, content_hash: str) -> CaseResult | None:
        for r in self.results:
            if r.case.content_hash() == content_hash:
                return r
        return None


def _dedup(cases: list[Case]) -> list[Case]:
    seen: set[str] = set()
    out: list[Case] = []
    for c in cases:
        h = c.content_hash()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out

def _estimate(cases: list[Case]) -> Estimate:
    """Priced as if every alpha passes every gate: the most the run can cost."""
    points = sum(c.conditions.point_count() for c in cases)
    thermal = [c for c in cases if c.thermal is not None]
    thermal_points = sum(c.conditions.point_count() for c in thermal)
    seconds = (
        len(cases) * SECONDS_PER_CONTAINER + points * SECONDS_PER_ALPHA_POINT
        + len(thermal) * SECONDS_PER_THERMAL_CONTAINER + thermal_points * SECONDS_PER_THERMAL_POINT
    )
    return Estimate(case_count=len(cases), alpha_points=points,
                    expected_seconds=round(seconds, 1), thermal_cases=len(thermal))

def _with_thermal(aero: CaseResult, timeout: float) -> CaseResult:
    """Run the thermal stage on a finished aero result and attach it.

    The aero gate is checked here first: a failed aero result never
    launches a thermal container. The model's summary gets the verdict;
    the per-alpha detail goes to data.
    """
    if aero.status in ("error", "empty"):
        thermal = ThermalResult.failure(
            "input", f"upstream aero stage did not pass (status {aero.status})")
    else:
        thermal = dispatch.run_thermal(aero, timeout=timeout)

    detail = thermal.model_dump(mode="json")
    summary = {**aero.summary, "thermal": {k: detail[k] for k in THERMAL_SUMMARY_KEYS}}
    data = {**(aero.data or {}), "thermal": detail}
    return aero.model_copy(update={"summary": summary, "data": data})

def _approval_hash(cases: list[Case]) -> str:
    joined = "\n".join(c.content_hash() for c in cases)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]

def evaluate(case: Case, timeout: float = 180.0) -> CaseResult:
    """Run every stage a case asks for, in order: aero, then thermal if it has a heater."""
    result = dispatch.run_case(case, timeout=timeout)
    if case.thermal is not None:
        result = _with_thermal(result, timeout)
    return result


def dry_run(source: str, timeout: float = 30.0) -> DryRunReport:
    outcome = sandbox.run_campaign_source(source, timeout=timeout)
    cases = _dedup(outcome.cases)
    return DryRunReport(
        cases=cases,
        rejected=outcome.rejected,
        errors=outcome.errors,
        estimate=_estimate(cases),
        approval_hash=_approval_hash(cases),
        killed=outcome.killed,
        infrastructure=outcome.infrastructure,
    )


def submit(source: str, approval_hash: str, case_timeout: float = 180.0,
           dry_run_timeout: float = 30.0) -> BatchResult:
    """Run an approved campaign.

    The dry run is repeated rather than cached so that the case list actually
    executed is the one just produced from the source, and the hash check
    proves it is the one that was approved.
    """
    report = dry_run(source, timeout=dry_run_timeout)
    if not report.runnable:
        reasons = "; ".join(report.errors + report.rejected) or "campaign() returned no cases"
        raise CampaignNotRunnable(
            f"campaign is not runnable: {reasons}" + (" (killed)" if report.killed else ""),
            failure_kind="infrastructure" if report.infrastructure else "input",
        )
    if report.approval_hash != approval_hash:
        raise ApprovalMismatch(
            f"approval hash {approval_hash} does not match current campaign "
            f"{report.approval_hash}; re-run dry_run and approve again"
        )

    results = [_evaluate_contained(case, case_timeout) for case in report.cases]
    return BatchResult(results=results, approval_hash=approval_hash)


def _evaluate_contained(case: Case, timeout: float) -> CaseResult:
    """`evaluate`, with anything nobody anticipated turned into that case's
    own result. One case must not cost the batch the results of the others."""
    try:
        return evaluate(case, timeout=timeout)
    except Exception as exc:
        return CaseResult(
            case=case, status="error", failure_kind="infrastructure",
            summary={"error": f"unexpected {type(exc).__name__} while evaluating this case: {exc}"},
            provenance={"content_hash": case.content_hash()},
        )
