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
from dataclasses import dataclass, field

from xfoil_mcp import dispatch, sandbox
from xfoil_mcp.schema import Case, CaseResult

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

    @property
    def runnable(self) -> bool:
        return bool(self.cases) and not self.errors and not self.rejected and not self.killed


@dataclass
class BatchResult:
    results: list[CaseResult]
    approval_hash: str

    @property
    def submitted(self) -> int:
        return len(self.results)

    @property
    def complete(self) -> bool:
        """Every case reached a terminal state. Sequential dispatch means this
        is always true once submit returns; it is here because a queued
        implementation will need it, and the payload contract includes it."""
        return all(r.status in ("ok", "partial", "empty", "error") for r in self.results)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"ok": 0, "partial": 0, "empty": 0, "error": 0}
        for r in self.results:
            out[r.status] += 1
        return out

    def summaries(self) -> list[dict]:
        """What the model reads. Full data stays out."""
        return [
            {
                "content_hash": r.case.content_hash(),
                "label": r.case.label,
                "status": r.status,
                "failure_kind": r.failure_kind,
                "retry_could_help": r.retry_could_help,
                **r.summary,
            }
            for r in self.results
        ]

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
        thermal = {
            "status": "error", "failure_kind": "input",
            "errors": [f"upstream aero stage did not pass (status {aero.status})"],
            "coverage": 0.0, "worst_case": None, "points": [], "excluded": [],
        }
    else:
        thermal = dispatch.run_thermal(aero, timeout=timeout)

    summary = {**aero.summary, "thermal": {k: thermal.get(k) for k in THERMAL_SUMMARY_KEYS}}
    data = {**(aero.data or {}), "thermal": thermal}
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
    )


def submit(source: str, approval_hash: str, case_timeout: float = 180.0) -> BatchResult:
    """Run an approved campaign.

    The dry run is repeated rather than cached so that the case list actually
    executed is the one just produced from the source, and the hash check
    proves it is the one that was approved.
    """
    report = dry_run(source)
    if not report.runnable:
        raise ApprovalMismatch(
            "campaign is not runnable: "
            + "; ".join(report.errors + report.rejected)
            + (" (killed)" if report.killed else "")
        )
    if report.approval_hash != approval_hash:
        raise ApprovalMismatch(
            f"approval hash {approval_hash} does not match current campaign "
            f"{report.approval_hash}; re-run dry_run and approve again"
        )

    results = [evaluate(case, timeout=case_timeout) for case in report.cases]
    return BatchResult(results=results, approval_hash=approval_hash)
