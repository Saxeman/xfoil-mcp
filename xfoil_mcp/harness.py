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


class ApprovalMismatch(Exception):
    """The campaign changed since it was approved, or was never approved."""


@dataclass(frozen=True)
class Estimate:
    case_count: int
    alpha_points: int
    expected_seconds: float


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
    points = sum(c.conditions.point_count() for c in cases)
    seconds = len(cases) * SECONDS_PER_CONTAINER + points * SECONDS_PER_ALPHA_POINT
    return Estimate(case_count=len(cases), alpha_points=points, expected_seconds=round(seconds, 1))


def _approval_hash(cases: list[Case]) -> str:
    joined = "\n".join(c.content_hash() for c in cases)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


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

    results = [dispatch.run_case(c, timeout=case_timeout) for c in report.cases]
    return BatchResult(results=results, approval_hash=approval_hash)
