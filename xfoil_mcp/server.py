"""MCP tools over the harness. Runs on the host, stdio transport.

Thin by design: every tool is a few lines calling the harness and shaping
the payload. The substance is in the docstrings, which are what the model
reads to decide when and how to call each tool.

Every response, from every tool, on success or failure, carries the same
envelope: status, failure_kind, retry_could_help, errors. A model learns
one rule for reading results, not one per tool.

Tools never raise for failures they anticipated. Expected failures are
return values with a failure_kind. Exceptions are reserved for the
genuinely unexpected.

Never write to stdout here. Over stdio, stdout is the protocol channel.
"""

from __future__ import annotations

import logging
import sys

from fastmcp import FastMCP
from pydantic import ValidationError

from xfoil_mcp import dispatch, harness
from xfoil_mcp.schema import Case, FailureKind

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("xfoil")

# Kept in memory for the lifetime of the server process. A queued
# implementation would put this in a store keyed by batch id.
_last_batch: harness.BatchResult | None = None


# --- envelope --------------------------------------------------------------

def _ok(**fields) -> dict:
    return {"status": "ok", "failure_kind": None, "retry_could_help": False, "errors": [], **fields}


def _error(kind: FailureKind, errors: list[str], **fields) -> dict:
    return {
        "status": "error",
        "failure_kind": kind,
        "retry_could_help": kind == "infrastructure",
        "errors": errors,
        **fields,
    }


def _validation_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]


def _case_view(cases: list[Case]) -> list[dict]:
    """What the model sees per case: enough to recognize it, not the whole object."""
    return [
        {
            "content_hash": c.content_hash(),
            "label": c.label,
            "naca": c.geometry.naca,
            "reynolds": c.conditions.reynolds,
            "alphas": f"{c.conditions.alpha_start}..{c.conditions.alpha_end} step {c.conditions.alpha_step}",
            "outputs": list(c.outputs),
        }
        for c in cases
    ]


# --- tools -----------------------------------------------------------------

@mcp.tool
def run_polar(
    airfoil: str,
    reynolds: float,
    alpha_start: float = 0.0,
    alpha_end: float = 10.0,
    alpha_step: float = 1.0,
    n_crit: float = 9.0,
    max_iter: int = 100,
) -> dict:
    """Run one viscous angle-of-attack sweep on a NACA airfoil with XFOIL.

    Use this for a single airfoil at a single Reynolds number. For anything
    with structure (parameter sweeps, sampling schemes, many airfoils),
    write a campaign and use dry_run_campaign instead.

    airfoil: NACA 4- or 5-digit designation, e.g. "2412" (2% camber at 40%
        chord, 12% thick) or "0012" (symmetric).
    reynolds: chord Reynolds number. ~5e4 is a small drone, ~1e6 a light
        aircraft, ~1e7 an airliner. Drag results are not comparable across
        Reynolds numbers.
    alpha_start, alpha_end, alpha_step: angle of attack sweep in degrees.
        Keep the step at 1 or less; larger steps lose warm starts and fail
        more often near stall. At most 200 points.
    n_crit: transition criterion. 9 = clean wind tunnel (default), ~4 =
        turbulent freestream or dirty surface. This is a modeling assumption,
        not a measurement.
    max_iter: viscous iteration limit. Raise toward 200 only for points that
        fail to converge near stall.

    Returns status, converged vs requested points, which alphas failed and
    whether the failures are consecutive, best L/D and where, and max CL
    within the sweep (not the airfoil's true max CL unless the sweep passes
    stall). A status of "partial" means some alphas did not converge.
    Retrying identically will not help; use a smaller step, start closer to
    the failed region, or accept the gap. Consecutive failures near the top
    of the sweep usually mean stall. A failure_kind of "input" means the
    arguments were invalid; fix them rather than retrying.
    """
    try:
        case = Case.model_validate({
            "geometry": {"naca": airfoil},
            "conditions": {
                "reynolds": reynolds, "alpha_start": alpha_start, "alpha_end": alpha_end,
                "alpha_step": alpha_step, "n_crit": n_crit, "max_iter": max_iter,
            },
        })
    except ValidationError as exc:
        return _error("input", _validation_errors(exc), content_hash=None)

    result = dispatch.run_case(case)
    if result.status == "error":
        return _error(
            result.failure_kind,
            [result.summary.get("error", "unknown error")],
            content_hash=case.content_hash(),
        )
    return {
        "status": result.status,
        "failure_kind": result.failure_kind,
        "retry_could_help": result.retry_could_help,
        "errors": [],
        "content_hash": case.content_hash(),
        **result.summary,
    }


@mcp.tool
def dry_run_campaign(source: str) -> dict:
    """Execute a campaign script in an isolated sandbox and report what it
    WOULD run. Runs no solver. Always call this before run_campaign.

    A campaign is Python source defining `campaign() -> list[Case]`. It may
    import only xfoil_mcp.schema (Case, Geometry, Flap, Conditions), numpy,
    and scipy. Use scipy.stats.qmc for Latin hypercube or Sobol sampling
    rather than writing a sampler by hand, and seed it. The sandbox has no
    network and a 30 second limit.

    Example:

        from xfoil_mcp.schema import Case, Conditions, Geometry
        def campaign():
            return [
                Case(geometry=Geometry(naca=n),
                     conditions=Conditions(reynolds=5e5, alpha_start=0,
                                           alpha_end=12, alpha_step=1),
                     label=n)
                for n in ("2412", "4412", "0012")
            ]

    On success returns the deduplicated case list, an estimate (case count,
    alpha points, expected seconds), and an approval_hash. Show the estimate
    to the user and get agreement before calling run_campaign with that
    hash. On failure returns the errors, the cases that did validate (so the
    fix can be local), and no approval_hash. A failure_kind of "input" means
    the campaign code needs changing; "infrastructure" means the sandbox
    could not run and a retry may help.
    """
    report = harness.dry_run(source)

    if not report.runnable:
        infra = any("docker" in e.lower() for e in report.errors)
        errors = list(report.errors) + list(report.rejected)
        if report.killed:
            errors.append("campaign was killed by the sandbox timeout; it is probably looping")
        if not report.cases and not errors:
            errors.append("campaign() returned no cases")
        return _error(
            "infrastructure" if infra else "input",
            errors,
            cases=_case_view(report.cases),
            approval_hash=None,
            estimate=None,
        )

    return _ok(
        approval_hash=report.approval_hash,
        estimate=report.estimate.__dict__,
        cases=_case_view(report.cases),
    )


@mcp.tool
def run_campaign(source: str, approval_hash: str) -> dict:
    """Run a campaign that was approved via dry_run_campaign.

    approval_hash must be the value dry_run_campaign returned for this exact
    source. If the source changed, the hash will not match and this refuses;
    dry-run again. Returns one summary per case, never full polar tables;
    use get_case_detail for those.

    Read `complete` and `counts` before drawing conclusions. A case with
    failure_kind "numerical" will fail again if rerun identically; only a
    changed approach (smaller step, different range) can help. A case with
    failure_kind "infrastructure" may succeed on retry.
    """
    global _last_batch
    try:
        batch = harness.submit(source, approval_hash)
    except harness.ApprovalMismatch as exc:
        return _error("input", [str(exc)], complete=False, submitted=0, counts={}, results=[])

    _last_batch = batch
    return _ok(
        complete=batch.complete,
        submitted=batch.submitted,
        counts=batch.counts,
        results=batch.summaries(),
    )


@mcp.tool
def get_case_detail(content_hash: str) -> dict:
    """Fetch the full result for one case from the most recent campaign,
    including every converged polar point. Large; call only when the
    summary is not enough (for example, to plot the polar or inspect the
    lift curve near stall). content_hash comes from a run_campaign summary.
    """
    if _last_batch is None:
        return _error("input", ["no campaign has been run in this session"], result=None)
    result = _last_batch.get(content_hash)
    if result is None:
        return _error("input", [f"no case {content_hash} in the most recent campaign"], result=None)
    return _ok(result=result.model_dump(mode="json"))


if __name__ == "__main__":
    mcp.run()
