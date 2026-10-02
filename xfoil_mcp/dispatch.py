"""Launch one worker container per case. Runs on the host.

The host never imports wrapper.py; XFOIL exists only inside the worker image.
This module is how the host talks to it: a Case goes in as JSON on stdin, a
CaseResult comes back as JSON on stdout, and the result is rebuilt through
the schema on this side because the container's output is untrusted.
"""

from __future__ import annotations

import json
import os

from pydantic import ValidationError

from xfoil_mcp import containers
from xfoil_mcp.schema import Case, CaseResult, ThermalResult

WORKER_IMAGE = os.environ.get("XFOIL_WORKER_IMAGE", "xfoil-worker")
THERMAL_IMAGE = os.environ.get("XFOIL_THERMAL_IMAGE", "xfoil-thermal")

# The worker is cut off from the network and nothing more.
WORKER_FLAGS = ["--network", "none"]

# Same isolation as the sandbox: the code is ours, but its input crossed a
# container boundary on its way here.
THERMAL_FLAGS = [
    "--network", "none",
    "--read-only", "--tmpfs", "/tmp:size=64m",
    "--memory", "1g", "--cpus", "1", "--pids-limit", "64",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--user", "65534:65534",
]


def _infrastructure_failure(case: Case, message: str) -> CaseResult:
    return CaseResult(
        case=case,
        status="error",
        failure_kind="infrastructure",
        summary={"error": message},
        provenance={"content_hash": case.content_hash(), "image": WORKER_IMAGE},
    )

def _worker_refusal(stdout: str) -> str | None:
    """The worker's reason for refusing a case it could not parse, if that is
    what `stdout` holds: an error object with kind "input" and no case."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and "case" not in payload \
            and (payload.get("status"), payload.get("failure_kind")) == ("error", "input"):
        summary = payload.get("summary")
        return str(summary.get("error", "")) if isinstance(summary, dict) else ""
    return None


def _thermal_failure(message: str) -> ThermalResult:
    return ThermalResult.failure("infrastructure", message)


def run_thermal(aero: CaseResult, timeout: float = 120.0) -> ThermalResult:
    """Run the thermal stage for one finished aero result in a fresh container.

    Never raises. The output crossed a container boundary, so it is rebuilt
    through the schema here, and a result that ran must carry the content
    hash of the case that was sent.
    """
    run = containers.run(THERMAL_IMAGE, aero.model_dump_json(), timeout,
                         name="xfoil-thermal", flags=THERMAL_FLAGS)
    if run.proc is None:
        return _thermal_failure(run.reason("thermal worker"))
    proc = run.proc

    if proc.returncode != 0:
        return _thermal_failure(f"thermal worker exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    try:
        out = ThermalResult.model_validate_json(proc.stdout)
    except ValidationError as exc:
        return _thermal_failure(
            f"thermal output was not a ThermalResult: {exc}; stdout[:200]={proc.stdout[:200]!r}"
        )

    if out.status != "error" and out.content_hash != aero.case.content_hash():
        return _thermal_failure("thermal worker returned a result for a different case")
    return out


def run_case(case: Case, timeout: float = 180.0) -> CaseResult:
    """Run one case in a fresh worker container and return its result.

    Never raises. The container's output is not trusted: a non-zero exit,
    output that does not parse as a CaseResult, or a result for a different
    case each come back as an infrastructure failure, as does a timeout or
    docker failing to start.
    """
    run = containers.run(WORKER_IMAGE, case.model_dump_json(), timeout, name="xfoil-worker",
                         flags=WORKER_FLAGS, argv=("python", "-m", "xfoil_mcp.worker"))
    if run.proc is None:
        return _infrastructure_failure(case, run.reason("worker"))
    proc = run.proc

    if proc.returncode != 0:
        return _infrastructure_failure(
            case, f"worker exited {proc.returncode}: {proc.stderr.strip()[-500:]}"
        )

    try:
        result = CaseResult.model_validate_json(proc.stdout)
    except Exception as exc:
        refusal = _worker_refusal(proc.stdout)
        if refusal is not None:
            # The host built this case from the schema, so a worker that cannot
            # parse it was built from a different one. Retrying cannot help.
            return CaseResult(
                case=case, status="error", failure_kind="input",
                summary={"error": "the worker could not parse the case it was sent, which usually means "
                                  f"its image is older than the schema; rebuild {WORKER_IMAGE}. It said: {refusal}"},
                provenance={"content_hash": case.content_hash(), "image": WORKER_IMAGE},
            )
        return _infrastructure_failure(
            case, f"worker output was not a CaseResult: {exc}; stdout[:200]={proc.stdout[:200]!r}"
        )

    if result.case.content_hash() != case.content_hash():
        # The worker answered a different question than it was asked.
        return _infrastructure_failure(case, "worker returned a result for a different case")

    return result
