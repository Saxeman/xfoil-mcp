"""Launch one worker container per case. Runs on the host.

The host never imports wrapper.py; XFOIL exists only inside the worker image.
This module is how the host talks to it: a Case goes in as JSON on stdin, a
CaseResult comes back as JSON on stdout, and the result is rebuilt through
the schema on this side because the container's output is untrusted.
"""

from __future__ import annotations

import os
import subprocess
import uuid

from xfoil_mcp.schema import Case, CaseResult

WORKER_IMAGE = os.environ.get("XFOIL_WORKER_IMAGE", "xfoil-worker")


def _infrastructure_failure(case: Case, message: str) -> CaseResult:
    return CaseResult(
        case=case,
        status="error",
        failure_kind="infrastructure",
        summary={"error": message},
        provenance={"content_hash": case.content_hash(), "image": WORKER_IMAGE},
    )


def run_case(case: Case, timeout: float = 180.0) -> CaseResult:
    """..."""
    name = f"xfoil-worker-{uuid.uuid4().hex[:12]}"
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none",
        WORKER_IMAGE,
        "python", "-m", "xfoil_mcp.worker",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=case.model_dump_json(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Killing the docker client does not kill the container. A hung XFOIL
        # would otherwise keep running with no one listening.
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=10)
        return _infrastructure_failure(case, f"worker exceeded {timeout}s and was killed")
    except FileNotFoundError:
        return _infrastructure_failure(case, "docker not found on host")

    if proc.returncode != 0:
        return _infrastructure_failure(
            case, f"worker exited {proc.returncode}: {proc.stderr.strip()[-500:]}"
        )

    try:
        result = CaseResult.model_validate_json(proc.stdout)
    except Exception as exc:
        return _infrastructure_failure(
            case, f"worker output was not a CaseResult: {exc}; stdout[:200]={proc.stdout[:200]!r}"
        )

    if result.case.content_hash() != case.content_hash():
        # The worker answered a different question than it was asked.
        return _infrastructure_failure(case, "worker returned a result for a different case")

    return result
