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
import json

from xfoil_mcp.schema import Case, CaseResult

WORKER_IMAGE = os.environ.get("XFOIL_WORKER_IMAGE", "xfoil-worker")
THERMAL_IMAGE = os.environ.get("XFOIL_THERMAL_IMAGE", "xfoil-thermal")

# Same isolation as the sandbox: the code is ours, but its input crossed a
# container boundary on its way here.
THERMAL_FLAGS = [
    "--network", "none",
    "--read-only", "--tmpfs", "/tmp:size=64m",
    "--memory", "1g", "--cpus", "1", "--pids-limit", "64",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--user", "65534:65534",
]
THERMAL_KEYS = {"status", "failure_kind", "errors", "coverage", "worst_case", "points", "excluded"}
STATUSES = {"ok", "partial", "empty", "error"}


def _infrastructure_failure(case: Case, message: str) -> CaseResult:
    return CaseResult(
        case=case,
        status="error",
        failure_kind="infrastructure",
        summary={"error": message},
        provenance={"content_hash": case.content_hash(), "image": WORKER_IMAGE},
    )

def _thermal_failure(message: str) -> dict:
    return {
        "status": "error", "failure_kind": "infrastructure", "errors": [message],
        "coverage": 0.0, "worst_case": None, "points": [], "excluded": [],
    }


def run_thermal(aero: CaseResult, timeout: float = 120.0) -> dict:
    """Run the thermal stage for one finished aero result in a fresh container.

    Never raises. The output is checked here because it crossed a container
    boundary: the expected fields, a known status, and a content hash that
    matches the case that was sent.
    """
    name = f"xfoil-thermal-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", name, *THERMAL_FLAGS, THERMAL_IMAGE]
    try:
        try:
            proc = subprocess.run(
                cmd, input=aero.model_dump_json(), capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _thermal_failure(f"thermal worker exceeded {timeout}s and was killed")
        except FileNotFoundError:
            return _thermal_failure("docker not found on host")

        if proc.returncode != 0:
            return _thermal_failure(f"thermal worker exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return _thermal_failure(f"thermal output was not JSON: {proc.stdout[:200]!r}")

        if not isinstance(out, dict) or not THERMAL_KEYS <= out.keys() or out["status"] not in STATUSES:
            return _thermal_failure("thermal output did not have the expected shape")
        if out["status"] != "error" and out.get("content_hash") != aero.case.content_hash():
            return _thermal_failure("thermal worker returned a result for a different case")
        return out
    finally:
        rm = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=15)


def run_case(case: Case, timeout: float = 180.0) -> CaseResult:
    """Run one case in a fresh worker container and return its result.

    The container's output is not trusted: a non-zero exit, output that does
    not parse as a CaseResult, or a result for a different case each come
    back as an infrastructure failure. A timeout kills the container by
    name, because killing the docker client alone leaves it running.
    """
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
