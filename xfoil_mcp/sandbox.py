"""Run agent-written campaign code in an isolated container. Runs on the host.

The control plane never executes campaign code in its own process. The
source goes into a container with no network, a read-only filesystem, and
hard caps on memory, CPU, processes, and wall time. What comes back is a
list of cases as JSON, and every one is rebuilt through the schema here
because the code that produced it was untrusted.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field

from xfoil_mcp.schema import Case

SANDBOX_IMAGE = os.environ.get("XFOIL_SANDBOX_IMAGE", "xfoil-sandbox")

# Every flag answers a specific attack.
SANDBOX_FLAGS = [
    "--network", "none",                    # no exfiltration, no downloads
    "--read-only",                          # nothing persists between runs
    "--tmpfs", "/tmp:size=64m",             # scratch space that cannot fill the disk
    "--memory", "512m",                     # memory exhaustion stays in the container
    "--cpus", "1",                          # cannot starve the host
    "--pids-limit", "64",                   # fork bombs
    "--cap-drop", "ALL",                    # no privileged operations
    "--security-opt", "no-new-privileges",  # no setuid escalation
    "--user", "65534:65534",                # nobody, not root
]


@dataclass
class SandboxOutcome:
    cases: list[Case] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # host-side validation failures
    errors: list[str] = field(default_factory=list)     # from the campaign or the container
    killed: bool = False
    wall_seconds: float = 0.0


def run_campaign_source(source: str, timeout: float = 30.0) -> SandboxOutcome:
    name = f"xfoil-sandbox-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", name, *SANDBOX_FLAGS, SANDBOX_IMAGE]
    started = time.monotonic()

    try:
        try:
            proc = subprocess.run(
                cmd, input=source, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxOutcome(
                killed=True,
                errors=[f"campaign exceeded {timeout}s and was killed"],
                wall_seconds=time.monotonic() - started,
            )
        except FileNotFoundError:
            return SandboxOutcome(errors=["docker not found on host"])

        wall = time.monotonic() - started

        if proc.returncode != 0:
            return SandboxOutcome(
                killed=True,
                errors=[f"sandbox exited {proc.returncode}: {proc.stderr.strip()[-500:]}"],
                wall_seconds=wall,
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return SandboxOutcome(
                errors=[f"sandbox output was not JSON: {proc.stdout[:200]!r}"],
                wall_seconds=wall,
            )

        outcome = SandboxOutcome(errors=list(payload.get("errors", [])), wall_seconds=wall)

        for i, raw in enumerate(payload.get("cases", [])):
            try:
                outcome.cases.append(Case.model_validate(raw))
            except Exception as exc:
                outcome.rejected.append(f"case {i}: {exc}")

        return outcome

    finally:
        # Every path out of this function ends here, including the early
        # returns above. --rm handles a clean exit; this handles the rest.
        # Best effort: a failed cleanup must not replace the outcome above,
        # including when docker itself is missing or hangs.
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

