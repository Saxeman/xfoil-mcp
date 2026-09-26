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

# Every flag answers a specific attack. See ARCHITECTURE.md for the table.
SANDBOX_FLAGS = [
    "--network", "none",
    "--read-only",
    "--tmpfs", "/tmp:size=64m",
    "--memory", "512m",
    "--cpus", "1",
    "--pids-limit", "64",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--user", "65534:65534",
]


@dataclass
class SandboxOutcome:
    cases: list[Case] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # host-side validation failures
    errors: list[str] = field(default_factory=list)     # from the campaign or the container
    killed: bool = False
    wall_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.killed and not self.errors and not self.rejected

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
        rm = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=15)
        if rm.returncode != 0 and "No such container" not in rm.stderr:
            logging.getLogger(__name__).warning("failed to remove %s: %s", name, rm.stderr.strip())

