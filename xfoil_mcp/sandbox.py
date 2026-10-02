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
import time
from dataclasses import dataclass, field

from xfoil_mcp import containers
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


# Exit codes docker itself uses when it could not start the container or its
# command. The campaign never ran, so they are not the campaign's fault.
DOCKER_COULD_NOT_START = {125, 126, 127}
OUT_OF_MEMORY = 137         # 128 + SIGKILL: what the kernel does at the --memory limit


@dataclass
class SandboxOutcome:
    cases: list[Case] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # host-side validation failures
    errors: list[str] = field(default_factory=list)     # from the campaign or the container
    killed: bool = False            # the time limit was hit: the campaign is probably looping
    infrastructure: bool = False    # docker could not run the sandbox: the campaign is not at fault
    wall_seconds: float = 0.0


def _exit_error(returncode: int, stderr: str) -> str:
    """Say what a non-zero exit most likely means, because the three causes
    call for three different responses."""
    detail = stderr.strip()[-500:]
    if returncode in DOCKER_COULD_NOT_START:
        return f"docker could not start the sandbox (exit {returncode}): {detail}"
    if returncode == OUT_OF_MEMORY:
        limit = SANDBOX_FLAGS[SANDBOX_FLAGS.index("--memory") + 1]
        return f"sandbox exited {returncode}: killed, most likely for exceeding the {limit} memory limit. {detail}".rstrip()
    return f"sandbox exited {returncode}: the campaign ended the process, or the interpreter crashed. {detail}".rstrip()


def run_campaign_source(source: str, timeout: float = 30.0) -> SandboxOutcome:
    started = time.monotonic()
    run = containers.run(SANDBOX_IMAGE, source, timeout, name="xfoil-sandbox", flags=SANDBOX_FLAGS)
    wall = time.monotonic() - started

    if run.proc is None:
        return SandboxOutcome(killed=run.timed_out, infrastructure=not run.timed_out,
                              errors=[run.reason("campaign")], wall_seconds=wall)
    proc = run.proc

    if proc.returncode != 0:
        return SandboxOutcome(
            infrastructure=proc.returncode in DOCKER_COULD_NOT_START,
            errors=[_exit_error(proc.returncode, proc.stderr)],
            wall_seconds=wall,
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return SandboxOutcome(
            errors=[f"sandbox output was not JSON: {proc.stdout[:200]!r}"],
            wall_seconds=wall,
        )

    # The payload was written by a process that ran campaign code, so its own
    # shape is checked before anything in it is used.
    cases, errors = (payload.get(k, []) for k in ("cases", "errors")) if isinstance(payload, dict) else (None, None)
    if not isinstance(cases, list) or not isinstance(errors, list):
        return SandboxOutcome(
            errors=[f"sandbox output did not have the expected shape: {proc.stdout[:200]!r}"],
            wall_seconds=wall,
        )

    outcome = SandboxOutcome(errors=[str(e) for e in errors], wall_seconds=wall)

    for i, raw in enumerate(cases):
        try:
            outcome.cases.append(Case.model_validate(raw))
        except Exception as exc:
            outcome.rejected.append(f"case {i}: {exc}")

    return outcome
