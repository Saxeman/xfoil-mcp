"""Prove the sandbox limits bite. Needs Docker and the xfoil-sandbox image.

    pytest -m docker tests/test_sandbox.py

Each hostile campaign targets one limit. The test asserts the specific
outcome, not just "it failed": a limit that fails for the wrong reason is
not a limit you understand.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from xfoil_mcp import sandbox

CAMPAIGNS = Path(__file__).parent.parent / "campaigns"
HOSTILE = CAMPAIGNS / "hostile"

pytestmark = pytest.mark.docker


def run(name: str, timeout: float = 30.0) -> sandbox.SandboxOutcome:
    return sandbox.run_campaign_source((HOSTILE / name).read_text(), timeout=timeout)


# --- benign baseline -------------------------------------------------------

def test_benign_campaign_returns_cases():
    outcome = sandbox.run_campaign_source((CAMPAIGNS / "single_polar.py").read_text())
    assert not (outcome.killed or outcome.errors or outcome.rejected), (outcome.errors, outcome.rejected)
    assert len(outcome.cases) == 1
    assert outcome.cases[0].geometry.naca == "2412"


def test_lhs_campaign_uses_scipy_inside_sandbox():
    outcome = sandbox.run_campaign_source((CAMPAIGNS / "lhs_naca4.py").read_text())
    assert not (outcome.killed or outcome.errors or outcome.rejected), (outcome.errors, outcome.rejected)
    assert len(outcome.cases) == 12
    assert all(len(c.geometry.naca) == 4 for c in outcome.cases)


def test_campaign_is_deterministic_across_runs():
    """Seeded sampling plus content hashing means the same source gives the
    same case list, which is what makes an approval hash meaningful."""
    src = (CAMPAIGNS / "lhs_naca4.py").read_text()
    a = [c.content_hash() for c in sandbox.run_campaign_source(src).cases]
    b = [c.content_hash() for c in sandbox.run_campaign_source(src).cases]
    assert a == b


# --- hostile ---------------------------------------------------------------

def running_sandboxes() -> set[str]:
    ps = subprocess.run(["docker", "ps", "--filter", "name=xfoil-sandbox-", "-q"],
                        capture_output=True, text=True)
    return set(ps.stdout.split())


def test_infinite_loop_is_killed_by_timeout():
    before = running_sandboxes()        # another run or a live server may have its own
    outcome = run("infinite_loop.py", timeout=5.0)
    assert outcome.killed
    assert outcome.wall_seconds >= 4.5
    # The container must be gone, not detached and still spinning.
    assert running_sandboxes() <= before, "sandbox container leaked after timeout"

def test_network_egress_is_blocked():
    outcome = run("network_egress.py")
    assert not outcome.killed
    assert outcome.cases == []
    assert outcome.errors
    joined = "\n".join(outcome.errors)
    assert "campaign() raised" in joined
    assert "OSError" in joined or "Network is unreachable" in joined or "timed out" in joined


def test_fork_bomb_hits_pid_limit():
    outcome = run("fork_bomb.py", timeout=20.0)
    assert outcome.cases == []
    # make sure that we exited when fork bomb occurs
    assert not outcome.killed, outcome.errors
    joined = "\n".join(outcome.errors)
    assert "campaign() raised" in joined
    assert "BlockingIOError" in joined or "Resource temporarily unavailable" in joined


def test_filesystem_write_is_refused():
    outcome = run("filesystem_write.py")
    assert not outcome.killed
    assert outcome.cases == []
    joined = "\n".join(outcome.errors)
    assert "Read-only file system" in joined or "Permission denied" in joined


def test_forbidden_import_fails_because_module_is_absent():
    """The harness is not installed in the sandbox image. The rule is
    enforced by what exists, not by a loader check."""
    outcome = run("forbidden_import.py")
    assert outcome.cases == []
    joined = "\n".join(outcome.errors)
    assert "campaign failed to load" in joined
    assert "ModuleNotFoundError" in joined or "ImportError" in joined
