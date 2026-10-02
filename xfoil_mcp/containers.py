"""Run one command in a fresh Docker container. Runs on the host.

Every container the host starts goes through `run`: the sandbox for campaign
code, the worker for XFOIL, and the thermal stage. One place owns the parts
that are easy to get wrong: the time limit, the cleanup, and never letting
either of them raise into the caller.
"""

from __future__ import annotations

import contextlib
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerRun:
    """How one `docker run` ended.

    `proc` is set when the container ran to an exit, whatever its exit code.
    Otherwise `error` says why it did not, and `timed_out` says whether that
    was the host's time limit rather than docker failing to start.
    """

    proc: subprocess.CompletedProcess | None = None
    error: str | None = None
    timed_out: bool = False

    def reason(self, what: str) -> str:
        """Why there is no `proc`, naming what was running if the limit hit."""
        return f"{what} {self.error}" if self.timed_out else self.error


def run(image: str, stdin: str, timeout: float, *, name: str,
        flags: Sequence[str] = (), argv: Sequence[str] = ()) -> ContainerRun:
    """Run `image` with `stdin` piped in and its output captured. Never raises.

    `name` is a prefix: a random suffix makes the container's name unique, so
    it can be removed by name afterwards. `flags` go before the image and
    `argv` after it.
    """
    container = f"{name}-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", container, *flags, image, *argv]
    try:
        return ContainerRun(proc=subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, timeout=timeout,
        ))
    except subprocess.TimeoutExpired:
        return ContainerRun(error=f"exceeded {timeout}s and was killed", timed_out=True)
    except FileNotFoundError:
        return ContainerRun(error="docker not found on host")
    except OSError as exc:
        return ContainerRun(error=f"could not run docker: {exc}")
    finally:
        # --rm removes the container after a clean exit. This covers the rest:
        # a timeout kills only the docker client, and the container runs on.
        # Best effort, because a failed cleanup must not replace the result
        # above. stdin is detached: the server's own stdin is the MCP stream.
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["docker", "rm", "-f", container],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
