"""Drive XFOIL from Python and return structured, validated results.

XFOIL is an interactive Fortran program written for a human at a terminal.
It has no API, no exit codes worth trusting, and several silent failure
modes. This module isolates all of that behind a plain Python interface so
the rest of the system can treat an airfoil analysis as a function call.

Failure modes handled here, all observed experimentally:
  1. Graphics: XFOIL aborts a sweep if it cannot open an X display.
     Suppressed with PLOP/G before any analysis.
  2. Non-convergence: a failed point is absent from the polar file and is
     announced only on stdout as "VISCAL:  Convergence failed".
  3. Warm-start cascade: each alpha starts from the previous solution, so a
     failure degrades the points after it. Failures are reported with their
     positions so callers can see runs of them.
  4. Desynchronization: one unanswered prompt shifts every later command,
     and exhausting stdin crashes the process with a Fortran backtrace.
"""

# TODO: ADD THIS IN TO FORCE FORTRAN OVERFLOW INTO /DEV/NULL
# If using a Python wrapper script:
# import os
# try:
#     os.symlink('/dev/null', '/tmp/:00.bl')
# except FileExistsError:
#     pass # Already exists from a previous run


from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

XFOIL_BIN = os.environ.get("XFOIL_BIN", "xfoil")

# Alphas are floats echoed back through a text file, so requested and
# returned values are matched within this tolerance rather than by equality.
ALPHA_TOL = 1e-6

# Define failure modes
_VISCAL_FAILURE = re.compile(r"VISCAL:\s+Convergence failed")
_MRCHDU_FAILURE = re.compile(r"MRCH(DU|UE):\s+Convergence failed")
_ITER_LIMIT_ECHO = re.compile(r"Current iteration limit:\s+(\d+)")
_DISPLAY_ABORT = re.compile(r"Cannot open display")


class XfoilError(Exception):
    """Raised only for failures that make the whole run meaningless.

    Non-convergence is NOT an error - it is a result. It comes back inside
    PolarResult so the caller can see which alphas are missing and decide
    what that means.
    """


@dataclass(frozen=True)
class PolarPoint:
    alpha: float
    cl: float
    cd: float
    cdp: float
    cm: float
    top_xtr: float
    bot_xtr: float

    # Define cl/cd ratio with error checking
    @property
    def ld(self) -> float:
        return self.cl / self.cd if self.cd else float("nan")


@dataclass
class PolarResult:
    """Everything a caller needs to judge whether to trust these numbers."""

    airfoil: str
    reynolds: float
    mach: float
    n_crit: float
    max_iter: int

    requested_alphas: list[float]
    points: list[PolarPoint]

    failed_alphas: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    returncode: int | None = None
    stdout: str = ""

    @property
    def converged_count(self) -> int:
        return len(self.points)

    @property
    def requested_count(self) -> int:
        return len(self.requested_alphas)

    @property
    def status(self) -> str:
        if not self.points:
            return "empty"
        if self.failed_alphas:
            return "partial"
        return "ok"

    @property
    def failure_runs(self) -> list[list[float]]:
        """Consecutive failed alphas, grouped.

        A run of length > 1 suggests a warm-start cascade rather than
        independent hard points, which is a different thing to report.
        """
        runs: list[list[float]] = []
        index = {round(a, 6): i for i, a in enumerate(self.requested_alphas)}
        for alpha in self.failed_alphas:
            i = index[round(alpha, 6)]
            if runs and index[round(runs[-1][-1], 6)] == i - 1:
                runs[-1].append(alpha)
            else:
                runs.append([alpha])
        return runs

    def summary(self) -> dict:
        """Compact form - what to hand a model or a dashboard.

        The full point table is large and mostly uninteresting; this is the
        part a caller usually reasons over.
        """
        best = max(self.points, key=lambda p: p.ld, default=None)
        peak_cl = max(self.points, key=lambda p: p.cl, default=None)
        return {
            "status": self.status,
            "airfoil": self.airfoil,
            "reynolds": self.reynolds,
            "requested_points": self.requested_count,
            "converged_points": self.converged_count,
            "failed_alphas": self.failed_alphas,
            "failure_runs": self.failure_runs,
            "best_ld": None if best is None else round(best.ld, 1),
            "best_ld_alpha": None if best is None else best.alpha,
            "max_cl": None if peak_cl is None else round(peak_cl.cl, 4),
            "max_cl_alpha": None if peak_cl is None else peak_cl.alpha,
            "warnings": self.warnings,
            "runtime_seconds": round(self.runtime_seconds, 2),
        }


def _expected_alphas(start: float, end: float, step: float) -> list[float]:
    """Reproduce ASEQ's sequence without accumulating float error."""
    if step <= 0:
        raise ValueError("alpha_step must be positive")
    if end < start:
        raise ValueError("alpha_end must be >= alpha_start")
    n = int(round((end - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def _build_commands(
    airfoil: str,
    reynolds: float,
    mach: float,
    n_crit: float,
    max_iter: int,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    polar_path: str,
) -> str:
    """Assemble the command script.

    Every blank line here answers a prompt. Miss one and every later command
    is consumed as an answer to the wrong question, so they are written
    explicitly rather than generated.
    """
    return "\n".join(
        [
            "PLOP",          # plotting options
            "G",             # toggle graphics off - required when headless
            "",              # leave PLOP
            f"NACA {airfoil}",
            "OPER",
            f"ITER {max_iter}",
            "VPAR",          # viscous parameters
            f"N {n_crit}",
            "",              # leave VPAR
            f"VISC {reynolds}",
            f"MACH {mach}",
            "PACC",
            polar_path,      # polar save file
            "",              # decline the dump file
            f"ASEQ {alpha_start} {alpha_end} {alpha_step}",
            "PACC",          # stop accumulating, flushes the file
            "",              # leave OPER
            "QUIT",
            "",              # trailing newline so stdin is never exhausted mid-prompt
        ]
    )


def _parse_polar_file(path: Path) -> list[PolarPoint]:
    """Parse the fixed-width polar table.

    The header length varies with settings, so the dashed rule is used as the
    anchor rather than a hardcoded line count.
    """
    if not path.exists():
        return []

    lines = path.read_text(errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if set(line.strip()) <= {"-", " "} and "---" in line:
            start = i + 1
            break
    if start is None:
        return []

    points: list[PolarPoint] = []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            values = [float(f) for f in fields[:7]]
        except ValueError:
            continue
        points.append(
            PolarPoint(
                alpha=values[0],
                cl=values[1],
                cd=values[2],
                cdp=values[3],
                cm=values[4],
                top_xtr=values[5],
                bot_xtr=values[6],
            )
        )
    return points


def _scan_stdout(stdout: str, requested_max_iter: int) -> list[str]:
    """Pull failure evidence out of stdout.

    The polar file records only successes. Everything about what went wrong
    lives here and nowhere else.
    """
    warnings: list[str] = []

    if _DISPLAY_ABORT.search(stdout):
        warnings.append(
            "XFOIL aborted trying to open a display; graphics suppression failed"
        )

    viscal = len(_VISCAL_FAILURE.findall(stdout))
    if viscal:
        warnings.append(f"{viscal} point(s) reported VISCAL convergence failure")

    mrchdu = len(_MRCHDU_FAILURE.findall(stdout))
    if mrchdu:
        warnings.append(
            f"{mrchdu} boundary-layer march failure(s) during solves "
            "(some points may have converged from a poor path)"
        )

    # Verify the iteration limit was actually applied rather than assuming
    # the command landed. XFOIL echoes the value when it changes.
    echoes = _ITER_LIMIT_ECHO.findall(stdout)
    if echoes and int(echoes[-1]) != requested_max_iter:
        warnings.append(
            f"iteration limit reads {echoes[-1]}, requested {requested_max_iter}"
        )

    if "not recognized" in stdout:
        warnings.append("XFOIL rejected a command; the input sequence may have desynchronized")

    if "Fortran runtime error" in stdout:
        warnings.append("XFOIL crashed with a Fortran runtime error")

    return warnings


def run_polar(
    airfoil: str,
    reynolds: float,
    alpha_start: float = 0.0,
    alpha_end: float = 10.0,
    alpha_step: float = 1.0,
    mach: float = 0.0,
    n_crit: float = 9.0,
    max_iter: int = 100,
    timeout: float = 120.0,
) -> PolarResult:
    """Run a viscous alpha sweep and return a validated result.

    airfoil is a NACA 4- or 5-digit designation, e.g. "2412".
    reynolds is the chord Reynolds number, e.g. 1e6.

    Non-convergence is reported inside the result, not raised. Only failures
    that invalidate the whole run raise XfoilError.
    """
    requested = _expected_alphas(alpha_start, alpha_end, alpha_step)

    # A fresh directory per run: XFOIL appends to an existing polar file and
    # scatters files it was told not to write.
    with tempfile.TemporaryDirectory(prefix="xfoil_") as workdir:
        polar_path = Path(workdir) / "polar.txt"
        commands = _build_commands(
            airfoil, reynolds, mach, n_crit, max_iter,
            alpha_start, alpha_end, alpha_step, polar_path.name,
        )

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [XFOIL_BIN],
                input=commands,
                capture_output=True,
                text=True,
                cwd=workdir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise XfoilError(
                f"XFOIL exceeded {timeout}s for {airfoil} at Re={reynolds:g}; "
                "it was killed, so no results are available"
            ) from exc
        except FileNotFoundError as exc:
            raise XfoilError(
                f"XFOIL binary {XFOIL_BIN!r} not found; set XFOIL_BIN"
            ) from exc

        runtime = time.monotonic() - started
        stdout = proc.stdout or ""
        points = _parse_polar_file(polar_path)

    returned = {round(p.alpha, 6) for p in points}
    failed = [a for a in requested if a not in returned]
    warnings = _scan_stdout(stdout, max_iter)

    if not points:
        warnings.append("no converged points; check warnings above")

    return PolarResult(
        airfoil=airfoil,
        reynolds=reynolds,
        mach=mach,
        n_crit=n_crit,
        max_iter=max_iter,
        requested_alphas=requested,
        points=points,
        failed_alphas=failed,
        warnings=warnings,
        runtime_seconds=runtime,
        returncode=proc.returncode,
        stdout=stdout,
    )


if __name__ == "__main__":
    import json

    clean = run_polar("2412", 1e6, 0, 10, 1)
    print(json.dumps(clean.summary(), indent=2))

#    cascade = run_polar("2412", 1e6, 0, 20, 1, max_iter=5)
#    print(json.dumps(cascade.summary(), indent=2))
