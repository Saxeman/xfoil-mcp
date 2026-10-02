import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry
from xfoil_mcp.wrapper import _parse_bl_file

BL_A4 = Path(__file__).parent / "fixtures" / "bl_2412_re1e6_a4.txt"
THERMAL = dict(
    chord_m=0.5, air_temperature_k=263.15, heater_width=0.1,
    heater_power_w_per_m=500.0, skin_thickness_m=0.001, skin_conductivity_w_mk=200.0,
)

def aero(requested=(4.0,), converged=(4.0,), status="ok", thermal=None) -> CaseResult:
    """An aero result shaped like the worker's, using the fixture's boundary
    layer for every converged alpha."""
    case = Case(
        geometry=Geometry(naca="2412"),
        conditions=Conditions(reynolds=1e6, alpha_start=min(requested),
                              alpha_end=max(requested), alpha_step=2),
        outputs=("forces", "bl"),
        thermal=thermal or THERMAL,
    )
    layer = dataclasses.asdict(_parse_bl_file(BL_A4, alpha=4.0))
    points = [dict(alpha=a, cl=0.7146, cd=0.00693, cdp=0.00107, cm=-0.0573,
                   top_xtr=0.398, bot_xtr=1.0) for a in converged]
    data = {
        "requested_alphas": list(requested),
        "points": points,
        "failed_alphas": [a for a in requested if a not in converged],
        "bl": {str(a): layer for a in converged},
    }
    return CaseResult(
        case=case, status=status,
        failure_kind=None if status == "ok" else "numerical",
        data=None if status == "empty" else data,
    )


class FakeDocker:
    """Stands in for subprocess.run and records every call.

    `run_result` answers `docker run`: a CompletedProcess to return or an
    exception to raise. `rm_result` does the same for the cleanup call.
    """

    def __init__(self, run_result=None, rm_result=None):
        self.run_result = run_result if run_result is not None else self.exited(0, "out")
        self.rm_result = rm_result if rm_result is not None else self.exited(0)
        self.calls = []

    @staticmethod
    def exited(code, stdout="", stderr=""):
        return subprocess.CompletedProcess(["docker"], code, stdout=stdout, stderr=stderr)

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        result = self.rm_result if cmd[:2] == ["docker", "rm"] else self.run_result
        if isinstance(result, BaseException):
            raise result
        return result

    @property
    def run_call(self):
        return next(c for c in self.calls if c[0][:2] == ["docker", "run"])

    @property
    def rm_call(self):
        return next(c for c in self.calls if c[0][:2] == ["docker", "rm"])


def needs_source_change(issue: str, why: str):
    """Marks a test that states the behaviour we want and fails today because
    of a known gap in the source (`issue` is its id in docs/ISSUE_REPORT.MD).

    Strict: the day the source is fixed the test passes, pytest reports that
    as a failure, and the marker has to be removed.
    """
    return pytest.mark.xfail(strict=True, reason=f"needs source change ({issue}): {why}")


FIXTURES = Path(__file__).parent / "fixtures"

_FAKE_XFOIL = '''#!{python}
"""A stand-in for the xfoil binary. It reads the command script on stdin and
writes the files XFOIL would, behaving as the constants below say."""
import shutil
import sys
import time

MODE, SKIP, NAN, FIXTURES = {mode!r}, {skip!r}, {nan!r}, {fixtures!r}

lines = sys.stdin.read().split("\\n")
if MODE == "hang":
    time.sleep(30)
if MODE == "desync":
    print(' XYZ   command not recognized.  Type a "?" for list')
    sys.exit(0)

polar = lines[lines.index("PACC") + 1]
rows, code, limit = [], 0, 20
for line in lines:
    word, _, arg = line.partition(" ")
    if word == "ITER" and arg:
        limit = int(arg)
    elif word == "ITER":
        print(f".OPERi   c>   Current iteration limit:          {{limit}}")
    elif word == "FLAP":
        x, y, _ = arg.split()
        print(" Top    surface:  y =  0.0516     y/t = 1.0")
        print(" Bottom surface:  y = -0.0216     y/t = 0.0")
        print(f" Flap hinge: x,y =  {{float(x):.5f}}  {{float(y):.5f}}")
    elif word == "EXEC" and MODE != "flap_stays_in_buffer":
        print(" Current airfoil nodes set from buffer airfoil nodes ( 160 )")
    elif word == "PSAV":
        shutil.copy(FIXTURES + "/naca2412_flap_none.dat", arg)
    elif word == "DUMP":
        shutil.copy(FIXTURES + "/bl_2412_re1e6_a4.txt", arg)
    elif word == "CPWR":
        shutil.copy(FIXTURES + "/cp_2412_re1e6_a4.txt", arg)
    elif word == "ALFA":
        alpha = float(arg)
        if MODE == "crash" and len(rows) == 2:
            sys.stderr.write("At line 2015 of file xoper.f\\nFortran runtime error: End of file\\n")
            code = 2
            break
        if alpha in SKIP:
            print(" VISCAL:  Convergence failed")
        else:
            rows.append(alpha)

with open(polar, "w") as f:
    f.write(" Calculated polar for: NACA 2412\\n\\n"
            "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\\n"
            "  ------ -------- --------- --------- -------- -------- --------\\n")
    for alpha in rows:
        cl = "NaN" if alpha in NAN else f"{{0.2 + 0.1 * alpha:.4f}}"
        f.write(f"  {{alpha:7.3f}}  {{cl:>7}}   0.00600   0.00050  -0.0500   0.5000   1.0000\\n")
sys.exit(code)
'''


def fake_xfoil(directory: Path, mode: str = "ok", skip=(), nan=()) -> str:
    """Write a stand-in xfoil executable into `directory` and return its path.

    Point `wrapper.XFOIL_BIN` at it to run `run_polar` on the host. It solves
    nothing: every alpha gets cl = 0.2 + 0.1 alpha and cd = 0.006, and field
    files are copies of the fixtures.

    mode: "ok"; "crash" (dies with a Fortran runtime error on stderr after
        two points, exit 2); "hang" (sleeps 30 s); "desync" (rejects a command
        and writes no polar); "flap_stays_in_buffer" (never confirms EXEC).
    skip: alphas that do not converge (announced on stdout, absent from the polar).
    nan: alphas whose polar row holds NaN.
    """
    path = Path(directory) / "fake_xfoil"
    path.write_text(_FAKE_XFOIL.format(python=sys.executable, mode=mode, skip=tuple(skip),
                                       nan=tuple(nan), fixtures=str(FIXTURES)))
    path.chmod(0o755)
    return str(path)
