"""Worker entry point. Runs inside the xfoil-worker container.

    stdin:  one Case as JSON
    stdout: one CaseResult as JSON
    exit:   0, unless the process itself is broken

Never writes anything but the result to stdout. The host parses that stream,
so a stray print corrupts the payload. Diagnostics go to stderr.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from xfoil_mcp.schema import Case, CaseResult
from xfoil_mcp.wrapper import InvalidGeometry, PolarResult, XfoilError, run_polar

SOLVER = "xfoil 6.99"


def _provenance(case: Case, runtime: float) -> dict:
    return {
        "content_hash": case.content_hash(),
        "solver": SOLVER,
        "image": os.environ.get("XFOIL_IMAGE", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime, 3),
    }

def _serialize(result: PolarResult) -> dict:
    """Full data, minus stdout. The stdout log can be tens of kilobytes and
    belongs in a log store, not a result payload.

    Field outputs are keyed by alpha, a float. JSON object keys must be
    strings, so 4.0 becomes "4.0" here, in one place. The host reads them
    back with float(key).
    """
    d = dataclasses.asdict(result)
    d.pop("stdout", None)
    d["bl"] = {str(alpha): layer for alpha, layer in d["bl"].items()}
    d["cp"] = {str(alpha): dist for alpha, dist in d["cp"].items()}
    return d

def run_case(case: Case) -> CaseResult:
    started = time.monotonic()

    c = case.conditions
    f = case.geometry.flap
    flap = None if f is None else (f.x_hinge, f.y_hinge, f.deflection)

    
    try:
        polar = run_polar(
            airfoil=case.geometry.naca,
            reynolds=c.reynolds,
            alpha_start=c.alpha_start,
            alpha_end=c.alpha_end,
            alpha_step=c.alpha_step,
            mach=c.mach,
            n_crit=c.n_crit,
            max_iter=c.max_iter,
            outputs=case.outputs,
            flap=flap,
        )
    except InvalidGeometry as exc:
        return CaseResult(
            case=case,
            status="error",
            failure_kind="input",
            summary={"error": str(exc)},
            provenance=_provenance(case, time.monotonic() - started),
        )
    except XfoilError as exc:
        return CaseResult(
            case=case,
            status="error",
            failure_kind="infrastructure",
            summary={"error": str(exc)},
            provenance=_provenance(case, time.monotonic() - started),
        )

    runtime = time.monotonic() - started
    status = polar.status
    kind = None if status == "ok" else "numerical"

    return CaseResult(
        case=case,
        status=status,
        failure_kind=kind,
        summary=polar.summary(),
        data=_serialize(polar),
        provenance=_provenance(case, runtime),
    )


def main() -> int:
    raw = sys.stdin.read()
    try:
        case = Case.model_validate_json(raw)
    except Exception as exc:
        # The host validated this before sending, so reaching here means the
        # contract is broken somewhere. Still return a result, not a crash.
        print(
            '{"status": "error", "failure_kind": "input", '
            f'"summary": {{"error": "unparseable case: {str(exc)!r}"}}}}',
        )
        return 0

    try:
        result = run_case(case)
    except Exception:
        result = CaseResult(
            case=case,
            status="error",
            failure_kind="infrastructure",
            summary={"error": traceback.format_exc()},
            provenance=_provenance(case, 0.0),
        )

    sys.stdout.write(result.model_dump_json())
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
