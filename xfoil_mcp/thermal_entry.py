"""Thermal worker entry point. Runs inside the xfoil-thermal container.

    stdin:  one aero CaseResult as JSON, for a case with a thermal block
    stdout: one thermal summary as JSON
    exit:   0, unless the process itself is broken

Only the result goes to stdout; the host parses that stream.
"""

from __future__ import annotations

import json
import sys
import traceback

from xfoil_mcp.schema import CaseResult
from xfoil_mcp.thermal_stage import evaluate_thermal


def _error(kind: str, message: str) -> dict:
    return {
        "status": "error", "failure_kind": kind, "errors": [message],
        "coverage": 0.0, "worst_case": None, "points": [], "excluded": [],
    }


def main() -> int:
    raw = sys.stdin.read()
    try:
        aero = CaseResult.model_validate_json(raw)
    except Exception as exc:
        out = _error("input", f"input was not an aero CaseResult: {exc}")
    else:
        try:
            out = evaluate_thermal(aero)
            out["content_hash"] = aero.case.content_hash()
        except ValueError as exc:             # e.g. a case with no thermal block
            out = _error("input", str(exc))
        except Exception:                      # anything unexpected: a bug, not a bad request
            out = _error("infrastructure", traceback.format_exc())
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
