"""Thermal worker entry point. Runs inside the xfoil-thermal container.

    stdin:  one aero CaseResult as JSON, for a case with a thermal block
    stdout: one thermal summary as JSON
    exit:   0, unless the process itself is broken

Only the result goes to stdout; the host parses that stream.
"""

from __future__ import annotations

import sys
import traceback

from xfoil_mcp.schema import CaseResult, ThermalResult
from xfoil_mcp.thermal_stage import NoThermalBlock, evaluate_thermal


def main() -> int:
    raw = sys.stdin.read()
    try:
        aero = CaseResult.model_validate_json(raw)
    except Exception as exc:
        out = ThermalResult.failure("input", f"input was not an aero CaseResult: {exc}")
    else:
        try:
            out = evaluate_thermal(aero)
        except NoThermalBlock as exc:                  # the request was wrong
            out = ThermalResult.failure("input", str(exc))
        except (ArithmeticError, ValueError):          # the numbers were: the same input fails the same way
            out = ThermalResult.failure("numerical", traceback.format_exc())
        except Exception:                              # anything else is a bug, not a bad request
            out = ThermalResult.failure("infrastructure", traceback.format_exc())
    sys.stdout.write(out.model_dump_json())
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
