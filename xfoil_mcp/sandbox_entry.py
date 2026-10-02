"""Sandbox entry point. Runs inside the xfoil-sandbox container.

    stdin:  campaign source (Python) defining `campaign() -> list[Case]`
    stdout: {"cases": [...], "errors": [...]} as JSON

Executes the source in a fresh namespace, calls campaign(), checks the
result is a list of Case, serializes. Every exception is captured into
`errors` rather than raised. This image has no XFOIL and no harness, so a
campaign cannot run anything; it can only describe what it would run.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback

from xfoil_mcp.schema import Case

# Recorded at import, before any campaign code runs, so main() can tell the
# original process from children a campaign forked.
_ORIGINAL_PID = os.getpid()


def _contained(what: str, action):
    """Run one piece of campaign code. Returns (its value, None) or (None, error).

    Stdout belongs to the payload, so anything the campaign prints is sent to
    stderr instead of corrupting it. sys.exit() is reported like any other
    failure: SystemExit is not an Exception, and uncaught it would end the
    process before the payload was written.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            return action(), None
    except SystemExit as exc:
        return None, f"{what}: it called sys.exit({exc.code!r}). A campaign returns its cases; it must not exit"
    except Exception:
        return None, f"{what}:\n{traceback.format_exc()}"


def _run(source: str) -> dict:
    errors: list[str] = []
    namespace: dict = {"__name__": "campaign_module"}

    _, error = _contained("campaign failed to load",
                          lambda: exec(compile(source, "<campaign>", "exec"), namespace))
    if error:
        return {"cases": [], "errors": [error]}

    fn = namespace.get("campaign")
    if not callable(fn):
        return {"cases": [], "errors": ["campaign source must define campaign() -> list[Case]"]}

    produced, error = _contained("campaign() raised", fn)
    if error:
        return {"cases": [], "errors": [error]}

    if not isinstance(produced, (list, tuple)):
        return {"cases": [], "errors": [f"campaign() must return a list, got {type(produced).__name__}"]}

    cases: list[dict] = []
    for i, item in enumerate(produced):
        if not isinstance(item, Case):
            errors.append(f"item {i} is {type(item).__name__}, not Case")
            continue
        cases.append(item.model_dump(mode="json"))

    return {"cases": cases, "errors": errors}


def main() -> int:
    source = sys.stdin.read()
    payload = _run(source)

    # A campaign that forked leaves child processes that will also reach
    # this line. Only the original process owns stdout.
    if os.getpid() != _ORIGINAL_PID:
        os._exit(0)

    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    sys.exit(main())
