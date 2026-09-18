"""Sandbox entry point. Runs inside the xfoil-sandbox container.

    stdin:  campaign source (Python) defining `campaign() -> list[Case]`
    stdout: {"cases": [...], "errors": [...]} as JSON

Executes the source in a fresh namespace, calls campaign(), checks the
result is a list of Case, serializes. Every exception is captured into
`errors` rather than raised. This image has no XFOIL and no harness, so a
campaign cannot run anything; it can only describe what it would run.
"""

from __future__ import annotations

import json
import sys
import traceback

from xfoil_mcp.schema import Case


def _run(source: str) -> dict:
    errors: list[str] = []
    namespace: dict = {"__name__": "campaign_module"}

    try:
        exec(compile(source, "<campaign>", "exec"), namespace)
    except Exception:
        return {"cases": [], "errors": [f"campaign failed to load:\n{traceback.format_exc()}"]}

    fn = namespace.get("campaign")
    if not callable(fn):
        return {"cases": [], "errors": ["campaign source must define campaign() -> list[Case]"]}

    try:
        produced = fn()
    except Exception:
        return {"cases": [], "errors": [f"campaign() raised:\n{traceback.format_exc()}"]}

    if not isinstance(produced, (list, tuple)):
        return {"cases": [], "errors": [f"campaign() must return a list, got {type(produced).__name__}"]}

    cases: list[dict] = []
    for i, item in enumerate(produced):
        if not isinstance(item, Case):
            errors.append(f"item {i} is {type(item).__name__}, not Case")
            continue
        cases.append(item.model_dump(mode="json"))

    return {"cases": cases, "errors": errors}

import os
_ORIGINAL_PID = os.getpid()


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
