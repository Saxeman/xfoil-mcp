"""The sandbox container's entry point, run on the host.

It needs only pydantic, so nothing here needs Docker. The container around it
(no network, read-only, capped) is tested in test_sandbox.py; this file tests
what the entry point does with each kind of campaign source.

These campaigns are harmless. The fork test runs in a child process because
forking the test runner itself would duplicate it.
"""

import io
import json
import subprocess
import sys

import pytest

from xfoil_mcp import sandbox_entry
from xfoil_mcp.sandbox_entry import _run
from xfoil_mcp.schema import Case

GOOD = '''
from xfoil_mcp.schema import Case, Conditions, Geometry

def campaign():
    return [Case(geometry=Geometry(naca=n),
                 conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=4, alpha_step=2))
            for n in ("2412", "0012")]
'''


def run_main(monkeypatch, source: str) -> str:
    """Feed `source` to main() and return what it wrote to stdout."""
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(source))
    monkeypatch.setattr("sys.stdout", stdout)
    assert sandbox_entry.main() == 0
    return stdout.getvalue()


def test_a_campaign_returns_its_cases_as_json_ready_dicts():
    payload = _run(GOOD)
    assert payload["errors"] == []
    assert [c["geometry"]["naca"] for c in payload["cases"]] == ["2412", "0012"]
    assert [Case.model_validate(c).geometry.naca for c in payload["cases"]] == ["2412", "0012"]


def test_a_tuple_of_cases_is_accepted_like_a_list():
    assert len(_run(GOOD.replace("return [", "return tuple([").replace('"0012")]', '"0012")])'))["cases"]) == 2


@pytest.mark.parametrize("source, expected", [
    ("def campaign(:\n    pass\n", "campaign failed to load"),                  # syntax error
    ("raise RuntimeError('at import')\n", "campaign failed to load"),
    ("from xfoil_mcp import harness_that_is_not_here\n", "campaign failed to load"),
    ("x = 1\n", "must define campaign()"),
    ("campaign = 5\n", "must define campaign()"),                               # not callable
    ("def campaign():\n    raise ValueError('boom')\n", "campaign() raised"),
    ("def campaign():\n    return 5\n", "campaign() must return a list, got int"),
    ("def campaign():\n    return {'a': 1}\n", "campaign() must return a list, got dict"),
])
def test_a_campaign_that_cannot_produce_cases_reports_why(source, expected):
    payload = _run(source)
    assert payload["cases"] == []
    assert len(payload["errors"]) == 1 and expected in payload["errors"][0]


def test_a_failure_carries_the_traceback_so_the_agent_can_fix_it():
    payload = _run("def campaign():\n    raise ValueError('the reason')\n")
    assert "ValueError: the reason" in payload["errors"][0]
    assert '"<campaign>"' in payload["errors"][0]                # the source is named, not a temp file


def test_an_invalid_case_is_reported_by_the_schema_inside_the_sandbox():
    source = GOOD.replace('naca=n', 'naca="24a2"')
    payload = _run(source)
    assert payload["cases"] == []
    assert "campaign() raised" in payload["errors"][0] and "naca must be 4 or 5 digits" in payload["errors"][0]


def test_items_that_are_not_cases_are_named_and_the_rest_are_kept():
    source = GOOD.replace("return [", "return ['not a case', 7] + [")
    payload = _run(source)
    assert payload["errors"] == ["item 0 is str, not Case", "item 1 is int, not Case"]
    assert len(payload["cases"]) == 2


def test_the_campaign_runs_in_a_fresh_namespace():
    """It sees nothing of the entry point's globals, and a second run sees
    nothing of the first."""
    assert "NameError" in _run("def campaign():\n    return [_ORIGINAL_PID]\n")["errors"][0]
    _run("leaked = 1\ndef campaign():\n    return []\n")
    assert "NameError" in _run("def campaign():\n    return [leaked]\n")["errors"][0]


def test_main_writes_one_json_payload(monkeypatch):
    payload = json.loads(run_main(monkeypatch, GOOD))
    assert sorted(payload) == ["cases", "errors"]
    assert len(payload["cases"]) == 2


def test_main_reports_a_broken_campaign_and_still_exits_zero(monkeypatch):
    payload = json.loads(run_main(monkeypatch, "def campaign():\n    raise ValueError('boom')\n"))
    assert payload["cases"] == [] and "campaign() raised" in payload["errors"][0]


def test_a_campaign_that_forks_still_yields_exactly_one_payload():
    """Every forked child reaches the end of main() too. Only the original
    process may write the payload, or the host would read several."""
    source = "import os\nos.fork()\n" + GOOD
    proc = subprocess.run([sys.executable, "-B", "-m", "xfoil_mcp.sandbox_entry"],
                          input=source, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)                           # one document, not two
    assert len(payload["cases"]) == 2


def test_a_campaign_that_prints_does_not_corrupt_the_payload(monkeypatch):
    out = run_main(monkeypatch, "print('checking cases')\n" + GOOD)
    assert len(json.loads(out)["cases"]) == 2


@pytest.mark.parametrize("code", [0, 3])
def test_a_campaign_that_calls_sys_exit_is_reported_not_obeyed(code):
    payload = _run(f"import sys\nsys.exit({code})\n")
    assert payload["cases"] == [] and payload["errors"]


def test_what_a_campaign_prints_goes_to_stderr(monkeypatch, capsys):
    out = run_main(monkeypatch, "print('checking cases')\n" + GOOD)
    assert "checking cases" not in out
    assert "checking cases" in capsys.readouterr().err


def test_an_exit_is_named_with_its_code_and_where_it_happened():
    at_load = _run("import sys\nsys.exit(3)\n")["errors"][0]
    in_call = _run("import sys\ndef campaign():\n    sys.exit('giving up')\n")["errors"][0]
    assert at_load.startswith("campaign failed to load: it called sys.exit(3)")
    assert in_call.startswith("campaign() raised: it called sys.exit('giving up')")
