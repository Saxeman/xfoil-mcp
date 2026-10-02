"""The thermal container's entry point: an aero CaseResult on stdin, one
ThermalResult on stdout, exit 0. Run in-process here; needs the thermal extra."""

import io
import json

import pytest

pytest.importorskip("skfem")     # the thermal extra

from builders import aero
from xfoil_mcp import thermal_entry
from xfoil_mcp.schema import ThermalResult


def run_entry(monkeypatch, stdin: str) -> ThermalResult:
    """Feed `stdin` to main() and parse what it wrote, as the host would."""
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    monkeypatch.setattr("sys.stdout", stdout)
    assert thermal_entry.main() == 0
    return ThermalResult.model_validate_json(stdout.getvalue())


def test_a_good_aero_result_gives_a_result_that_names_its_case(monkeypatch):
    result = aero()
    out = run_entry(monkeypatch, result.model_dump_json())
    assert (out.status, out.coverage) == ("ok", 1.0)
    assert out.content_hash == result.case.content_hash()
    assert out.worst_case["min_heated_temperature_c"] == pytest.approx(24.8, abs=0.5)


def test_input_that_is_not_a_case_result_is_an_input_error(monkeypatch):
    out = run_entry(monkeypatch, "not json")
    assert (out.status, out.failure_kind) == ("error", "input")
    assert "not an aero CaseResult" in out.errors[0]


def test_a_case_without_a_thermal_block_is_an_input_error(monkeypatch):
    result = aero()
    bare = result.model_copy(update={"case": result.case.model_copy(update={"thermal": None})})
    out = run_entry(monkeypatch, bare.model_dump_json())
    assert (out.status, out.failure_kind) == ("error", "input")
    assert "no thermal block" in out.errors[0]


def test_an_unexpected_exception_is_reported_not_raised(monkeypatch):
    """A bug in the stage still produces one well-formed result and exit 0."""
    def broken(aero):
        raise RuntimeError("the stage has a bug")
    monkeypatch.setattr(thermal_entry, "evaluate_thermal", broken)
    out = run_entry(monkeypatch, aero().model_dump_json())
    assert (out.status, out.failure_kind) == ("error", "infrastructure")
    assert "the stage has a bug" in out.errors[0]


def test_the_output_is_strict_json_even_when_a_number_is_not_finite(monkeypatch):
    """NaN and Infinity are not JSON. A strict parser on the host must accept
    whatever the entry point writes."""
    def blown_up(aero_result):
        return ThermalResult(status="ok", failure_kind=None, errors=[], coverage=1.0,
                             worst_case={"min_heated_temperature_c": float("nan")},
                             points=[{"alpha": 4.0, "peak_temperature_c": float("inf")}], excluded=[])
    monkeypatch.setattr(thermal_entry, "evaluate_thermal", blown_up)
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(aero().model_dump_json()))
    monkeypatch.setattr("sys.stdout", stdout)
    thermal_entry.main()

    def refuse(token):
        raise AssertionError(f"non-standard JSON token {token}")
    payload = json.loads(stdout.getvalue(), parse_constant=refuse)
    assert payload["worst_case"]["min_heated_temperature_c"] is None


def test_a_deterministic_numerical_failure_is_not_called_retryable(monkeypatch):
    def divides_by_zero(aero_result):
        raise ZeroDivisionError("float division by zero")
    monkeypatch.setattr(thermal_entry, "evaluate_thermal", divides_by_zero)
    out = run_entry(monkeypatch, aero().model_dump_json())
    assert out.failure_kind == "numerical"
