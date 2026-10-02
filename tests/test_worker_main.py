"""The worker's entry point on the host: a Case on stdin, a CaseResult on
stdout, exit 0. XFOIL is replaced by `builders.fake_xfoil`.

test_worker.py runs worker.run_case against the real solver in the container.
"""

import io
import json
import subprocess

import pytest

from builders import FakeDocker, fake_xfoil
from xfoil_mcp import dispatch, worker, wrapper
from xfoil_mcp.schema import Case, CaseResult, Conditions, Flap, Geometry
from xfoil_mcp.wrapper import XfoilError


def make_case(outputs=("forces",), flap=None) -> Case:
    return Case(geometry=Geometry(naca="2412", flap=flap), outputs=outputs,
                conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=4, alpha_step=2))


@pytest.fixture
def xfoil(tmp_path, monkeypatch):
    def install(**kwargs):
        monkeypatch.setattr(wrapper, "XFOIL_BIN", fake_xfoil(tmp_path, **kwargs))
    return install


def run_main(monkeypatch, stdin: str) -> str:
    """Feed `stdin` to worker.main() and return what it wrote to stdout."""
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    monkeypatch.setattr("sys.stdout", stdout)
    assert worker.main() == 0
    return stdout.getvalue()


# --- run_case -------------------------------------------------------------------

def test_a_clean_run_is_an_ok_result_with_summary_data_and_provenance(xfoil):
    xfoil()
    case = make_case()
    result = worker.run_case(case)
    assert (result.status, result.failure_kind) == ("ok", None)
    assert result.summary["converged_points"] == 3
    assert [p["alpha"] for p in result.data["points"]] == [0.0, 2.0, 4.0]
    assert result.provenance["content_hash"] == case.content_hash()
    assert result.provenance["solver"] == worker.SOLVER
    assert result.provenance["runtime_seconds"] >= 0


@pytest.mark.parametrize("skip, status", [((2.0,), "partial"), ((0.0, 2.0, 4.0), "empty")])
def test_missing_alphas_are_numerical_failures(xfoil, skip, status):
    xfoil(skip=skip)
    result = worker.run_case(make_case())
    assert (result.status, result.failure_kind, result.retry_could_help) == (status, "numerical", False)


def test_the_stdout_log_stays_out_of_the_result(xfoil):
    xfoil(skip=(2.0,))
    result = worker.run_case(make_case())
    assert "stdout" not in result.data
    assert "VISCAL:  Convergence failed" not in result.model_dump_json()     # the raw log line


def test_field_outputs_are_keyed_by_alpha_as_a_string(xfoil):
    """JSON object keys must be strings. The thermal stage reads them back with float()."""
    xfoil()
    result = worker.run_case(make_case(outputs=("forces", "bl", "cp")))
    assert sorted(result.data["bl"]) == ["0.0", "2.0", "4.0"]
    assert sorted(result.data["cp"]) == ["0.0", "2.0", "4.0"]
    assert CaseResult.model_validate_json(result.model_dump_json()) == result


def test_the_flap_reaches_the_solver(xfoil):
    xfoil()
    result = worker.run_case(make_case(flap=Flap(x_hinge=0.7, deflection=10.0)))
    assert result.status == "ok"


def test_a_hinge_outside_the_section_is_an_input_failure(xfoil):
    """The request was wrong, so a retry cannot help."""
    xfoil()
    result = worker.run_case(make_case(flap=Flap(x_hinge=0.7, y_hinge=0.3, deflection=10.0)))
    assert (result.status, result.failure_kind, result.retry_could_help) == ("error", "input", False)
    assert "outside the airfoil" in result.summary["error"]


def test_a_solver_that_cannot_run_is_an_infrastructure_failure(monkeypatch):
    monkeypatch.setattr(wrapper, "XFOIL_BIN", "/nonexistent/xfoil")
    result = worker.run_case(make_case())
    assert (result.status, result.failure_kind, result.retry_could_help) == ("error", "infrastructure", True)
    assert "not found" in result.summary["error"]
    assert result.data is None


# --- main ---------------------------------------------------------------------

def test_main_writes_exactly_one_case_result(xfoil, monkeypatch):
    xfoil()
    case = make_case()
    out = run_main(monkeypatch, case.model_dump_json())
    result = CaseResult.model_validate_json(out)
    assert result.status == "ok"
    assert result.case.content_hash() == case.content_hash()       # what the host checks


def test_main_turns_an_unexpected_exception_into_a_result(monkeypatch):
    """A bug in the worker still produces a result the host can parse."""
    def broken(case):
        raise RuntimeError("the worker has a bug")
    monkeypatch.setattr(worker, "run_case", broken)
    case = make_case()
    result = CaseResult.model_validate_json(run_main(monkeypatch, case.model_dump_json()))
    assert (result.status, result.failure_kind) == ("error", "infrastructure")
    assert "the worker has a bug" in result.summary["error"]
    assert result.case == case


def test_main_returns_zero_for_a_solver_failure(monkeypatch):
    def failing(**kwargs):
        raise XfoilError("XFOIL exceeded 120s")
    monkeypatch.setattr(worker, "run_polar", failing)
    result = CaseResult.model_validate_json(run_main(monkeypatch, make_case().model_dump_json()))
    assert (result.status, result.failure_kind) == ("error", "infrastructure")


@pytest.mark.parametrize("stdin", ["", "not json", '{"geometry": {"naca": "24a2"}}'])
def test_an_unparseable_case_still_produces_valid_json(monkeypatch, stdin):
    payload = json.loads(run_main(monkeypatch, stdin))
    assert (payload["status"], payload["failure_kind"]) == ("error", "input")


def test_an_unparseable_case_reaches_the_host_as_an_input_failure(monkeypatch):
    """A worker image built before a schema change receives a Case it cannot
    parse. The host should hear "input", not "infrastructure, retry"."""
    case = make_case()
    stale = json.dumps({**json.loads(case.model_dump_json()), "field_this_worker_predates": 1})
    worker_stdout = run_main(monkeypatch, stale)
    monkeypatch.setattr(subprocess, "run", FakeDocker(run_result=FakeDocker.exited(0, stdout=worker_stdout)))
    result = dispatch.run_case(case)
    assert (result.failure_kind, result.retry_could_help) == ("input", False)
