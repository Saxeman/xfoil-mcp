"""Launching the worker container from the host, and judging what comes back.

Docker is replaced by a fake, so these run anywhere. The worker itself is
tested inside its container (test_worker.py).
"""

import subprocess

import pytest

from builders import FakeDocker
from xfoil_mcp import dispatch
from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry


def make_case(naca="2412") -> Case:
    return Case(geometry=Geometry(naca=naca),
                conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=1))


def answer(case: Case) -> subprocess.CompletedProcess:
    """What a healthy worker writes for `case`."""
    result = CaseResult(case=case, status="ok", summary={"best_ld": 104.4})
    return FakeDocker.exited(0, stdout=result.model_dump_json())


def run_with(monkeypatch, case, timeout=180.0, **fake_kwargs):
    fake = FakeDocker(**fake_kwargs)
    monkeypatch.setattr(subprocess, "run", fake)
    return dispatch.run_case(case, timeout=timeout), fake


def assert_infrastructure_failure(result, case, containing):
    assert (result.status, result.failure_kind, result.retry_could_help) == ("error", "infrastructure", True)
    assert containing in result.summary["error"]
    assert result.case == case                      # a failure still says which case it was


def test_a_good_result_comes_back_rebuilt_through_the_schema(monkeypatch):
    case = make_case()
    result, _ = run_with(monkeypatch, case, run_result=answer(case))
    assert isinstance(result, CaseResult)
    assert (result.status, result.summary) == ("ok", {"best_ld": 104.4})


def test_the_case_goes_in_on_stdin_to_the_worker_module(monkeypatch):
    case = make_case()
    _, fake = run_with(monkeypatch, case, timeout=42.0, run_result=answer(case))
    cmd, kwargs = fake.run_call
    assert Case.model_validate_json(kwargs["input"]) == case
    assert kwargs["timeout"] == 42.0
    assert cmd[cmd.index("--name") + 1].startswith("xfoil-worker-")
    image = cmd.index(dispatch.WORKER_IMAGE)
    assert cmd[image - len(dispatch.WORKER_FLAGS):image] == ["--network", "none"]
    assert cmd[image + 1:] == ["python", "-m", "xfoil_mcp.worker"]


def test_a_result_for_a_different_case_is_rejected(monkeypatch):
    """The worker answered a question it was not asked."""
    case = make_case("2412")
    result, _ = run_with(monkeypatch, case, run_result=answer(make_case("4412")))
    assert_infrastructure_failure(result, case, "different case")


@pytest.mark.parametrize("stdout", [
    "",
    "not json",
    '{"status": "ok"}',                                              # no case
])
def test_output_that_is_not_a_case_result_is_rejected(monkeypatch, stdout):
    case = make_case()
    result, _ = run_with(monkeypatch, case, run_result=FakeDocker.exited(0, stdout=stdout))
    assert_infrastructure_failure(result, case, "not a CaseResult")


def test_a_non_zero_exit_reports_the_code_and_the_end_of_stderr(monkeypatch):
    case = make_case()
    result, _ = run_with(monkeypatch, case, run_result=FakeDocker.exited(125, stderr="Unable to find image"))
    assert_infrastructure_failure(result, case, "worker exited 125: Unable to find image")


def test_a_timeout_is_reported_and_the_container_is_removed(monkeypatch):
    case = make_case()
    result, fake = run_with(monkeypatch, case, timeout=5.0,
                            run_result=subprocess.TimeoutExpired("docker", 5.0))
    assert_infrastructure_failure(result, case, "worker exceeded 5.0s and was killed")
    assert fake.rm_call[0][:3] == ["docker", "rm", "-f"]


def test_missing_docker_is_a_result_not_an_exception(monkeypatch):
    case = make_case()
    result, _ = run_with(monkeypatch, case, run_result=FileNotFoundError("docker"),
                         rm_result=FileNotFoundError("docker"))
    assert_infrastructure_failure(result, case, "docker not found on host")


def test_docker_that_cannot_be_executed_is_a_result_not_an_exception(monkeypatch):
    case = make_case()
    result, _ = run_with(monkeypatch, case, run_result=PermissionError(13, "Permission denied"))
    assert_infrastructure_failure(result, case, "could not run docker")


@pytest.mark.parametrize("rm_result", [
    subprocess.TimeoutExpired("docker", 15),
    FileNotFoundError("docker"),
    FakeDocker.exited(1, stderr="Cannot connect to the Docker daemon"),
])
def test_a_failed_cleanup_does_not_lose_the_result(monkeypatch, rm_result):
    case = make_case()
    result, _ = run_with(monkeypatch, case, run_result=answer(case), rm_result=rm_result)
    assert result.status == "ok"


def test_a_wedged_docker_after_a_timeout_still_returns(monkeypatch):
    """The time limit is often hit because docker itself is stuck, and then
    the cleanup hangs too. That must not turn into an exception."""
    case = make_case()
    stuck = subprocess.TimeoutExpired("docker", 5.0)
    result, _ = run_with(monkeypatch, case, timeout=5.0, run_result=stuck, rm_result=stuck)
    assert_infrastructure_failure(result, case, "worker exceeded 5.0s and was killed")


def test_a_worker_that_could_not_parse_the_case_is_an_input_failure_naming_the_image(monkeypatch):
    """The worker has no Case to answer with, so it cannot send a CaseResult.
    The host recognises its refusal and says what to do: a retry cannot help."""
    case = make_case()
    refusal = '{"status": "error", "failure_kind": "input", "summary": {"error": "unparseable case: extra field"}}'
    result, _ = run_with(monkeypatch, case, run_result=FakeDocker.exited(0, stdout=refusal))
    assert (result.status, result.failure_kind, result.retry_could_help) == ("error", "input", False)
    assert "rebuild" in result.summary["error"] and "extra field" in result.summary["error"]
    assert result.case == case


def test_an_error_object_with_another_kind_is_still_not_a_case_result(monkeypatch):
    """Only the worker's own refusal is recognised. Anything else that is not
    a CaseResult stays an infrastructure failure."""
    case = make_case()
    other = '{"status": "error", "failure_kind": "numerical", "summary": {}}'
    result, _ = run_with(monkeypatch, case, run_result=FakeDocker.exited(0, stdout=other))
    assert_infrastructure_failure(result, case, "not a CaseResult")
