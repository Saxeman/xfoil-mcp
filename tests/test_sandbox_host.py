"""How the host handles what comes back from the sandbox container.

Docker is replaced by a fake, so these run anywhere. test_sandbox.py runs the
real container against the reference and hostile campaigns.
"""

import json
import subprocess

import pytest

from builders import FakeDocker
from xfoil_mcp import sandbox

GOOD_CASE = {"geometry": {"naca": "2412"},
             "conditions": {"reynolds": 1e6, "alpha_start": 0, "alpha_end": 5, "alpha_step": 1}}


def run_with(monkeypatch, **fake_kwargs):
    fake = FakeDocker(**fake_kwargs)
    monkeypatch.setattr(subprocess, "run", fake)
    return sandbox.run_campaign_source("the campaign source", timeout=30.0), fake


def payload(cases=(), errors=()):
    return FakeDocker.exited(0, stdout=json.dumps({"cases": list(cases), "errors": list(errors)}))


def test_cases_from_the_sandbox_are_rebuilt_through_the_schema(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=payload([GOOD_CASE]))
    assert [c.geometry.naca for c in outcome.cases] == ["2412"]
    assert (outcome.errors, outcome.rejected, outcome.killed) == ([], [], False)


def test_malformed_cases_from_sandbox_are_rejected_on_host(monkeypatch):
    """If the sandbox somehow emits an out-of-bounds case, the host rejects
    it rather than running it."""
    bad = {**GOOD_CASE, "conditions": {**GOOD_CASE["conditions"], "reynolds": -1}}
    outcome, _ = run_with(monkeypatch, run_result=payload([bad]))
    assert outcome.cases == []
    assert len(outcome.rejected) == 1
    assert "reynolds" in outcome.rejected[0]


def test_errors_the_campaign_reported_are_passed_on(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=payload(errors=["campaign() raised:\nboom"]))
    assert outcome.errors == ["campaign() raised:\nboom"]
    assert not outcome.killed


def test_the_source_goes_in_on_stdin_under_the_isolation_flags(monkeypatch):
    _, fake = run_with(monkeypatch, run_result=payload())
    cmd, kwargs = fake.run_call
    assert kwargs["input"] == "the campaign source"
    assert cmd[cmd.index("--name") + 1].startswith("xfoil-sandbox-")
    image = cmd.index(sandbox.SANDBOX_IMAGE)
    assert cmd[image - len(sandbox.SANDBOX_FLAGS):image] == sandbox.SANDBOX_FLAGS
    assert cmd[image + 1:] == []                    # the image's own command runs


def test_a_timeout_is_a_killed_campaign(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=subprocess.TimeoutExpired("docker", 30.0))
    assert outcome.killed
    assert outcome.errors == ["campaign exceeded 30.0s and was killed"]
    assert outcome.cases == []


def test_missing_docker_is_an_error_but_not_a_killed_campaign(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=FileNotFoundError("docker"),
                          rm_result=FileNotFoundError("docker"))
    assert not outcome.killed
    assert outcome.errors == ["docker not found on host"]


def test_an_out_of_memory_kill_names_the_memory_limit_and_is_not_a_timeout(monkeypatch):
    """Exit 137 is the kernel killing the container at its memory limit.
    `killed` means the time limit; this is a different thing to fix."""
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(137, stderr="x" * 600 + " Killed"))
    assert not outcome.killed and not outcome.infrastructure
    assert outcome.errors[0].startswith("sandbox exited 137: killed, most likely for exceeding the 512m memory limit")
    assert outcome.errors[0].endswith(" Killed") and len(outcome.errors[0]) < 620     # the end of stderr, not all of it


def test_any_other_exit_code_is_reported_as_the_campaign_ending_the_process(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(3))
    assert not outcome.killed and not outcome.infrastructure
    assert outcome.errors == ["sandbox exited 3: the campaign ended the process, or the interpreter crashed."]


@pytest.mark.parametrize("code", [125, 126, 127])
def test_docker_failing_to_start_the_container_is_infrastructure(monkeypatch, code):
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(code, stderr="Unable to find image"))
    assert outcome.infrastructure and not outcome.killed
    assert outcome.errors == [f"docker could not start the sandbox (exit {code}): Unable to find image"]


def test_only_docker_failures_are_infrastructure(monkeypatch):
    """A timeout is the campaign's doing; a missing docker is not."""
    timed_out, _ = run_with(monkeypatch, run_result=subprocess.TimeoutExpired("docker", 30.0))
    missing, _ = run_with(monkeypatch, run_result=FileNotFoundError("docker"), rm_result=FileNotFoundError("docker"))
    clean, _ = run_with(monkeypatch, run_result=payload(errors=["campaign() raised:\nboom"]))
    assert (timed_out.killed, timed_out.infrastructure) == (True, False)
    assert (missing.killed, missing.infrastructure) == (False, True)
    assert (clean.killed, clean.infrastructure) == (False, False)


def test_output_that_is_not_json_is_an_error(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(0, stdout="hello\n{}"))
    assert not outcome.killed
    assert outcome.errors[0].startswith("sandbox output was not JSON")


@pytest.mark.parametrize("rm_result", [subprocess.TimeoutExpired("docker", 15), FileNotFoundError("docker")])
def test_a_failed_cleanup_does_not_lose_the_cases(monkeypatch, rm_result):
    outcome, _ = run_with(monkeypatch, run_result=payload([GOOD_CASE]), rm_result=rm_result)
    assert len(outcome.cases) == 1


@pytest.mark.parametrize("stdout", [
    "[]",                                   # not an object
    '{"cases": 5, "errors": []}',           # cases is not a list
    '{"cases": [], "errors": 7}',           # errors is not a list
])
def test_a_payload_of_the_wrong_shape_is_an_error_not_an_exception(monkeypatch, stdout):
    """A campaign can write its own JSON and call os._exit(0). Whatever it
    writes, the host must return an outcome."""
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(0, stdout=stdout))
    assert outcome.cases == [] and outcome.errors


def test_errors_that_are_not_a_list_of_strings_are_not_split_into_characters(monkeypatch):
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(0, stdout='{"cases": [], "errors": "abc"}'))
    assert outcome.errors != ["a", "b", "c"]


def test_a_container_that_could_not_start_is_not_a_killed_campaign(monkeypatch):
    """Exit 125 is docker failing to start the container (image missing,
    daemon down). The campaign never ran, so it was not killed."""
    outcome, _ = run_with(monkeypatch, run_result=FakeDocker.exited(125, stderr="Unable to find image"))
    assert outcome.errors and not outcome.killed
