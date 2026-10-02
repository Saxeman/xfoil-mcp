"""Launching the thermal container from the host.

Most tests replace docker with a fake, so they check how the host handles
each kind of output without needing an image. The last runs the real one.
"""

import json
import subprocess

import pytest

from builders import THERMAL, FakeDocker, aero
from xfoil_mcp import dispatch


def fake_docker(stdout="", raises=None, rm_result=None):
    """A FakeDocker whose `docker run` writes `stdout`, or raises `raises`."""
    fake = FakeDocker(run_result=raises or FakeDocker.exited(0, stdout=stdout, stderr="boom"),
                      rm_result=rm_result)
    return fake, fake.calls


def good_output(for_result):
    return json.dumps({
        "status": "ok", "failure_kind": None, "errors": [], "coverage": 1.0,
        "worst_case": {}, "points": [], "excluded": [],
        "content_hash": for_result.case.content_hash(),
    })


def test_good_output_passes_through(monkeypatch):
    result = aero()
    run, _ = fake_docker(good_output(result))
    monkeypatch.setattr(subprocess, "run", run)
    assert dispatch.run_thermal(result).status == "ok"


def test_result_for_a_different_case_is_rejected(monkeypatch):
    other = aero(thermal={**THERMAL, "chord_m": 0.4})
    run, _ = fake_docker(good_output(other))
    monkeypatch.setattr(subprocess, "run", run)
    out = dispatch.run_thermal(aero())
    assert out.failure_kind == "infrastructure"
    assert "different case" in out.errors[0]


def output_with(for_result, **changes):
    return json.dumps({**json.loads(good_output(for_result)), **changes})


@pytest.mark.parametrize("stdout", [
    "not json",
    json.dumps({"status": "ok"}),                           # the core fields are missing
    json.dumps([1, 2]),
    output_with(aero(), failure_kind="banana"),             # a kind the envelope does not know
    output_with(aero(), failure_kind="numerical"),          # ok cannot carry a kind
    output_with(aero(), errors="boom"),                     # not a list
    output_with(aero(), coverage=7.5),                      # not a fraction
    output_with(aero(), surprise=1),                        # a field the schema does not have
])
def test_malformed_output_is_an_infrastructure_failure(monkeypatch, stdout):
    run, _ = fake_docker(stdout)
    monkeypatch.setattr(subprocess, "run", run)
    assert dispatch.run_thermal(aero()).failure_kind == "infrastructure"


def test_timeout_still_removes_the_container(monkeypatch):
    run, calls = fake_docker(raises=subprocess.TimeoutExpired("docker", 1))
    monkeypatch.setattr(subprocess, "run", run)
    out = dispatch.run_thermal(aero(), timeout=1)
    assert "killed" in out.errors[0]
    assert any(cmd[:3] == ["docker", "rm", "-f"] for cmd, _ in calls)


@pytest.mark.parametrize("rm_result", [
    subprocess.TimeoutExpired("docker", 15),
    FileNotFoundError("docker"),
    FakeDocker.exited(1, stderr="Cannot connect to the Docker daemon"),
])
def test_a_failed_cleanup_does_not_lose_the_thermal_result(monkeypatch, rm_result):
    """run_thermal promises never to raise. The cleanup used to break that."""
    result = aero()
    run, _ = fake_docker(good_output(result), rm_result=rm_result)
    monkeypatch.setattr(subprocess, "run", run)
    assert dispatch.run_thermal(result).status == "ok"


def test_missing_docker_is_a_thermal_failure_not_an_exception(monkeypatch):
    missing = FileNotFoundError("docker")
    run, _ = fake_docker(raises=missing, rm_result=missing)
    monkeypatch.setattr(subprocess, "run", run)
    out = dispatch.run_thermal(aero())
    assert out.failure_kind == "infrastructure"
    assert out.errors == ["docker not found on host"]


def test_a_non_zero_exit_is_an_infrastructure_failure_with_the_reason(monkeypatch):
    fake = FakeDocker(run_result=FakeDocker.exited(137, stderr="Killed"))
    monkeypatch.setattr(subprocess, "run", fake)
    out = dispatch.run_thermal(aero())
    assert (out.status, out.failure_kind) == ("error", "infrastructure")
    assert out.errors == ["thermal worker exited 137: Killed"]


def test_the_thermal_container_runs_under_the_isolation_flags(monkeypatch):
    result = aero()
    fake = FakeDocker(run_result=FakeDocker.exited(0, stdout=good_output(result)))
    monkeypatch.setattr(subprocess, "run", fake)
    dispatch.run_thermal(result, timeout=77.0)
    cmd, kwargs = fake.run_call
    image = cmd.index(dispatch.THERMAL_IMAGE)
    assert cmd[image - len(dispatch.THERMAL_FLAGS):image] == dispatch.THERMAL_FLAGS
    assert cmd[image + 1:] == []                                # the image's own command runs
    assert cmd[cmd.index("--name") + 1].startswith("xfoil-thermal-")
    assert kwargs["timeout"] == 77.0
    for flag in ("--network", "--read-only", "--memory", "--cpus", "--pids-limit", "--cap-drop", "--user"):
        assert flag in dispatch.THERMAL_FLAGS


def test_an_error_result_from_the_container_is_passed_on_without_a_hash(monkeypatch):
    """The stage refuses some cases itself (a failed aero stage, too fast).
    Those results carry no content hash and must still come through."""
    refusal = json.dumps({"status": "error", "failure_kind": "input", "errors": ["too fast"],
                          "coverage": 0.0, "worst_case": None, "points": [], "excluded": []})
    monkeypatch.setattr(subprocess, "run", FakeDocker(run_result=FakeDocker.exited(0, stdout=refusal)))
    out = dispatch.run_thermal(aero())
    assert (out.status, out.failure_kind, out.errors) == ("error", "input", ["too fast"])


@pytest.mark.docker
def test_thermal_runs_in_its_container():
    """The real image, end to end. Needs `docker build ... -t xfoil-thermal`."""
    out = dispatch.run_thermal(aero())
    assert out.status == "ok"
    assert out.worst_case["min_heated_temperature_c"] == pytest.approx(24.8, abs=0.5)
