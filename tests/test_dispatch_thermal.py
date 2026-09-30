"""Launching the thermal container from the host.

Most tests replace docker with a fake, so they check how the host handles
each kind of output without needing an image. The last runs the real one.
"""

import json
import subprocess

import pytest

from builders import THERMAL, aero
from xfoil_mcp import dispatch


def fake_docker(stdout="", returncode=0, raises=None):
    """Stand in for subprocess.run: answers `docker rm` quietly, and gives
    `docker run` the output we want to test against. Records every call."""
    calls = []

    def run(cmd, *args, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="boom")

    return run, calls


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
    assert dispatch.run_thermal(result)["status"] == "ok"


def test_result_for_a_different_case_is_rejected(monkeypatch):
    other = aero(thermal={**THERMAL, "chord_m": 0.4})
    run, _ = fake_docker(good_output(other))
    monkeypatch.setattr(subprocess, "run", run)
    out = dispatch.run_thermal(aero())
    assert out["failure_kind"] == "infrastructure"
    assert "different case" in out["errors"][0]


@pytest.mark.parametrize("stdout", ["not json", json.dumps({"status": "ok"}), json.dumps([1, 2])])
def test_malformed_output_is_an_infrastructure_failure(monkeypatch, stdout):
    run, _ = fake_docker(stdout)
    monkeypatch.setattr(subprocess, "run", run)
    assert dispatch.run_thermal(aero())["failure_kind"] == "infrastructure"


def test_timeout_still_removes_the_container(monkeypatch):
    run, calls = fake_docker(raises=subprocess.TimeoutExpired("docker", 1))
    monkeypatch.setattr(subprocess, "run", run)
    out = dispatch.run_thermal(aero(), timeout=1)
    assert "killed" in out["errors"][0]
    assert any(c[:3] == ["docker", "rm", "-f"] for c in calls)


@pytest.mark.docker
def test_thermal_runs_in_its_container():
    """The real image, end to end. Needs `docker build ... -t xfoil-thermal`."""
    out = dispatch.run_thermal(aero())
    assert out["status"] == "ok"
    assert out["worst_case"]["min_heated_temperature_c"] == pytest.approx(24.8, abs=0.5)
