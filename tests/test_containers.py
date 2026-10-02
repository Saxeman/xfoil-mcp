"""The one place the host starts containers. Docker is replaced by a fake, so
these run anywhere; the docker-marked tests exercise the real thing."""

import subprocess

import pytest

from builders import FakeDocker
from xfoil_mcp import containers


@pytest.fixture
def docker(monkeypatch):
    def install(**kwargs):
        fake = FakeDocker(**kwargs)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake
    return install


def launch(**kwargs):
    return containers.run("some-image", "the input", 30.0, name="xfoil-test", **kwargs)


def test_command_puts_flags_before_the_image_and_argv_after(docker):
    fake = docker()
    launch(flags=["--network", "none"], argv=("python", "-m", "mod"))
    cmd, kwargs = fake.run_call
    name = cmd[cmd.index("--name") + 1]
    assert cmd == ["docker", "run", "--rm", "-i", "--name", name, "--network", "none",
                   "some-image", "python", "-m", "mod"]
    assert name.startswith("xfoil-test-") and len(name) == len("xfoil-test-") + 12
    assert (kwargs["input"], kwargs["timeout"]) == ("the input", 30.0)


def test_each_run_gets_its_own_container_name(docker):
    fake = docker()
    launch(), launch()
    names = {cmd[cmd.index("--name") + 1] for cmd, _ in fake.calls if cmd[1] == "run"}
    assert len(names) == 2


@pytest.mark.parametrize("code", [0, 1, 137])
def test_a_container_that_exits_comes_back_whatever_its_exit_code(docker, code):
    docker(run_result=FakeDocker.exited(code, stdout="payload", stderr="why"))
    result = launch()
    assert (result.proc.returncode, result.proc.stdout, result.proc.stderr) == (code, "payload", "why")
    assert result.error is None and not result.timed_out


def test_a_timeout_is_reported_and_the_container_is_removed_by_name(docker):
    fake = docker(run_result=subprocess.TimeoutExpired("docker", 30.0))
    result = launch()
    assert result.proc is None and result.timed_out
    assert result.reason("worker") == "worker exceeded 30.0s and was killed"
    started = fake.run_call[0][fake.run_call[0].index("--name") + 1]
    assert fake.rm_call[0] == ["docker", "rm", "-f", started]


def test_missing_docker_is_an_error_not_an_exception(docker):
    """The cleanup call hits the same missing binary. Neither may escape."""
    docker(run_result=FileNotFoundError("docker"), rm_result=FileNotFoundError("docker"))
    result = launch()
    assert result.proc is None and not result.timed_out
    assert result.reason("worker") == "docker not found on host"


def test_any_other_failure_to_start_docker_is_an_error_too(docker):
    docker(run_result=PermissionError(13, "Permission denied"))
    result = launch()
    assert result.proc is None and not result.timed_out
    assert result.error.startswith("could not run docker:") and "Permission denied" in result.error


@pytest.mark.parametrize("rm_result", [
    subprocess.TimeoutExpired("docker", 15),                    # the daemon hangs
    FileNotFoundError("docker"),                                # docker vanished mid-run
    FakeDocker.exited(1, stderr="Cannot connect to the Docker daemon"),
])
def test_a_failed_cleanup_never_replaces_the_result(docker, rm_result):
    docker(run_result=FakeDocker.exited(0, stdout="payload"), rm_result=rm_result)
    assert launch().proc.stdout == "payload"


def test_cleanup_always_runs_and_does_not_inherit_stdin(docker):
    """The server's stdin is the MCP protocol stream. No child may hold it."""
    fake = docker()
    launch()
    cmd, kwargs = fake.rm_call
    assert cmd[:3] == ["docker", "rm", "-f"]
    assert kwargs["stdin"] is subprocess.DEVNULL
