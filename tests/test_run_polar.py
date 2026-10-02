"""run_polar on the host, against a stand-in for the xfoil binary.

The real solver is tested inside the worker container (test_wrapper.py).
These tests cover what run_polar itself does around the solver: bookkeeping
of converged and failed alphas, field files, warnings, and the ways the
process can fail. The stand-in is `builders.fake_xfoil`.
"""

import pytest

from builders import fake_xfoil
from xfoil_mcp import wrapper
from xfoil_mcp.wrapper import InvalidGeometry, XfoilError, run_polar


@pytest.fixture
def xfoil(tmp_path, monkeypatch):
    """Install a stand-in binary: xfoil(mode=..., skip=..., nan=...)."""
    def install(**kwargs):
        monkeypatch.setattr(wrapper, "XFOIL_BIN", fake_xfoil(tmp_path, **kwargs))
    return install


def test_a_clean_sweep_returns_every_requested_alpha(xfoil):
    xfoil()
    result = run_polar("2412", 1e6, 0, 4, 2)
    assert result.status == "ok"
    assert result.requested_alphas == [0.0, 2.0, 4.0]
    assert [p.alpha for p in result.points] == [0.0, 2.0, 4.0]
    assert (result.failed_alphas, result.warnings, result.returncode) == ([], [], 0)
    assert (result.bl, result.cp, result.geometry) == ({}, {}, None)     # nothing asked for


def test_identical_calls_give_identical_results(xfoil):
    """Each run gets a fresh directory, so nothing from the first can leak
    into the second. XFOIL appends to a polar file that already exists."""
    xfoil()
    first, second = run_polar("2412", 1e6, 0, 4, 2), run_polar("2412", 1e6, 0, 4, 2)
    assert first.points == second.points


def test_an_alpha_missing_from_the_polar_is_a_failure_not_an_exception(xfoil):
    xfoil(skip=(2.0,))
    result = run_polar("2412", 1e6, 0, 4, 2)
    assert result.status == "partial"
    assert (result.failed_alphas, result.failure_runs) == ([2.0], [[2.0]])
    assert [p.alpha for p in result.points] == [0.0, 4.0]
    assert "1 point(s) reported VISCAL convergence failure" in result.warnings


def test_consecutive_failures_are_grouped_into_runs(xfoil):
    xfoil(skip=(1.0, 2.0, 4.0))
    result = run_polar("2412", 1e6, 0, 5, 1)
    assert result.failure_runs == [[1.0, 2.0], [4.0]]


def test_no_converged_point_is_an_empty_result_with_a_warning(xfoil):
    xfoil(skip=(0.0, 2.0, 4.0))
    result = run_polar("2412", 1e6, 0, 4, 2)
    assert result.status == "empty"
    assert result.failed_alphas == [0.0, 2.0, 4.0]
    assert "no converged points; check warnings above" in result.warnings
    assert result.summary()["best_ld"] is None


def test_a_polar_row_holding_nan_is_not_a_converged_point(xfoil):
    xfoil(nan=(2.0,))
    result = run_polar("2412", 1e6, 0, 4, 2)
    assert result.status == "partial"
    assert result.failed_alphas == [2.0]
    assert any("non-finite" in w and "2.0" in w for w in result.warnings)


def test_field_files_are_attached_only_for_converged_alphas(xfoil):
    """XFOIL writes a DUMP for every alpha, converged or not. Only the polar
    says which ones to believe."""
    xfoil(skip=(2.0,))
    result = run_polar("2412", 1e6, 0, 4, 2, outputs=("forces", "bl", "cp"))
    assert sorted(result.bl) == [0.0, 4.0]
    assert sorted(result.cp) == [0.0, 4.0]
    assert result.bl[4.0].n_surface == 160
    assert result.summary()["field_outputs"] == {"bl": [0.0, 4.0], "cp": [0.0, 4.0]}


def test_the_outline_is_returned_when_geometry_is_requested(xfoil):
    xfoil()
    result = run_polar("2412", 1e6, 0, 0, 1, outputs=("forces", "geometry"))
    assert len(result.geometry) == 160
    assert result.geometry[0] == pytest.approx((1.0, 0.00126))


def test_a_rejected_command_is_reported_as_a_possible_desync(xfoil):
    xfoil(mode="desync")
    result = run_polar("2412", 1e6, 0, 4, 2)
    assert result.status == "empty"
    assert "XFOIL rejected a command; the input sequence may have desynchronized" in result.warnings


def test_a_hung_xfoil_is_killed_and_raised(xfoil):
    xfoil(mode="hang")
    with pytest.raises(XfoilError, match="exceeded 0.5s"):
        run_polar("2412", 1e6, 0, 4, 2, timeout=0.5)


def test_a_missing_binary_is_raised_with_the_fix(monkeypatch):
    monkeypatch.setattr(wrapper, "XFOIL_BIN", "/nonexistent/xfoil")
    with pytest.raises(XfoilError, match="not found; set XFOIL_BIN"):
        run_polar("2412", 1e6, 0, 4, 2)


def test_a_crash_mid_sweep_is_reported_as_a_crash(xfoil):
    """gfortran writes its runtime errors to stderr and exits non-zero. The
    alphas after the crash were never attempted; the caller must be told
    that, not that they failed to converge."""
    xfoil(mode="crash")
    result = run_polar("2412", 1e6, 0, 4, 1)
    assert result.returncode == 2
    assert [p.alpha for p in result.points] == [0.0, 1.0]            # what was written survives
    assert any("exit" in w.lower() or "crash" in w.lower() for w in result.warnings)


# --- flaps --------------------------------------------------------------------

def test_a_flap_that_xfoil_confirms_is_accepted(xfoil):
    xfoil()
    result = run_polar("2412", 1e6, 0, 0, 1, flap=(0.7, 0.0, 10.0))
    assert result.status == "ok"


def test_a_flap_that_never_left_the_buffer_is_an_error(xfoil):
    xfoil(mode="flap_stays_in_buffer")
    with pytest.raises(XfoilError, match="never reached the analysis"):
        run_polar("2412", 1e6, 0, 0, 1, flap=(0.7, 0.0, 10.0))


def test_a_hinge_outside_the_section_is_invalid_geometry(xfoil):
    """The stand-in reports the surface between -0.0216 and 0.0516 at the hinge."""
    xfoil()
    with pytest.raises(InvalidGeometry, match="outside the airfoil"):
        run_polar("2412", 1e6, 0, 0, 1, flap=(0.7, 0.3, 10.0))


def test_a_clean_exit_with_noise_on_stderr_is_not_a_crash():
    """Real XFOIL exits 0 and still writes a floating-point note to stderr."""
    assert wrapper._crash_warning(0, "Note: The following floating-point exceptions are signalling: IEEE_DIVIDE_BY_ZERO") is None
    assert "exit status 2" in wrapper._crash_warning(2, "Fortran runtime error: End of file")
    assert wrapper._crash_warning(0, "Fortran runtime error: End of file") is not None     # whatever the exit code


def test_a_crashed_sweep_reaches_the_summary_the_model_reads(xfoil):
    xfoil(mode="crash")
    summary = run_polar("2412", 1e6, 0, 4, 1).summary()
    assert summary["status"] == "partial"
    assert any("never attempted" in w for w in summary["warnings"])
