"""Command-script tests. Pure string checks; no XFOIL needed.

The script is the only interface to XFOIL, and every line in it answers a
prompt. These tests pin its shape so a change to one branch cannot quietly
shift the others.
"""

import pytest

from xfoil_mcp.wrapper import _build_commands, _expected_alphas, _field_filename

def script(outputs=("forces",), start=0, end=4, step=2, flap=None) -> list[str]:
    return _build_commands(
        airfoil="2412", reynolds=1e6, mach=0.0, n_crit=9.0, max_iter=100,
        alpha_start=start, alpha_end=end, alpha_step=step,
        polar_path="polar.txt", outputs=outputs, flap=flap,
    ).split("\n")


def test_forces_only_solves_each_alpha_without_writing_files():
    lines = script()
    assert not any(l.startswith(("ASEQ", "DUMP", "CPWR")) for l in lines)
    assert [l for l in lines if l.startswith("ALFA")] == ["ALFA 0.0", "ALFA 2.0", "ALFA 4.0"]


def test_bl_output_solves_each_alpha_individually():
    lines = script(outputs=("forces", "bl"))
    assert not any(l.startswith("ASEQ") for l in lines)
    assert [l for l in lines if l.startswith("ALFA")] == ["ALFA 0.0", "ALFA 2.0", "ALFA 4.0"]


def test_each_alfa_is_followed_by_its_dump():
    lines = script(outputs=("bl",))
    for i, alpha in enumerate(_expected_alphas(0, 4, 2)):
        k = lines.index(f"ALFA {alpha}")
        assert lines[k + 1] == f"DUMP {_field_filename('bl', i)}"


def test_dump_files_are_named_by_position_not_alpha():
    lines = script(outputs=("bl",), start=-2, end=2, step=1)
    dumps = [l.split()[1] for l in lines if l.startswith("DUMP")]
    assert dumps == ["bl_000.txt", "bl_001.txt", "bl_002.txt", "bl_003.txt", "bl_004.txt"]

@pytest.mark.parametrize("outputs", [
    ("forces",), ("forces", "bl"), ("cp",), ("forces", "bl", "cp"),
])
def test_script_frame_is_unchanged(outputs):
    """The lines around the sweep must be identical in both modes: graphics
    off first, polar accumulation opened with its two prompt answers, and
    the exact unwind at the end. One missing blank line desynchronizes
    everything after it."""
    lines = script(outputs=outputs)
    assert lines[:3] == ["PLOP", "G", ""]
    k = lines.index("PACC")
    assert lines[k + 1] == "polar.txt" and lines[k + 2] == ""
    assert lines[-4:] == ["PACC", "", "QUIT", ""]

def test_cp_output_writes_cpwr_after_each_alfa():
    lines = script(outputs=("cp",))
    assert not any(l.startswith(("ASEQ", "DUMP")) for l in lines)
    for i, alpha in enumerate(_expected_alphas(0, 4, 2)):
        k = lines.index(f"ALFA {alpha}")
        assert lines[k + 1] == f"CPWR {_field_filename('cp', i)}"


def test_bl_and_cp_are_both_written_for_each_alfa():
    lines = script(outputs=("forces", "bl", "cp"))
    for i, alpha in enumerate(_expected_alphas(0, 4, 2)):
        k = lines.index(f"ALFA {alpha}")
        assert lines[k + 1] == f"DUMP {_field_filename('bl', i)}"
        assert lines[k + 2] == f"CPWR {_field_filename('cp', i)}"

def test_no_flap_means_no_geometry_menu():
    assert "GDES" not in script()


def test_flap_is_applied_and_repaneled_before_analysis():
    lines = script(flap=(0.7, 0.0, 10.0))
    k = lines.index("GDES")
    assert lines[k:k + 5] == ["GDES", "FLAP 0.7 0.0 10.0", "EXEC", "", "PANE"]
    assert lines.index("NACA 2412") < k < lines.index("OPER")

def test_geometry_output_saves_the_outline_before_solving():
    lines = script(outputs=("forces", "geometry"), flap=(0.7, 0.0, 10.0))
    k = lines.index("PSAV geometry.txt")
    assert lines.index("PANE") < k < lines.index("OPER")    # after the flap is final


def test_no_geometry_output_means_no_psav():
    assert not any(l.startswith("PSAV") for l in script())


def full_script(**overrides) -> list[str]:
    args = dict(airfoil="2412", reynolds=1e6, mach=0.0, n_crit=9.0, max_iter=100,
                alpha_start=0, alpha_end=4, alpha_step=2, polar_path="polar.txt")
    return _build_commands(**{**args, **overrides}).split("\n")


def test_solver_settings_are_sent_in_the_order_xfoil_prompts_for_them():
    """Between OPER and PACC every line is a setting or the answer to a
    prompt. The frame test above does not look at these."""
    lines = full_script(reynolds=5e5, mach=0.2, n_crit=4.0, max_iter=150)
    k = lines.index("OPER")
    assert lines[k:k + 10] == ["OPER", "ITER 150", "ITER", "", "VPAR", "N 4.0", "",
                               "VISC 500000.0", "MACH 0.2", "PACC"]


def test_the_airfoil_is_loaded_before_anything_else_is_asked_of_it():
    lines = full_script(airfoil="0012")
    assert lines[3] == "NACA 0012"
    assert lines.index("NACA 0012") < lines.index("OPER")


def test_a_mach_number_reaches_the_script():
    assert "MACH 0.3" in full_script(mach=0.3)
    assert "MACH 0.0" in full_script()


def test_every_requested_alpha_is_solved_once_and_in_order():
    lines = full_script(alpha_start=-2, alpha_end=2, alpha_step=1)
    assert [l for l in lines if l.startswith("ALFA")] == ["ALFA -2.0", "ALFA -1.0", "ALFA 0.0", "ALFA 1.0", "ALFA 2.0"]


def test_the_iteration_limit_is_set_and_then_asked_back():
    """`ITER n` sets the limit silently. A bare ITER makes XFOIL print the
    current limit and prompt for a new one; the blank line keeps it."""
    lines = full_script(max_iter=37)
    k = lines.index("ITER 37")
    assert lines[k + 1:k + 3] == ["ITER", ""]
