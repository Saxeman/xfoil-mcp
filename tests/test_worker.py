"""Worker tests: a Case goes in, a CaseResult comes out. Needs XFOIL.

These call worker.run_case directly, the same function the container's
entry point calls with the case it reads from stdin.
"""

import shutil

import pytest

from xfoil_mcp.schema import Case, CaseResult, Conditions, Flap, Geometry
from xfoil_mcp.worker import run_case
from xfoil_mcp.wrapper import XFOIL_BIN

pytestmark = pytest.mark.skipif(
    shutil.which(XFOIL_BIN) is None,
    reason="needs xfoil; run inside the worker container",
)


def case(outputs=("forces",), flap=None) -> Case:
    return Case(
        geometry=Geometry(naca="2412", flap=flap),
        conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=0, alpha_step=1),
        outputs=outputs,
    )


def test_flap_reaches_xfoil_through_the_worker():
    """The worker is the next place flap=flap could be forgotten."""
    clean = run_case(case())
    flapped = run_case(case(flap=Flap(x_hinge=0.7, deflection=10)))
    assert flapped.data["points"][0]["cl"] > clean.data["points"][0]["cl"] + 0.5


def test_field_outputs_reach_the_result():
    result = run_case(case(outputs=("forces", "bl", "cp")))
    assert result.status == "ok"
    assert result.summary["field_outputs"] == {"bl": [0.0], "cp": [0.0]}
    assert set(result.data["bl"]) == {"0.0"}
    assert set(result.data["cp"]) == {"0.0"}


def test_result_survives_the_json_round_trip():
    """What the host receives must be exactly what the worker produced."""
    result = run_case(case(outputs=("bl", "cp")))
    back = CaseResult.model_validate_json(result.model_dump_json())
    assert back == result
    assert back.data["bl"]["0.0"]["n_surface"] == 160


def test_hinge_outside_the_airfoil_is_an_input_failure():
    """The schema allows y_hinge up to 0.5; the wrapper's guard catches it."""
    result = run_case(case(flap=Flap(x_hinge=0.7, y_hinge=0.3, deflection=10)))
    assert result.status == "error"
    assert "outside the airfoil" in result.summary["error"]
    assert result.failure_kind == "input"
    assert result.retry_could_help is False

def test_geometry_reaches_the_result():
    result = run_case(case(outputs=("forces", "geometry"), flap=Flap(x_hinge=0.7, deflection=10)))
    assert len(result.data["geometry"]) == 160


@pytest.mark.parametrize("naca", ["6412", "4406", "23012"])
def test_a_flap_with_no_hinge_height_works_on_cambered_sections(naca):
    """On these sections the chord line is outside the airfoil aft of
    mid-chord, so a hinge at y = 0 was refused as invalid geometry. Left out,
    the hinge goes on the camber line, which XFOIL accepts."""
    flapped = Case(geometry=Geometry(naca=naca, flap=Flap(x_hinge=0.7, deflection=10)),
                   conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=2, alpha_step=2))
    clean = Case(geometry=Geometry(naca=naca), conditions=flapped.conditions)
    result, baseline = run_case(flapped), run_case(clean)
    assert result.status != "error", result.summary
    assert result.data["points"][0]["cl"] > baseline.data["points"][0]["cl"] + 0.3     # the flap really applied


def test_a_hinge_forced_onto_the_chord_line_of_a_cambered_section_is_still_refused():
    """Giving y_hinge explicitly is honoured, and the wrapper's guard still catches it."""
    result = run_case(Case(geometry=Geometry(naca="6412", flap=Flap(x_hinge=0.7, y_hinge=0.0, deflection=10)),
                           conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=0, alpha_step=1)))
    assert (result.status, result.failure_kind) == ("error", "input")
