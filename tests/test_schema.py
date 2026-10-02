"""Schema tests. No XFOIL, no Docker; these run in milliseconds.

What they prove: invalid cases cannot be constructed, hashes are stable and
mean what they should, and result status/kind stay consistent.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError


from xfoil_mcp.schema import Case, CaseResult, Conditions, Flap, Geometry, ThermalResult


def make_case(**overrides) -> Case:
    cond = dict(reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=1)
    cond.update(overrides.pop("conditions", {}))
    return Case(
        geometry=Geometry(naca=overrides.pop("naca", "2412")),
        conditions=Conditions(**cond),
        **overrides,
    )


# --- geometry --------------------------------------------------------------

@pytest.mark.parametrize("naca", ["2412", "0012", "23012", " 4412 "])
def test_valid_naca_designations(naca):
    assert Geometry(naca=naca).naca == naca.strip()


@pytest.mark.parametrize("naca", ["241", "241222", "24a2", "2400", ""])
def test_invalid_naca_designations(naca):
    with pytest.raises(ValidationError):
        Geometry(naca=naca)


def test_flap_bounds():
    Flap(x_hinge=0.7, deflection=10)
    with pytest.raises(ValidationError):
        Flap(x_hinge=1.0, deflection=10)
    with pytest.raises(ValidationError):
        Flap(x_hinge=0.7, deflection=60)

def test_geometry_is_an_output():
    assert "geometry" in make_case(outputs=("forces", "geometry")).outputs


# --- conditions ------------------------------------------------------------

def test_reynolds_must_be_positive():
    with pytest.raises(ValidationError):
        make_case(conditions=dict(reynolds=0))


def test_mach_must_be_subsonic():
    with pytest.raises(ValidationError):
        make_case(conditions=dict(mach=1.0))


def test_n_crit_physical_range():
    with pytest.raises(ValidationError):
        make_case(conditions=dict(n_crit=0.5))
    with pytest.raises(ValidationError):
        make_case(conditions=dict(n_crit=20))


def test_sweep_must_run_forward():
    with pytest.raises(ValidationError):
        make_case(conditions=dict(alpha_start=10, alpha_end=0))


def test_sweep_point_limit():
    make_case(conditions=dict(alpha_start=0, alpha_end=19.9, alpha_step=0.1))
    with pytest.raises(ValidationError):
        make_case(conditions=dict(alpha_start=0, alpha_end=30, alpha_step=0.1))


def test_alphas_match_aseq_without_float_drift():
    c = make_case(conditions=dict(alpha_start=0, alpha_end=1, alpha_step=0.1))
    assert c.alphas() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert c.conditions.point_count() == 11


# --- case ------------------------------------------------------------------

def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Conditions(reynolds=1e6, alpha_start=0, alpha_end=5, alpha_step=1, reynold=2e6)


def test_outputs_must_be_non_empty_and_unique():
    with pytest.raises(ValidationError):
        make_case(outputs=())
    with pytest.raises(ValidationError):
        make_case(outputs=("forces", "forces"))
    with pytest.raises(ValidationError):
        make_case(outputs=("pressure",))


def test_outputs_are_canonicalized():
    assert make_case(outputs=("cp", "forces")).outputs == ("cp", "forces")
    assert make_case(outputs=("forces", "cp")).outputs == ("cp", "forces")


def test_cases_are_immutable():
    c = make_case()
    with pytest.raises(ValidationError):
        c.label = "changed"


# --- hashing ---------------------------------------------------------------

def test_hash_is_stable_across_constructions():
    assert make_case().content_hash() == make_case().content_hash()


def test_hash_excludes_label():
    assert make_case(label="a").content_hash() == make_case(label="b").content_hash()


def test_hash_changes_with_outputs():
    assert make_case(outputs=("forces",)).content_hash() != \
        make_case(outputs=("forces", "cp")).content_hash()


def test_hash_changes_with_conditions():
    assert make_case().content_hash() != make_case(conditions=dict(reynolds=2e6)).content_hash()


def test_hash_ignores_output_order():
    assert make_case(outputs=("cp", "forces")).content_hash() == \
        make_case(outputs=("forces", "cp")).content_hash()


def test_hash_treats_equal_floats_equally():
    a = make_case(conditions=dict(reynolds=1e6))
    b = make_case(conditions=dict(reynolds=1000000))
    assert a.content_hash() == b.content_hash()


# --- serialization ---------------------------------------------------------

def test_case_round_trips_through_json():
    c = make_case(label="round trip", outputs=("forces", "cp"))
    restored = Case.model_validate_json(c.model_dump_json())
    assert restored == c
    assert restored.content_hash() == c.content_hash()


# --- results ---------------------------------------------------------------

def test_ok_result_cannot_carry_failure_kind():
    with pytest.raises(ValidationError):
        CaseResult(case=make_case(), status="ok", failure_kind="numerical")


def test_partial_result_is_numerical():
    r = CaseResult(case=make_case(), status="partial", failure_kind="numerical")
    assert not r.retry_could_help
    with pytest.raises(ValidationError):
        CaseResult(case=make_case(), status="partial", failure_kind="input")


def test_error_result_needs_a_kind():
    with pytest.raises(ValidationError):
        CaseResult(case=make_case(), status="error")


def test_only_infrastructure_failures_are_retryable():
    kinds = {"input": False, "infrastructure": True, "numerical": False}
    for kind, expected in kinds.items():
        r = CaseResult(case=make_case(), status="error", failure_kind=kind)
        assert r.retry_could_help is expected


def test_result_round_trips_through_json():
    r = CaseResult(
        case=make_case(), status="partial", failure_kind="numerical",
        summary={"converged_points": 17}, data={"points": []},
        provenance={"content_hash": make_case().content_hash()},
    )
    assert CaseResult.model_validate_json(r.model_dump_json()) == r

def test_alpha_step_has_a_floor():
    """Finer than 0.01 deg, requested alphas crowd within the polar file's
    printed precision and can no longer be told apart."""
    make_case(conditions=dict(alpha_start=0, alpha_end=1, alpha_step=0.01))
    with pytest.raises(ValidationError):
        make_case(conditions=dict(alpha_start=0, alpha_end=1, alpha_step=0.005))


def test_sweep_never_passes_alpha_end():
    c = make_case(conditions=dict(alpha_start=0, alpha_end=11, alpha_step=4))
    assert c.alphas() == [0.0, 4.0, 8.0]
    c = make_case(conditions=dict(alpha_start=29, alpha_end=30, alpha_step=0.6))
    assert c.alphas() == [29.0, 29.6]


def test_sweep_keeps_endpoint_despite_float_division():
    # 0.3 / 0.1 is 2.9999999999999996; a bare floor would drop 0.3.
    c = make_case(conditions=dict(alpha_start=0, alpha_end=0.3, alpha_step=0.1))
    assert c.alphas() == [0.0, 0.1, 0.2, 0.3]


@pytest.mark.parametrize("start,end,step", [
    (0, 11, 4), (29, 30, 0.6), (0, 0.3, 0.1), (-5, 12, 1), (0, 10, 3), (-2.5, 2.5, 0.25),
])
def test_schema_and_wrapper_sweeps_agree(start, end, step):
    from xfoil_mcp.wrapper import _expected_alphas
    c = make_case(conditions=dict(alpha_start=start, alpha_end=end, alpha_step=step))
    assert c.alphas() == _expected_alphas(start, end, step)

# --- thermal ----------------------------------------------------------------

def thermal(**overrides) -> dict:
    base = dict(
        chord_m=0.5, air_temperature_k=263.15, heater_width=0.1,
        heater_power_w_per_m=500.0, skin_thickness_m=0.001, skin_conductivity_w_mk=200.0,
    )
    base.update(overrides)
    return base


def test_thermal_block_is_optional():
    assert make_case().thermal is None


def test_valid_thermal_block():
    case = make_case(outputs=("forces", "bl"), thermal=thermal())
    assert case.thermal.chord_m == 0.5
    assert case.thermal.pressure_pa == 101325.0


def test_thermal_requires_boundary_layer_output():
    with pytest.raises(ValidationError, match="bl"):
        make_case(outputs=("forces",), thermal=thermal())


@pytest.mark.parametrize("field, bad", [
    ("chord_m", 0.0),
    ("air_temperature_k", 150.0),
    ("pressure_pa", 5000.0),
    ("heater_width", 0.8),
    ("heater_power_w_per_m", -1.0),
    ("skin_thickness_m", 0.1),
    ("skin_conductivity_w_mk", 0.0),
])
def test_thermal_bounds(field, bad):
    with pytest.raises(ValidationError):
        make_case(outputs=("bl",), thermal=thermal(**{field: bad}))


def test_unknown_thermal_field_is_rejected():
    """A campaign that writes chord= instead of chord_m= must fail, not default."""
    with pytest.raises(ValidationError):
        make_case(outputs=("bl",), thermal=thermal(chord=0.5))


def test_hash_changes_with_thermal_inputs():
    base = make_case(outputs=("bl",), thermal=thermal())
    assert base.content_hash() != make_case(outputs=("bl",)).content_hash()
    assert base.content_hash() != make_case(
        outputs=("bl",), thermal=thermal(heater_power_w_per_m=600.0)).content_hash()


# --- thermal results ---------------------------------------------------------

THERMAL_OK = dict(status="ok", failure_kind=None, errors=[], coverage=1.0,
                  worst_case={"min_heated_temperature_c": 24.8, "at_alpha": 4.0},
                  points=[{"alpha": 4.0}], excluded=[])


def test_valid_thermal_result():
    result = ThermalResult(**THERMAL_OK, heater_power_w_per_m=500.0, content_hash="abc")
    assert (result.status, result.coverage, result.velocity_m_s) == ("ok", 1.0, None)


def test_thermal_failure_is_an_error_with_a_kind_and_nothing_else():
    result = ThermalResult.failure("infrastructure", "docker not found on host")
    assert (result.status, result.failure_kind, result.errors) == ("error", "infrastructure", ["docker not found on host"])
    assert (result.coverage, result.worst_case, result.points, result.excluded) == (0.0, None, [], [])
    assert result.content_hash is None


@pytest.mark.parametrize("change", [
    {"failure_kind": "numerical"},                          # ok with a kind
    {"status": "partial"},                                  # partial without one
    {"status": "empty", "failure_kind": "input"},           # empty is numerical
    {"status": "error"},                                    # error without a kind
    {"status": "banana"},
    {"failure_kind": "banana"},
])
def test_thermal_status_and_kind_must_agree(change):
    """The same rule CaseResult enforces, so the envelope can trust both."""
    with pytest.raises(ValidationError):
        ThermalResult(**{**THERMAL_OK, **change})


@pytest.mark.parametrize("coverage", [-0.1, 1.5, "most"])
def test_thermal_coverage_is_a_fraction(coverage):
    with pytest.raises(ValidationError):
        ThermalResult(**{**THERMAL_OK, "coverage": coverage})


@pytest.mark.parametrize("missing", ["status", "failure_kind", "errors", "coverage",
                                     "worst_case", "points", "excluded"])
def test_thermal_result_needs_every_core_field(missing):
    """A container that leaves one out did not produce a thermal result."""
    with pytest.raises(ValidationError):
        ThermalResult(**{k: v for k, v in THERMAL_OK.items() if k != missing})


@pytest.mark.parametrize("bad", [{"errors": "boom"}, {"points": "none"}, {"worst_case": "n/a"}, {"surprise": 1}])
def test_thermal_result_rejects_wrong_types_and_unknown_fields(bad):
    with pytest.raises(ValidationError):
        ThermalResult(**{**THERMAL_OK, **bad})


def test_thermal_result_round_trips_through_json():
    result = ThermalResult(**THERMAL_OK, heater_power_w_per_m=500.0, velocity_m_s=24.84, content_hash="abc")
    assert ThermalResult.model_validate_json(result.model_dump_json()) == result


# --- gaps in the tests above --------------------------------------------------

def conditions(**changes):
    return Conditions(**{**dict(reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=1), **changes})


@pytest.mark.parametrize("y_hinge, valid", [(-0.5, True), (0.5, True), (-0.51, False), (0.51, False)])
def test_hinge_height_bounds(y_hinge, valid):
    if valid:
        assert Flap(x_hinge=0.7, y_hinge=y_hinge, deflection=5).y_hinge == y_hinge
    else:
        with pytest.raises(ValidationError):
            Flap(x_hinge=0.7, y_hinge=y_hinge, deflection=5)


def test_a_flap_that_names_no_hinge_height_is_hinged_on_the_camber_line():
    """Left out means "on the camber line", which Geometry works out because
    it depends on the airfoil."""
    flap = Flap(x_hinge=0.7, deflection=5)
    assert flap.y_hinge is None
    assert Geometry(naca="0012", flap=flap).hinge_y() == 0.0                    # symmetric: the chord line
    assert Geometry(naca="2412", flap=flap).hinge_y() == pytest.approx(0.015, abs=1e-5)
    assert Geometry(naca="23012", flap=flap).hinge_y() == pytest.approx(0.00663, abs=1e-5)


def test_a_hinge_height_that_is_given_is_used_as_given():
    assert Geometry(naca="6412", flap=Flap(x_hinge=0.7, y_hinge=0.0, deflection=5)).hinge_y() == 0.0
    assert Geometry(naca="2412").hinge_y() is None                              # no flap, no hinge


def test_label_length_limit():
    geometry, cond = Geometry(naca="2412"), conditions()
    assert len(Case(geometry=geometry, conditions=cond, label="x" * 80).label) == 80
    with pytest.raises(ValidationError):
        Case(geometry=geometry, conditions=cond, label="x" * 81)


def test_empty_result_is_numerical():
    case = Case(geometry=Geometry(naca="2412"), conditions=conditions())
    assert CaseResult(case=case, status="empty", failure_kind="numerical").retry_could_help is False
    for kind in (None, "input", "infrastructure"):
        with pytest.raises(ValidationError):
            CaseResult(case=case, status="empty", failure_kind=kind)


@pytest.mark.parametrize("field", ["reynolds", "alpha_step", "n_crit", "mach", "alpha_start"])
def test_nan_is_never_a_valid_condition(field):
    with pytest.raises(ValidationError):
        conditions(**{field: float("nan")})


@pytest.mark.parametrize("field", ["reynolds", "alpha_step"])
def test_infinity_is_never_a_valid_condition(field):
    with pytest.raises(ValidationError):
        conditions(**{field: float("inf")})


def test_a_step_larger_than_the_sweep_is_refused():
    with pytest.raises(ValidationError):
        conditions(alpha_start=0, alpha_end=10, alpha_step=100)


@pytest.mark.parametrize("naca", ["21012", "23012", "25018"])
def test_five_digit_designations_xfoil_can_build_are_accepted(naca):
    """XFOIL's NACA5 builds the 210, 220, 230, 240 and 250 mean lines."""
    assert Geometry(naca=naca).naca == naca


@pytest.mark.parametrize("naca", [
    "12345", "99999",       # not a mean line XFOIL implements
    "63415",                # a 6-series name without its dash
    "23000",                # zero thickness, which the 4-digit check would have caught
    "01234", "00012",       # XFOIL reads these as the 4-digit 1234 and 0012
])
def test_five_digit_designations_xfoil_cannot_build_are_refused(naca):
    with pytest.raises(ValidationError):
        Geometry(naca=naca)


@pytest.mark.parametrize("naca", ["\uff12\uff14\uff11\uff12", "\u0662\u0664\u0661\u0662"])     # full-width, Arabic-Indic
def test_only_ascii_digits_are_a_designation(naca):
    with pytest.raises(ValidationError):
        Geometry(naca=naca)


def surfaces_at(naca: str, x: float) -> tuple[float, float]:
    """(lower, upper) surface of a NACA 4-digit section at chord fraction x,
    from the textbook formula, written out here independently of the schema.
    Thickness is taken vertically, which is within a fraction of a percent of chord."""
    m, p, t = int(naca[0]) / 100, int(naca[1]) / 10, int(naca[2:]) / 100
    camber = (m / p**2 * (2 * p * x - x * x) if x < p
              else m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x * x)) if m else 0.0
    half = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    return camber - half, camber + half


@pytest.mark.parametrize("naca", ["0012", "2412", "4412", "6412", "4406", "6409"])
@pytest.mark.parametrize("x_hinge", [0.5, 0.7, 0.9])
def test_the_default_hinge_lies_inside_the_section(naca, x_hinge):
    """evaluate_design and run_polar cannot set the hinge height, so the
    default has to be inside every section they can be asked for."""
    lower, upper = surfaces_at(naca, x_hinge)
    hinge = Geometry(naca=naca, flap=Flap(x_hinge=x_hinge, deflection=10)).hinge_y()
    assert lower < hinge < upper


@pytest.mark.parametrize("naca", ["6412", "4406", "6409"])
def test_the_chord_line_is_outside_strongly_cambered_sections(naca):
    """Why the default is not 0: on these sections the lower surface is above the chord line."""
    lower, _ = surfaces_at(naca, 0.7)
    assert lower > 0.0


def test_negative_zero_and_zero_sweeps_are_the_same_case():
    a = Case(geometry=Geometry(naca="2412"), conditions=conditions(alpha_start=0.0))
    b = Case(geometry=Geometry(naca="2412"), conditions=conditions(alpha_start=-0.0))
    assert a.alphas() == b.alphas()                 # identical solver input
    assert a.content_hash() == b.content_hash()


def test_a_heater_is_refused_on_a_case_xfoil_would_solve_at_speed():
    """The thermal model assumes slow air. Refusing at construction means a
    dry run rejects the case before anything is approved or run."""
    heater = dict(chord_m=0.5, air_temperature_k=263.15, heater_width=0.1, heater_power_w_per_m=500.0,
                  skin_thickness_m=0.001, skin_conductivity_w_mk=200.0)
    fields = dict(geometry=Geometry(naca="2412"), outputs=("forces", "bl"), thermal=heater)
    assert Case(conditions=conditions(mach=0.3), **fields).conditions.mach == 0.3
    with pytest.raises(ValidationError, match="thermal requires mach"):
        Case(conditions=conditions(mach=0.31), **fields)
    assert Case(geometry=Geometry(naca="2412"), conditions=conditions(mach=0.6)).thermal is None   # aero only is fine
