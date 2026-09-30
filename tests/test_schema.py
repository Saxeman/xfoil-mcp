"""Schema tests. No XFOIL, no Docker; these run in milliseconds.

What they prove: invalid cases cannot be constructed, hashes are stable and
mean what they should, and result status/kind stay consistent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xfoil_mcp.schema import Case, CaseResult, Conditions, Flap, Geometry


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
