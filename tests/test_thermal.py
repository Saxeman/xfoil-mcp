"""Skin temperature tests. Host-side; needs the thermal extra (scikit-fem).

The fixture pipeline is steps 1-10 end to end: DUMP file -> boundary
layer -> stagnation point -> h(s) -> skin temperature.
"""

import math
from pathlib import Path

import pytest

pytest.importorskip("skfem")     # the thermal extra

import numpy as np

from xfoil_mcp.coupling import Air, heat_transfer, stagnation_point, velocity_for
from xfoil_mcp.thermal import build_mesh_nodes, skin_temperature
from xfoil_mcp.wrapper import _parse_bl_file

BL_A4 = Path(__file__).parent / "fixtures" / "bl_2412_re1e6_a4.txt"
T_AIR = 263.15                        # -10 C
AIR = Air.at(T_AIR)
CHORD = 0.5
V = velocity_for(1e6, CHORD, AIR)
ALUMINIUM = 200.0 * 0.001             # conductivity x thickness, 1 mm skin
COMPOSITE = 1.0 * 0.001


@pytest.fixture(scope="module")
def surface():
    layer = _parse_bl_file(BL_A4, alpha=4.0)
    n = layer.n_surface
    s, x, y, ue, cf = layer.s[:n], layer.x[:n], layer.y[:n], layer.ue[:n], layer.cf[:n]
    st = stagnation_point(s, x, y, ue)
    ht = heat_transfer(s, x, ue, cf, st, xtr_upper=0.398, xtr_lower=1.0,
                       chord_m=CHORD, velocity=V, air=AIR)
    return s, ht.h, st.s


def skin(surface, power=500.0, kt=ALUMINIUM, width=0.1, refine=4):
    s, h, s_stag = surface
    return skin_temperature(s, h, s_stag, CHORD, width, power, kt, T_AIR, refine)


def test_constant_h_matches_the_exact_strip():
    """Step 12's answer key, through the real function."""
    kt, h, q, a = ALUMINIUM, 100.0, 10_000.0, 0.025
    lam = math.sqrt(kt / h)
    s = np.linspace(0.0, 2.0, 401)
    result = skin_temperature(s, [h] * len(s), 1.0, 0.5, 0.1, q * 2 * a, kt, T_AIR)
    x = np.abs(np.array(result.z_m))
    rise = np.array(result.temperature_k) - T_AIR
    exact = np.where(x <= a,
                     (q / h) * (1 - np.exp(-a / lam) * np.cosh(x / lam)),
                     (q / h) * np.sinh(a / lam) * np.exp(-x / lam))
    assert np.max(np.abs(rise - exact)) < 0.05


def test_heat_out_equals_heat_in(surface):
    assert skin(surface).balance_error < 0.01


def test_a_point_a_hair_from_a_node_does_not_crack_the_mesh():
    """Step 12's bug, as a regression test."""
    nodes = build_mesh_nodes(np.linspace(-1, 1, 11), [0.0, 0.2 + 1e-15], refine=1)
    gaps = np.diff(nodes)
    assert gaps.min() > 0.25 * np.median(gaps)
    assert 0.2 + 1e-15 in nodes


def test_a_narrow_heater_does_not_move_the_stagnation_point():
    nodes = build_mesh_nodes(np.linspace(-1, 1, 21), [0.0, -1e-3, 1e-3], refine=1)
    assert 0.0 in nodes and -1e-3 in nodes and 1e-3 in nodes


def test_hottest_inside_the_heater_and_nearly_air_temperature_at_the_tails(surface):
    result = skin(surface)
    T, z = np.array(result.temperature_k), np.array(result.z_m)
    assert abs(z[np.argmax(T)]) <= 0.025
    peak_rise = T.max() - T_AIR
    assert T[0] - T_AIR < 0.05 * peak_rise
    assert T[-1] - T_AIR < 0.05 * peak_rise


def test_more_power_is_warmer(surface):
    assert (skin(surface, power=800).min_heated_temperature_k
            > skin(surface, power=500).min_heated_temperature_k)


def test_aluminium_spreads_heat_further_than_composite(surface):
    al, co = skin(surface, kt=ALUMINIUM), skin(surface, kt=COMPOSITE)
    assert al.ice_free_upper_m + al.ice_free_lower_m > co.ice_free_upper_m + co.ice_free_lower_m


def test_more_mesh_points_barely_change_the_answer(surface):
    coarse, fine = skin(surface, refine=4), skin(surface, refine=8)
    assert abs(coarse.min_heated_temperature_k - fine.min_heated_temperature_k) < 0.1


# --- checks the energy balance cannot make ----------------------------------

def test_energy_balance_holds_at_every_resolution_so_it_cannot_judge_the_mesh(surface):
    """The Galerkin form conserves the heater power exactly. This pins that
    fact: a balance within 1% says the solve ran, not that it is resolved."""
    for refine in (1, 4, 16):
        assert skin(surface, kt=COMPOSITE, refine=refine).balance_error < 1e-9


@pytest.mark.parametrize("kt", [ALUMINIUM, COMPOSITE])
def test_a_heated_skin_is_never_colder_than_the_air(surface, kt):
    """The maximum principle: heat only enters, so no point can be below air temperature."""
    assert min(skin(surface, kt=kt).temperature_k) >= T_AIR - 1e-6


def test_a_very_thin_skin_is_never_colder_than_the_air_either(surface):
    """0.1 mm of a k = 0.05 W/mK skin is inside the schema's bounds."""
    assert min(skin(surface, kt=0.05 * 0.0001).temperature_k) >= T_AIR - 1e-6


@pytest.mark.parametrize("kt", [ALUMINIUM, COMPOSITE])
def test_a_finer_mesh_barely_moves_the_temperature_or_the_ice_free_extent(surface, kt):
    """The convergence check above covers one material and one number. A
    low-conductivity skin has the sharper gradients, and the ice-free extent
    is what the model is asked about."""
    coarse, fine = skin(surface, kt=kt, refine=4), skin(surface, kt=kt, refine=16)
    assert abs(coarse.min_heated_temperature_k - fine.min_heated_temperature_k) < 0.05
    assert abs(coarse.ice_free_upper_m - fine.ice_free_upper_m) < 1e-3          # within a millimetre
    assert abs(coarse.ice_free_lower_m - fine.ice_free_lower_m) < 1e-3


def test_a_skin_that_never_thaws_reports_no_ice_free_extent(surface):
    result = skin(surface, power=1.0)
    assert result.min_heated_temperature_k < 273.15
    assert (result.ice_free_upper_m, result.ice_free_lower_m) == (None, None)
