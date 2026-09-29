"""Coupling tests: from boundary-layer data to heat-transfer inputs.

Host-side, no XFOIL. Surface arrays come from the fixture captured in
step 1, parsed with the step 2 parser.
"""

from pathlib import Path

import pytest
import math


from xfoil_mcp.coupling import Air, heat_transfer, smith_spalding, stagnation_point, velocity_for
from xfoil_mcp.wrapper import _parse_bl_file

BL_A4 = Path(__file__).parent / "fixtures" / "bl_2412_re1e6_a4.txt"


@pytest.fixture
def surface():
    layer = _parse_bl_file(BL_A4, alpha=4.0)
    n = layer.n_surface
    return layer.s[:n], layer.x[:n], layer.y[:n], layer.ue[:n]


def test_stagnation_is_where_ue_crosses_zero(surface):
    st = stagnation_point(*surface)
    assert st.s == pytest.approx(1.03819, abs=1e-4)
    assert st.x == pytest.approx(0.00394, abs=1e-4)


def test_stagnation_lies_between_its_bracketing_nodes(surface):
    s, _, _, ue = surface
    st = stagnation_point(*surface)
    assert ue[st.index] > 0 >= ue[st.index + 1]
    assert s[st.index] <= st.s <= s[st.index + 1]


def test_stagnation_is_on_the_lower_surface_at_positive_alpha(surface):
    st = stagnation_point(*surface)
    assert st.surface == "lower"
    assert st.y < 0


def test_stagnation_is_not_the_geometric_leading_edge(surface):
    s, x, _, _ = surface
    le = min(range(len(x)), key=x.__getitem__)
    st = stagnation_point(*surface)
    assert st.s - s[le] == pytest.approx(0.011, abs=1e-3)


def test_symmetric_flow_stagnates_at_the_leading_edge():
    s = [0.0, 1.0, 2.0, 3.0, 4.0]
    x = [1.0, 0.5, 0.0, 0.5, 1.0]
    y = [0.1, 0.05, 0.0, -0.05, -0.1]
    ue = [1.0, 0.5, 0.0, -0.5, -1.0]
    st = stagnation_point(s, x, y, ue)
    assert st.s == pytest.approx(2.0)
    assert st.surface == "leading edge"


def test_no_crossing_is_an_error():
    with pytest.raises(ValueError):
        stagnation_point([0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [0.1, 0.0, -0.1], [1.0, 0.5, 0.2])


def test_two_crossings_is_an_error():
    with pytest.raises(ValueError):
        stagnation_point([0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 1.0],
                         [0.1, 0.0, 0.0, -0.1], [1.0, -1.0, 1.0, -1.0])

# --- heat transfer coefficient ---------------------------------------------

AIR = Air.at(263.15)                 # -10 C, sea level
CHORD = 0.5                          # m
V = velocity_for(1e6, CHORD, AIR)    # the speed that makes Re = 1e6 at this chord


def test_velocity_is_whatever_makes_the_reynolds_number_match():
    assert V == pytest.approx(24.8, abs=0.1)


def test_laminar_method_matches_the_flat_plate_answer():
    """Constant speed: the textbook laminar result is Nu = 0.332 Re^0.5 Pr^(1/3)."""
    U = 20.0
    d = [0.001 * i for i in range(1, 201)]
    h = smith_spalding(d, [U] * len(d), AIR)
    nusselt = h[100] * d[100] / AIR.k
    reynolds = U * d[100] / AIR.nu
    assert nusselt / reynolds**0.5 == pytest.approx(0.332 * AIR.pr ** (1 / 3), rel=0.02)


def test_laminar_method_matches_the_cylinder_stagnation_answer():
    """Flow around a cylinder: Nu_D = 1.14 Re_D^0.5 Pr^0.4 at the front."""
    r = 0.01
    d = [r * 0.001 * i for i in range(1, 51)]
    h = smith_spalding(d, [2 * V * math.sin(q / r) for q in d], AIR)
    re_d = V * 2 * r / AIR.nu
    assert h[0] * 2 * r / AIR.k / re_d**0.5 == pytest.approx(1.14 * AIR.pr**0.4, rel=0.02)


@pytest.fixture
def ht():
    layer = _parse_bl_file(BL_A4, alpha=4.0)
    n = layer.n_surface
    s, x, y, ue, cf = layer.s[:n], layer.x[:n], layer.y[:n], layer.ue[:n], layer.cf[:n]
    st = stagnation_point(s, x, y, ue)
    result = heat_transfer(s, x, ue, cf, st, xtr_upper=0.398, xtr_lower=1.0,
                           chord_m=CHORD, velocity=V, air=AIR)
    return x, st, result


def test_h_is_positive_and_finite_everywhere(ht):
    _, _, result = ht
    assert all(math.isfinite(h) and h > 0 for h in result.h)


def test_the_nose_loses_heat_fastest(ht):
    x, st, result = ht
    mid = min(range(st.index + 1), key=lambda i: abs(x[i] - 0.2))
    assert result.h[st.index] > 4 * result.h[mid]


def test_heat_transfer_jumps_at_transition(ht):
    x, st, result = ht
    before = min(range(st.index + 1), key=lambda i: abs(x[i] - 0.37))
    after = min(range(st.index + 1), key=lambda i: abs(x[i] - 0.53))
    assert result.regime[before] == "laminar"
    assert result.regime[after] == "turbulent"
    assert result.h[after] > 3 * result.h[before]


def test_lower_surface_stays_laminar_at_4_degrees(ht):
    _, st, result = ht
    assert set(result.regime[st.index + 1:]) == {"laminar"}
