"""Coupling tests: from boundary-layer data to heat-transfer inputs.

Host-side, no XFOIL. Surface arrays come from the fixture captured in
step 1, parsed with the step 2 parser.
"""

from pathlib import Path

import pytest

from xfoil_mcp.coupling import stagnation_point
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
