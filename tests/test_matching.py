"""Matching polar rows to requested alphas. No XFOIL needed.

The polar file prints alpha to three decimals, so a solved 0.0625 is
written as 0.062. Exact matching reported such points as failed.
"""

import pytest

from xfoil_mcp.wrapper import (
    MIN_ALPHA_STEP, POLAR_ALPHA_RESOLUTION, PolarPoint, _expected_alphas, _match_alphas,
)

def pt(alpha: float) -> PolarPoint:
    return PolarPoint(alpha=alpha, cl=0.5, cd=0.01, cdp=0.005, cm=-0.05,
                      top_xtr=0.4, bot_xtr=1.0)


def test_three_decimal_printing_still_matches():
    matched = _match_alphas([0.0625, 0.1875], [pt(0.062), pt(0.188)])
    assert set(matched) == {0.0625, 0.1875}


def test_neighbouring_printed_value_does_not_match():
    assert _match_alphas([0.0625], [pt(0.061)]) == {}


def test_missing_row_is_absent_from_the_match():
    matched = _match_alphas([0.0, 0.5, 1.0], [pt(0.0), pt(1.0)])
    assert set(matched) == {0.0, 1.0}


def test_each_requested_alpha_maps_to_its_own_row():
    matched = _match_alphas([0.0, 0.0625, 0.125], [pt(0.0), pt(0.062), pt(0.125)])
    assert matched[0.0625].alpha == 0.062


def test_step_finer_than_the_file_can_distinguish_is_rejected():
    with pytest.raises(ValueError):
        _expected_alphas(0, 0.01, 0.001)



def test_tolerance_cannot_match_one_row_to_two_requests():
    """Two requested alphas are at least MIN_ALPHA_STEP apart. A row can
    only be claimed by both if the tolerance reaches halfway between them."""
    assert POLAR_ALPHA_RESOLUTION < MIN_ALPHA_STEP / 2
