"""Coupling: turn XFOIL's boundary layer into heat-transfer inputs.

Pure functions over plain lists: no XFOIL, no Docker, no imports from the
rest of the package, so this runs wherever the thermal solve runs, and its
tests run on the host.

Surface arrays run upper trailing edge -> leading edge -> lower trailing
edge, as XFOIL writes them. Ue is signed along s: positive on the upper
surface, negative on the lower.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stagnation:
    """Where the oncoming air stops: Ue = 0, interpolated between nodes.

    Not the geometric leading edge. At positive alpha it sits just behind
    the tip on the lower surface. It is where droplets strike and ice forms,
    so it is where the heater is centered.
    """

    s: float
    x: float
    y: float
    index: int      # the crossing lies between index and index + 1
    surface: str    # "upper", "lower", or "leading edge"


def stagnation_point(
    s: list[float], x: list[float], y: list[float], ue: list[float]
) -> Stagnation:
    """Locate the stagnation point on the surface.

    Finds the one pair of neighboring nodes where Ue goes from positive to
    zero or negative, and interpolates linearly to where it is exactly zero.
    Anything other than exactly one crossing means the flow is not what this
    model assumes, and a heater placed on a guess would fail silently, so it
    raises instead.
    """
    crossings = [i for i in range(len(ue) - 1) if ue[i] > 0 >= ue[i + 1]]
    if len(crossings) != 1:
        raise ValueError(
            f"expected exactly one positive-to-negative crossing of Ue on the surface, "
            f"found {len(crossings)}"
        )
    i = crossings[0]
    t = ue[i] / (ue[i] - ue[i + 1])          # fraction of the way from node i to i + 1

    def at_crossing(values: list[float]) -> float:
        return values[i] + t * (values[i + 1] - values[i])

    s_stag = at_crossing(s)
    le = min(range(len(x)), key=x.__getitem__)
    if s_stag > s[le]:
        surface = "lower"
    elif s_stag < s[le]:
        surface = "upper"
    else:
        surface = "leading edge"

    return Stagnation(s=s_stag, x=at_crossing(x), y=at_crossing(y), index=i, surface=surface)
