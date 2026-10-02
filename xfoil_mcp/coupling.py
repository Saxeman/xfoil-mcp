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
import math


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

# --- air and flight condition ----------------------------------------------

@dataclass(frozen=True)
class Air:
    """Air properties, SI units. Ideal gas for density, Sutherland's law for
    viscosity, a power-law fit for conductivity, constant cp. Good to a few
    percent over the -40 to +40 C range icing cares about."""

    rho: float      # density, kg/m^3
    mu: float       # dynamic viscosity, Pa s
    k: float        # thermal conductivity, W/m K
    cp: float = 1006.0   # specific heat, J/kg K

    @classmethod
    def at(cls, temperature_k: float, pressure_pa: float = 101325.0) -> Air:
        T = temperature_k
        return cls(
            rho=pressure_pa / (287.05 * T),
            mu=1.458e-6 * T**1.5 / (T + 110.4),
            k=0.0241 * (T / 273.15) ** 0.81,
        )

    @property
    def nu(self) -> float:
        """Kinematic viscosity, m^2/s."""
        return self.mu / self.rho

    @property
    def pr(self) -> float:
        """Prandtl number: how fast momentum spreads relative to heat. ~0.72 for air."""
        return self.mu * self.cp / self.k


def velocity_for(reynolds: float, chord_m: float, air: Air) -> float:
    """The airspeed that gives this Reynolds number at this chord and air.

    Derived rather than chosen, so the thermal model can never disagree
    with the conditions XFOIL actually solved.
    """
    return reynolds * air.nu / chord_m


# --- heat transfer coefficient -----------------------------------------------

def _power_integral(u0: float, u1: float, ds: float) -> float:
    """Exact integral of u^1.87 across a segment where u varies linearly.

    A plain trapezoid is badly wrong on the first segment out of the
    stagnation point, where u starts at zero: it overestimates this integral
    by 44% and underestimates h there by about 17%.
    """
    if abs(u1 - u0) < 1e-12:
        return u0**1.87 * ds
    return ds * (u1**2.87 - u0**2.87) / (2.87 * (u1 - u0))


def smith_spalding(distances: list[float], speeds: list[float], air: Air) -> list[float]:
    """Laminar heat transfer coefficient along a surface, W/m^2 K.

    distances: metres from the stagnation point, increasing.
    speeds: edge velocity magnitude at each distance, m/s.

    Thermal layer thickness from an integral of the edge speed, then
    h = 2k / thickness. Accounts for acceleration, so it holds at the
    stagnation point, where the Reynolds analogy does not. Reproduces the
    flat-plate and cylinder-stagnation results to about 1%.

    A node where the air is at rest gets the limit of the same formula. Near
    a stagnation point u = a d, so thickness^2 tends to 46.72 nu / (2.87 a):
    finite, where the formula itself reads 0 / 0. XFOIL prints Ue to five
    decimals, so a node at the stagnation point reads exactly zero.
    """
    h: list[float] = []
    integral, d_prev, u_prev = 0.0, 0.0, 0.0
    for k, (d, u) in enumerate(zip(distances, speeds)):
        integral += _power_integral(u_prev, u, d - d_prev)
        if u > 0.0:
            thickness = math.sqrt(46.72 * air.nu * integral / u**2.87)
        else:
            thickness = math.sqrt(46.72 * air.nu / (2.87 * _velocity_gradient(distances, speeds, k)))
        h.append(2 * air.k / thickness)
        d_prev, u_prev = d, u
    return h


def _velocity_gradient(distances: list[float], speeds: list[float], k: int) -> float:
    """du/dd at node k, where the air is at rest: the slope to the nearest
    node where it moves, looking downstream first and then upstream."""
    for j in (*range(k + 1, len(speeds)), *range(k - 1, -1, -1)):
        if speeds[j] > 0.0 and distances[j] != distances[k]:
            return speeds[j] / abs(distances[j] - distances[k])
    raise ValueError("the edge velocity is zero at every node on this side of the stagnation point")


def reynolds_analogy(cf: float, ue_ratio: float, velocity: float, air: Air) -> float:
    """Turbulent heat transfer coefficient from skin friction, W/m^2 K.

    Heat and momentum are carried to the wall by the same mixing, so
    St = (Cf/2) Pr^(-2/3), with Cf and St both based on the local edge
    speed. XFOIL's Cf is based on the freestream speed (xoper.f:
    CF = TAU/(0.5*QINF**2)), so wall shear is recovered from it first.
    abs() because separated flow gives negative Cf; the analogy is weak
    there, but heat still leaves.
    """
    wall_shear = abs(cf) * 0.5 * air.rho * velocity**2
    edge_speed = abs(ue_ratio) * velocity
    if edge_speed == 0.0:
        return 0.0          # nothing to base the analogy on; the caller's laminar floor applies
    return air.cp * wall_shear * air.pr ** (-2 / 3) / edge_speed


@dataclass(frozen=True)
class HeatTransfer:
    h: list[float]          # W/m^2 K, one per surface node
    regime: list[str]       # "laminar" or "turbulent"


def heat_transfer(
    s: list[float], x: list[float], ue: list[float], cf: list[float],
    stagnation: Stagnation, xtr_upper: float, xtr_lower: float,
    chord_m: float, velocity: float, air: Air,
) -> HeatTransfer:
    """h along the whole surface, walking from the stagnation point to each tail.

    Laminar (Smith-Spalding) until transition; turbulent (Reynolds analogy)
    after. The turbulent value is never allowed below the laminar one: at the
    transition node XFOIL's Cf is still at its laminar low, and turbulence
    only increases heat transfer. A transition location of 1.0 means the
    side stays laminar.
    """
    n = len(s)
    le = min(range(n), key=x.__getitem__)
    h = [0.0] * n
    regime = [""] * n
    sides = (
        # upper side: stagnation node back to the upper tail; x only counts once past the tip
        (list(range(stagnation.index, -1, -1)), xtr_upper, lambda i: i <= le),
        # lower side: next node on to the lower tail
        (list(range(stagnation.index + 1, n)), xtr_lower, lambda i: i >= le),
    )
    for nodes, xtr, past_tip in sides:
        laminar = smith_spalding(
            [abs(s[i] - stagnation.s) * chord_m for i in nodes],
            [abs(ue[i]) * velocity for i in nodes],
            air,
        )
        turbulent = False
        for i, h_laminar in zip(nodes, laminar):
            turbulent = turbulent or (xtr < 1.0 and past_tip(i) and x[i] >= xtr)
            if turbulent:
                h[i] = max(reynolds_analogy(cf[i], ue[i], velocity, air), h_laminar)
                regime[i] = "turbulent"
            else:
                h[i] = h_laminar
                regime[i] = "laminar"
    return HeatTransfer(h=h, regime=regime)
