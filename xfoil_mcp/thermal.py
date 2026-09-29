"""Skin temperature along the airfoil surface, with a leading-edge heater.

The skin is treated as a thin strip unrolled along the surface. Heat from
a heater band centred on the stagnation point spreads along the skin by
conduction and leaves to the air at the local rate h(s). One-dimensional,
steady, dry air: no droplets, evaporation, or ice.

Needs numpy and scikit-fem (the thermal extra). Takes plain lists, so it
depends on nothing else in the package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import Basis, BilinearForm, ElementLineP1, Functional, LinearForm, MeshLine, solve
from skfem.helpers import dot, grad

FREEZING_K = 273.15

# A required point closer than this fraction of its gap to an existing mesh
# point moves that point instead of adding a new one. No gap can then be
# much smaller than its neighbours, which is what cracked the mesh in step 12.
SNAP_FRACTION = 0.25


def _place(nodes: np.ndarray, point: float, placed: list[float]) -> np.ndarray:
    """Make `point` a mesh node without creating a sliver of an element."""
    i = int(np.searchsorted(nodes, point))
    if i == 0 or i == len(nodes):
        raise ValueError(f"point {point} lies outside the surface")
    left, right = nodes[i - 1], nodes[i]
    nearest = i - 1 if point - left <= right - point else i
    close = abs(nodes[nearest] - point) < SNAP_FRACTION * (right - left)
    if close and nodes[nearest] not in placed:
        moved = nodes.copy()
        moved[nearest] = point
        return moved
    return np.insert(nodes, i, point)


def build_mesh_nodes(
    z_surface: list[float], required: list[float], refine: int = 4
) -> np.ndarray:
    """XFOIL's surface points, each gap split into `refine` pieces, plus the
    required points (stagnation, heater edges) placed exactly. Points placed
    earlier are never moved by later ones."""
    z = np.asarray(z_surface, dtype=float)
    pieces = [np.linspace(z[j], z[j + 1], refine + 1)[:-1] for j in range(len(z) - 1)]
    nodes = np.concatenate(pieces + [z[-1:]])
    placed: list[float] = []
    for point in required:
        nodes = _place(nodes, point, placed)
        placed.append(point)
    return nodes


@dataclass(frozen=True)
class SkinResult:
    z_m: list[float]                        # distance along the skin from stagnation; upper < 0 < lower
    temperature_k: list[float]
    heat_in_w_per_m: float
    heat_out_w_per_m: float
    min_heated_temperature_k: float         # coldest point inside the heater band
    ice_free_upper_m: float | None          # how far above-freezing skin extends from stagnation, each way
    ice_free_lower_m: float | None          # None if the stagnation point itself is below freezing

    @property
    def balance_error(self) -> float:
        """Relative mismatch between heat out and heat in. The thermal gate."""
        return abs(self.heat_out_w_per_m - self.heat_in_w_per_m) / self.heat_in_w_per_m


def skin_temperature(
    s: list[float], h: list[float], stagnation_s: float, chord_m: float,
    heater_width: float, heater_power_w_per_m: float, kt: float,
    air_temperature_k: float, refine: int = 4,
) -> SkinResult:
    """Steady skin temperature with a heater band centred on the stagnation point.

    s and h are XFOIL's surface points and step 10's heat transfer
    coefficient at each. heater_width is a fraction of chord; kt is skin
    conductivity times thickness. The two trailing-edge ends are treated as
    separate insulated ends; in reality they meet, but almost no heat
    reaches them.
    """
    z_surface = (np.asarray(s, dtype=float) - stagnation_s) * chord_m
    h_surface = np.asarray(h, dtype=float)
    half = 0.5 * heater_width * chord_m
    flux = heater_power_w_per_m / (2 * half)

    nodes = build_mesh_nodes(z_surface, [0.0, -half, half], refine)
    basis = Basis(MeshLine(nodes), ElementLineP1())

    def h_at(z):
        """h anywhere along the skin, by straight-line interpolation from XFOIL's points."""
        return np.interp(z, z_surface, h_surface)

    @BilinearForm
    def conduction_and_loss(u, v, w):
        return kt * dot(grad(u), grad(v)) + h_at(w.x[0]) * u * v

    @LinearForm
    def heater(v, w):
        return np.where(np.abs(w.x[0]) <= half, flux, 0.0) * v

    @Functional
    def loss(w):
        return h_at(w.x[0]) * w["rise"]

    rise = solve(conduction_and_loss.assemble(basis), heater.assemble(basis))
    heat_out = float(loss.assemble(basis, rise=basis.interpolate(rise)))
    T = air_temperature_k + rise

    in_band = np.abs(nodes) <= half + 1e-12
    upper, lower = _ice_free_extent(nodes, T)
    return SkinResult(
        z_m=nodes.tolist(),
        temperature_k=T.tolist(),
        heat_in_w_per_m=heater_power_w_per_m,
        heat_out_w_per_m=heat_out,
        min_heated_temperature_k=float(T[in_band].min()),
        ice_free_upper_m=upper,
        ice_free_lower_m=lower,
    )


def _ice_free_extent(nodes: np.ndarray, T: np.ndarray) -> tuple[float | None, float | None]:
    """Walk outward from the stagnation point while the skin is above freezing."""
    warm = T > FREEZING_K
    centre = int(np.argmin(np.abs(nodes)))
    if not warm[centre]:
        return None, None
    j = centre
    while j > 0 and warm[j - 1]:
        j -= 1
    k = centre
    while k < len(nodes) - 1 and warm[k + 1]:
        k += 1
    return float(-nodes[j]), float(nodes[k])
