"""From an analysed outline to a printable part.

Takes the outline XFOIL analysed (step 17a) and returns an STL, a hash of
those exact bytes, and the stats a person needs before approving a print.
Plain lists in; imports CadQuery, so it runs on the host (cad extra).
"""

from __future__ import annotations

import bisect
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass

import cadquery as cq

# Calibrated from a Bambu Studio slice of two of these parts (X1C, 0.4 mm
# nozzle, 0.2 mm layers, default profile): 148.1 cm^3 took 55.27 g and 64 min
# of printing, plus 7.25 min of preparation per plate. Linear in volume, so
# an estimate, not a slice: real print time also depends on surface area.
GRAMS_PER_CM3 = 55.27 / 148.1
PRINT_MINUTES_PER_CM3 = 64.0 / 148.1
PREP_MINUTES = 7.25

# Both dimensions must fit the printer the estimates were calibrated on (256 mm
# build volume), with a margin. Below 1 mm there is nothing to print.
MIN_DIMENSION_MM = 1.0
MAX_DIMENSION_MM = 250.0


class CadError(ValueError):
    """The outline cannot become a valid solid."""


@dataclass(frozen=True)
class SectionPart:
    stl: bytes          # binary STL, exactly what gets printed
    sha256: str         # of those bytes; the print approval binds to this
    stats: dict         # dimensions, volume, and estimates, for the approval page


def _checked(outline) -> list[tuple[float, float]]:
    """The outline as (x, y) pairs of finite numbers. It came out of a
    container, so its shape is checked before the CAD kernel sees it."""
    points = []
    for i, point in enumerate(outline):
        try:
            x, y = point
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            raise CadError(f"outline point {i} is not an (x, y) pair of numbers: {point!r}") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CadError(f"outline point {i} is not finite: {point!r}")
        points.append((x, y))
    if len(points) < 3:
        raise CadError(f"an outline needs at least 3 points, got {len(points)}")
    return points


def _height_at(surface: list[tuple[float, float]], x: float) -> float:
    """y on a surface (points sorted by x) at x, by straight-line interpolation."""
    xs = [p[0] for p in surface]
    i = bisect.bisect_left(xs, x)
    if i == 0:
        return surface[0][1]
    if i == len(surface):
        return surface[-1][1]
    (x0, y0), (x1, y1) = surface[i - 1], surface[i]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _max_thickness(points: list[tuple[float, float]]) -> float:
    """The greatest vertical distance between the upper and lower surfaces.

    The outline runs upper trailing edge -> leading edge -> lower trailing
    edge, so the leftmost point splits it into the two surfaces. This is the
    section's thickness; the bounding box is taller once there is camber or
    a deflected flap.
    """
    nose = min(range(len(points)), key=lambda i: points[i][0])
    upper, lower = sorted(points[:nose + 1]), sorted(points[nose:])
    return max((y - _height_at(lower, x) for x, y in upper), default=0.0)


def build_section(
    outline: list[tuple[float, float]], chord_mm: float = 150.0, span_mm: float = 40.0
) -> SectionPart:
    """Scale a chord-fraction outline to millimetres, extrude it, and export an STL.

    The section lies in the XY plane and the span runs along Z, which is also
    the print orientation: every layer is one slice of the airfoil.
    chord_mm and span_mm must each lie between MIN_DIMENSION_MM and
    MAX_DIMENSION_MM.
    """
    for name, value in (("chord_mm", chord_mm), ("span_mm", span_mm)):
        if not MIN_DIMENSION_MM <= value <= MAX_DIMENSION_MM:     # also refuses NaN
            raise CadError(
                f"{name} must be between {MIN_DIMENSION_MM:g} and {MAX_DIMENSION_MM:g} mm, got {value}"
            )
    points = [(x * chord_mm, y * chord_mm) for x, y in _checked(outline)]
    try:
        section = cq.Workplane("XY").polyline(points).close().extrude(span_mm)
        solid = section.val()
        valid, volume_mm3 = solid.isValid(), solid.Volume()
    except Exception as exc:        # the kernel raises its own types for outlines it cannot build
        raise CadError(f"the CAD kernel could not build a solid from this outline: {exc}") from exc
    if not valid:
        raise CadError("the outline does not make a valid closed solid")
    if volume_mm3 <= 0:
        raise CadError("the outline encloses no area")

    # CadQuery exports to a file, so go through a temporary one.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "part.stl")
        cq.exporters.export(section, path)
        with open(path, "rb") as f:
            stl = f.read()

    box = solid.BoundingBox()
    volume_cm3 = volume_mm3 / 1000.0
    stats = {
        "chord_mm": round(box.xlen, 2),                     # projected length, shorter with a flap down
        "thickness_mm": round(_max_thickness(points), 2),   # of the section itself
        "height_mm": round(box.ylen, 2),                    # bounding box: thickness plus camber and flap drop
        "span_mm": round(box.zlen, 2),
        "volume_cm3": round(volume_cm3, 2),
        "estimated_mass_g": round(volume_cm3 * GRAMS_PER_CM3, 1),
        "estimated_print_minutes": round(PREP_MINUTES + volume_cm3 * PRINT_MINUTES_PER_CM3),
        "estimate_basis": "volume, calibrated against one Bambu Studio slice",
    }
    return SectionPart(stl=stl, sha256=hashlib.sha256(stl).hexdigest(), stats=stats)
