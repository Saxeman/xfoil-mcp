"""From an analysed outline to a printable part.

Takes the outline XFOIL analysed (step 17a) and returns an STL, a hash of
those exact bytes, and the stats a person needs before approving a print.
Plain lists in; imports CadQuery, so it runs on the host (cad extra).
"""

from __future__ import annotations

import hashlib
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


class CadError(ValueError):
    """The outline cannot become a valid solid."""


@dataclass(frozen=True)
class SectionPart:
    stl: bytes          # binary STL, exactly what gets printed
    sha256: str         # of those bytes; the print approval binds to this
    stats: dict         # dimensions, volume, and estimates, for the approval page


def build_section(
    outline: list[tuple[float, float]], chord_mm: float = 150.0, span_mm: float = 40.0
) -> SectionPart:
    """Scale a chord-fraction outline to millimetres, extrude it, and export an STL.

    The section lies in the XY plane and the span runs along Z, which is also
    the print orientation: every layer is one slice of the airfoil.
    """
    if len(outline) < 3:
        raise CadError(f"an outline needs at least 3 points, got {len(outline)}")

    points = [(x * chord_mm, y * chord_mm) for x, y in outline]
    section = cq.Workplane("XY").polyline(points).close().extrude(span_mm)
    solid = section.val()
    if not solid.isValid():
        raise CadError("the outline does not make a valid closed solid")

    # CadQuery exports to a file, so go through a temporary one.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "part.stl")
        cq.exporters.export(section, path)
        with open(path, "rb") as f:
            stl = f.read()

    box = solid.BoundingBox()
    volume_cm3 = solid.Volume() / 1000.0
    stats = {
        "chord_mm": round(box.xlen, 2),
        "thickness_mm": round(box.ylen, 2),
        "span_mm": round(box.zlen, 2),
        "volume_cm3": round(volume_cm3, 2),
        "estimated_mass_g": round(volume_cm3 * GRAMS_PER_CM3, 1),
        "estimated_print_minutes": round(PREP_MINUTES + volume_cm3 * PRINT_MINUTES_PER_CM3),
        "estimate_basis": "volume, calibrated against one Bambu Studio slice",
    }
    return SectionPart(stl=stl, sha256=hashlib.sha256(stl).hexdigest(), stats=stats)
