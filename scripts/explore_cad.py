"""Step 16: XFOIL coordinates to a printable 3D section.

Scales the outline to a 150 mm chord, extrudes it 40 mm into a solid, and
exports an STL for the slicer.
"""

import cadquery as cq

CHORD_MM = 150.0
SPAN_MM = 40.0


def load(path: str) -> list[tuple[float, float]]:
    """Read PSAV output: one 'x y' pair per line, in chord fractions."""
    return [tuple(float(v) * CHORD_MM for v in line.split())
            for line in open(path) if line.strip()]


for name in ("naca2412_flap_none", "naca2412_flap_10"):
    points = load(f"tests/fixtures/{name}.dat")

    # Trace the outline on a flat plane, close it, and push it up into a solid.
    section = cq.Workplane("XY").polyline(points).close().extrude(SPAN_MM)
    solid = section.val()
    box = solid.BoundingBox()

    cq.exporters.export(section, f"{name}.stl")
    te_thickness = abs(points[0][1] - points[-1][1])
    print(f"{name}: valid={solid.isValid()}  volume={solid.Volume() / 1000:.1f} cm^3  "
          f"x {box.xmin:.1f} to {box.xmax:.1f} mm  y {box.ymin:.1f} to {box.ymax:.1f} mm  "
          f"trailing edge {te_thickness:.2f} mm thick")
