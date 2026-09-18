"""Reference campaign: one case. Proves the loop end to end."""

from xfoil_mcp.schema import Case, Conditions, Geometry


def campaign() -> list[Case]:
    return [
        Case(
            geometry=Geometry(naca="2412"),
            conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=1),
            label="baseline 2412",
        )
    ]
