"""A NACA 2412 with and without a 10 degree flap, each with a leading-edge heater."""

from xfoil_mcp.schema import Case, Conditions, Flap, Geometry, Thermal


def campaign() -> list[Case]:
    heater = Thermal(
        chord_m=0.5, air_temperature_k=263.15, heater_width=0.1,
        heater_power_w_per_m=500.0, skin_thickness_m=0.001, skin_conductivity_w_mk=200.0,
    )
    return [
        Case(
            geometry=Geometry(naca="2412", flap=None if deflection == 0 else Flap(x_hinge=0.7, deflection=deflection)),
            conditions=Conditions(reynolds=1e6, alpha_start=0, alpha_end=10, alpha_step=2),
            outputs=("forces", "bl"),
            thermal=heater,
            label=f"2412, flap {deflection} deg",
        )
        for deflection in (0, 10)
    ]
