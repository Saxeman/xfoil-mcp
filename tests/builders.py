import dataclasses
from pathlib import Path

from xfoil_mcp.schema import Case, CaseResult, Conditions, Geometry
from xfoil_mcp.wrapper import _parse_bl_file

BL_A4 = Path(__file__).parent / "fixtures" / "bl_2412_re1e6_a4.txt"
THERMAL = dict(
    chord_m=0.5, air_temperature_k=263.15, heater_width=0.1,
    heater_power_w_per_m=500.0, skin_thickness_m=0.001, skin_conductivity_w_mk=200.0,
)

def aero(requested=(4.0,), converged=(4.0,), status="ok", thermal=None) -> CaseResult:
    """An aero result shaped like the worker's, using the fixture's boundary
    layer for every converged alpha."""
    case = Case(
        geometry=Geometry(naca="2412"),
        conditions=Conditions(reynolds=1e6, alpha_start=min(requested),
                              alpha_end=max(requested), alpha_step=2),
        outputs=("forces", "bl"),
        thermal=thermal or THERMAL,
    )
    layer = dataclasses.asdict(_parse_bl_file(BL_A4, alpha=4.0))
    points = [dict(alpha=a, cl=0.7146, cd=0.00693, cdp=0.00107, cm=-0.0573,
                   top_xtr=0.398, bot_xtr=1.0) for a in converged]
    data = {
        "requested_alphas": list(requested),
        "points": points,
        "failed_alphas": [a for a in requested if a not in converged],
        "bl": {str(a): layer for a in converged},
    }
    return CaseResult(
        case=case, status=status,
        failure_kind=None if status == "ok" else "numerical",
        data=None if status == "empty" else data,
    )
