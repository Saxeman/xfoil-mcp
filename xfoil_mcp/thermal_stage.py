"""The thermal stage: from a finished aero result to heater performance.

Chains steps 9, 10 and 13 for every requested angle of attack, behind the
aero gate, and reports the worst case with coverage. One alpha failing
never fails the case: it is excluded with a reason, and coverage says how
much of the requested sweep the worst case actually covers.
"""

from __future__ import annotations

import math

from xfoil_mcp.coupling import Air, heat_transfer, stagnation_point, velocity_for
from xfoil_mcp.schema import CaseResult
from xfoil_mcp.thermal import skin_temperature

MAX_MACH = 0.3              # both XFOIL's solve here and the thermal model assume slow air
BALANCE_TOLERANCE = 0.01    # heat out must match heater power within 1%
ALPHA_MATCH = 1e-3          # polar rows print alpha to 3 decimals (step 5)


def _refuse(kind: str, message: str) -> dict:
    return {
        "status": "error", "failure_kind": kind, "errors": [message],
        "coverage": 0.0, "worst_case": None, "points": [], "excluded": [],
    }


def evaluate_thermal(aero: CaseResult) -> dict:
    """Heater performance at every requested alpha of a finished aero case."""
    case = aero.case
    thermal = case.thermal
    if thermal is None:
        raise ValueError("case has no thermal block")

    # The aero gate: nothing thermal runs on a failed aero result.
    if aero.status in ("error", "empty") or not aero.data:
        return _refuse("input", f"upstream aero stage did not pass (status {aero.status})")

    air = Air.at(thermal.air_temperature_k, thermal.pressure_pa)
    velocity = velocity_for(case.conditions.reynolds, thermal.chord_m, air)
    mach = velocity / math.sqrt(1.4 * 287.05 * thermal.air_temperature_k)
    if mach > MAX_MACH:
        return _refuse(
            "input",
            f"Reynolds {case.conditions.reynolds:g} at a {thermal.chord_m} m chord implies "
            f"{velocity:.0f} m/s (Mach {mach:.2f}); the thermal model assumes low speed",
        )

    requested = aero.data["requested_alphas"]
    polar = aero.data["points"]
    layers = aero.data.get("bl", {})
    kt = thermal.skin_conductivity_w_mk * thermal.skin_thickness_m

    points: list[dict] = []
    excluded: list[dict] = []
    for alpha in requested:
        layer = next((v for k, v in layers.items() if abs(float(k) - alpha) <= ALPHA_MATCH), None)
        row = next((p for p in polar if abs(p["alpha"] - alpha) <= ALPHA_MATCH), None)
        if layer is None or row is None:
            excluded.append({"alpha": alpha, "reason": "aero did not converge"})
            continue

        n = layer["n_surface"]
        s, x, y, ue, cf = (layer[f][:n] for f in ("s", "x", "y", "ue", "cf"))
        try:
            stag = stagnation_point(s, x, y, ue)
        except ValueError as exc:
            excluded.append({"alpha": alpha, "reason": f"no stagnation point: {exc}"})
            continue

        h = heat_transfer(s, x, ue, cf, stag, row["top_xtr"], row["bot_xtr"],
                          thermal.chord_m, velocity, air)
        skin = skin_temperature(s, h.h, stag.s, thermal.chord_m, thermal.heater_width,
                                thermal.heater_power_w_per_m, kt, thermal.air_temperature_k)

        # The thermal gate: a solve whose heat does not balance is not a result.
        if skin.balance_error > BALANCE_TOLERANCE:
            excluded.append({"alpha": alpha, "reason": f"energy balance off by {skin.balance_error:.1%}"})
            continue

        def mm(metres):
            return None if metres is None else round(metres * 1000, 1)

        points.append({
            "alpha": alpha,
            "min_heated_temperature_c": round(skin.min_heated_temperature_k - 273.15, 2),
            "peak_temperature_c": round(max(skin.temperature_k) - 273.15, 2),
            "ice_free_upper_mm": mm(skin.ice_free_upper_m),
            "ice_free_lower_mm": mm(skin.ice_free_lower_m),
            "balance_error": skin.balance_error,
        })

    # Fan-in. Coverage is against requested alphas: the ones the aero stage
    # dropped are usually the hardest, so a worst case over survivors alone
    # would flatter the design.
    coverage = len(points) / len(requested) if requested else 0.0
    worst = None
    if points:
        coldest = min(points, key=lambda p: p["min_heated_temperature_c"])
        worst = {
            "min_heated_temperature_c": coldest["min_heated_temperature_c"],
            "at_alpha": coldest["alpha"],
            "ice_free_upper_mm": min(p["ice_free_upper_mm"] or 0.0 for p in points),
            "ice_free_lower_mm": min(p["ice_free_lower_mm"] or 0.0 for p in points),
        }

    status = "ok" if coverage == 1.0 else "partial" if points else "empty"
    return {
        "status": status,
        "failure_kind": None if status == "ok" else "numerical",
        "errors": [],
        "heater_power_w_per_m": thermal.heater_power_w_per_m,
        "velocity_m_s": round(velocity, 2),
        "coverage": coverage,
        "worst_case": worst,
        "points": points,
        "excluded": excluded,
    }
