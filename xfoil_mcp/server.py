"""MCP tools over the harness. Runs on the host, stdio transport.

Thin by design: every tool is a few lines calling the harness and shaping
the payload. The substance is in the docstrings, which are what the model
reads to decide when and how to call each tool.

Every response, from every tool, on success or failure, carries the same
envelope: status, failure_kind, retry_could_help, errors. A model learns
one rule for reading results, not one per tool.

Tools never raise for failures they anticipated. Expected failures are
return values with a failure_kind. Exceptions are reserved for the
genuinely unexpected.

Never write to stdout here. Over stdio, stdout is the protocol channel.
"""

from __future__ import annotations

import logging
import sys
import threading

from fastmcp import FastMCP
from pydantic import ValidationError

from xfoil_mcp import harness
from xfoil_mcp.schema import Case, FailureKind

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("xfoil")

# Kept in memory for the lifetime of the server process. A queued
# implementation would put this in a store keyed by batch id.
_last_batch: harness.BatchResult | None = None

# Validation errors name schema fields; the caller used tool arguments. Say
# which argument was wrong, in the caller's own terms.
_ARGUMENT_NAMES = {
    "geometry.naca": "airfoil",
    "geometry.flap.x_hinge": "flap_hinge",
    "geometry.flap.deflection": "flap_deflection_deg",
    "conditions.reynolds": "reynolds",
    "conditions.alpha_start": "alpha_start",
    "conditions.alpha_end": "alpha_end",
    "conditions.alpha_step": "alpha_step",
    "thermal.chord_m": "chord_m",
    "thermal.air_temperature_k": "air_temperature_c (checked in kelvin)",
    "thermal.heater_width": "heater_width",
    "thermal.heater_power_w_per_m": "heater_power_w_per_m",
    "thermal.skin_thickness_m": "skin_thickness_mm (checked in metres)",
    "thermal.skin_conductivity_w_mk": "skin_conductivity_w_mk",
}

_AERO_HIDDEN = {"thermal", "airfoil", "reynolds", "field_outputs"}
_THERMAL_SHOWN = ("status", "failure_kind", "errors", "coverage", "worst_case", "excluded")

_designs: dict = {}          # run_polar and evaluate_design results this session, by content hash
_print_queue = None          # created on the first print request
_print_site = None
_print_base_url = None
_print_lock = threading.Lock()   # tools run in worker threads; only one may start the queue

# --- envelope --------------------------------------------------------------

def _envelope(status: str, kind: FailureKind | None, errors: list[str], /, **fields) -> dict:
    """The four keys every response carries, then the tool's own fields.

    The envelope is authoritative. Fields often come from a container's
    summary, so one that shares a name with an envelope key is dropped
    rather than allowed to change the verdict. The parameters are
    positional-only so that such a field cannot collide with them either.
    """
    envelope = {"status": status, "failure_kind": kind,
                "retry_could_help": kind == "infrastructure", "errors": errors}
    return {**envelope, **{k: v for k, v in fields.items() if k not in envelope}}


def _ok(**fields) -> dict:
    return _envelope("ok", None, [], **fields)


def _error(kind: FailureKind, errors: list[str], /, **fields) -> dict:
    return _envelope("error", kind, errors, **fields)


def _validation_errors(exc: ValidationError, names: dict[str, str] | None = None) -> list[str]:
    """One "location: message" line per error.

    `names` maps schema paths to the tool's argument names, so the caller is
    told which argument was wrong in its own terms. A path it does not list
    is reported as the schema path.
    """
    names = names or {}
    lines = []
    for e in exc.errors():
        location = ".".join(str(p) for p in e["loc"])
        lines.append(f"{names.get(location, location)}: {e['msg']}")
    return lines


def _flap_text(flap) -> str:
    return "none" if flap is None else f"{flap.deflection:g} deg at {flap.x_hinge:.0%} chord"


def _case_view(cases: list[Case]) -> list[dict]:
    """What the model, and through it the person approving, sees per case.

    Everything that makes one solver run differ from another is here, so two
    cases never look alike unless they are. The label is written by the
    agent and cannot be what tells them apart.
    """
    return [
        {
            "content_hash": c.content_hash(),
            "label": c.label,
            "naca": c.geometry.naca,
            "flap": _flap_text(c.geometry.flap),
            "reynolds": c.conditions.reynolds,
            "mach": c.conditions.mach,
            "n_crit": c.conditions.n_crit,
            "max_iter": c.conditions.max_iter,
            "alphas": f"{c.conditions.alpha_start}..{c.conditions.alpha_end} step {c.conditions.alpha_step}",
            "outputs": list(c.outputs),
            "thermal": None if c.thermal is None else c.thermal.model_dump(mode="json"),
        }
        for c in cases
    ]


def _per_alpha(result) -> list[dict]:
    """Aero and thermal results side by side, one row per converged alpha."""
    data = result.data or {}
    thermal = {round(p["alpha"], 3): p for p in (data.get("thermal") or {}).get("points", [])}
    rows = []
    for p in data.get("points", []):
        t = thermal.get(round(p["alpha"], 3), {})
        rows.append({
            "alpha": p["alpha"], "cl": p["cl"], "cd": p["cd"], "upper_transition": p["top_xtr"],
            "min_heated_temperature_c": t.get("min_heated_temperature_c"),
            "ice_free_upper_mm": t.get("ice_free_upper_mm"),
            "ice_free_lower_mm": t.get("ice_free_lower_mm"),
        })
    return rows


def _design_response(case, result) -> dict:
    """The stage trace. Overall status comes from the first stage that did not pass."""
    aero = {"stage": "aero", "status": result.status, "failure_kind": result.failure_kind}
    aero.update((k, v) for k, v in result.summary.items() if k not in _AERO_HIDDEN and k not in aero)
    if result.status in ("error", "empty"):
        thermal = {"stage": "thermal", "status": "not_run",
                   "reason": f"aero stage did not pass (status {result.status})"}
    else:
        raw = (result.data or {}).get("thermal") or {}
        thermal = {"stage": "thermal", **{k: raw.get(k) for k in _THERMAL_SHOWN}}
    fields = dict(content_hash=case.content_hash(), stages=[aero, thermal],
                  per_alpha=_per_alpha(result))

    if result.status in ("error", "empty"):
        return _error(result.failure_kind or "numerical",
                      [result.summary.get("error") or f"aero stage returned {result.status}"], **fields)
    if thermal["status"] == "error":
        return _error(thermal["failure_kind"] or "infrastructure",
                      thermal["errors"] or ["thermal stage failed"], **fields)
    if "partial" in (result.status, thermal["status"]) or thermal["status"] == "empty":
        return _envelope("partial", "numerical", [], **fields)
    return _ok(**fields)

# --- printing ----------------------------------------------------------------

def _default_outbox():
    """outbox/ at the repository root, found from this file's location.

    Not the working directory: Claude Desktop starts the server from one we
    don't control. This file is <repo>/xfoil_mcp/server.py, so the repo root
    is two levels up. XFOIL_PRINT_OUTBOX still overrides it.
    """
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "outbox"

def _printing():
    """Start the print queue and the approval page on first use.

    Imported lazily: CadQuery takes seconds to load, and the MCP server should
    start instantly. The outbox defaults to a fixed folder because Claude
    Desktop starts this server from an unpredictable working directory.

    Nothing is published until the page has bound its port, so a failed
    start leaves no half-built state and the next call tries again. Raises
    OSError if the port cannot be bound and ValueError if XFOIL_PRINT_PORT
    is not a number.
    """
    global _print_queue, _print_site, _print_base_url
    with _print_lock:
        if _print_queue is None:
            import os
            from pathlib import Path

            from xfoil_mcp.print_site import start_site
            from xfoil_mcp.printing import PrintQueue

            outbox = Path(os.environ.get("XFOIL_PRINT_OUTBOX", _default_outbox()))
            port = int(os.environ.get("XFOIL_PRINT_PORT", "8765"))
            queue = PrintQueue(_backend(os.environ, outbox))
            site, base_url = start_site(queue, port)
            _print_queue, _print_site, _print_base_url = queue, site, base_url
        return _print_queue, _print_base_url

def _print_unavailable(exc: Exception, **fields) -> dict:
    """The response when the queue, its backend or its review page could not be started."""
    if isinstance(exc, OSError):
        message = (f"could not start the print review page: {exc}; "
                   "set XFOIL_PRINT_PORT to a free port and try again")
    else:
        message = f"the print setup is invalid: {exc}"
    return _error("infrastructure", [message], **fields)

def _backend(env, outbox):
    """The dry run unless XFOIL_PRINT_BACKEND=bambu switches on the real printer.

    Opt-in, so that nothing outside a deliberate setup can reach the printer.
    ValueError names a bad setting.
    """
    from xfoil_mcp.printing import DryBackend

    choice = env.get("XFOIL_PRINT_BACKEND", "dry")
    if choice == "dry":
        return DryBackend(outbox)
    if choice == "bambu":
        from xfoil_mcp.bambu import PrinterBackend, PrinterConfig

        return PrinterBackend(outbox, PrinterConfig.from_env(env))
    raise ValueError(f"XFOIL_PRINT_BACKEND is {choice!r}; use 'dry' or 'bambu'")


def _find_design(content_hash: str):
    """A design evaluated this session, or a case from the most recent campaign."""
    if content_hash in _designs:
        return _designs[content_hash]
    return _last_batch.get(content_hash) if _last_batch is not None else None

# --- tools -----------------------------------------------------------------

@mcp.tool
def run_polar(
    airfoil: str,
    reynolds: float,
    alpha_start: float = 0.0,
    alpha_end: float = 10.0,
    alpha_step: float = 1.0,
    n_crit: float = 9.0,
    max_iter: int = 100,
    flap_deflection_deg: float | None = None,
    flap_hinge: float = 0.7,
) -> dict:
    """Run one viscous angle-of-attack sweep on a NACA airfoil with XFOIL.

    Use this for a single airfoil at a single Reynolds number. For anything
    with structure (parameter sweeps, sampling schemes, many airfoils),
    write a campaign and use dry_run_campaign instead.

    airfoil: NACA 4- or 5-digit designation, e.g. "2412" (2% camber at 40%
        chord, 12% thick) or "0012" (symmetric).
    reynolds: chord Reynolds number. ~5e4 is a small drone, ~1e6 a light
        aircraft, ~1e7 an airliner. Drag results are not comparable across
        Reynolds numbers.
    alpha_start, alpha_end, alpha_step: angle of attack sweep in degrees.
        Keep the step at 1 or less; larger steps lose warm starts and fail
        more often near stall. At most 200 points.
    n_crit: transition criterion. 9 = clean wind tunnel (default), ~4 =
        turbulent freestream or dirty surface. This is a modeling assumption,
        not a measurement.
    max_iter: viscous iteration limit. Raise toward 200 only for points that
        fail to converge near stall.
    flap_deflection_deg: omit for no flap; positive is trailing edge down,
        which adds lift. flap_hinge is the hinge position as a fraction of
        chord (0.5 to 0.9).

    Returns status, converged vs requested points, which alphas failed and
    whether the failures are consecutive, best L/D and where, and max CL
    within the sweep (not the airfoil's true max CL unless the sweep passes
    stall). A status of "partial" means some alphas did not converge.
    Retrying identically will not help; use a smaller step, start closer to
    the failed region, or accept the gap. Consecutive failures near the top
    of the sweep usually mean stall. A failure_kind of "input" means the
    arguments were invalid; fix them rather than retrying.

    The outline XFOIL analysed is kept with the result, so this airfoil can
    be printed: pass the returned content_hash to request_print.
    """
    try:
        case = Case.model_validate({
            "geometry": {
                "naca": airfoil,
                "flap": None if flap_deflection_deg is None
                        else {"x_hinge": flap_hinge, "deflection": flap_deflection_deg},
            },
            "conditions": {
                "reynolds": reynolds, "alpha_start": alpha_start, "alpha_end": alpha_end,
                "alpha_step": alpha_step, "n_crit": n_crit, "max_iter": max_iter,
            },
            # The outline costs one XFOIL command and makes the result printable.
            "outputs": ["forces", "geometry"],
        })
    except ValidationError as exc:
        return _error("input", _validation_errors(exc), content_hash=None)

    result = harness.evaluate(case)
    _designs[case.content_hash()] = result
    if result.status == "error":
        return _error(
            result.failure_kind,
            [result.summary.get("error", "unknown error")],
            content_hash=case.content_hash(),
        )
    return _envelope(result.status, result.failure_kind, [],
                     content_hash=case.content_hash(), **result.summary)




@mcp.tool
def evaluate_design(
    airfoil: str,
    reynolds: float,
    chord_m: float,
    air_temperature_c: float,
    heater_power_w_per_m: float,
    heater_width: float = 0.1,
    skin_thickness_mm: float = 1.0,
    skin_conductivity_w_mk: float = 200.0,
    flap_deflection_deg: float | None = None,
    flap_hinge: float = 0.7,
    alpha_start: float = 0.0,
    alpha_end: float = 10.0,
    alpha_step: float = 2.0,
) -> dict:
    """Evaluate one wing section with a leading-edge heater: XFOIL aerodynamics,
    then skin temperature from a heater centred on the stagnation point.

    Use this for a single design. To compare many designs or sample a design
    space, write a campaign with a thermal block and use dry_run_campaign.

    airfoil: NACA 4- or 5-digit, e.g. "2412".
    reynolds: chord Reynolds number. Airspeed is derived from this, the chord
        and the air, so it is not an argument.
    chord_m: wing chord in metres.
    air_temperature_c: freestream air temperature in Celsius, e.g. -10.
    heater_power_w_per_m: total heater power per metre of span.
    heater_width: heater band width as a fraction of chord (0.1 = 10%).
    skin_thickness_mm: skin thickness in millimetres (default 1).
    skin_conductivity_w_mk: about 200 for aluminium, about 1 for carbon
        composite. Aluminium spreads heat far beyond the band; composite
        keeps it near the band and runs much hotter.
    flap_deflection_deg: omit for no flap; positive is trailing edge down.
    flap_hinge: hinge position as a fraction of chord, 0.5 to 0.9 (default 0.7).
    alpha_start, alpha_end, alpha_step: the sweep, in degrees.

    Returns a stage-by-stage trace (aero, then thermal) and a per-alpha table
    with lift coefficient, upper-surface transition, coldest temperature in
    the heater, and ice-free extent on each side. If aero fails, thermal is
    not run and the trace says why.

    The thermal worst_case takes the worst of each field across alphas, so
    its numbers can come from different angles. Coverage is converged alphas
    over requested: below 1.0, the hardest angles are missing from the worst
    case. To compare a flapped and unflapped design fairly, compare at equal
    lift coefficient (cl in per_alpha), not equal angle: a flap makes lift
    without raising the nose, which keeps transition further back.

    Model limits to state when reporting: 2D, steady, dry air (no droplets
    or evaporation, so it underestimates the power a real heater needs).
    """
    try:
        case = Case.model_validate({
            "geometry": {
                "naca": airfoil,
                "flap": None if flap_deflection_deg is None
                        else {"x_hinge": flap_hinge, "deflection": flap_deflection_deg},
            },
            "conditions": {"reynolds": reynolds, "alpha_start": alpha_start,
                           "alpha_end": alpha_end, "alpha_step": alpha_step},
            "outputs": ["forces", "bl", "geometry"],
            "thermal": {
                "chord_m": chord_m,
                "air_temperature_k": air_temperature_c + 273.15,
                "heater_width": heater_width,
                "heater_power_w_per_m": heater_power_w_per_m,
                "skin_thickness_m": skin_thickness_mm / 1000.0,
                "skin_conductivity_w_mk": skin_conductivity_w_mk,
            },
        })
    except ValidationError as exc:
        return _error("input", _validation_errors(exc, _ARGUMENT_NAMES), content_hash=None, stages=[], per_alpha=[])
    result = harness.evaluate(case)
    _designs[case.content_hash()] = result
    return _design_response(case, result)



@mcp.tool
def request_print(content_hash: str, chord_mm: float = 150.0, span_mm: float = 40.0) -> dict:
    """Prepare a printed section of a design and ask the user to approve it.

    content_hash comes from run_polar or evaluate_design (or from a campaign
    whose cases included "geometry" in outputs). To print an airfoil that
    has not been analysed yet, call run_polar first. Builds the part from
    the exact outline XFOIL analysed, then serves a review page on the
    user's machine showing the part in 3D, its stats, and the file's hash.

    chord_mm: printed chord length in millimetres (default 150).
    span_mm: printed span, the extrusion height, in millimetres (default 40).
        Each must be between 1 and 250 mm to fit the printer.

    This does NOT print anything. Give the user the url and ask them to
    review and approve the part there. Only the user can approve, on that
    page; never say it is approved unless they tell you so. When they
    confirm, call start_print with the request_id.
    """
    result = _find_design(content_hash)
    if result is None:
        return _error("input", [f"no evaluated design {content_hash!r} in this session"])
    geometry = (result.data or {}).get("geometry")
    if not geometry:
        return _error("input", ["this design has no geometry; run it with run_polar or "
                                "evaluate_design, or include \"geometry\" in the campaign's outputs"])

    from xfoil_mcp.cad import CadError, build_section
    try:
        part = build_section([tuple(p) for p in geometry], chord_mm=chord_mm, span_mm=span_mm)
    except CadError as exc:
        return _error("input", [str(exc)])

    case = result.case
    flap = case.geometry.flap
    thermal = (result.summary or {}).get("thermal") or {}
    design = {
        "label": case.label or f"NACA {case.geometry.naca}",
        "airfoil": case.geometry.naca,
        "flap": _flap_text(flap),
        "reynolds": f"{case.conditions.reynolds:g}",
        "coldest_heater_temperature_c": (thermal.get("worst_case") or {}).get("min_heated_temperature_c"),
        "design_hash": content_hash[:12],
    }
    try:
        queue, base = _printing()
    except (OSError, ValueError) as exc:
        return _print_unavailable(exc)
    request = queue.create(design, part)
    return _ok(request_id=request.id, url=f"{base}/print/{request.id}",
               sha256=part.sha256[:12], stats=part.stats, print_status=request.status)


@mcp.tool
def start_print(request_id: str) -> dict:
    """Send an approved print request to the print backend.

    Refuses unless the user has approved this request on its review page,
    for the exact file shown there. If it is still awaiting approval, ask
    the user to approve it at the url from request_print, then try again.
    An approval is used once and expires after 30 minutes.

    With the dry backend (the default) this writes the approved STL and a
    record of the approval to the outbox folder. With the printer backend
    it slices the part, uploads it and starts a physical print, then
    returns the slicer's time and weight estimates and the printer's state.
    If a send fails, the request is closed: request the print again once
    the cause is fixed, and the user approves the new request.
    """
    from xfoil_mcp.printing import PrintRefused

    try:
        queue, _ = _printing()
    except (OSError, ValueError) as exc:
        return _print_unavailable(exc, request_id=request_id, print_status=None)
    try:
        sent = queue.start(request_id)
    except PrintRefused as exc:
        try:
            status = queue.get(request_id).status
        except PrintRefused:
            status = None
        return _error(exc.failure_kind, [str(exc)], request_id=request_id, print_status=status)
    return _ok(request_id=request_id, print_status="sent", **sent)

@mcp.tool
def dry_run_campaign(source: str) -> dict:
    """Execute a campaign script in an isolated sandbox and report what it
    WOULD run. Runs no solver. Always call this before run_campaign.

    A campaign is Python source defining `campaign() -> list[Case]`. It may
    import only xfoil_mcp.schema (Case, Geometry, Flap, Conditions, Thermal),
    numpy, and scipy. Use scipy.stats.qmc for Latin hypercube or Sobol sampling
    rather than writing a sampler by hand, and seed it. The sandbox has no
    network and a 30 second limit.

    Each Case can set outputs: "forces" (default), "bl", "cp", "geometry".
    Add "geometry" to any case you might want to print (request_print needs
    it). A Case with a thermal=Thermal(...) block runs the heater analysis
    after XFOIL and requires "bl" in outputs. Thermal cases cost two to three
    times as much as aero-only; screen with aero first, add thermal to a
    shortlist.

    Example:

        from xfoil_mcp.schema import Case, Conditions, Geometry
        def campaign():
            return [
                Case(geometry=Geometry(naca=n),
                     conditions=Conditions(reynolds=5e5, alpha_start=0,
                                           alpha_end=12, alpha_step=1),
                     outputs=("forces", "geometry"),
                     label=n)
                for n in ("2412", "4412", "0012")
            ]

    On success returns the deduplicated case list, an estimate (case count,
    alpha points, expected seconds), and an approval_hash. Show the estimate
    to the user and get agreement before calling run_campaign with that
    hash. On failure returns the errors, the cases that did validate (so the
    fix can be local), and no approval_hash. A failure_kind of "input" means
    the campaign code needs changing; "infrastructure" means the sandbox
    could not run and a retry may help.
    """
    report = harness.dry_run(source)

    if not report.runnable:
        errors = list(report.errors) + list(report.rejected)
        if report.killed:
            errors.append("campaign was killed by the sandbox timeout; it is probably looping")
        if not report.cases and not errors:
            errors.append("campaign() returned no cases")
        return _error(
            "infrastructure" if report.infrastructure else "input",
            errors,
            cases=_case_view(report.cases),
            approval_hash=None,
            estimate=None,
        )

    return _ok(
        approval_hash=report.approval_hash,
        estimate=report.estimate.__dict__,
        cases=_case_view(report.cases),
    )


@mcp.tool
def run_campaign(source: str, approval_hash: str) -> dict:
    """Run a campaign that was approved via dry_run_campaign.

    approval_hash must be the value dry_run_campaign returned for this exact
    source. If the source changed, the hash will not match and this refuses;
    dry-run again. Returns one summary per case, never full polar tables;
    use get_case_detail for those.

    status is "ok" only when every case passed every stage it asked for,
    "partial" when some did, and "error" when none produced a usable result.
    Read `complete` and `counts` before drawing conclusions: counts are of
    each case's overall status, so a case whose aero passed and whose heater
    analysis failed counts as an error. A case with failure_kind "numerical"
    will fail again if rerun identically; only a changed approach (smaller
    step, different range) can help. A case with failure_kind
    "infrastructure" may succeed on retry.
    """
    global _last_batch
    refused = dict(complete=False, submitted=0, counts={}, results=[])
    try:
        batch = harness.submit(source, approval_hash)
    except harness.CampaignNotRunnable as exc:
        return _error(exc.failure_kind, [str(exc)], **refused)
    except harness.ApprovalMismatch as exc:
        return _error("input", [str(exc)], **refused)

    _last_batch = batch
    counts = batch.counts
    fields = dict(complete=batch.complete, submitted=batch.submitted, counts=counts,
                  results=batch.summaries())
    if counts["ok"] == batch.submitted:
        return _ok(**fields)
    if counts["ok"] + counts["partial"] == 0:
        return _error(batch.failure_kind,
                      ["no case produced a usable result; each case's status and error are in results"],
                      **fields)
    return _envelope("partial", batch.failure_kind, [], **fields)


@mcp.tool
def get_case_detail(content_hash: str) -> dict:
    """Fetch the full result for one case from the most recent campaign,
    including every converged polar point. Large; call only when the
    summary is not enough (for example, to plot the polar or inspect the
    lift curve near stall). content_hash comes from a run_campaign summary.
    """
    if _last_batch is None:
        return _error("input", ["no campaign has been run in this session"], result=None)
    result = _last_batch.get(content_hash)
    if result is None:
        return _error("input", [f"no case {content_hash} in the most recent campaign"], result=None)
    return _ok(result=result.model_dump(mode="json"))


if __name__ == "__main__":
    mcp.run()
