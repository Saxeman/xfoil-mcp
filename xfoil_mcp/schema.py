"""The contract between agent-written campaigns, the harness, and the workers.

This module is installed in every environment (host, worker, sandbox,
thermal) and is the only module present in all four. It knows nothing about
XFOIL, Docker, or MCP. It defines what a Case is, what a CaseResult is, and
refuses to construct either when the values are not physically meaningful.

Validation happens at construction. An invalid Case cannot exist, so every
consumer downstream can trust the values without re-checking them.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_ALPHA_POINTS = 200

Output = Literal["forces", "cp", "bl", "geometry"]
Status = Literal["ok", "partial", "empty", "error"]
FailureKind = Literal["input", "infrastructure", "numerical"]


class _Strict(BaseModel):
    """Reject unknown fields. A typo in a campaign should fail loudly, not be
    silently dropped."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Flap(_Strict):
    """A simple trailing-edge flap, applied through XFOIL's GDES menu.

    x_hinge and y_hinge are chord fractions. deflection is in degrees,
    positive for trailing edge down (which increases lift).
    """

    x_hinge: float = Field(ge=0.5, le=0.9)
    y_hinge: float = Field(default=0.0, ge=-0.5, le=0.5)
    deflection: float = Field(ge=-45.0, le=45.0)

class Thermal(_Strict):
    """A leading-edge heater and the flight condition it operates in.

    Evaluated once per converged alpha from the aero sweep. Units are in the
    field names so a campaign cannot confuse millimetres with metres.

    Airspeed is not a field: it follows from the Reynolds number, the chord,
    and the air (coupling.velocity_for), so the thermal model can never
    describe a flight condition XFOIL did not solve.
    """

    chord_m: float = Field(gt=0.0, le=10.0)
    air_temperature_k: float = Field(ge=200.0, le=320.0)
    pressure_pa: float = Field(default=101325.0, ge=20000.0, le=110000.0)
    heater_width: float = Field(
        gt=0.0, le=0.5,
        description="Band width along the surface, as a fraction of chord, "
                    "centred on the stagnation point",
    )
    heater_power_w_per_m: float = Field(
        gt=0.0, le=100000.0, description="Total heater power per metre of span",
    )
    skin_thickness_m: float = Field(gt=0.0, le=0.01)
    skin_conductivity_w_mk: float = Field(
        gt=0.0, le=500.0, description="About 200 for aluminium, about 1 for composite",
    )


class Geometry(_Strict):
    """What airfoil to analyze. NACA 4- and 5-digit designations only, for now."""

    naca: str
    flap: Flap | None = None

    @field_validator("naca")
    @classmethod
    def _valid_naca(cls, value: str) -> str:
        value = value.strip()
        if not value.isdigit() or len(value) not in (4, 5):
            raise ValueError(f"naca must be 4 or 5 digits, got {value!r}")
        if len(value) == 4 and int(value[2:]) == 0:
            raise ValueError("4-digit NACA thickness must be non-zero")
        return value


class Conditions(_Strict):
    """Flow conditions and sweep definition for a viscous polar."""

    reynolds: float = Field(gt=0.0, description="Chord Reynolds number, e.g. 1e6")
    mach: float = Field(default=0.0, ge=0.0, lt=1.0)
    n_crit: float = Field(
        default=9.0, ge=1.0, le=15.0,
        description="Transition criterion; 9 is a clean tunnel, ~4 is turbulent freestream",
    )
    alpha_start: float = Field(ge=-30.0, le=30.0)
    alpha_end: float = Field(ge=-30.0, le=30.0)
    alpha_step: float = Field(
        ge=0.01,
        description="Degrees. Finer than 0.01 exceeds the polar file's printed precision",
    )
    max_iter: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def _sweep_is_sane(self) -> Conditions:
        if self.alpha_end < self.alpha_start:
            raise ValueError("alpha_end must be >= alpha_start")
        if self.point_count() > MAX_ALPHA_POINTS:
            raise ValueError(
                f"sweep has {self.point_count()} points; limit is {MAX_ALPHA_POINTS}"
            )
        return self

    def point_count(self) -> int:
        # Floor, never round: rounding up adds a point past alpha_end. The
        # epsilon keeps an exact endpoint that division lands a hair under.
        return math.floor((self.alpha_end - self.alpha_start) / self.alpha_step + 1e-9) + 1

    def alphas(self) -> list[float]:
        """The sweep the wrapper runs, one ALFA per value, without float drift.

        Kept in step with wrapper._expected_alphas, which cannot import this
        module. test_schema asserts they agree.
        """
        return [
            round(self.alpha_start + i * self.alpha_step, 6)
            for i in range(self.point_count())
        ]


class Case(_Strict):
    """One solver run: a geometry, a set of conditions, and what to extract.

    `outputs` is decided here, before the run, because field data (cp, bl)
    can only be extracted while the solver session holds the solution.
    """

    geometry: Geometry
    conditions: Conditions
    outputs: tuple[Output, ...] = ("forces",)
    label: str | None = Field(default=None, max_length=80)
    thermal: Thermal | None = None

    @field_validator("outputs")
    @classmethod
    def _outputs_non_empty_unique(cls, value: tuple[Output, ...]) -> tuple[Output, ...]:
        if not value:
            raise ValueError("outputs must name at least one output")
        if len(set(value)) != len(value):
            raise ValueError("outputs must not repeat")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _thermal_needs_boundary_layer(self) -> Case:
        """The heat transfer model is built from XFOIL's skin friction and
        edge velocity, which only exist when the case asks for them."""
        if self.thermal is not None and "bl" not in self.outputs:
            raise ValueError("thermal requires 'bl' in outputs")
        return self

    def content_hash(self) -> str:
        """Stable identity for dedup and provenance.

        Excludes `label`: two cases that would produce identical solver runs
        must collide regardless of what the agent called them.
        """
        payload = self.model_dump(exclude={"label"}, mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def alphas(self) -> list[float]:
        return self.conditions.alphas()


class CaseResult(_Strict):
    """What a worker returns for one Case.

    Lives here rather than in the harness because the worker constructs it
    and the host deserializes it. Both need the same definition.
    """

    case: Case
    status: Status
    failure_kind: FailureKind | None = None
    summary: dict = Field(default_factory=dict)
    data: dict | None = None
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_and_kind_agree(self) -> CaseResult:
        if self.status == "ok" and self.failure_kind is not None:
            raise ValueError("an ok result cannot carry a failure_kind")
        if self.status in ("partial", "empty") and self.failure_kind != "numerical":
            raise ValueError(f"{self.status} results are numerical failures")
        if self.status == "error" and self.failure_kind is None:
            raise ValueError("error results must say what kind of failure")
        return self

    @property
    def retry_could_help(self) -> bool:
        """Only infrastructure failures are worth retrying identically.

        Numerical failures are deterministic: the same call gives the same
        failure. Input failures will never succeed.
        """
        return self.failure_kind == "infrastructure"
