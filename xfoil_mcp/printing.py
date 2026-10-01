"""The print queue: when a part may be printed, and who decides.

A request moves pending -> approved -> sent, or pending -> rejected. An
approval can expire, and a request whose file or backend fails at send time
ends as failed. Approval names the file by its hash, is used once, and the
bytes are re-hashed at send time. The web page and the MCP tools both call
into this; the rules live only here.

Backends do the sending. DryBackend writes to an outbox folder; the printer
backend replaces only that.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from xfoil_mcp.cad import SectionPart

APPROVAL_TTL_S = 30 * 60


class PrintRefused(Exception):
    """A request cannot move to the state asked for. The message says why."""


@dataclass
class PrintRequest:
    id: str
    design: dict                    # what was designed, for the page and the record
    part: SectionPart               # the exact bytes to print, with their hash
    created_at: float
    status: str = "pending"         # pending, approved, rejected, sent, expired, failed
    approved_at: float | None = None
    result: dict | None = None      # what the backend reported


class Backend(Protocol):
    name: str

    def send(self, request: PrintRequest) -> dict: ...


class DryBackend:
    """Writes the approved file and a record of it to an outbox folder."""

    name = "dry"

    def __init__(self, outbox: Path):
        self.outbox = Path(outbox)

    def send(self, request: PrintRequest) -> dict:
        self.outbox.mkdir(parents=True, exist_ok=True)
        stem = f"{request.id}_{request.part.sha256[:12]}"
        stl_path = self.outbox / f"{stem}.stl"
        stl_path.write_bytes(request.part.stl)
        (self.outbox / f"{stem}.json").write_text(json.dumps({
            "request_id": request.id,
            "sha256": request.part.sha256,
            "design": request.design,
            "stats": request.part.stats,
            "approved_at": request.approved_at,
        }, indent=2))
        return {"backend": self.name, "stl_path": str(stl_path)}


class PrintQueue:
    """Holds print requests and enforces the rules for moving between states.

    Thread-safe: the web page and the MCP server call in from different
    threads. The clock is injectable so tests can move time forward.
    """

    def __init__(self, backend: Backend, clock: Callable[[], float] = time.time,
                 ttl_s: float = APPROVAL_TTL_S):
        self._backend = backend
        self._clock = clock
        self._ttl = ttl_s
        self._requests: dict[str, PrintRequest] = {}
        self._lock = threading.Lock()

    def create(self, design: dict, part: SectionPart) -> PrintRequest:
        request = PrintRequest(id=uuid.uuid4().hex[:12], design=design, part=part,
                               created_at=self._clock())
        with self._lock:
            self._requests[request.id] = request
        return request

    def get(self, request_id: str) -> PrintRequest:
        with self._lock:
            return self._find(request_id)

    def approve(self, request_id: str, sha256: str) -> PrintRequest:
        """A person approves the file they were shown, named by its hash."""
        with self._lock:
            request = self._find(request_id)
            if request.status != "pending":
                raise PrintRefused(f"request is {request.status}, not pending")
            if sha256 != request.part.sha256:
                raise PrintRefused("the approved file is not the file in this request")
            request.status = "approved"
            request.approved_at = self._clock()
            return request

    def reject(self, request_id: str) -> PrintRequest:
        with self._lock:
            request = self._find(request_id)
            if request.status != "pending":
                raise PrintRefused(f"request is {request.status}, not pending")
            request.status = "rejected"
            return request

    def start(self, request_id: str) -> dict:
        """Send an approved request to the backend, once."""
        with self._lock:
            request = self._find(request_id)
            if request.status == "pending":
                raise PrintRefused("awaiting approval: a person has to approve this on the print page")
            if request.status != "approved":
                raise PrintRefused(f"request is {request.status}; it cannot be printed")
            if self._clock() - request.approved_at > self._ttl:
                request.status = "expired"
                raise PrintRefused("the approval expired; request the print again")
            if hashlib.sha256(request.part.stl).hexdigest() != request.part.sha256:
                request.status = "failed"
                raise PrintRefused("the file no longer matches its hash")
            request.status = "sent"        # claimed before sending, so it can't be sent twice

        try:
            request.result = self._backend.send(request)
        except Exception as exc:
            request.status = "failed"
            raise PrintRefused(f"the {self._backend.name} backend failed: {exc}") from exc
        return request.result

    def _find(self, request_id: str) -> PrintRequest:
        """Look up a request. Callers hold the lock."""
        request = self._requests.get(request_id)
        if request is None:
            raise PrintRefused(f"no print request {request_id!r}")
        return request
