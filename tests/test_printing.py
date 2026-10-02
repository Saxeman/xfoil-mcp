"""The print queue: the rules for when a part can be printed. Host-side."""

import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("cadquery")

from xfoil_mcp.cad import build_section
from xfoil_mcp import printing
from xfoil_mcp.printing import DryBackend, PrintQueue, PrintRefused

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def part():
    lines = (FIXTURES / "naca2412_flap_10.dat").read_text().splitlines()
    return build_section([tuple(float(v) for v in l.split()) for l in lines if l.strip()])


class FakeClock:
    """A clock the test can move forward instead of waiting."""
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup(tmp_path, part):
    clock = FakeClock()
    queue = PrintQueue(DryBackend(tmp_path / "outbox"), clock=clock, ttl_s=600)
    request = queue.create({"label": "2412, flap 10 deg"}, part)
    return queue, request, clock


def test_a_new_request_is_pending(setup):
    _, request, _ = setup
    assert request.status == "pending"


def test_nothing_prints_without_approval(setup):
    queue, request, _ = setup
    with pytest.raises(PrintRefused, match="awaiting approval"):
        queue.start(request.id)


def test_approval_must_name_the_file_in_the_request(setup):
    queue, request, _ = setup
    with pytest.raises(PrintRefused, match="not the file"):
        queue.approve(request.id, "0" * 64)
    assert queue.get(request.id).status == "pending"


def test_an_approved_request_sends_exactly_the_approved_bytes(setup, part):
    queue, request, _ = setup
    queue.approve(request.id, part.sha256)
    result = queue.start(request.id)
    assert Path(result["stl_path"]).read_bytes() == part.stl
    assert queue.get(request.id).status == "sent"


def test_an_approval_is_used_once(setup, part):
    queue, request, _ = setup
    queue.approve(request.id, part.sha256)
    queue.start(request.id)
    with pytest.raises(PrintRefused, match="sent"):
        queue.start(request.id)


def test_a_rejected_request_cannot_be_printed(setup):
    queue, request, _ = setup
    queue.reject(request.id)
    with pytest.raises(PrintRefused, match="rejected"):
        queue.start(request.id)


def test_an_approval_expires(setup, part):
    queue, request, clock = setup
    queue.approve(request.id, part.sha256)
    clock.now += 601
    with pytest.raises(PrintRefused, match="expired"):
        queue.start(request.id)


def test_an_unknown_request_is_refused(setup):
    queue, _, _ = setup
    with pytest.raises(PrintRefused, match="no print request"):
        queue.start("nope")


# --- the rest of the state machine -------------------------------------------

def test_bytes_that_changed_after_approval_are_not_sent(tmp_path, part):
    """The approval names a hash. If the bytes no longer match it at send
    time, nothing goes to the backend and the request is spent."""
    tampered = dataclasses.replace(part, stl=part.stl + b"extra")       # same hash, different bytes
    queue = PrintQueue(DryBackend(tmp_path / "outbox"))
    request = queue.create({"label": "tampered"}, tampered)
    queue.approve(request.id, part.sha256)
    with pytest.raises(PrintRefused, match="no longer matches its hash"):
        queue.start(request.id)
    assert queue.get(request.id).status == "failed"
    assert not (tmp_path / "outbox").exists()


class BrokenBackend:
    name = "broken"

    def send(self, request):
        raise OSError("printer unreachable")


def test_a_backend_failure_fails_the_request_and_says_which_backend(part):
    queue = PrintQueue(BrokenBackend())
    request = queue.create({"label": "x"}, part)
    queue.approve(request.id, part.sha256)
    with pytest.raises(PrintRefused, match="the broken backend failed: printer unreachable"):
        queue.start(request.id)
    assert queue.get(request.id).status == "failed"
    with pytest.raises(PrintRefused, match="failed; it cannot be printed"):
        queue.start(request.id)                                         # and it stays failed


@pytest.mark.parametrize("first", ["approve", "reject"])
@pytest.mark.parametrize("second", ["approve", "reject"])
def test_a_decision_cannot_be_made_twice_or_reversed(setup, part, first, second):
    queue, request, _ = setup
    decide = {"approve": lambda: queue.approve(request.id, part.sha256),
              "reject": lambda: queue.reject(request.id)}
    decide[first]()
    settled = queue.get(request.id).status
    with pytest.raises(PrintRefused, match=f"request is {settled}, not pending"):
        decide[second]()
    assert queue.get(request.id).status == settled


def test_an_approval_is_good_up_to_and_including_its_time_limit(setup, part):
    queue, request, clock = setup                   # ttl 600 s
    queue.approve(request.id, part.sha256)
    clock.now += 600
    assert queue.start(request.id)["backend"] == "dry"


def test_an_approval_one_second_past_its_limit_has_expired(setup, part):
    queue, request, clock = setup
    queue.approve(request.id, part.sha256)
    clock.now += 601
    with pytest.raises(PrintRefused, match="expired"):
        queue.start(request.id)
    assert queue.get(request.id).status == "expired"
    with pytest.raises(PrintRefused, match="expired; it cannot be printed"):
        queue.start(request.id)                                         # expiry is final


def test_the_default_time_limit_is_thirty_minutes(tmp_path, part):
    """The tool description promises 30 minutes."""
    assert printing.APPROVAL_TTL_S == 30 * 60
    clock = FakeClock()
    queue = PrintQueue(DryBackend(tmp_path / "outbox"), clock=clock)
    request = queue.create({"label": "x"}, part)
    queue.approve(request.id, part.sha256)
    clock.now += 30 * 60 + 1
    with pytest.raises(PrintRefused, match="expired"):
        queue.start(request.id)


def test_the_outbox_record_names_the_design_the_hash_and_the_approval_time(setup, part):
    queue, request, clock = setup
    queue.approve(request.id, part.sha256)
    stl_path = Path(queue.start(request.id)["stl_path"])
    record = json.loads(stl_path.with_suffix(".json").read_text())
    assert stl_path.name == f"{request.id}_{part.sha256[:12]}.stl"
    assert (record["request_id"], record["sha256"], record["approved_at"]) == (request.id, part.sha256, clock.now)
    assert record["design"] == {"label": "2412, flap 10 deg"}


def test_every_request_gets_its_own_id(setup, part):
    queue, request, _ = setup
    assert queue.create({"label": "again"}, part).id != request.id
