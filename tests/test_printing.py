"""The print queue: the rules for when a part can be printed. Host-side."""

from pathlib import Path

import pytest

pytest.importorskip("cadquery")

from xfoil_mcp.cad import build_section
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
