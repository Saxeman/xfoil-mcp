"""The approval page, tested over real HTTP on a free local port."""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("cadquery")

from xfoil_mcp.cad import build_section
from xfoil_mcp.print_site import start_site
from xfoil_mcp.printing import DryBackend, PrintQueue

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def part():
    lines = (FIXTURES / "naca2412_flap_10.dat").read_text().splitlines()
    return build_section([tuple(float(v) for v in l.split()) for l in lines if l.strip()])


@pytest.fixture
def site(tmp_path, part):
    queue = PrintQueue(DryBackend(tmp_path / "outbox"))
    request = queue.create({"label": "2412, flap 10 deg"}, part)
    server, base = start_site(queue, port=0)       # port 0: the OS picks a free one
    yield queue, request, base
    server.shutdown()
    server.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def post(url, body, content_type="application/json"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_shows_the_request_and_its_hash(site, part):
    _, request, base = site
    code, _, body = get(f"{base}/print/{request.id}")
    text = body.decode()
    assert code == 200
    assert part.sha256 in text
    assert "2412, flap 10 deg" in text
    assert "pending" in text


def test_download_is_exactly_the_file(site, part):
    _, request, base = site
    code, headers, body = get(f"{base}/print/{request.id}/part.stl")
    assert code == 200
    assert body == part.stl
    assert "attachment" in headers["Content-Disposition"]


def test_approving_with_the_displayed_hash_approves(site, part):
    queue, request, base = site
    code, body = post(f"{base}/print/{request.id}/approve", {"sha256": part.sha256})
    assert code == 200 and body["status"] == "approved"
    assert queue.get(request.id).status == "approved"


def test_approving_a_different_file_is_refused(site):
    queue, request, base = site
    code, _ = post(f"{base}/print/{request.id}/approve", {"sha256": "0" * 64})
    assert code == 409
    assert queue.get(request.id).status == "pending"


def test_a_plain_form_post_cannot_approve(site, part):
    """What another website could send from your browser. Must not count."""
    queue, request, base = site
    code, _ = post(f"{base}/print/{request.id}/approve", {"sha256": part.sha256},
                   content_type="application/x-www-form-urlencoded")
    assert code == 415
    assert queue.get(request.id).status == "pending"


def test_reject(site):
    queue, request, base = site
    code, _ = post(f"{base}/print/{request.id}/reject", {})
    assert code == 200
    assert queue.get(request.id).status == "rejected"


def test_unknown_request_is_404(site):
    _, _, base = site
    assert get(f"{base}/print/nope")[0] == 404


def test_site_never_writes_to_stdout(site, capfd):
    """Inside the MCP server, stdout is the protocol stream."""
    _, request, base = site
    get(f"{base}/print/{request.id}")
    assert capfd.readouterr().out == ""
