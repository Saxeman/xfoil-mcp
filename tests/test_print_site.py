"""The approval page, tested over real HTTP on a free local port."""

import http.client
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


@pytest.fixture(scope="module")
def running_site(tmp_path_factory):
    """One server for the module: stopping it takes half a second each time."""
    queue = PrintQueue(DryBackend(tmp_path_factory.mktemp("outbox")))
    server, base = start_site(queue, port=0)       # port 0: the OS picks a free one
    yield queue, base
    server.shutdown()
    server.server_close()


@pytest.fixture
def site(running_site, part):
    """A fresh pending request on the running site."""
    queue, base = running_site
    return queue, queue.create({"label": "2412, flap 10 deg"}, part), base


# No proxies: urllib would otherwise send requests for 127.0.0.1 through
# whatever HTTP_PROXY names, and the tests would depend on the environment.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url):
    try:
        with DIRECT.open(url) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def post(url, body, content_type="application/json"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with DIRECT.open(req) as r:
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


# --- requests the page itself never sends ------------------------------------

def raw_post(base, path, body: bytes, headers: dict) -> tuple[int, dict]:
    """POST exactly these bytes and headers. http.client adds nothing we did not ask for."""
    host, port = base.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    try:
        conn.putrequest("POST", path)
        for name, value in headers.items():
            conn.putheader(name, value)
        conn.endheaders()
        conn.send(body)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def json_headers(body: bytes) -> dict:
    return {"Content-Type": "application/json", "Content-Length": str(len(body))}


def test_a_text_plain_post_cannot_approve(site, part):
    """The other content type a browser will send cross-site without asking."""
    queue, request, base = site
    code, body = post(f"{base}/print/{request.id}/approve", {"sha256": part.sha256}, content_type="text/plain")
    assert code == 415 and "JSON" in body["error"]
    assert queue.get(request.id).status == "pending"


def test_a_json_content_type_with_a_charset_is_still_json(site, part):
    queue, request, base = site
    code, _ = post(f"{base}/print/{request.id}/approve", {"sha256": part.sha256},
                   content_type="application/json; charset=utf-8")
    assert code == 200 and queue.get(request.id).status == "approved"


def test_a_body_that_is_not_json_is_a_400(site):
    queue, request, base = site
    body = b"sha256=abc"
    code, payload = raw_post(base, f"/print/{request.id}/approve", body, json_headers(body))
    assert code == 400 and payload == {"error": "body was not JSON"}
    assert queue.get(request.id).status == "pending"


def test_an_approval_without_a_hash_is_refused(site):
    queue, request, base = site
    code, body = post(f"{base}/print/{request.id}/approve", {})
    assert code == 409 and "not the file" in body["error"]
    assert queue.get(request.id).status == "pending"


def test_a_second_decision_is_refused_with_the_reason(site, part):
    queue, request, base = site
    assert post(f"{base}/print/{request.id}/approve", {"sha256": part.sha256})[0] == 200
    code, body = post(f"{base}/print/{request.id}/reject", {})
    assert code == 409 and body == {"error": "request is approved, not pending"}
    assert queue.get(request.id).status == "approved"


@pytest.mark.parametrize("path", ["/print/{id}/print", "/print/{id}", "/print/{id}/approve/extra", "/elsewhere"])
def test_a_post_to_anything_but_approve_or_reject_is_404(site, path):
    queue, request, base = site
    code, _ = post(base + path.format(id=request.id), {})
    assert code == 404
    assert queue.get(request.id).status == "pending"


def test_a_decision_on_an_unknown_request_is_refused(site, part):
    _, _, base = site
    code, body = post(f"{base}/print/nope/approve", {"sha256": part.sha256})
    assert code in (404, 409) and "no print request" in body["error"]


def test_an_unknown_page_under_a_request_is_404(site):
    _, request, base = site
    assert get(f"{base}/print/{request.id}/secrets")[0] == 404
    assert get(f"{base}/")[0] == 404


def test_the_download_carries_the_request_and_hash_in_its_filename(site, part):
    _, request, base = site
    _, headers, _ = get(f"{base}/print/{request.id}/part.stl")
    assert headers["Content-Disposition"] == f'attachment; filename="{request.id}_{part.sha256[:12]}.stl"'


def test_text_from_the_agent_is_escaped_on_the_page(running_site, part):
    """The label is written by the agent. It must not become markup on the
    page where a person approves a print."""
    queue, base = running_site
    hostile = '<script>document.getElementById("approve").click()</script>'
    request = queue.create({"label": hostile, "<b>key</b>": "v"}, part)
    page = get(f"{base}/print/{request.id}")[2].decode()
    assert hostile not in page and "<b>key</b>" not in page
    assert "&lt;script&gt;document.getElementById(&quot;approve&quot;).click()&lt;/script&gt;" in page


def test_a_decided_request_shows_its_buttons_disabled(site, part):
    queue, request, base = site
    assert 'id="approve" disabled' not in get(f"{base}/print/{request.id}")[2].decode()
    queue.approve(request.id, part.sha256)
    page = get(f"{base}/print/{request.id}")[2].decode()
    assert 'id="approve" disabled' in page and 'id="reject" disabled' in page
    assert "Status: approved" in page


def test_site_writes_nothing_to_stderr_either(site, part, capfd):
    """http.server logs every request to stderr by default. Stdout is the
    protocol stream; stderr is the server's log, and should not fill with page hits."""
    _, request, base = site
    get(f"{base}/print/{request.id}")
    post(f"{base}/print/{request.id}/approve", {"sha256": part.sha256})
    captured = capfd.readouterr()
    assert (captured.out, captured.err) == ("", "")


@pytest.mark.parametrize("body", [b"[1, 2]", b'"approve"', b"7", b"null"])
def test_a_json_body_that_is_not_an_object_is_a_400(site, body):
    queue, request, base = site
    code, _ = raw_post(base, f"/print/{request.id}/approve", body, json_headers(body))
    assert code == 400
    assert queue.get(request.id).status == "pending"


def test_a_content_length_that_is_not_a_number_is_a_400(site):
    _, request, base = site
    code, _ = raw_post(base, f"/print/{request.id}/approve", b"{}",
                       {"Content-Type": "application/json", "Content-Length": "abc"})
    assert code == 400


@pytest.mark.parametrize("length", ["-1", "4097", "99999999"])
def test_a_content_length_out_of_range_is_a_400_and_nothing_is_read(site, length):
    """A negative length would make the handler read until the client hung
    up, and a huge one would make it wait for bytes that never come."""
    queue, request, base = site
    code, body = raw_post(base, f"/print/{request.id}/approve", b"{}",
                          {"Content-Type": "application/json", "Content-Length": length})
    assert code == 400 and "at most 4096 bytes" in body["error"]
    assert queue.get(request.id).status == "pending"


def test_a_body_that_is_not_utf8_is_a_400(site):
    queue, request, base = site
    body = b"\xff\xfe{}"
    code, payload = raw_post(base, f"/print/{request.id}/approve", body, json_headers(body))
    assert code == 400 and payload == {"error": "body was not JSON"}
