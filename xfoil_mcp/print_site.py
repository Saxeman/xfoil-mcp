"""The local page where a person reviews and approves a print.

Serves one page per print request, bound to localhost only: the part in 3D,
the design it came from, its stats, and the hash of the exact file, with
Approve, Reject, and Download STL. Approving sends back the hash the page
displayed, so the approval names the file the person looked at. The rules
live in PrintQueue; this only shows requests and passes decisions on.

Never writes to stdout: the MCP server's stdout is its protocol stream.
"""

from __future__ import annotations

import html
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from xfoil_mcp.printing import PrintQueue, PrintRefused

log = logging.getLogger(__name__)

THREE = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
STL_LOADER = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"

MAX_BODY_BYTES = 4096       # a decision is {"sha256": 64 hex characters}; nothing larger is read


def _rows(items: dict) -> str:
    e = html.escape
    return "".join(f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in items.items())


def _page(request) -> str:
    """The review page. Every value from the request is escaped before it goes into HTML."""
    e = html.escape
    sha = request.part.sha256
    decided = "disabled" if request.status != "pending" else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Print request {e(request.id)}</title>
<style>
 body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
 #view {{ width: 100%; height: 360px; background: #f3f3f1; border-radius: 8px; }}
 table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; }} td {{ padding: 3px 16px 3px 0; }}
 code {{ font-size: 13px; word-break: break-all; }}
 button {{ font-size: 15px; padding: 8px 18px; margin-right: 8px; }}
 #status {{ margin-top: 1rem; font-weight: 500; }}
</style></head><body>
<h1>Print request {e(request.id)}</h1>
<div id="view"></div>
<h2>Design</h2><table>{_rows(request.design)}</table>
<h2>Part</h2><table>{_rows(request.part.stats)}</table>
<p>File hash (SHA-256): <code>{e(sha)}</code></p>
<p><button id="approve" {decided}>Approve</button>
<button id="reject" {decided}>Reject</button>
<a href="/print/{e(request.id)}/part.stl">Download STL</a></p>
<div id="status">Status: {e(request.status)}</div>
<script src="{THREE}"></script>
<script src="{STL_LOADER}"></script>
<script>
const id = {json.dumps(request.id)}, sha = {json.dumps(sha)};

async function decide(action) {{
  const r = await fetch(`/print/${{id}}/${{action}}`, {{
    method: "POST", headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{sha256: sha}})}});
  const body = await r.json();
  document.getElementById("status").textContent =
    r.ok ? `Status: ${{body.status}}` : `Refused: ${{body.error}}`;
  if (r.ok) document.querySelectorAll("button").forEach(b => b.disabled = true);
}}
document.getElementById("approve").onclick = () => decide("approve");
document.getElementById("reject").onclick = () => decide("reject");

try {{
  const el = document.getElementById("view");
  const renderer = new THREE.WebGLRenderer({{antialias: true}});
  renderer.setSize(el.clientWidth, el.clientHeight);
  renderer.setClearColor(0xf3f3f1);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, el.clientWidth / el.clientHeight, 1, 2000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7);
  sun.position.set(80, 120, 160);
  scene.add(sun);
  new THREE.STLLoader().load(`/print/${{id}}/part.stl`, geo => {{
    geo.center();
    scene.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({{color: 0x1d9e75}})));
    let yaw = -0.8, pitch = 0.4, drag = null;
    const place = () => {{
      camera.position.set(260 * Math.cos(pitch) * Math.sin(yaw), 260 * Math.sin(pitch),
                          260 * Math.cos(pitch) * Math.cos(yaw));
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    }};
    el.onpointerdown = ev => drag = {{x: ev.clientX, y: ev.clientY, yaw, pitch}};
    window.onpointerup = () => drag = null;
    el.onpointermove = ev => {{
      if (!drag) return;
      yaw = drag.yaw - (ev.clientX - drag.x) * 0.008;
      pitch = Math.max(-1.3, Math.min(1.3, drag.pitch + (ev.clientY - drag.y) * 0.008));
      place();
    }};
    place();
  }});
}} catch (err) {{
  document.getElementById("view").textContent =
    "3D preview unavailable; download the STL to inspect it.";
}}
</script></body></html>"""


def _handler(queue: PrintQueue):
    """Build a request handler class bound to this queue."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # The default writes every request to the terminal. Not in this process.
            log.debug("print site: " + fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str, headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _route(self) -> tuple[str, str] | None:
            """'/print/<id>' or '/print/<id>/<action>', as (id, action)."""
            parts = self.path.strip("/").split("/")
            if len(parts) in (2, 3) and parts[0] == "print":
                return parts[1], parts[2] if len(parts) == 3 else ""
            return None

        def do_GET(self):
            route = self._route()
            if route is None:
                return self._json(404, {"error": "not found"})
            request_id, action = route
            try:
                request = queue.get(request_id)
            except PrintRefused as exc:
                return self._json(404, {"error": str(exc)})
            if action == "":
                return self._send(200, _page(request).encode(), "text/html; charset=utf-8")
            if action == "part.stl":
                filename = f"{request.id}_{request.part.sha256[:12]}.stl"
                return self._send(200, request.part.stl, "model/stl",
                                  {"Content-Disposition": f'attachment; filename="{filename}"'})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            route = self._route()
            if route is None or route[1] not in ("approve", "reject"):
                return self._json(404, {"error": "not found"})
            # Browsers let any website submit a plain form to localhost, but not
            # a JSON request. Requiring JSON means only this page can decide.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json(415, {"error": "decisions must be sent as JSON"})
            # Anything but a small JSON object is refused with a reply. Left
            # to raise, a bad header or body closes the connection without one.
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "Content-Length was not a number"})
            if not 0 <= length <= MAX_BODY_BYTES:
                return self._json(400, {"error": f"body must be at most {MAX_BODY_BYTES} bytes"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:              # not JSON, or not UTF-8
                return self._json(400, {"error": "body was not JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "body must be a JSON object"})

            request_id, action = route
            try:
                if action == "approve":
                    request = queue.approve(request_id, str(body.get("sha256", "")))
                else:
                    request = queue.reject(request_id)
            except PrintRefused as exc:
                return self._json(409, {"error": str(exc)})
            return self._json(200, {"status": request.status})

    return Handler


def start_site(queue: PrintQueue, port: int = 8765) -> tuple[ThreadingHTTPServer, str]:
    """Serve the approval pages on localhost in a background thread.

    Returns the server (call .shutdown() to stop it) and its base URL.
    port=0 lets the operating system pick a free port.
    """
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(queue))
    threading.Thread(target=server.serve_forever, daemon=True, name="print-site").start()
    host, bound_port = server.server_address[:2]
    return server, f"http://{host}:{bound_port}"
