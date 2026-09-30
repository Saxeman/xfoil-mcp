"""Open the approval page for the flapped 2412 fixture, to look at it by hand."""

from pathlib import Path

from xfoil_mcp.cad import build_section
from xfoil_mcp.print_site import start_site
from xfoil_mcp.printing import DryBackend, PrintQueue

lines = Path("tests/fixtures/naca2412_flap_10.dat").read_text().splitlines()
part = build_section([tuple(float(v) for v in l.split()) for l in lines if l.strip()])
queue = PrintQueue(DryBackend(Path("outbox")))
request = queue.create({"label": "2412, flap 10 deg", "airfoil": "2412",
                        "flap": "10 deg at 70% chord"}, part)
server, base = start_site(queue)
print(f"Open {base}/print/{request.id}  (press Enter to stop)")
input()
print("Final status:", queue.get(request.id).status)
server.shutdown()
