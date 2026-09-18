"""Must fail: the sandbox has no network."""
import socket
from xfoil_mcp.schema import Case

def campaign() -> list[Case]:
    s = socket.create_connection(("1.1.1.1", 53), timeout=3)
    s.close()
    return []
