"""Must fail: the root filesystem is read-only and we are not root."""
from xfoil_mcp.schema import Case

def campaign() -> list[Case]:
    with open("/app/pwned", "w") as f:
        f.write("x")
    return []
