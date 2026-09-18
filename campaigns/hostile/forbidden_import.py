"""Must fail: the harness is not installed in the sandbox image."""
from xfoil_mcp import harness  # noqa: F401
from xfoil_mcp.schema import Case

def campaign() -> list[Case]:
    return []
