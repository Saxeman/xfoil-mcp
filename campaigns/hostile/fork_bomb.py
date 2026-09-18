"""Must hit the pid limit rather than take the host down."""
import os
from xfoil_mcp.schema import Case

def campaign() -> list[Case]:
    children = 0
    while children < 500:
        os.fork()
        children += 1
    return []
