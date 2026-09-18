"""Reference campaign: Latin hypercube over NACA 4-digit parameters.

Samples (max camber %, camber position tenths, thickness %) with a seeded
LHS, at fixed conditions. This is the shape of thing an agent would write
when asked to explore a design space rather than a handful of named
airfoils. Uses scipy's sampler rather than a hand-rolled one.
"""

import numpy as np
from scipy.stats import qmc

from xfoil_mcp.schema import Case, Conditions, Geometry

N_SAMPLES = 12
SEED = 42

# camber 0-6 %, camber position 2-6 (tenths of chord), thickness 8-18 %
LOWER = [0, 2, 8]
UPPER = [6, 6, 18]


def campaign() -> list[Case]:
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    unit = sampler.random(N_SAMPLES)
    scaled = qmc.scale(unit, LOWER, UPPER)

    cases = []
    for m, p, t in np.round(scaled).astype(int):
        # A symmetric section has no meaningful camber position; NACA
        # convention writes it as 00xx.
        if m == 0:
            p = 0
        naca = f"{m}{p}{t:02d}"
        cases.append(
            Case(
                geometry=Geometry(naca=naca),
                conditions=Conditions(reynolds=5e5, alpha_start=0, alpha_end=12, alpha_step=1),
                label=f"lhs {naca}",
            )
        )
    return cases
