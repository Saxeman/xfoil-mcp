"""Step 12: scikit-fem on a problem with a known exact answer.

A long thin strip of skin with a heater band in the middle. Heat spreads
along the skin by conduction and leaves through the surface to the air.
For an infinitely long strip the answer is known exactly, so the solver
can be checked before it is trusted with h(s) from XFOIL.
"""

import numpy as np
from scipy.integrate import trapezoid
from skfem import Basis, BilinearForm, ElementLineP1, LinearForm, MeshLine, solve
from skfem.helpers import dot, grad

kt = 200.0 * 0.001      # skin conductivity x thickness, W/K (1 mm aluminium)
h = 100.0               # heat transfer coefficient, W/m^2 K
q = 10_000.0            # heater flux, W/m^2
a = 0.025               # heater half-width, m (a 50 mm band)
L = 0.5                 # strip half-length, m
lam = np.sqrt(kt / h)   # spreading distance, m


def exact(x):
    """Temperature rise above the air for an infinitely long strip."""
    x = np.abs(x)
    inside = (q / h) * (1 - np.exp(-a / lam) * np.cosh(x / lam))
    outside = (q / h) * np.sinh(a / lam) * np.exp(-x / lam)
    return np.where(x <= a, inside, outside)


def solve_strip(x):
    basis = Basis(MeshLine(x), ElementLineP1())

    @BilinearForm
    def conduction_and_loss(u, v, w):
        return kt * dot(grad(u), grad(v)) + h * u * v

    @LinearForm
    def heater(v, w):
        return np.where(np.abs(w.x[0]) <= a, q, 0.0) * v

    return solve(conduction_and_loss.assemble(basis), heater.assemble(basis))


print(f"spreading distance {lam * 1000:.1f} mm; exact centre rise {exact(0.0):.3f} K\n")
for n in (201, 401, 801, 1601):
    # Each of these n puts a node on the heater's edges (x = +/-a), so the band
    # starts and stops cleanly. Other values of n would not.
    x = np.linspace(-L, L, n)
    theta = solve_strip(x)
    centre = theta[np.argmin(np.abs(x))]
    error = np.max(np.abs(theta - exact(x)))
    heat_out = trapezoid(h * theta, x)
    print(f"n={n:5d}  centre {centre:7.3f} K   max error {error:.2e} K   "
          f"heat out {heat_out:6.1f} W/m (in {q * 2 * a:.0f})")
