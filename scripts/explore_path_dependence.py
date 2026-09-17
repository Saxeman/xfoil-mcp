"""Characterize XFOIL's path dependence.

Run inside the container:  python scripts/explore_path_dependence.py

This is an EXPERIMENT, not a test. It has no correct answer, and its output
could legitimately change with a different XFOIL build or platform without
anything being broken. Asserting on it would produce a flaky test that fails
for uninteresting reasons, so it lives here rather than in tests/.

What it characterizes
---------------------
XFOIL does not solve each alpha from scratch. Each point starts from the
previous converged solution, so results depend on the order points were
computed in. Two consequences, which are easy to conflate and are worth
keeping straight:

  Converged VALUES are not path-dependent. The viscous-inviscid coupling
  solves a system of equations with one solution; warm starting changes the
  route to that solution, not the solution itself.

  CONVERGENCE ITSELF is path-dependent. Whether a point converges at all,
  and how many iterations it takes, depend on where the solve started.

The practical consequence: re-running a failed point identically is
pointless, but approaching it differently - smaller steps, a different
starting alpha - genuinely changes the odds.
"""

from __future__ import annotations

from xfoil_mcp.wrapper import run_polar

# Re 5e4 is drone / model-aircraft territory. The boundary layer is mostly
# laminar and separates readily, so convergence is fragile enough that step
# size visibly changes which points survive.
FRAGILE_RE = 5e4


def values_are_path_independent() -> None:
    """Compare shared alphas between a coarse and a fine sweep at an easy Re.

    Expectation: identical to the polar file's precision. Any difference here
    would overturn the claim above.
    """
    print("== do converged values depend on the path? ==")
    coarse = run_polar("2412", 1e6, 0, 10, 2)
    fine = run_polar("2412", 1e6, 0, 10, 1)
    fine_by_alpha = {p.alpha: p for p in fine.points}

    differences = 0
    for p in coarse.points:
        q = fine_by_alpha.get(p.alpha)
        if q is None:
            continue
        if abs(p.cl - q.cl) > 1e-9 or abs(p.cd - q.cd) > 1e-9:
            differences += 1
            print(f"  alpha {p.alpha}: CL {p.cl} vs {q.cl}, CD {p.cd} vs {q.cd}")

    if differences == 0:
        print(f"  no differences across {len(coarse.points)} shared alphas")
        print("  -> warm starting changes the route, not the destination")


def convergence_depends_on_the_path() -> None:
    """Same target alphas, different approach, different failure sets."""
    print("\n== does convergence depend on the path? ==")
    for step in (1, 2):
        result = run_polar("2412", FRAGILE_RE, 0, 14, step, max_iter=30)
        print(f"  {step}-deg steps: "
              f"{result.converged_count}/{result.requested_count} converged, "
              f"failed {result.failed_alphas}")
    print("  -> an alpha can converge or fail depending only on how it was reached")


def a_failure_degrades_its_successors() -> None:
    """A tight iteration limit makes the cascade visible.

    Failures that arrive in consecutive runs are not independent hard points:
    the failed solve leaves a poor starting state for the next one.
    """
    print("\n== do failures cascade? ==")
    result = run_polar("2412", 1e6, 0, 20, 1, max_iter=5)
    print(f"  failed alphas: {result.failed_alphas}")
    print(f"  grouped into runs: {result.failure_runs}")
    for run in result.failure_runs:
        if len(run) > 1:
            print(f"  -> {run} is consecutive, consistent with a cascade "
                  "rather than {len(run)} independent hard points")


def retrying_identically_is_pointless() -> None:
    """Determinism means a plain retry cannot help. Worth demonstrating,
    because retry-on-failure is the reflex an agent will reach for first."""
    print("\n== does retrying an identical call help? ==")
    kwargs = dict(airfoil="2412", reynolds=1e6, alpha_start=0, alpha_end=20,
                  alpha_step=1, max_iter=5)
    first, second = run_polar(**kwargs), run_polar(**kwargs)
    print(f"  attempt 1 failed at {first.failed_alphas}")
    print(f"  attempt 2 failed at {second.failed_alphas}")
    print("  -> identical, so a retry strategy must CHANGE something to matter")


if __name__ == "__main__":
    values_are_path_independent()
    convergence_depends_on_the_path()
    a_failure_degrades_its_successors()
    retrying_identically_is_pointless()
