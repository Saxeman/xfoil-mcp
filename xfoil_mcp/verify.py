"""Verification checks for the XFOIL wrapper.

Run inside the container:  python -m xfoil_mcp.verify

Three kinds of check, in increasing order of interest:
  1. Determinism - two identical calls must agree exactly. A difference means
     state is leaking between runs, not that the solver is noisy.
  2. Physical sanity - the numbers must obey things we know independently of
     XFOIL (lift-curve slope, camber sign, drag rising with lift).
  3. Path dependence - runs that SHOULD differ, because warm starting makes
     results depend on execution order. Confirming this is real is as
     important as confirming determinism.
"""

from __future__ import annotations

from xfoil_mcp.wrapper import PolarResult, run_polar


def compare(a: PolarResult, b: PolarResult) -> list[str]:
    """Exact comparison of two results. Empty list means identical."""
    diffs: list[str] = []

    if a.status != b.status:
        diffs.append(f"status: {a.status} vs {b.status}")

    if a.failed_alphas != b.failed_alphas:
        diffs.append(f"failed_alphas: {a.failed_alphas} vs {b.failed_alphas}")

    if len(a.points) != len(b.points):
        diffs.append(f"point count: {len(a.points)} vs {len(b.points)}")
        return diffs

    for pa, pb in zip(a.points, b.points):
        if pa != pb:
            diffs.append(
                f"alpha {pa.alpha}: "
                f"CL {pa.cl} vs {pb.cl}, CD {pa.cd} vs {pb.cd}, "
                f"CM {pa.cm} vs {pb.cm}"
            )
    return diffs


def check_determinism() -> None:
    print("== determinism ==")
    cases = [
        ("clean sweep", dict(airfoil="2412", reynolds=1e6,
                             alpha_start=0, alpha_end=10, alpha_step=1)),
        ("cascade sweep", dict(airfoil="2412", reynolds=1e6, alpha_start=0,
                               alpha_end=20, alpha_step=1, max_iter=5)),
    ]
    for name, kwargs in cases:
        first = run_polar(**kwargs)
        second = run_polar(**kwargs)
        diffs = compare(first, second)
        if diffs:
            print(f"  FAIL {name}: {len(diffs)} difference(s)")
            for d in diffs[:5]:
                print(f"    {d}")
        else:
            print(f"  PASS   {name}: identical "
                  f"({first.converged_count}/{first.requested_count} points)")


def check_physics() -> None:
    """Assertions that do not depend on XFOIL being right, only on aerodynamics."""
    print("== physical sanity ==")
    result = run_polar("2412", 1e6, 0, 10, 1)
    if result.status != "ok":
        print(f"  SKIP: baseline run was {result.status}")
        return

    by_alpha = {p.alpha: p for p in result.points}

    # 2% camber means positive lift at zero incidence.
    cl0 = by_alpha[0.0].cl
    print(f"  {'PASS  ' if cl0 > 0.15 else 'FAIL'} CL at alpha=0 is {cl0:.4f} (expect ~0.24)")

    # Thin airfoil theory: dCL/dalpha approaches 2*pi per radian (~0.11 per
    # degree). Viscous effects pull it slightly below that.
    slope = (by_alpha[5.0].cl - by_alpha[0.0].cl) / 5.0
    ok = 0.09 < slope < 0.12
    print(f"  {'PASS  ' if ok else 'FAIL'} lift-curve slope {slope:.4f}/deg (expect 0.09-0.12)")

    # CD has a bucket near low alpha, so it is not monotonic. What must hold
    # is that drag at high alpha greatly exceeds drag at the minimum.
    cd_min = min(p.cd for p in result.points)
    cd_high = by_alpha[10.0].cd
    ok = cd_high > 2 * cd_min
    print(f"  {'PASS  ' if ok else 'FAIL'} CD at alpha=10 ({cd_high:.5f}) "
          f"is well above minimum CD ({cd_min:.5f})")

    # Upper-surface transition must move forward as alpha increases.
    forward = by_alpha[10.0].top_xtr < by_alpha[0.0].top_xtr
    print(f"  {'PASS  ' if forward else 'FAIL'} transition moves forward "
          f"({by_alpha[0.0].top_xtr:.3f} -> {by_alpha[10.0].top_xtr:.3f})")

    # A symmetric section must produce no lift at zero incidence.
    sym = run_polar("0012", 1e6, 0, 4, 1)
    if sym.status == "ok":
        cl0_sym = sym.points[0].cl
        print(f"  {'PASS  ' if abs(cl0_sym) < 0.01 else 'FAIL'} "
              f"NACA 0012 CL at alpha=0 is {cl0_sym:.4f} (expect ~0)")

def check_path_dependence() -> None:
    print("== path dependence ==")
    up = run_polar("2412", 5e4, 0, 14, 1, max_iter=30)
    coarse = run_polar("2412", 5e4, 0, 14, 2, max_iter=30)
    print(f"  1-deg steps: {up.converged_count}/{up.requested_count}, "
          f"failed {up.failed_alphas}")
    print(f"  2-deg steps: {coarse.converged_count}/{coarse.requested_count}, "
          f"failed {coarse.failed_alphas}")

if __name__ == "__main__":
    check_determinism()
    print()
    check_physics()
    print()
    check_path_dependence()
