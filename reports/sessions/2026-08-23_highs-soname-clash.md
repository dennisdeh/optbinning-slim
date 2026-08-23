# 2026-08-23 — the highspy/ortools HIGHS clash

## What was asked

Fix the highspy/ortools HIGHS conflict, the last entry in `OPEN_ITEMS.md`.

## What it actually is

Not "ortools loads its HiGHS first" as originally filed, but a **SONAME
collision**: ortools 9.15.6755 ships `libhighs.so.1` built from HiGHS 1.12.0,
highspy 1.15.1 ships `libhighs.so.1` built from HiGHS 1.15.1, and the dynamic
linker reuses whichever is already loaded. One HiGHS per process; the other
library gets a version it was not built against.

## What was tried

- **Reverse the import order** so highspy's newer library wins. cvxpy then gets
  HIGHS — and ortools dies: `libortools.so.9: undefined symbol:
  _Z19setLocalOptionValue...HighsLogOptions...`. `import optbinning` raises.
  This is worse than the problem, and it establishes that today's order is the
  only working one.
- **Align the versions down**, `highspy==1.12.0`. Both import cleanly and cvxpy
  reports HIGHS available. But cvxpy 1.9.2 requires `highspy>=1.14.0`, so the
  install is an unresolvable environment and pip reports a conflict; declaring
  it in `pyproject.toml` is not shippable.
- **Align up**: there is no ortools release bundling HiGHS >= 1.14. 9.15.6755 is
  the newest and carries 1.12.0.

## What shipped

No code change — there is no fix available from this package. What shipped is
the part that was missing: the constraint is now enforced instead of implicit.

`tests/test_package.py` pins that ortools still solves a CP-SAT model after
`import optbinning`, in a subprocess so import order means something. Prepending
`import cvxpy` to `optbinning/__init__.py` makes both tests in that file fail,
so the check has teeth rather than merely passing. `CLAUDE.md` warns against
reordering those imports, `DECISIONS.md` records why the order is load-bearing,
and the `OPEN_ITEMS.md` entry now carries the version table, both tested
directions, and the condition under which it can be deleted.

## Result

212 passed on Python 3.13.15, flake8 gate 0. The impact of the remaining
limitation is unchanged and does not touch results: optbinning's own
`solver="highs"` goes through ropwr and scipy and works; what is lost is cvxpy's
HIGHS backend, which nothing here selects, plus one log line in test output.

At the time of writing this work was on branch `fix-highs-clash` and nothing had
been merged.
