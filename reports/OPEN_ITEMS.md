# Open items

Things the code does that it should not. One entry per defect; remove the entry
in the commit that fixes it.

## cvxpy cannot use its HIGHS backend in any optbinning process

*Last updated: 2026-08-23*

**Root cause: a SONAME collision between two wheels.** Both ship a HiGHS shared
library named `libhighs.so.1`, built from different versions:

| wheel | file | HiGHS version |
|---|---|---|
| ortools 9.15.6755 | `ortools/.libs/libhighs.so.1` | 1.12.0 |
| highspy 1.15.1 | `highspy/libhighs.so.1` | 1.15.1 |

The dynamic linker resolves by SONAME and reuses whatever is already loaded, so
exactly one of the two HiGHS builds exists per process and the other library
gets the wrong one. Neither is optional here: ortools is a direct dependency,
and highspy arrives through ropwr -> cvxpy.

**Both orders were tested on 2026-08-23, and only one works.**

- *ortools first* (what `import optbinning` does): ortools works. highspy's
  extension then fails with `undefined symbol: _ZN5Highs13releaseMemoryEv` —
  a method HiGHS 1.12 does not have — so cvxpy logs
  `Encountered unexpected exception importing solver HIGHS` and drops HIGHS
  from its solver list.
- *highspy first*: cvxpy gets HIGHS, and **ortools dies** —
  `libortools.so.9: undefined symbol: _Z19setLocalOptionValue...HighsLogOptions...`,
  a symbol HiGHS 1.15 no longer exports. `import optbinning` itself raises.

So the current behaviour is the safe configuration, not an oversight, and the
import order in `optbinning/__init__.py` is load-bearing — see `DECISIONS.md`
and `tests/test_package.py`.

**It cannot be fixed from this package.** Aligning the versions does work:
`highspy==1.12.0` matches ortools' bundled HiGHS, and then both import cleanly
and cvxpy reports HIGHS available (verified 2026-08-23). But cvxpy 1.9.2
requires `highspy>=1.14.0`, so pinning that is an unresolvable environment, and
declaring it in `pyproject.toml` would make `pip install` conflict. There is no
ortools release bundling HiGHS >= 1.14 — 9.15.6755 is the newest and carries
1.12.0.

**Impact is limited and does not touch results.** optbinning's own
`solver="highs"` is unaffected: it reaches HiGHS through ropwr and scipy, and
`OptimalPWBinning(solver="highs", objective="l1")` fits normally (verified
2026-08-23, IV 6.8833236302). What is lost is cvxpy's HIGHS backend, which
neither optbinning nor ropwr selects, plus one log line in test output.

**Re-check when either wheel moves**: the entry can be deleted once ortools
bundles HiGHS >= 1.14, or once either wheel adopts a version-qualified SONAME.
If cvxpy's HIGHS backend is needed in the meantime, the only route is a
separate process, or `pip install highspy==1.12.0` and living with pip's
dependency-conflict warning.
