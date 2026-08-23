# Open items

Things the code does that it should not. One entry per defect; remove the entry
in the commit that fixes it.

## cvxpy cannot use its HIGHS backend in any optbinning process

*Last updated: 2026-08-23*

`ortools` bundles its own HiGHS library, and once it is loaded `highspy`'s
extension module resolves against it and fails:

```
ImportError: .../highspy/_core.cpython-313-x86_64-linux-gnu.so:
             undefined symbol: _ZN5Highs13releaseMemoryEv
```

Reproduced 2026-08-23 with ortools 9.15.6755 / highspy 1.15.1 / cvxpy 1.9.2 on
Python 3.13.15 — `import highspy` alone succeeds, and so does
`import scipy.optimize; import highspy`, but
`from ortools.linear_solver import pywraplp; import highspy` raises the above.
optbinning imports ortools before cvxpy, so cvxpy always logs
`Encountered unexpected exception importing solver HIGHS` and drops HIGHS from
its solver list.

Impact is limited but real:

- optbinning's own `solver="highs"` is **not** affected; it reaches HiGHS
  through ropwr/scipy, and `OptimalPWBinning(solver="highs", objective="l1")`
  fits normally (verified 2026-08-23, IV 6.8833236302).
- cvxpy's HIGHS backend is unavailable, and the log line appears in test output.

Neither library is ours; the fix belongs upstream in highspy or ortools
packaging. Recheck when either is upgraded, and delete this entry when the
symbol clash is gone.
