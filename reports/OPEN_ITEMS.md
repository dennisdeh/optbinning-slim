# Open items

Things the code does that it should not. One entry per defect; remove the entry
in the commit that fixes it.

## `OptimalPWBinning.fit` fits the caller's estimator in place

*Last updated: 2026-08-24*

`OptimalPWBinning._fit` aliases the `estimator` constructor parameter rather
than copying it, and then fits it. Verified 2026-08-24:

```python
est = LogisticRegression()
optb = OptimalPWBinning(estimator=est).fit(x, y)
optb.estimator is est     # True
hasattr(est, "coef_")     # True -- the caller's object was fitted in place
```

This is the same class as the "`fit()` must not mutate its constructor
parameters" entry in `DECISIONS.md`, which was fixed for `user_splits_fixed` on
2026-08-24 in the four estimators that accept it. It is filed rather than fixed
because the right answer is not obvious: `sklearn.base.clone` would give the
caller a fresh estimator and match the sklearn contract, but a user who passes a
pre-configured estimator and then inspects it afterwards is relying on today's
behaviour, and nothing documents either reading.

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

## The CP model's weight scaling overflows int64 for extreme weight ratios

*Last updated: 2026-08-24*

`optbinning/binning/cp.py`, in the scenario branch, makes the scenario weights
integral for CP-SAT with

```python
sw = 10 ** np.abs(np.log10(np.min(w)))
w = [np.int64(w[s] * sw) for s in range(n_scenarios)]
```

The scale factor is chosen to bring the *smallest* weight up to about 1, with
no regard for what it does to the largest. Once the ratio between the largest
and smallest weight exceeds roughly 1e18, the product exceeds `int64`'s range
and the cast is undefined.

Measured 2026-08-24 (numpy 2.5.2, Linux x86-64), for `weights=[min, 1.0]`:

| min | scale factor | largest scaled weight | int64 cast |
|---|---|---|---|
| 1e-17 | 1e17 | 1e17 | 100000000000000000 — correct |
| 1e-18 | 1e18 | 1e18 | 1000000000000000000 — correct |
| 1e-19 | 1e19 | 1e19 | **-9223372036854775808** |
| 1e-30 | 1e30 | 1e30 | **-9223372036854775808** |

These weights are numeric, finite and strictly positive, so they pass the
validation added on 2026-08-24 (see `DECISIONS.md`) and reach the model. The
non-finite cases — negative and zero weights — are now rejected before this
point, which is why this is the *remaining* part of the defect rather than the
whole of it.

Not fixed here because the repair is a change to the CP model's numerics, not
to validation: the scale factor should be chosen from the *ratio* (and clamped
so the largest scaled weight fits `int64`), and any change to it moves the
integer weights the objective is computed from. That needs its own measurement
of the objective before and after, on a fixture where the weights differ.

**A weight ratio above 1e18 is not a plausible accident**, which is why this is
filed rather than treated as urgent: it needs weights spanning nineteen orders
of magnitude.
