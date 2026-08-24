# Open items

Things the code does that it should not. One entry per defect; remove the entry
in the commit that fixes it.

## `time_limit` is documented as `int`, but floats are honoured

*Last updated: 2026-08-24*

Every `time_limit` entry reads `time_limit : int (default=...)`. This has been
inaccurate since before the fork: `_check_parameters` validates
`numbers.Number`, and `solver="cp"` has always honoured a fractional value. As
of 2026-08-24 `solver="mip"` honours one too, rounding to the nearest
millisecond, so `int or float` is now the only accurate type.

Sites: `binning/binning.py`, `binning/continuous_binning.py`,
`binning/multiclass_binning.py`, `binning/distributed/binning_sketch.py`,
`binning/uncertainty/binning_scenarios.py`,
`binning/multidimensional/binning_2d.py`,
`binning/multidimensional/continuous_binning_2d.py` and
`scorecard/counterfactual/counterfactual.py`. The validators at those eight
sites were corrected on 2026-08-24 (see `DECISIONS.md`); only the documented
type is still wrong.

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

## A dict `metric_special` passes validation and then raises

*Last updated: 2026-08-24*

`transformations.py::_check_metric_special_missing` has a dedicated
`elif isinstance(metric_special, dict):` branch that validates every value is a
number, and its fall-through message advertises "a dict" as an allowed form.
`_apply_transform` then cannot use one. Verified 2026-08-24:
`optb.transform(x, metric="woe", metric_special={"a": 0.5})` raises
`TypeError: float() argument must be a string or a real number, not 'dict'`.

Every public `transform` docstring says `metric_special : float or str
(default=0)`, so the docstrings and the validator disagree with each other as
well. Either the validator's dict branch should go, or `_apply_transform` should
map a named special code to its value — which is the reading the dict form of
`special_codes` suggests. This was examined twice on 2026-08-24: once reported
as a defect and refuted on the grounds that per-key semantics are undocumented,
then re-judged, because the refutation does not explain why the validator has a
dict branch at all.

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
