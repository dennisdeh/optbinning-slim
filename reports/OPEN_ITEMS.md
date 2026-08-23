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

## `read_json` cannot restore a categorical binning

*Last updated: 2026-08-23*

`to_dict` persists `table.splits`, which for `dtype="categorical"` is the list
of **grouped categories** — the output of `bin_categorical`. Restoring a working
estimator needs its **input**: the numeric split positions over the ordered
category index (`self._splits_optimal`). Those are never written, so no change
to `read_json` can rebuild a categorical estimator from the saved file; the
format itself is lossy.

Affects `OptimalBinning` and `ContinuousOptimalBinning` with
`dtype="categorical"`. `MulticlassOptimalBinning` is numerical-only and
unaffected. The numerical case was fixed on 2026-08-23 (see `DECISIONS.md`);
`_restore_from_binning_table` restores what the table carries and does not
attempt the categorical case.

Fixing it means changing the on-disk format — add the split positions,
`_categories` and `_cat_others` to `to_dict` — which breaks files written by
earlier versions. That is a maintainer decision, not a silent fix.

## MDLP emits a split the MDLP criterion rejects

*Last updated: 2026-08-23*

`MDLP._recurse` appends the best candidate split **before** consulting
`_terminate`, and then uses `_terminate` only to decide whether to recurse:

```python
split = self._find_split(u_x, x, y)
if split is not None:
    self._splits.append(split)          # <- unconditional
    t = np.searchsorted(x, split, side="right")
    if not self._terminate(n_x, n_y, y, y[:t], y[t:]):
        self._recurse(...)
```

The Fayyad-Irani stopping criterion is meant to decide whether the split is
accepted at all. As written, every node that has any candidate satisfying
`min_samples_leaf` contributes a split, so the discretisation always has at
least one cut.

Reproduced 2026-08-23:

- `x = np.arange(200.)`, `y` random 0/1 (no information): 1 split at 174.58,
  where correct MDLP returns none.
- `min_samples_split=500` on 200 samples: 1 split. The parameter is documented
  as "the minimum number of samples required to split an internal node" and is
  only ever read inside `_terminate`, so it cannot prevent the first split.

Blast radius: the public `MDLP` estimator, and `prebinning_method="mdlp"` in
`OptimalBinning` and `OptimalBinning2D` (`ContinuousOptimalBinning` and
`MulticlassOptimalBinning` reject `"mdlp"` during parameter validation). The
downstream effect is a prebinning with one extra cut per terminal branch, which
the optimiser may or may not keep.

Not fixed: moving the append inside the `if not self._terminate(...)` branch
changes the output of a public algorithm and of every `prebinning_method="mdlp"`
fit. It needs a maintainer decision, and the fix must ship with the
noise-returns-no-splits test above.

## MDLP rejects a float target

*Last updated: 2026-08-23*

`MDLP().fit(x, y.astype(float))` raises
`TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
according to the rule 'safe'` from `np.bincount` inside `_recurse`. A float
0.0/1.0 target is what `df["target"].values` gives, and every sibling estimator
accepts it. `_check_parameters` and `check_array` both pass it through, so the
failure surfaces from numpy with no hint that the target is the problem.

Blast radius: direct `MDLP` users only — `PreBinning` passes the target it was
given, and `OptimalBinning` has already coerced the target by then.

## `ContinuousOptimalBinning2D` accepts a prebinning method it cannot use

*Last updated: 2026-08-23*

`_check_parameters` in `continuous_binning_2d.py` allows
`prebinning_method="mdlp"`, but MDLP is binary-target only, so `fit` later
raises `ValueError: mdlp method can only handle binary classification
problems.` from `PreBinning`. The 1D `ContinuousOptimalBinning` gets this right:
`"mdlp"` is absent from its allowed values and the constructor's own validation
rejects it with the list of what is allowed.

Verified 2026-08-23. Cost is a late and less informative error, not a wrong
result.

## `PWContinuousBinningTable` publishes none of its statistics

*Last updated: 2026-08-23*

`analysis()` computes `_quality_score`, `_hhi`, `_hhi_norm` and the regression
metrics, and `PWContinuousBinningTable` defines no properties at all, so none of
it is reachable. Its three sibling tables — `BinningTable`,
`MulticlassBinningTable`, `ContinuousBinningTable` — each expose
`quality_score`, and `PWBinningTable` inherits `BinningTable`'s, so
`OptimalPWBinning(...).binning_table.quality_score` works while
`ContinuousOptimalPWBinning(...).binning_table.quality_score` raises
`AttributeError`.

The values are computed either way; only the accessors are missing. Adding them
is new public API, hence not done unasked.
`tests/test_continuous_binning_piecewise.py::test_binning_table_analysis`
asserts on the printed report instead, and says why.
