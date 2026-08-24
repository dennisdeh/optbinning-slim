# Changelog

Release notes for `optbinning-slim`, a maintenance fork of
[optbinning](https://github.com/guillermo-navas-palencia/optbinning). Versions
before 0.22.0 are upstream's; their notes are in the
[upstream documentation](http://gnpalencia.org/optbinning/release_notes.html).

The fork adds no algorithms. Everything below is packaging, currency with the
Python and dependency stack, defect fixes and test coverage.

## Unreleased

Getting CI green for the first time. Three of the four defects below surfaced
only off this project's development machine — found by reading the CI logs of
the seven consecutive red runs that followed the 0.22.0 release, none of them
reproducing on Linux x86-64 with CPython 3.13.15 or 3.14.7, which is why the
suite was green locally throughout. The fourth, the `MDLP` refit, was found
while fixing the others.

### Fixed

- **`MDLP` gave different discretisations on different machines.**
  `_find_split` read its candidate cut points off the rows where the label
  changes, so with ties in `x` the answer depended on the order
  `np.argsort` left tied rows in — and that order is neither stable nor the
  same across CPU architectures. It now applies the Fayyad-Irani boundary rule
  per *distinct value* of `x`: the midpoint between two adjacent values is a
  candidate cut unless every observation at both values carries the same
  single label. The result is now a function of the `(x, y)` pairs alone.

  Measured 2026-08-23 on breast-cancer `"mean radius"`, permuting the input
  rows moved the IV of `OptimalBinning(prebinning_method="mdlp")` over the
  range 3.85 to 4.82; the Linux jobs landed on 4.76862756 and the macOS arm64
  jobs on 3.92842062. The fix holds that IV at 4.76862756 on every platform.
  Split *values* move in the third decimal for data with ties; split counts
  and IV on the breast-cancer fixture do not.

- **`MDLP.fit` accumulated splits instead of resetting them.** `_splits` was
  only ever appended to, so refitting an instance returned the previous fit's
  splits alongside the new ones — and since `splits` sorts them, interleaved
  rather than obviously doubled. Nothing inside optbinning reused an instance,
  so this only affected callers who did. `fit` now resets, after validation,
  so a rejected call leaves the estimator on its last good fit.

- **`Logger.close` leaked a file handle on CPython 3.14.6.** It iterated
  `logger.handlers` while removing from it, which skips every second handler
  wherever `logging.Logger.removeHandler` mutates that list in place. Whether
  it mutates or rebinds (gh-79366) varies by *micro* version — 3.13.15 and
  3.14.7 rebind, 3.14.6 does not — so the supported range straddles the
  boundary. It now iterates a copy.

- **The test suite no longer needs a GUI toolkit.** `tests/conftest.py` pins
  matplotlib's non-interactive `Agg` backend. Matplotlib otherwise picks a GUI
  backend wherever it believes there is a display — always, on Windows — and
  the Windows runners' CPython 3.13.15 ships a Tcl tree `tkinter` cannot
  initialise, which killed `test_multiclass_binning.py::test_numerical_default`
  there. Nothing in the library changed; this affects the suite only.

### Fixed — the 2026-08-24 coverage pass

A test-coverage pass took statement coverage from 89% to 99% and the suite from
216 to over a thousand tests. Everything below is a defect those tests exposed;
each fix ships a regression test that was seen failing against the unfixed code.
Most of them existed in three or four of the near-parallel estimators at once.

**Wrong numbers, returned silently.** These are the ones worth reading first: no
exception was raised, the output merely was not what it claimed to be.

- **`BinningProcess` leaked a per-variable transform option onto every later
  variable.** A `binning_transform_params={"v0": {"metric": "indices"}}` made the
  override the default for every variable after `v0` in `variable_names` order,
  and on the disk path for every later chunk. Configuring one variable silently
  changed the output of the rest.
- **`OptimalPWBinning.transform` returned the *non-event* rate** for special and
  missing buckets with `metric_special`/`metric_missing="empirical"`, because
  `transform_binary_target`'s signature took its four counts in the opposite
  order to all three of its callers. With `metric="woe"` the WoE for those
  buckets had the wrong sign.
- **A bin holding only events reported an event rate of 0.0.** The event rate was
  gated on a bin having *both* classes, which is the condition WoE needs, not the
  condition an event rate needs. Any pure bin was affected — a special-code
  bucket that happens to be all-event, a tail bin — not only degenerate targets.
  Fixed in all four binning tables and all four transforms, and
  `binning_table.build()` and `transform(metric="event_rate")` are now pinned to
  agree.
- **`analysis()` counted special buckets as ordinary bins.** With dict-form
  `special_codes`, all but one named bucket entered the significance tests, the
  monotonic-trend report and Cramer's V, so a strictly ascending binning could be
  reported as `"peak"`. The categorical `"others"` bin had the same problem.
- **`gamma` had no effect on `ContinuousOptimalBinning`.** The dominating-bin
  penalty was subtracted from the CP objective without the constraints that give
  it meaning, so it changed the reported objective and never the bins.
- **`split_digits` was ignored** by both 2D estimators and by the sketch
  estimators, though validated and documented by all four.
- **`monotonic_trend` values `"convex"`, `"concave"`, `"peak_heuristic"` and
  `"valley_heuristic"` were silently dropped** by the multiclass model, which
  compared the whole trend *list* against a string. They passed validation and
  returned the unconstrained binning as `"OPTIMAL"`.
- **`BCatSketch.merge` never checked that the two sketches were compatible** (it
  has a `_mergeable` method that nothing called) and **aliased the other
  sketch's count lists**, so a distributed workflow that merged a worker into an
  aggregator and kept using the worker got wrong counts on the worker.

**Crashes on ordinary or degenerate input.**

- A **single-class binary target** was accepted by `fit()` as `"OPTIMAL"` and
  then made `binning_table.build()` — and `BinningProcess.fit`, which calls it —
  raise sklearn's `Found array with 0 sample(s)`. Such a table now builds, with
  IV, JS, Hellinger, Triangular, KS and Gini all 0 (Gini was `nan`).
- **`user_splits=[]` could not complete a fit** in any of the four estimators
  that accept it, each failing differently. It now means "no split points".
- **`MulticlassOptimalBinning` could not use `min_bin_size`/`max_bin_size` at
  all** — every combination died inside OR-Tools with `NotImplementedError`.
  `BinningProcess` on a multiclass target inherited that.
- **`special_codes` as a numpy array of more than one code** was accepted by
  `fit()` and then raised `The truth value of an array with more than one element
  is ambiguous` from `transform()`.
- **`OptimalPWBinning.fit` raised `TypeError`** whenever the missing-value bucket
  held both events and non-events — an entirely ordinary input.
- **`ContinuousOptimalPWBinning.fit_transform` dropped `lb`/`ub`** and passed
  `check_input` positionally into `lb`.
- **A fractional `time_limit` crashed `solver="mip"`** with a SWIG `TypeError`
  while `solver="cp"` honoured it, in the 1D, 2D and counterfactual models; and
  `time_limit=0` meant "no time" to one backend and "no limit" to the other.
- **`Counterfactual.generate` raised `IndexError`** whenever every feature of the
  scorecard happened to yield the same number of candidate bins.
- **`SBOptimalBinning` raised `IndexError`** whenever pre-binning left no splits,
  and passed the wrong `user_splits_fixed` array to the solver.
- **`OptimalBinningSketch.solve()` raised `IndexError`** on a single-class or
  single-row stream, and `information()` raised `AttributeError` when the solver
  had not run.
- **`strategy="cart"` could not fit inputs `strategy="grid"` fits**, raising
  sklearn's `InvalidParameterError` when pre-binning left an axis unsplit.
- **`strategy="cart"` returned a coarser binning than it should**, and on small
  prebinning grids a single whole-grid bin with IV 0. The CART leaf budget was
  `n_splits_x * n_splits_y`, but a cart bin merges two or more of the tree's
  leaves, so *b* bins need 2*b* leaves and any budget below four could only
  return the union of all of them. The budget is now the prebinning grid's cell
  count. **This changes cart results at the default `max_n_prebins`**: measured
  2026-08-24, a synthetic 300-row logistic target went from 7 bins / IV 2.5925
  to 9 / 2.7376, and breast-cancer (mean texture, mean area) from 7 / 5.4471 to
  8 / 5.5412. Inputs whose grid is large enough that `min_prebin_size` binds
  first are unchanged, and cart remains far cheaper than grid (0.34 s vs 11.84 s
  at `max_n_prebins` 20x20).

**Estimator contract.**

- **`fit()` no longer mutates its constructor parameters.** All four estimators
  that accept `user_splits_fixed` rewrote it to a numpy `bool_` array, so
  refitting the same instance raised `ValueError: user_splits_fixed must be list
  of boolean` and `sklearn.base.clone` round-trips were unsound.
- **`BinningProcess.update_binned_variable`'s binary guard** accepted
  `ContinuousOptimalBinning`, `MulticlassOptimalBinning` and the 2D estimators,
  because they subclass `OptimalBinning`; `transform()` then returned that
  binning's own metric instead of WoE.
- **`BinningProcess.fit` did not validate `fixed_variables` names**, unlike
  `fit_disk`. A bogus name was silently ignored.
- **`BinningProcessSketch` ignored `selection_criteria`**, dying with
  `AttributeError` for any non-`None` value, and rejected the `"indices"` and
  `"bins"` metrics its own docstring and body support.
- **Mixing `"indices"` with another transform metric** silently truncated floats
  to integers; it now raises.
- Unfitted `RangeDetector` / `ModifiedZScoreDetector` / `YQuantileDetector` raise
  `NotFittedError` from `get_support()` rather than `AttributeError`, and a 2D
  binning table raises `NotFittedError` rather than `AttributeError` when
  `analysis()` is called before `build()`.
- **`Scorecard(scaling_method="min_max", rounding=True)`** now lands on the
  requested range instead of merely inside it, and a degenerate scorecard raises
  instead of returning an all-`NaN` one from a successful `fit()`.

**Validation and messages.**

- **A dict `metric_special` passed validation and then raised.** The parameter
  checker has always had a branch validating one number per key — it arrived
  with named `special_codes` in upstream `51445f0` — but the transform ignored
  the keys and handed the dict to numpy, so `transform(metric_special={"a":
  0.5})` died with `TypeError: float() argument must be a string or a real
  number, not 'dict'`. A dict now does what the checker always implied: it gives
  each **named** special bucket its own value, for the binary, continuous,
  multiclass and piecewise estimators and for `BinningProcess`. Passing a dict
  where `special_codes` is not a dict, or omitting a bucket, now raises a
  `ValueError` naming the problem instead of failing inside numpy.

- **`time_limit=float("nan")` and `time_limit=float("inf")` passed validation**
  at all eight sites that accept the parameter, because neither is `< 0`. A
  `nan` budget then produced a silently empty binning (`solver="cp"` reported
  `MODEL_INVALID` as an ordinary status), and `inf` worked under `solver="cp"`
  while dying under `solver="mip"` with `OverflowError: cannot convert float
  infinity to integer`. Non-finite values are now rejected with a clear
  `ValueError`. `time_limit=0` remains valid for the binning estimators, where
  it means "no budget" and both backends return the single-bin fallback, and
  remains invalid for `Counterfactual.generate`, which has no such fallback —
  and each message now states its own rule instead of all eight claiming the
  value must be "positive" while seven accepted 0.

- `BSketch` accepted an out-of-range `eps` — the guard used `and` where every
  sibling uses `or`.
- `split_data` silently dropped `fix_ub` whenever `fix_lb` was also given.
- A stray `print` wrote numpy dtypes to stdout on every categorical
  `ContinuousOptimalBinning` fit that produced an `"others"` group.
- `jensen_shannon_multivariate` rejected the array-like it documents.
- Error messages that named values the code rejects, and docstrings that
  advertised `outlier_detector="zcore"` where only `"zscore"` is accepted.
- **`time_limit` was documented as `int`** in all eight docstrings that carry
  it, while every validator has always accepted any number and both solver
  backends honour a fractional value. It now reads `int or float`, and says that
  a fractional budget is honoured to a resolution of one millisecond — the MIP
  models round to whole milliseconds and clamp a positive sub-millisecond budget
  up to one, rather than discarding it.

### Continuous integration

- **`fail-fast: false`** on the test matrix. Cancelling the other five jobs on
  the first failure hid two further failures — the 3.13 one behind the 3.14
  one, and the Windows one behind both — for seven runs, and left an
  annotation naming only whichever job finished first.
- `actions/checkout`, `actions/setup-python` and `actions/upload-artifact`
  bumped to v5, v6 and v5, clearing the Node 20 deprecation warnings.

## 0.22.0 (2026-08-23)

First release of the fork, branched at upstream 0.21.0. On PyPI as
[`optbinning-slim`](https://pypi.org/project/optbinning-slim/0.22.0/); the import
package is `optbinning`.

### Requirements

- **Python >= 3.13.** Tested on 3.13 and 3.14; both are in CI and in the trove
  classifiers. Upstream 0.21.0 declared `>=3.7` and tested up to 3.12.
- **Dependency floors raised**, and the `ortools` upper cap **dropped**:
  `matplotlib>=3.10`, `numpy>=2.3`, `ortools>=9.14` (was `>=9.4,<9.12`),
  `pandas>=2.3`, `ropwr>=1.2`, `scikit-learn>=1.7`, `scipy>=1.15`. The cap was
  not protecting against anything the suite can reach: on OR-Tools 9.15 every
  solver status, split and IV is unchanged.

### Packaging

- `setup.py`, `requirements.txt` and `test_requirements.txt` replaced by a
  single PEP 621 `pyproject.toml`. `environment.yml` builds the conda
  environment and restates no dependency.
- The distribution is named **`optbinning-slim`**; the import package stays
  `optbinning`, so it is a drop-in replacement. The two distributions claim the
  same import package and must not share an environment.
- Upstream's Sphinx sources (`doc/`) and the unused Travis configuration were
  removed. Docstrings remain the user-facing documentation; upstream's rendered
  documentation still describes the library.

### Removed

- **`solver="ls"` and the LocalSolver integration.** `localsolver` is not
  distributed on PyPI and the `hexaly` package that succeeds it provides no
  `localsolver` module, so the option could not be used by anyone installing
  this package. `OptimalBinning` now accepts `("cp", "mip")`, matching every
  other estimator.

### Added

- `OptimalPWBinning` and `ContinuousOptimalPWBinning` accept
  **`solver="clarabel"`**. `ropwr` 1.2 resolves `solver="auto"` to Clarabel on
  constrained problems, and the estimators previously rejected the solver they
  were themselves using.
- `OptimalBinningSketch.plot_progress` takes **`savefig`** and `save_kwargs`,
  like every other plot method. It previously always called `plt.show()`, which
  blocks on an interactive backend.
- `PWContinuousBinningTable.quality_score`, which `analysis()` computed but did
  not publish, unlike the three sibling binning tables.

### Fixed

- **pandas 3.0 support.** A variable whose column carries pandas' string dtype
  was classified as numerical and its strings reached `check_array`, raising
  `could not convert string to float`. `BinningProcess` now tests
  `is_numeric_dtype` directly. As a consequence a categorical column is binned
  as categorical, where it was previously treated as numerical.
- **`read_json` produced an unusable estimator.** It rebuilt the binning table
  but restored none of the state `transform` and `splits` read, so a loaded
  estimator returned `None` from `.splits` and raised `TypeError` from
  `.transform`. Affected `OptimalBinning`, `ContinuousOptimalBinning` and
  `MulticlassOptimalBinning`.
- **`to_json` could not write a categorical binning at all** (`Object of type
  ... is not JSON serializable`), and the payload could not represent one: it
  saved the grouped categories rather than the split positions needed to
  rebuild them. Both are fixed, and the payload gained `splits_optimal`. Files
  written by earlier versions raise a `ValueError` naming the problem rather
  than loading a broken estimator.
- **MDLP applied its stopping criterion to the recursion only**, so every node
  with a viable candidate contributed a split and `min_samples_split` could not
  prevent the first one. The Fayyad-Irani criterion now decides whether the
  split is accepted. **This changes results**: on breast-cancer "mean radius"
  the default fit goes from 7 splits to 3, and
  `OptimalBinning(prebinning_method="mdlp")` from IV 4.79132740 to 4.76862756.
  A target carrying no information now yields no splits.
- **MDLP rejected an integral float target** with a numpy cast error that did
  not mention the target. It is now accepted; non-integral or negative labels
  raise a `ValueError` that names it.
- `ContinuousOptimalBinning2D` accepted `prebinning_method="mdlp"` and failed
  during `fit`; it is now rejected by parameter validation, like the 1D
  estimator.
- `OptimalBinningSketch`'s docstring advertised `solver="ls"`, which its own
  validation rejected.

### Testing

- 176 tests to **212**, and coverage from 85% to **89%**. The suite passes on
  Python 3.13 and 3.14.
- `MDLP`'s only two algorithmic tests were commented out upstream, leaving the
  recursion, split search and stopping criterion uncovered; they are replaced
  by property-based tests. JSON persistence had no tests at all.
- `tests/test_package.py` pins that ortools still solves after
  `import optbinning`. ortools and highspy ship colliding `libhighs.so.1`
  libraries, so the import order in `optbinning/__init__.py` is load-bearing —
  see `reports/OPEN_ITEMS.md`.

### Known issues

- cvxpy's HIGHS backend is unavailable in any process that imports optbinning,
  because ortools and highspy ship different HiGHS builds under the same
  SONAME. optbinning's own `solver="highs"` is unaffected. Full diagnosis in
  `reports/OPEN_ITEMS.md`.
