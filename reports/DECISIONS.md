# Decisions

Examined, found correct or deliberately chosen, not to be re-raised. Each entry
records the reasoning, so the next session does not reopen it. Divergences from
upstream `guillermo-navas-palencia/optbinning` belong here — this file is what
makes the next upstream merge tractable.

## Variable dtype detection tests for numeric, not for object

*Last updated: 2026-08-23*

`_check_variable_dtype` (`optbinning/binning/binning_process.py`) now returns
`"numerical"` when `pandas.api.types.is_numeric_dtype` holds and
`"categorical"` otherwise. Upstream tests `x.dtype == object`.

Reason: since pandas 3.0 a column of strings gets pandas' own string dtype
(`StringDtype`, backed by `ArrowStringArray`), not object dtype. Under the
upstream test such a column is classified `"numerical"` and the strings reach
`sklearn.utils.check_array`, which raises `ValueError: could not convert string
to float: 'A'`. This is what broke `tests/test_scorecard.py::test_missing_metrics`
on pandas 3.0.5 (2026-08-23).

Two consequences of inverting the test rather than adding string dtype to it,
both accepted deliberately:

- A **categorical** dtype column is now `"categorical"`. Under the upstream
  test it was `"numerical"`, which was almost certainly not intended.
- A **datetime64** column is now `"categorical"`, where upstream called it
  `"numerical"`. The library has no datetime handling anywhere (`grep -rn
  datetime optbinning/` is empty as of 2026-08-23) and no test covers it, so
  neither answer is supported; the direct expression of the question was
  preferred over preserving an untested accident.

Pinned by `tests/test_binning_process.py::test_string_dtype_variable_is_categorical`.

## The multiclass non-numerical guard is correct as it stands

*Last updated: 2026-08-23*

`MulticlassOptimalBinning.fit` guards with `x_clean.dtype == np.dtype("object")`
and this **looks** like the same defect as the one above. It is not: `split_data`
(`optbinning/binning/preprocessing.py`) normalises a pandas string array back to
an object-dtype ndarray before the guard runs, so the guard still fires and the
caller still gets `"must be numerical"`. Verified 2026-08-23 on pandas 3.0.5.

A change here was written, found to have no failing test to justify it, and
reverted. `tests/test_multiclass_binning.py::test_string_dtype_not_numerical`
was kept as a pin on the normalisation, and is explicitly labelled as a pin
rather than a regression test — it has never been red.

## Dependency floors, and Python >= 3.13

*Last updated: 2026-08-23*

The fork requires Python >= 3.13 and raises every dependency floor:

| package | upstream 0.21.0 | here | resolved 2026-08-23 |
|---|---|---|---|
| matplotlib | (none) | >=3.10 | 3.11.1 |
| numpy | >=1.16.1 | >=2.3 | 2.5.2 |
| ortools | >=9.4,<9.12 | >=9.14 | 9.15.6755 |
| pandas | (none) | >=2.3 | 3.0.5 |
| ropwr | >=1.0.0 | >=1.2 | 1.2.0 |
| scikit-learn | >=1.6.0 | >=1.7 | 1.9.0 |
| scipy | >=1.6.0 | >=1.15 | 1.18.1 |

The `ortools<9.12` upper cap is **dropped**. It was not protecting against
anything reachable by the suite: on 9.15.6755 the whole suite passes, solver
statuses are unchanged, and no IV or split expectation moved. Restore the cap
only with a failing test that justifies it.

Python 3.14 is supported, not merely allowed: the suite passes on 3.14.7 with
the same resolved dependency versions (176 passed, 2026-08-23), so 3.14 is in
the CI matrix and in the classifiers.

The `ecos` extra is now also part of the `test` extra, because
`test_binning_piecewise.py::test_solvers` fails without ECOS installed. The
standalone `ecos` extra is kept — it is documented in the README.

## The piecewise `"auto"` solver is Clarabel, and `"clarabel"` is now accepted

*Last updated: 2026-08-23*

`ropwr` 1.2 changed what `solver="auto"` resolves to: on a constrained problem
it is now Clarabel, where it used to be ECOS. Two consequences, both handled
here rather than worked around:

**`"clarabel"` is accepted by the piecewise estimators.** `_check_parameters`
in `optbinning/binning/piecewise/base.py` listed
`("auto", "ecos", "osqp", "direct", "scs", "highs")` — every ropwr solver except
the one `"auto"` actually selects, so a user could not name it explicitly.
`"clarabel"` was added, with the docstrings of `OptimalPWBinning` and
`ContinuousOptimalPWBinning` updated to match. Pinned by
`tests/test_binning_piecewise.py::test_solver_clarabel`, which was red
(`ValueError: Invalid value for solver`) before the change.

**The bounded-transform tolerances were widened from `rel=1e-6` to `rel=1e-5`**
in `test_bounds_transform` and `test_bounds_fit_transform`. The frozen values
are ECOS output — measured 2026-08-23, ECOS reproduces them to 7.1e-9 while
Clarabel differs by 6.9e-06, i.e. the literals encode the solver, not the
library. Clarabel's answer is a correct optimum at its own tolerance: it
respects the `lb=0.001` / `ub=0.999` bounds to 2.4e-08, and the fitted IV agrees
with ECOS's to 2.7e-06 relative.

The values were widened rather than regenerated deliberately, so that the test
keeps failing if the *library* changes while tolerating the solver swap. The
better long-term fix is to pin the objective instead of four transformed
values — filed in `IMPROVEMENT_SUGGESTIONS.md`.

Note that `"auto"` is not always Clarabel: on the unconstrained problem in
`test_solvers` it resolves to the direct solver, which is why that test still
asserts `rel=1e-6` while `test_solver_clarabel` needs `rel=5e-4`.

## `.travis.yml` is deleted

*Last updated: 2026-08-23*

Upstream still carries `.travis.yml`; this fork does not. It configured Travis
CI — which the project does not use, the CI badge and the gating suite both
being GitHub Actions — on `dist: xenial` with Python 3.6 and 3.7, against a
package that now declares `python_requires>=3.13`. Nothing referenced it: no
badge, no workflow, no documentation.

Removed 2026-08-23 at the maintainer's request. If Travis is ever wanted again,
write a new file rather than reviving this one; every version and platform in it
was obsolete.

## `read_json` restores the estimator state, not just the table

*Last updated: 2026-08-23*

`read_json` set `_is_fitted = True` and rebuilt the binning table, but
`transform` and `splits` read the estimator's own attributes, which stayed
`None`. A loaded estimator therefore returned `None` from `.splits` and raised
`TypeError: object of type 'NoneType' has no len()` from `.transform` — that is,
persistence did not survive a round trip in any of the three estimators that
offer it.

`OptimalBinning._restore_from_binning_table` now copies back what the table
carries, driven by the `_restored_from_table` mapping, and all three
`read_json` methods call it. The mapping is `hasattr`-guarded because the three
tables carry different statistics: binary has `n_event`/`n_nonevent`,
continuous has `n_records`/`sums`, multiclass has `n_event` only.

`dtype` and `special_codes` are constructor parameters, and restoring them from
the file mutates them on load. That is deliberate: a loaded estimator must
describe the bins in the file, not the arguments that happened to be passed to
`__init__`. The categorical case cannot be restored at all — see
`OPEN_ITEMS.md`.

Pinned by `test_json_round_trip` in `tests/test_binning.py`,
`tests/test_continuous_binning.py` and `tests/test_multiclass_binning.py`, all
three red before the change.

## `Logger.close` was **not** correct — the Python floor is not the boundary

*Last updated: 2026-08-23*

An earlier revision of this file argued that

```python
for handler in self.logger.handlers:
    handler.close()
    self.logger.removeHandler(handler)
```

was safe on this project's supported Pythons, because `removeHandler`
**replaces** the handler list rather than mutating it (gh-79366) from 3.13
onward, so the loop keeps iterating the original list. That reasoning treated a
*minor* version as the boundary. It is a **micro** version boundary, and the
supported range straddles it:

| interpreter | `removeHandler` | `test_close` |
|---|---|---|
| 3.13.15 | rebinds | passes |
| **3.14.6** | **mutates in place** | **fails** |
| 3.14.7 | rebinds | passes |

The macOS CI runners were on CPython 3.14.6, which is inside `requires-python
>= 3.13`, and `test_close` failed there — one `FileHandler` survived `close()`
and leaked the file handle. `close()` now iterates `list(...)`, which is
correct under either semantics.

The lesson generalises: a `>=X.Y` floor says nothing about which micro version
of `X.Y` a user or a runner has, so it cannot license depending on a change
that shipped in a point release. Pinned by
`tests/test_logging.py::test_close_when_removehandler_mutates_in_place`, which
monkeypatches the mutating semantics so the property holds regardless of the
interpreter the suite happens to run on; it is red against the old loop on
every Python.

## `MDLP.fit` resets `_splits`, and does so after validation

*Last updated: 2026-08-24*

`_splits` was initialised in `__init__` and only ever appended to, by
`_recurse`, so refitting an instance returned the previous fit's splits as
well as the new ones — and because `splits` returns `np.sort(self._splits)`,
they came back interleaved rather than obviously doubled. Measured 2026-08-23
on breast-cancer `"mean radius"`: 3 splits, then 6.

Nothing inside optbinning hit it — `PreBinning` builds a fresh `MDLP` per fit
— so it only ever affected users reusing an instance, and no test covered a
refit.

The reset sits **after** parameter and target validation rather than at the
top of `_fit`, so a call rejected for a bad `min_samples_leaf` or a
non-integral target leaves the estimator on its last good fit instead of
silently emptying it. Pinned by
`tests/test_mdlp.py::test_refit_replaces_previous_splits` and
`::test_refit_on_different_data`, the second of which is the sharper test: it
refits on a separable target and used to return the first fit's three splits
with `99.5` appended.

## The test suite pins matplotlib's Agg backend

*Last updated: 2026-08-23*

`tests/conftest.py` calls `matplotlib.use("Agg", force=True)`. The plotting
tests only ever save figures, so the backend is not part of what they assert,
and letting matplotlib choose costs correctness rather than buying anything:
it picks a GUI backend wherever it believes there is a display, which on
Windows is always. The hosted CPython 3.13.15 on the Windows runners ships a
Tcl tree that `tkinter` cannot initialise — `TclError: Can't find a usable
init.tcl` — and `test_multiclass_binning.py::test_numerical_default` died on
it. The 3.14 Windows job passed, so this is per-runner-image, not per-Python.

It had never been reported because `fail-fast: true` cancelled the Windows
jobs before they finished, every run. Pinning the backend also means a
contributor on a headless box gets the same suite CI does.

## MDLP candidate cuts are read per distinct value, not per row

*Last updated: 2026-08-23*

`_find_split` took its candidate cut points from the rows where the label
changes:

```python
u_x = np.unique(0.5 * (x[1:] + x[:-1])[(y[1:] - y[:-1]) != 0])
```

With ties in `x` this depends on the order `np.argsort(x)` leaves tied rows in
— and `np.argsort` defaults to a **non-stable** quicksort whose tie order is
SIMD- and architecture-dependent. The same data therefore discretised
differently on different machines.

Measured 2026-08-23 on breast-cancer `"mean radius"` (569 rows, 456 distinct
values, 24 tied values carrying both labels): permuting the input rows moved
the IV of `OptimalBinning(prebinning_method="mdlp")` over the range **3.85 to
4.82**. The Linux x86-64 jobs happened to land on 4.76862756 and the macOS
arm64 jobs on 3.92842062, which is the whole of the `test_prebinning_method`
CI failure.

The fix reads the labels **per distinct value of x** and applies the
Fayyad-Irani boundary rule: the midpoint between two adjacent values is a
candidate cut unless every observation at both values carries the same single
label. Because it only ever consults the set of labels present at a value, the
candidate set — and so the whole discretisation — is a function of the
`(x, y)` pairs alone.

Two alternatives were measured and rejected:

- `np.argsort(x, kind="stable")` — deterministic across machines, but the
  answer still depends on the order the rows arrive in, and the IV falls to
  3.86987386.
- `np.lexsort((y, x))` — order-independent, but it fixes an arbitrary
  convention rather than removing the arbitrariness, and the IV falls to
  3.84224645.

The boundary rule leaves `test_prebinning_method` at its documented IV of
**4.76862756**, so it corrects the defect without moving the number the earlier
MDLP work measured. Pinned by `tests/test_mdlp.py::test_row_order_independent`,
red against the old candidate set.

## MDLP split values are interpolated, so literal expectations are fragile

*Last updated: 2026-08-23*

`tests/test_mdlp.py` had its only two algorithmic tests commented out, leaving
the recursion, the split search and the stopping criterion at 33% coverage.
They were not restored with their literal values: those values no longer hold
(measured 2026-08-23, default `MDLP` on breast-cancer "mean radius" gives 7
splits where the commented test expected 6, and the shared values differ in the
third decimal).

The reason is in `_find_split`: when there are more candidates than
`max_candidates`, it takes `np.percentile(candidates, percentiles)` of the
candidate midpoints. Percentile interpolation returns values *between*
midpoints, so the split points are not observed midpoints and they move with
numpy's interpolation. Any value between two adjacent observations induces the
same partition, so this is not a defect — but it does mean a literal
expectation pins numpy, not optbinning.

This is the *remaining* source of fragility in the split values. The other one
— the candidate set itself moving with the order tied `x` values were sorted
into — was a defect, and is fixed; see the boundary-point entry above. The
counts quoted here are from before that work and before the stopping-criterion
fix: the default `MDLP` fit on "mean radius" gives 3 splits as of 2026-08-23.

The replacements assert properties instead: splits are strictly increasing,
strictly inside the range of x, each one actually partitions the sample, no leaf
is smaller than `min_samples_leaf`, a separable target yields exactly one split
at the boundary, a single-class target yields none, and `min_samples_leaf >= n`
yields none.

## The five 2026-08-23 defects, and how each was resolved

*Last updated: 2026-08-23*

All five were found by the coverage pass earlier the same day, filed in
`OPEN_ITEMS.md`, and fixed on request. Each fix ships a test that was red
against the unfixed code.

**The json format now carries categorical bins.** `to_dict` saved
`table.splits` — the *output* of `bin_categorical` — so nothing could rebuild
the estimator's split positions. It now also saves `splits_optimal`, the
positions themselves, and `read_json` restores them through
`_restore_json_payload`. Two further breakages surfaced while fixing it and are
part of the same repair: `to_json` could not write a categorical binning at all
(`TypeError: Object of type ArrowStringArray is not JSON serializable`, because
`categories` / `cat_others` / `user_splits` were never converted with
`tolist()`), and reading one back would have failed anyway because the
per-bin category arrays have different lengths and `np.array` cannot hold them
in one rectangular array. Files written before this change raise a `ValueError`
naming the problem rather than failing later — the format is not
backward compatible for categorical bins, and silently loading a broken one is
worse than refusing it.

**MDLP applies its stopping criterion to the split, not just to the
recursion.** `self._splits.append(split)` moved inside the
`if not self._terminate(...)` branch. This changes results: on breast-cancer
"mean radius" the default fit goes from 7 splits to 3, and
`OptimalBinning(prebinning_method="mdlp")` from IV 4.79132740 with 4 splits to
IV 4.76862756 with 3 (measured 2026-08-23). The old behaviour added one
unjustified cut per terminal branch; a target carrying no information now
yields no splits at all, and `min_samples_split` can prevent the first split,
which is what it documents.

**MDLP accepts an integral float target.** `np.bincount` needs integer labels,
so `_fit` now casts a float target whose values are integral and raises a
`ValueError` naming the target when they are not, or when a label is negative.
Previously a `df["target"].values` column produced a numpy cast error that did
not mention the target at all.

**`ContinuousOptimalBinning2D` rejects `prebinning_method="mdlp"` at parameter
validation**, matching the 1D estimator, instead of accepting it and failing
inside `PreBinning` during `fit`. Its docstring no longer lists mdlp.

**`PWContinuousBinningTable` publishes `quality_score`**, and `analysis()` sets
`_is_analyzed`, which it never did — so the property is guarded like the ones
on its three sibling tables. Only the accessor was missing; the value was
already computed.

While mapping the last one, `OptimalBinningSketch`'s docstring was found to
advertise `solver="ls"` although its `_check_parameters` allows only `"cp"` and
`"mip"`. The docstring was corrected.

## `solver="ls"` and the LocalSolver integration were removed

*Last updated: 2026-08-23*

`optbinning/binning/ls.py` is deleted, `OptimalBinning` accepts `("cp", "mip")`
only, and `information.py` no longer carries the `LSStatistics` import guard or
its two `solver_type == "ls"` branches. This is a divergence from upstream,
which still ships the integration.

Why, established by the investigation on 2026-08-23:

- **`localsolver` is not on PyPI.** `pip index versions localsolver` returns
  `No matching distribution found`. It was only ever distributed through the
  vendor's own installer, with a commercial licence.
- **Its successor does not provide the same import.** LocalSolver was renamed
  Hexaly; `hexaly` *is* on PyPI (15.0.20260812). Inspecting that wheel, its only
  top-level package is `hexaly` (`hexaly/optimizer.py`, `libhexaly150.so`) —
  there is no `localsolver` compatibility module, so
  `from localsolver import LocalSolver` could not be satisfied by it either.

So the option was unusable for anyone installing this package, which is why
`ls.py` sat at 9% coverage with no test able to reach it: 203 statements whose
only reachable line was the `ImportError` guard.

What went, in full: `ls.py`; in `binning.py` the `BinningLS` import, the
`("cp", "ls", "mip")` allowed list, the `elif self.solver == "ls"` dispatch and
four docstring mentions — including the note that `max_pvalue` was unsupported
under `"ls"` and the advice to raise `max_n_prebins` above 100 only with it, both
of which now describe nothing. Nothing in `tests/` referenced the solver; the
suite gained `tests/test_binning.py::test_params`' check that `solver="ls"` is
now rejected with `Invalid value for solver`.

Reintroducing it means restoring `ls.py` from history (`git show <rev>:optbinning/binning/ls.py`)
and porting it to the `hexaly` API, which cannot be tested here without a
licence. Prefer leaving it out.

## Packaging: `pyproject.toml` only, `environment.yml` for conda, no `doc/`

*Last updated: 2026-08-23*

`setup.py`, `requirements.txt` and `test_requirements.txt` were replaced by a single
PEP 621 `pyproject.toml` (setuptools backend, `setuptools>=77` for the SPDX
`license` expression). The three files declared the same dependency list three times;
CI installed two of them and then installed the project on top, so a floor could be
raised in one place and silently contradicted in another.

`environment.yml` takes `requirements.txt`'s place for conda users. It deliberately
restates **no** dependency: it pins only the interpreter from `conda-forge` and then
pip-installs `--editable .[dev]`, so `pyproject.toml` stays the single source. That
also matches how the `optbinning` env was actually built — `conda list` showed every
package except Python itself coming from pip.

The distribution is named **`optbinning-slim`**, not `optbinning`. `optbinning` on
PyPI is upstream's and cannot be claimed by this fork (checked 2026-08-23:
`optbinning` → 200, `optbinning-slim` → 404). The *import* package name is unchanged,
which keeps the fork a drop-in replacement but means the two distributions must never
be installed into the same environment — they own the same import path. The version is
still `0.21.0`, from `optbinning/_version.py`, read by setuptools as a literal.

`MANIFEST.in` ships `tests/` and `tests/data/` in the sdist so the released tarball can
be tested, and prunes `tests/results/`, whose contents are generated by a test run.

Upstream's Sphinx sources under `doc/` were deleted. They are *user* documentation of a
library this fork does not change the behaviour of, they were 4.7 MB of `.rst` that no
build in this repository consumed, and a stale copy is worse than a link. The fork now
points at <http://gnpalencia.org/optbinning/> and at upstream's tree. The seven images
`README.md` embeds were kept — moved to `assets/` and referenced by absolute
`raw.githubusercontent.com` URL, because relative links do not render on PyPI, which
serves the same `README.md` as the long description.

`optbinning.egg-info/` was deleted from the checkout. It is build metadata, is matched
by `*.egg-info/` in `.gitignore`, and was never tracked; an editable install
regenerates it (as `optbinning_slim.egg-info/` now).

## `plot_progress` takes `savefig`, unlike upstream's

*Last updated: 2026-08-23*

`OptimalBinningSketch.plot_progress` called `plt.show()` unconditionally and
took no arguments, so on an interactive backend it blocked until the window was
closed and there was no way to use it from a script or a test. Measured
2026-08-23 on this machine (matplotlib backend `tkagg`, `DISPLAY=:0`), the test
covering it took **1163 s of a 1344 s** suite; the same test now runs in 0.6 s.

It now takes `savefig` and `save_kwargs` and validates them exactly as
`BinningTable.plot` and every other plot method in the library do — string path
or `TypeError`, dict or `TypeError`, `plt.savefig(...)` then `plt.close()`.
`savefig=None` still shows the figure, so the default behaviour is unchanged
and this is additive for existing callers.

This is a divergence from upstream, which still has the no-argument version.
Pinned by `tests/test_binning_sketch.py::test_plot_progress` for the file path
and the two type guards, and `::test_plot_progress_show` for the display branch,
which stubs `plt.show` so it cannot block.

## The import order in `optbinning/__init__.py` is load-bearing

*Last updated: 2026-08-23*

`optbinning/__init__.py` imports `.binning` first, which pulls in ortools before
anything reaches ropwr or cvxpy. That order is not stylistic and must not be
"tidied": ortools and highspy each ship a `libhighs.so.1` built from a different
HiGHS version, and the first one loaded wins for the whole process. Prepending
`import cvxpy` to `__init__.py` on 2026-08-23 made `import optbinning` itself
raise `libortools.so.9: undefined symbol: ...HighsLogOptions...`. The full
diagnosis is in `OPEN_ITEMS.md`.

`tests/test_package.py::test_ortools_works_after_importing_optbinning` pins it.
It runs in a subprocess, because import order only means anything in a fresh
interpreter, and it solves a small CP-SAT model rather than merely importing —
an import alone would not prove the solver still works. The probe above makes
both tests in that file fail, so the check has teeth.

## The degenerate-input contract

*Last updated: 2026-08-24*

Before 2026-08-24 the estimators disagreed with each other about what a
degenerate input means, and each crashed somewhere different. A single-class
binary target made `binning_table.build()` raise sklearn's `Found array with 0
sample(s)` out of `jeffrey`, *after* `fit()` had already reported `"OPTIMAL"`;
`user_splits=[]` raised `TypeError` in the binary estimator, `UnboundLocalError`
in the continuous one, and `ValueError` from `check_array` in the multiclass and
scenario ones; the 2D estimators raised where the 1D ones fitted.

The contract now implemented everywhere, and the reasoning for each clause:

1. **A binary target carrying a single class is legal, not an error.** It is the
   natural result of filtering, of a rare event, or of a small fold in a
   cross-validation loop. `fit()` returns `"OPTIMAL"` with no splits.
2. **Event rate is gated on records; WoE, IV and JS are gated on mixedness.**
   These are different questions. An all-event bin *has* an event rate — 1.0 —
   and reporting 0.0 for it (which the mixedness gate did) is simply a wrong
   number. WoE genuinely is undefined without both classes, so that gate stays.
   This distinction is the single most repeated defect in the package: the same
   kernel is copy-pasted into the four binning tables and the four transforms,
   and on 2026-08-24 fixing only the tables made `build()` and `transform()`
   disagree for the same bin. **Both halves must move together.**
3. **Gini is 0, not nan,** for a single-class table. `metrics.gini` already
   returned 0 from its `n <= 1` branch for the same degeneracy; the multi-bin
   path divided by `te * tne == 0`. The function was internally inconsistent.
4. **No `RuntimeWarning` may escape** `build()`, `plot()`, `analysis()` or
   `transform()`. Guard the divides; do not silence them with `np.errstate`,
   which would hide the next one.
5. **1D is the reference implementation.** Where 2D and 1D disagreed, 2D was
   changed. There is no reason for the same degenerate input to be legal in one
   and an error in the other.
6. **`user_splits=[]` means "no split points", not an error** — a single bin.
   Nothing documents an empty list as unsupported, `_check_parameters` accepts
   it, and `_prebinning_refinement` in every estimator already early-returns the
   right empty arrays for it.

Pinned across `tests/test_binning_edge_cases.py`,
`tests/test_binning_statistics.py`, `tests/test_transformations.py`,
`tests/test_binning_2d_edge_cases.py`,
`tests/test_binning_piecewise_edge_cases.py` and the sibling edge-case modules.

## `fit()` must not mutate its constructor parameters

*Last updated: 2026-08-24*

Four estimators rewrote `self.user_splits_fixed` into a numpy `bool_` array
during `fit`, and pruned `self.user_splits` in place. Two consequences, both
measured 2026-08-24:

- **Refitting the same instance raised.** `_check_parameters` tests
  `isinstance(s, bool)`, which `np.bool_` fails, so the second `fit()` died with
  `ValueError: user_splits_fixed must be list of boolean` — on all four of
  `OptimalBinning`, `ContinuousOptimalBinning`, `MulticlassOptimalBinning` and
  `SBOptimalBinning`.
- **`sklearn.base.clone` and `get_params` round-trips were unsound**, and a
  refit after pre-binning had pruned a split silently used fewer splits than the
  caller asked for.

Each estimator now keeps its working copy in a private `_user_splits` /
`_user_splits_fixed` and leaves the public parameter as the caller passed it —
the sklearn contract, and the same class of defect as the `MDLP.fit` reset
recorded above.

The recurring shape is worth stating once: **a constructor parameter is input,
not scratch space.** When `fit` needs a normalised, sorted or pruned version, it
belongs in a private attribute.

## Sibling parity is the dominant failure mode in this package

*Last updated: 2026-08-24*

Of the defects found on 2026-08-24, very few were isolated. The binary,
continuous, multiclass, 2D, piecewise and sketch estimators are near-parallel
implementations, and the same defect was typically present in three or four of
them — because the code was copied and then only one copy was maintained.
Upstream's own history shows the mechanism: commit `c3ca5e7` (2021) converted
three `analysis()` methods to `n_bins - 1 - self._n_specials` and left the two
piecewise ones at `n_bins - 2`; commit `5efbfc0` (2022) then taught both
piecewise `build()` and `plot()` about `_n_specials` and again skipped
`analysis()`.

Practical consequence for anyone working here, beyond what `CLAUDE.md` already
says: after fixing a defect, grep the tree for the *kernel* — the two or three
lines of the computation — not for the symbol name. The copies have different
function names, different variable names and different surrounding code, but the
kernel is usually byte-identical.

## `time_limit` must be finite; 0 is valid for binning and not for counterfactuals

*Last updated: 2026-08-24*

The eight `_check_parameters` functions guarded `time_limit` with
`not isinstance(time_limit, numbers.Number) or time_limit < 0`. `nan < 0` and
`inf < 0` are both False, so both passed validation and reached the solver,
where what happened depended on the backend. Measured 2026-08-24 on
`OptimalBinning`:

| `time_limit` | `solver="cp"` | `solver="mip"` |
|---|---|---|
| `nan` | `MODEL_INVALID`, no splits | `UNKNOWN`, no splits |
| `inf` | `OPTIMAL`, 5 splits | `OverflowError: cannot convert float infinity to integer` |
| `-inf` | rejected (by `< 0`) | rejected (by `< 0`) |

Neither `nan` answer is usable and neither is an error the caller can act on:
`MODEL_INVALID` is reported as an ordinary status, so a `nan` budget produced a
silently empty binning. `inf` was worse than useless — it *worked* under `"cp"`
and crashed under `"mip"`, which is exactly the backend divergence the rest of
the 2026-08-24 `time_limit` work removed.

**All eight validators now reject a non-finite `time_limit`.** `inf` is rejected
rather than honoured as "no limit": nothing documents it as meaningful, the
parameter is documented in seconds, and the default is already a large finite
number. A caller who wants effectively no limit can pass a large finite one.

The finiteness test is written `not -np.inf < time_limit < np.inf` rather than
`math.isfinite(time_limit)`. Both reject `nan`, `inf` and `-inf`; the comparison
form additionally cannot raise `OverflowError` on an arbitrary-precision int,
where `math.isfinite(10**400)` does. A `10**400` budget still passes validation
and still fails inside the solver, exactly as before — that is untouched, not
overlooked.

**`time_limit=0` stays valid for the seven binning estimators and stays invalid
for `Counterfactual`.** This is a deliberate divergence between the two, not
drift:

- For a binning estimator 0 is a well-defined "no budget": both backends report
  `UNKNOWN` and return the single all-in-one-bin fallback, which is a coherent
  answer and is pinned by
  `tests/test_binning_solvers.py::test_zero_time_limit_is_accepted_and_means_no_budget`.
- A counterfactual search has no equivalent fallback to return, so
  `Counterfactual.generate` keeps `<= 0`, as it always has, and
  `tests/test_counterfactual.py` pins that rejection.

The shared message was the reason this looked like an inconsistency: all eight
said "time_limit must be a positive value in seconds" while seven of them
accepted 0. Each message now states its own rule — "a finite non-negative value"
for the binning estimators, "a finite positive value" for the counterfactual.

Pinned by `test_non_finite_time_limit_is_rejected_1d`,
`test_non_finite_time_limit_is_rejected_2d`,
`test_negative_time_limit_is_still_rejected` and
`test_zero_time_limit_is_accepted_and_means_no_budget` in
`tests/test_binning_solvers.py`, and by
`test_generate_rejects_a_non_finite_time_limit` in
`tests/test_counterfactual_edge_cases.py`. All were red before the change.

## A dict `metric_special` names one value per special bucket

*Last updated: 2026-08-24*

`_check_metric_special_missing` has always had an `elif isinstance(
metric_special, dict)` branch validating one number per key, and its
fall-through message advertised "a dict" as an allowed form — while
`_apply_transform` handed the dict itself to numpy, so any dict raised
`TypeError: float() argument must be a string or a real number, not 'dict'`.

Two readings were possible: delete the validator branch, or implement the
per-key semantics. **The per-key reading was taken**, on three pieces of
evidence:

- `git log -L` puts the dict branch in upstream `51445f0`, *"Support treatment
  of special codes separately"* — the same commit that introduced **named**
  (dict) `special_codes`. The two features arrived together and only one was
  finished.
- The validator is asymmetric: `metric_special` takes a dict, `metric_missing`
  does not. There is exactly one missing bucket and arbitrarily many named
  special buckets, which is only a meaningful distinction under the per-key
  reading.
- `_apply_transform` already loops `for i, (k, s) in enumerate(
  special_codes.items())` with the bucket name `k` bound — and never used it.
  `metric_special="empirical"` already resolves to a different value per
  bucket, so the addressing machinery was there; only the lookup was missing.

Deleting the branch would have thrown all three away to save four lines.

**Two cross-checks are enforced**, in `_check_metric_special_dict`, which runs
where `special_codes` and `metric_special` are first known together:

- a dict `metric_special` with non-dict `special_codes` raises — there are no
  names to map on to. This is what the 2D estimators always hit, since
  `special_codes_x` / `special_codes_y` accept only list and ndarray;
- a dict that omits a bucket raises, naming it. The alternative — silently
  substituting 0 — is the class of defect the 2026-08-24 work spent four rounds
  removing. It also catches a mistyped key, since the real key then goes
  missing. Extra keys are *not* rejected: they are harmless, and rejecting them
  would break passing one dict to a `BinningProcess` whose variables carry
  different special codes.

The `metric="indices"` rule is applied per key, exactly as the scalar form
applies it to all buckets: an int is used as given, anything else means "use the
bucket's own index".

Pinned by the five `test_metric_special_dict_*` tests in
`tests/test_transformations.py`, all red against the unfixed code. Note the
index test asserts 77 and 88 rather than the obvious 8 and 9 — an eight-bin fit
gives the two buckets indices 8 and 9, so that assertion passed without the dict
being read at all.

Fifteen `transform` / `fit_transform` docstrings now declare
`float, str or dict`. The eight that still say `float or str` are the estimators
whose `_check_parameters` does not accept a dict `special_codes` at all — the
sketch, scenario and 2D families — and are correct as they stand.
