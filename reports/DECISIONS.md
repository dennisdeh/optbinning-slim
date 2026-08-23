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

## `Logger.close` is correct, despite appearing to iterate what it mutates

*Last updated: 2026-08-23*

```python
for handler in self.logger.handlers:
    handler.close()
    self.logger.removeHandler(handler)
```

This reads as the classic mutate-while-iterating bug that would skip the second
handler and leak the file handle. It is not one on this project's supported
Pythons: since 3.13, `logging.Logger.removeHandler` **replaces** the handler
list rather than mutating it, so the `for` loop keeps iterating the original
list and visits every handler. Verified 2026-08-23 on 3.13.15 by watching
`id(logger.handlers)` change on each removal; both handlers are removed.

It *would* skip on 3.12 and earlier, where the list is mutated in place, but
`python_requires` is `>=3.13`. Pinned by `tests/test_logging.py::test_close`.
Do not "fix" it by iterating a copy without first checking the Python floor.

## MDLP split values are interpolated, so literal expectations are fragile

*Last updated: 2026-08-23*

`tests/test_mdlp.py` had its only two algorithmic tests commented out, leaving
the recursion, the split search and the stopping criterion at 33% coverage.
They were not restored with their literal values: those values no longer hold
(measured 2026-08-23, default `MDLP` on breast-cancer "mean radius" gives 7
splits where the commented test expected 6, and the shared values differ in the
third decimal).

The reason is in `_find_split`: when there are more candidates than
`max_candidates`, it takes `np.percentile(u_x, percentiles)` of the candidate
midpoints. Percentile interpolation returns values *between* midpoints, so the
split points are not observed midpoints and they move with numpy's
interpolation. Any value between two adjacent observations induces the same
partition, so this is not a defect — but it does mean a literal expectation
pins numpy, not optbinning.

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
