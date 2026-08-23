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
