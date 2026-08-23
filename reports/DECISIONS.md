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
