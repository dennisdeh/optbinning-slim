# 2026-08-23 — coverage pass, and what it found

## What was asked

Improve coverage across the package with new unit tests; for any bug found,
determine the root cause and the blast radius.

## Numbers

| | before | after |
|---|---|---|
| tests | 176 | 201 |
| statements missed | 1730 | 1447 |
| coverage | 85% | 87% |

Both runs on Python 3.13.15, `coverage run --source optbinning -m pytest`.
`flake8 --select=E9,F63,F7,F82 --exclude=.venv` reports 0; the E501s in
`binning.py` and `continuous_binning.py` and the E302 in
`test_binning_piecewise.py` all pre-date this work.

## Where the tests went

Coverage was measured first and the gaps ranked by missed statements. The ones
worth closing, and what was written for each:

- **`mdlp.py`, 33%** — the module's only two algorithmic tests were commented
  out, so the recursion, the split search and the stopping criterion never ran.
  Eight property-based tests replace them, plus the first test of
  `prebinning_method="mdlp"`, which is the only path into `PreBinning`'s mdlp
  branch. Why properties and not the original literals: `DECISIONS.md`.
- **`to_json` / `read_json`, no tests at all** — round-trip tests for all three
  estimators that offer persistence. These found the read_json defect below.
- **`binning_statistics.py`, 222 missed** — `plot(style="actual")`,
  `plot(add_special=False, add_missing=False)`, `show_bin_labels=True`, the
  categorical plot path, and the guard that rejects `style="actual"` for
  categorical bins.
- **`piecewise/binning_statistics.py`, 78 missed** — `analysis()` for both
  piecewise tables, including the `pvalue_test="fisher"` path and the two
  parameter guards.
- **`distributed/plots.py`, 13%** — `OptimalBinningSketch.plot_progress`.
- **`formatting.py`, `logging.py`** — new `tests/test_formatting.py` and
  `tests/test_logging.py`.

`ls.py` (9%) was deliberately left alone; see `DECISIONS.md`.

## Bugs

**Fixed — `read_json` did not restore the fitted state.** All three estimators
with JSON persistence returned `None` from `.splits` and raised `TypeError:
object of type 'NoneType' has no len()` from `.transform` after a load. Root
cause and the fix are in `DECISIONS.md`; the three round-trip tests were red
against the unfixed code.

**Filed, not fixed** — each needs a maintainer decision, and each is in
`OPEN_ITEMS.md` with its reproduction:

- `read_json` cannot restore a **categorical** binning at all: the file holds
  the grouped categories, not the split positions needed to rebuild them.
  Fixing it changes the on-disk format.
- **MDLP emits a split its own stopping criterion rejects** — noise data yields
  a split, and `min_samples_split` cannot prevent the first one. Fixing it
  changes the output of a public algorithm and of every
  `prebinning_method="mdlp"` fit.
- **MDLP rejects a float target** with a numpy cast error.
- **`ContinuousOptimalBinning2D`** accepts `prebinning_method="mdlp"` and fails
  at `fit`, where the 1D estimator rejects it at validation.
- **`PWContinuousBinningTable`** publishes none of the statistics `analysis()`
  computes, unlike its three sibling tables.

**Examined and cleared** — `Logger.close` iterates the handler list it removes
from, which is the classic skip-every-other bug on Python <= 3.12 and is
correct on this project's floor of 3.13. Reasoning in `DECISIONS.md` so it is
not re-raised.

At the time of writing this work was on branch `more-coverage` and nothing had
been merged.
