# Changelog

Release notes for `optbinning-slim`, a maintenance fork of
[optbinning](https://github.com/guillermo-navas-palencia/optbinning). Versions
before 0.22.0 are upstream's; their notes are in the
[upstream documentation](http://gnpalencia.org/optbinning/release_notes.html).

The fork adds no algorithms. Everything below is packaging, currency with the
Python and dependency stack, defect fixes and test coverage.

## 0.22.0 (2026-08-23)

First release of the fork, branched at upstream 0.21.0.

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
