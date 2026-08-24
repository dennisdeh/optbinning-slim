# Improvement suggestions

The code is correct and could be better. Not defects — those go to
`OPEN_ITEMS.md`.

## Removals to consider — proposed, not made

*Last updated: 2026-08-24*

This fork's standing rule is to **propose** removals rather than make them
(`CLAUDE.md`, goal 3). The 2026-08-24 coverage pass reached 99% and, in doing
so, identified code that nothing in the package calls. Each of these is
*reachable* by a user who imports the symbol directly, so none was deleted.

- **`BCatSketch._copy`** (`binning/distributed/bsketch.py`). `BSketch.merge`
  calls `BSketch._copy`; `BCatSketch.merge` merges its category dicts in place
  and never calls its own. Nothing in the package reaches it.
- **`binning/metrics.py::test_proportions`**. Nothing imports it.
  `binning/model_data.py` defines its own five-argument `test_proportions` that
  every caller uses. Note the name also makes it look like a test to a reader
  grepping for one.
- **`model_data(return_nonevent_event=True)`** (`binning/model_data.py`). No
  caller passes it, and it changes the *arity* of the return value (five items
  instead of six) while still computing the dropped one.
- **`continuous_model_data(scale=None)`** (same file). The sole library caller
  always passes `M = int(1e6)`; there is no continuous MIP model that would need
  the unscaled form.

## Keep — examined 2026-08-24 and deliberately not removed

*Last updated: 2026-08-24*

Recorded so a future coverage pass does not propose deleting them again:

- **Every solver-status branch** (`FEASIBLE`, `ABNORMAL`, `UNBOUNDED`,
  `MODEL_INVALID`, `NOT_SOLVED`) in `mip.py`, `mip_2d.py`, `rounding.py`,
  `counterfactual/mip.py` and `counterfactual/multi_mip.py`. These are
  unreachable from the public API today because these models are bounded and
  feasible, but they are the correct handling if a future OR-Tools returns one.
  Probed extensively on 2026-08-24 — NaN and infinite coefficients, 1e18
  magnitudes, zero time limits — and CBC answered `OPTIMAL` or `INFEASIBLE`
  every time.
- **The `pympler` and `tdigest` optional-import guards.** They can only be
  reached in an environment without the `distributed` extra.
- **`model_data_cart_2d.py::continuous_model_data_cart`'s `if sfr == 0:
  continue`.** Every sklearn leaf holds at least one sample, so the union of
  leaf regions is never empty.

## Other

*Last updated: 2026-08-24*

- **The suite takes ~4 minutes** and is dominated by real CP/MIP solver runs.
  The 2026-08-24 pass kept new tests cheap by preferring small synthetic arrays
  and by sharing fits through `functools.lru_cache` helpers, but the pre-existing
  modules still refit the breast-cancer dataset repeatedly. A session-scoped
  fixture for the common fits would cut wall clock without weakening anything.
- **`RoundingMIP` is the one MPSolver in the library with no time limit at all.**
  It never calls `SetTimeLimit`, so `Scorecard(rounding=True)` on a large
  scorecard has no bound on how long it can take, and the test suite has to reach
  into `round_mip.solver_` to set one by hand.
- **The reference values in `tests/test_binning_piecewise.py` are solver
  artifacts.** They were generated with ECOS and are asserted at a tolerance
  chosen to straddle whatever `solver="auto"` currently resolves to. Pinning the
  *objective* rather than four transformed values would not need re-tuning the
  next time ropwr changes its default solver. See `DECISIONS.md`.
- **Stale `.venv/` in the checkout** (Python 3.14.4, no packages). Gitignored, so
  it costs nothing in the repository, but flake8 walks it and reports ~21 `F821`s
  from vendored pip code. Deleting it locally would remove the need for
  `--exclude=.venv`.
- **The wider flake8 count is 28** (2026-08-24, down from 41). What remains is
  pre-existing `E501`/`E303`/`W291` in `continuous_binning.py`, `binning.py` and
  a few test modules. None is in the CI-gated selection.
