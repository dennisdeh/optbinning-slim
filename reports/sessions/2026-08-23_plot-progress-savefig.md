# 2026-08-23 — plot_progress gains savefig

## What was asked

Fix the `plot_progress` savefig issue filed in `IMPROVEMENT_SUGGESTIONS.md`.

## What changed

`OptimalBinningSketch.plot_progress` and the `plot_progress_divergence` helper
behind it now take `savefig` and `save_kwargs`, validated and applied exactly as
in `BinningTable.plot`, which was the reference for the wording of the guards
and the docstring. `savefig=None` keeps showing the figure, so nothing existing
breaks.

The test that covered it stubbed `plt.show` as a workaround for the blocking;
it now exercises the real path — writes a file and asserts it is non-empty,
passes `save_kwargs={"dpi": 50}`, and checks both `TypeError` guards. A second
test keeps covering the display branch, still stubbing `plt.show` since that
branch cannot be exercised safely on an interactive backend. Both were red
against the unfixed code with `plot_progress() got an unexpected keyword
argument 'savefig'`.

## Result

209 passed on Python 3.13.15, flake8 gate 0. The reasoning and the upstream
divergence are recorded in `DECISIONS.md`; the suggestion is removed from
`IMPROVEMENT_SUGGESTIONS.md` now that it is implemented.

At the time of writing this work was on branch `plot-progress-savefig` and
nothing had been merged.
