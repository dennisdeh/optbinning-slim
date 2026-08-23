# 2026-08-23 — the five open defects, and the LocalSolver question

## What was asked

Fix all five defects filed earlier the same day in `OPEN_ITEMS.md`, and
investigate whether localsolver can easily be removed.

## Result

209 passed on Python 3.13.15 (was 201), coverage 88%, `flake8
--select=E9,F63,F7,F82 --exclude=.venv` reports 0. The wider flake8 count over
`optbinning` and `tests` is 41 both before and after, so nothing new was
introduced.

All five are fixed and each fix has a test shown red first; the reasoning for
each is in `DECISIONS.md`. `OPEN_ITEMS.md` is down to one entry, the
ortools/highspy symbol clash, which is not ours to fix.

Two of the five were worse than the original diagnosis said:

- The categorical json path was not merely lossy on read — `to_json` could not
  **write** a categorical binning at all, and reading one back would have
  failed on the ragged per-bin category arrays even with the split positions
  restored. All three breakages are fixed together.
- `PWContinuousBinningTable.analysis` never set `_is_analyzed`, so the missing
  accessor was one of two problems, not one.

The MDLP fix changes results, deliberately: the criterion now decides whether a
split is accepted rather than only whether to recurse. On breast-cancer "mean
radius" the default fit drops from 7 splits to 3, and
`OptimalBinning(prebinning_method="mdlp")` from IV 4.79132740 to 4.76862756.
The literal in `tests/test_mdlp.py::test_prebinning_method` was updated in the
same commit for that reason.

Two extra defects were found and fixed. `OptimalBinningSketch`'s docstring
advertised `solver="ls"`, which its own parameter validation rejects. And the
verification run took 22 minutes instead of the usual 3.5: the durations named
`tests/test_binning_sketch.py::test_plot_progress`, added the previous session,
at **1163 s**. `plot_progress` calls `plt.show()` with no `savefig` option, so
on this machine's interactive backend it blocked on a real window. The test now
stubs `plt.show` and runs in 0.27 s. The library-side gap is filed in
`IMPROVEMENT_SUGGESTIONS.md`.

That slowdown was first put down to other test suites running on the same
machine. That was wrong, and the durations said so — worth remembering before
blaming a shared machine again.

## LocalSolver

`solver="ls"` cannot be used by anyone installing this package from PyPI:
`localsolver` is not distributed there, and the `hexaly` wheel that succeeds it
contains no `localsolver` module. Evidence and the full removal footprint are in
`DECISIONS.md` and `IMPROVEMENT_SUGGESTIONS.md`.

Nothing was removed. It is a documented public option of `OptimalBinning`, and
this fork's standing rule is to propose removals rather than make them.

At the time of writing this work was on branch `fix-open-items` and nothing had
been merged.
