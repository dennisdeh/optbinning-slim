# Improvement suggestions

The code is correct and could be better. Not defects — those go to
`OPEN_ITEMS.md`.

*Last updated: 2026-08-23*

- **Coverage is 87%; the next gains are in four files.** Measured 2026-08-23
  after the coverage pass, the largest remaining gaps outside `ls.py` (which is
  unreachable — see `DECISIONS.md`) are `binning_statistics.py` (182 missed),
  `binning_process.py` (92), `continuous_binning.py` (82) and `cp.py` (59).
  They are mostly `information()` reporting paths and solver-specific branches
  rather than untested logic, so the yield per test is lower than what the
  2026-08-23 pass found.

- **`OptimalBinningSketch.plot_progress` cannot be used non-interactively.**
  It calls `plt.show()` unconditionally and takes no `savefig` argument, unlike
  every other plot method in the library. On an interactive backend it blocks
  until the window is closed: measured 2026-08-23 on this machine (backend
  `tkagg`, `DISPLAY=:0`), the test covering it took **1163 s** of a 1344 s
  suite. `tests/test_binning_sketch.py::test_plot_progress` now stubs
  `plt.show`, which is a workaround in the test rather than a fix in the
  library. Giving it the `savefig` parameter its siblings have would fix it
  properly.

- **Stale `.venv/` in the checkout** (Python 3.14.4, no packages). It is
  gitignored so it costs nothing in the repository, but flake8 walks it and
  reports ~21 `F821`s from vendored pip code. Deleting it locally would remove
  the need for `--exclude=.venv`.
- **The reference values in `tests/test_binning_piecewise.py` are solver
  artifacts.** They were generated with ECOS and are asserted at a tolerance
  chosen to straddle whatever `solver="auto"` currently resolves to. A test
  that pinned the *objective* rather than four transformed values would not
  need re-tuning the next time ropwr changes its default solver. See
  `DECISIONS.md` for why the tolerance was widened rather than the values
  regenerated.

## Proposal: remove `solver="ls"` and the LocalSolver integration

*Last updated: 2026-08-23*

Why it is a candidate: the dependency is not installable from PyPI and its
successor does not provide the same import, so the option cannot be used by
anyone (evidence in `DECISIONS.md`). It is 203 statements at 9% coverage that
no test can reach.

Against removing it: `solver="ls"` is a documented public option of
`OptimalBinning`, and this fork's standing rule is not to strip subsystems. A
user with a legacy licensed LocalSolver install still has a working path.

What removal touches, in full — the footprint is small:

- `optbinning/binning/ls.py` — delete.
- `optbinning/binning/binning.py` — the `from .ls import BinningLS` import, the
  `("cp", "ls", "mip")` allowed list and its message, the `elif self.solver ==
  "ls"` dispatch, and four docstring or note mentions.
- `optbinning/information.py` — the `LSStatistics` import guard and the two
  `solver_type == "ls"` branches.
- Nothing in `tests/`: no test names the solver.
- Nothing in the documentation either, since `doc/` was dropped on 2026-08-23.
  Before that the footprint also covered `doc/source/tutorials.rst` and the
  notebook `tutorial_binary_localsolver.ipynb`.

Only `OptimalBinning` is affected. Every other estimator already allows
`("cp", "mip")` only.

A lighter alternative: keep the option and port `ls.py` to the `hexaly` API. It
cannot be tested here either — Hexaly needs a licence — so it would trade dead
code for untestable code.
