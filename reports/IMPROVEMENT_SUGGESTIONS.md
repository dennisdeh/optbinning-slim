# Improvement suggestions

The code is correct and could be better. Not defects — those go to
`OPEN_ITEMS.md`.

*Last updated: 2026-08-23*

- **Coverage is 89%; the next gains are in four files.** Measured 2026-08-23
  after removing `ls.py`, the largest remaining gaps are
  `binning_statistics.py` (181 missed), `binning_process.py` (92),
  `continuous_binning.py` (80) and `cp.py` (59). They are mostly
  `information()` reporting paths and solver-specific branches rather than
  untested logic, so the yield per test is lower than what the 2026-08-23
  coverage pass found.

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
