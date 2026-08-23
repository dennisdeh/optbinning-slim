# 2026-08-23 — removing the LocalSolver integration

## What was asked

Remove localsolver, following the investigation and the removal plan written
earlier the same day.

## What went

- `optbinning/binning/ls.py`, deleted in full.
- `optbinning/binning/binning.py`: the `BinningLS` import, the
  `("cp", "ls", "mip")` allowed list and its message, the
  `elif self.solver == "ls"` dispatch, and four docstring mentions. Two of those
  were notes that only made sense with the solver present — that `max_pvalue`
  was unsupported under `"ls"`, and that `max_n_prebins` above 100 was only
  recommended with it.
- `optbinning/information.py`: the `LSStatistics` import guard, the
  `solver_type == "ls"` branch of `solver_statistics` and the one in
  `print_solver_statistics`.

Nothing in `tests/` named the solver. `tests/test_binning.py::test_params` now
asserts that `solver="ls"` is rejected with `Invalid value for solver`, so the
removal is pinned rather than merely done.

Only `OptimalBinning` was affected. Every other estimator — continuous,
multiclass, both 2D, the sketch, the scenario binning — already allowed
`("cp", "mip")` only.

## Result

209 passed on Python 3.13.15, `flake8 --select=E9,F63,F7,F82 --exclude=.venv`
reports 0, and the wider flake8 count over `optbinning` and `tests` is 41, the
same as before the removal — one `E302` introduced by deleting the import guard
was fixed.

Coverage went from 87% to **89%**: 11509 statements with 1438 missed, to 11292
with 1256. Deleting 217 statements of which 194 were unreachable is most of
that; it is a smaller denominator, not more tested code.

The reasoning, the evidence that the dependency is uninstallable, and how to
reintroduce it if that ever changes, are in `DECISIONS.md`.

At the time of writing this work was on branch `remove-localsolver` and nothing
had been merged.
