# 2026-08-23 — dependency update, Python 3.13/3.14

## What was asked

Update all dependencies to the newest versions, subject to `pandas>=2.3` and
`python>=3.13`, and fix what breaks.

## What was done

Recreated the `optbinning` conda env on Python 3.13.15 and raised every
dependency floor, dropping the `ortools<9.12` upper cap. The floor table and
the reasoning are in `DECISIONS.md`. Declarations were updated in all four
places that carry them: `setup.py`, `requirements.txt`, `test_requirements.txt`
and the README dependency list, plus `python_requires`, the trove classifiers,
the GitHub Actions matrix and the README Python badge.

Also added a second env, `optbinning314` (Python 3.14.7), to check the upper end
of the supported range.

## What broke, and why

Only the piecewise binning tests, and the cause was **ropwr 1.2 changing what
`solver="auto"` resolves to** — Clarabel now, ECOS before.

The frozen WoE literals in `test_bounds_transform` / `test_bounds_fit_transform`
are ECOS output asserted at `rel=1e-6`; Clarabel differs by 6.9e-06. Evidence
(2026-08-23, bounded fit, `lb=0.001`, `ub=0.999`):

| solver | IV | max rel. diff from the frozen values |
|---|---|---|
| auto | 4.4803499947 | 6.899e-06 |
| ecos | 4.4803380870 | 7.123e-09 |
| osqp | 4.4803368381 | 4.547e-07 |
| scs | 4.4802358078 | 1.804e-05 |

Following that thread surfaced a real defect: `_check_parameters` in
`optbinning/binning/piecewise/base.py` rejected `solver="clarabel"`, so the
solver `"auto"` selects could not be named explicitly. Fixed, with docstrings
updated and `test_solver_clarabel` added — red with `ValueError: Invalid value
for solver` before the fix.

The bounded-transform tolerances were widened to `rel=1e-5`; see `DECISIONS.md`
for why widening beat regenerating.

## Result

**176 passed** on Python 3.13.15 (181.9 s) and **176 passed** on Python 3.14.7
(180.3 s); `flake8 --select=E9,F63,F7,F82 --exclude=.venv` reports 0. No
failures remain. The `OPEN_ITEMS.md` entry about piecewise tolerance drift,
filed earlier the same day, is resolved and removed.

One new open item: ortools' bundled HiGHS makes cvxpy's HIGHS backend
unimportable. It does not affect optbinning's own `solver="highs"`, which goes
through scipy.

At the time of writing this work was on branch `deps-python313` and nothing had
been merged.
