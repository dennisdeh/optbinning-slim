# 2026-08-24 — MDLP.fit resets its splits

## What was asked

Fix the `MDLP` refit defect filed in `OPEN_ITEMS.md` at the end of the
2026-08-23 cross-platform CI work.

## Result

216 passed on Python 3.13.15 and on 3.14.7 (was 214 + the 2 new regression
tests), coverage 89%, `flake8 --select=E9,F63,F7,F82 --exclude=.venv` reports
0. `optbinning/binning/mdlp.py` is at 99% statement coverage.

`_splits` was initialised in `__init__` and only ever appended to by
`_recurse`, so refitting an instance returned the previous fit's splits as
well — interleaved, because `splits` sorts them, rather than obviously
doubled. `_fit` now resets it.

The reset sits **after** validation, not at the top of `_fit`. A call rejected
for a bad parameter or a non-integral target therefore leaves the estimator on
its last good fit rather than silently emptying it; resetting first would turn
a rejected refit into a second, quieter failure.

Two tests, both shown red first:

- `test_refit_replaces_previous_splits` — fit twice on the same data.
- `test_refit_on_different_data` — fit on breast-cancer, then on a separable
  target. This is the sharper of the two: it returned the first fit's three
  splits with `99.5` appended, which is what the defect actually looks like to
  a caller.

`OPEN_ITEMS.md` is back to one entry, the ortools/highspy symbol clash, which
is not ours to fix.

## Note

No caller inside optbinning was affected — `PreBinning` builds a fresh `MDLP`
per fit — so this never reached the binning estimators. It affected users
reusing an instance, and nothing in the suite covered a refit, which is why it
survived the 2026-08-23 coverage pass at 99% coverage of the module: the line
that was wrong was executed by every test, just never twice.
