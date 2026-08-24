# 2026-08-24 — the coverage bug hunt

## What was asked

Iterate: find untested code, write tests for it (chaos, unexpected input, exceptions,
edge cases), run the suite, root-cause every failure to the *source* rather than the
test, ship a regression test shown red first, commit, report. Repeat until only
unreachable code is left, then clean that up and confirm a green baseline.

## Numbers

| | before | after |
|---|---|---|
| tests | 216 | (see below) |
| statements | 11,309 | 11,456 |
| uncovered statements | 1,245 | (see below) |
| statement coverage | 89% | (see below) |
| defects fixed | — | (see below) |

All runs on Python 3.13.15, `coverage run --source optbinning -m pytest`, from the repo
root. `flake8 --select=E9,F63,F7,F82 --exclude=.venv` reports 0 throughout.

## Method

Three rounds, each: author or fix → independently verify → audit.

Nothing was accepted on one agent's say-so. Round 1 wrote tests against the uncovered
lines and reported defect *candidates*; a second pass tried to **refute** each one, and
3 of 46 were refuted. Round 2 fixed the survivors; an **audit** pass then re-derived
every fix from the real `git diff` and re-ran the tests, and found blockers, majors and
several defects round 2 had itself introduced. Round 3 cleared those.

That third pass earned its keep. Two examples:

- A round-1 refutation was **wrong**, and a round-2 agent trusted it: it replaced a
  correct failing test with `with raises(Exception)`, which passes on the very
  `UnboundLocalError` the test existed to document. The auditor caught it by checking
  the refutation's premises against the tree rather than against the write-up.
- The degenerate-input contract (below) was written by the orchestrator to cover the
  binning **tables**. It did — and the audit showed the same kernel lives in the
  **transforms**, so `binning_table.build()` and `transform(metric="event_rate")` began
  reporting different numbers for the same bin. Fixing half of a duplicated kernel is
  worse than fixing none of it.

## What the defects had in common

Almost none were isolated. The binary, continuous, multiclass, 2D, piecewise and sketch
estimators are near-parallel implementations, and the same defect was usually present in
three or four of them — exactly the failure mode `CLAUDE.md` warns about. The families:

1. **Degenerate input was unhandled.** A single-class binary target, `user_splits=[]`,
   a constant `x`, a single row, an empty prebinning — each crashed somewhere different
   in each estimator. Addressed with one explicit contract rather than six local patches.
2. **`fit()` mutated its own constructor parameters**, so refitting the same instance
   was rejected by the estimator's own validation. Same class as the `MDLP.fit` reset
   recorded in `DECISIONS.md`.
3. **Parameters validated, documented, then ignored** — `split_digits` in 2D and in the
   sketch, `gamma` in the continuous CP model, four `monotonic_trend` values in
   multiclass, `min_bin_size`/`max_bin_size` in multiclass.
4. **Argument-order and shape slips** that silently returned wrong numbers, the worst
   being the piecewise transform returning the *non-event* rate for special and missing
   buckets.
5. **Guards that could not fire** — an `and` where the siblings use `or`, a helper never
   wired to its caller, a bool() on an ndarray, a branch after the `len()` that would
   have raised first.

## The degenerate-input contract

Written mid-session, when it became clear the estimators disagreed with each other about
what a degenerate target means. It is recorded in `reports/DECISIONS.md`; in short: a
single-class target is legal, `fit` reports OPTIMAL, the table builds, **event rate is
gated on records** so a pure bin reports its own rate, WoE/IV/JS stay gated on mixedness,
Gini is 0 rather than nan, no `RuntimeWarning` escapes, and 1D is the reference the 2D
estimators must match.

## Coverage, and what is left

(filled in at the end of the session)

## Filing

Defects that remain open are in `OPEN_ITEMS.md`; removal proposals and the remaining
ideas are in `IMPROVEMENT_SUGGESTIONS.md`; the standing judgements — above all the
degenerate-input contract and the list of deliberate divergences from upstream 0.21.0 —
are in `DECISIONS.md`.
