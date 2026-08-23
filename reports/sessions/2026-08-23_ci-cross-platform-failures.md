# 2026-08-23 — why CI had never once been green

## What was asked

"The github CI is not working." The report was the annotation list from a run:
one job failed with exit code 1, the other five said "canceled", plus Node 20
deprecation warnings.

## Result

214 passed on Python 3.13.15 and on 3.14.7 (was 212 + the 2 new regression
tests), `flake8 --select=E9,F63,F7,F82 --exclude=.venv` reports 0.

Two defects, both of which reproduce only off this development machine, which
is why the suite was green locally through all seven red runs:

- `MDLP._find_split` read candidate cuts per *row*, so ties in `x` made the
  result depend on the order `np.argsort` left tied rows in — not stable, and
  architecture-dependent.
- `Logger.close` iterated the list it was removing from, which only survives
  where CPython rebinds `Logger.handlers` (gh-79366). The macOS runners had
  3.14.6, which does not.

The reasoning and the measurements for both are in `DECISIONS.md`, which also
had to **retract** its own earlier "`Logger.close` is correct" entry: that
entry reasoned from the `>=3.13` floor, and the behaviour it relied on turns on
a *micro* version the supported range straddles.

## How the failure was found

Worth recording, because none of the usual moves worked.

The suite passed locally in every configuration tried: the project conda envs
on 3.13.15 and 3.14.7; a throwaway env built the way CI builds one
(`pip install -e ".[distributed,test]"`, same wheel versions as PyPI served
that day); pinned to 4 CPUs with `taskset` to match the runner; and headless
with `MPLBACKEND=Agg`, `CI=true` and a clean `HOME`, since the local matplotlib
backend is TkAgg and the runner's is Agg. 212 passed every time.

Reading the actual log needed a token — `GET
/repos/{owner}/{repo}/actions/jobs/{id}/logs` answers 403 "Must have admin
rights to Repository" unauthenticated, as do the artifact download and the log
archive, and there is no public HTML fallback. Only the check-run
*annotations* endpoint is public, and it carries nothing but "Process completed
with exit code 1".

**`fail-fast: true` was actively hiding half the evidence.** Each run reported
exactly one job and cancelled the other five, and which one survived was
whichever finished first — macOS in most runs, ubuntu in others. That made the
failure look platform-specific and rotating when it was neither: the logging
defect hits every 3.14.6 job and the MDLP defect hits every arm64 job. The
matrix is now `fail-fast: false`.

## What was left

- `MDLP.fit` accumulates splits across refits — found while in the file, filed
  in `OPEN_ITEMS.md` rather than fixed here, to keep this change to the defect
  it was chasing.
- No version bump. The fixes sit under **Unreleased** in `CHANGELOG.md`;
  0.22.0 had shipped to PyPI hours earlier and cutting 0.22.1 is a separate
  decision.
