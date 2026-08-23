# CLAUDE.md — optbinning-slim

*How to work in this repository. What the code is, read from the code.*

---

## Project overview  **[core]**

`optbinning-slim` is a fork of [optbinning](https://github.com/guillermo-navas-palencia/optbinning)
(currently at upstream **0.21.0**, no local divergence yet). It is a scikit-learn-style
Python library that solves the *optimal binning* problem — discretising a variable into
bins that are optimal with respect to a binary, continuous or multiclass target — via
mathematical programming (CP / MIP / local search over OR-Tools). It is consumed as an
importable library (`import optbinning`), not as an application.

The fork has three standing goals, in this order:

1. **Maintenance** — keep parity with upstream, fix bugs, keep it working on current
   Python (the upstream CI matrix stops at 3.12).
2. **Slim dependencies** — reduce and modernise the dependency surface and packaging
   (`setup.py` → `pyproject.toml`, the `ortools>=9.4,<9.12` pin, matplotlib as a hard
   requirement).
3. **No feature removal.** Despite the repo name, **do not strip subsystems.**
   `distributed`/sketch, `uncertainty`, `counterfactual`, `multidimensional`,
   `piecewise` and the plotting methods all stay. Everything exported from
   `optbinning/__init__.py` is public API and must keep working. If you believe
   something should be removed, *propose it and wait* — do not remove it as part of
   another task.

- **Primary language / runtime:** Python (library supports ≥3.7; CI tests 3.9–3.12)
- **Entry point:** there is no application entry point. The one command that exercises
  the whole thing is `pytest`, run from the repo root.
- **Central concept:** an **optimal binning estimator** — `OptimalBinning` and its
  siblings, all subclasses of `BaseOptimalBinning` (`optbinning/binning/base.py`),
  following the sklearn contract: `fit` / `transform` / `fit_transform`, plus the
  optbinning-specific `binning_table`, `splits`, `status`, `information()`.
  Configuration is constructor keyword arguments, validated by a module-level
  `_check_parameters` function in the same file as the estimator.

## Environment  **[core]**

```bash
source ~/miniforge3/etc/profile.d/conda.sh   # conda is NOT on the default PATH
conda activate optbinning
```

- The env is **`optbinning`** (conda, under `~/miniforge3/envs`). It may not exist on a
  fresh machine — create it, do not silently fall back to another env:

  ```bash
  conda create -n optbinning python=3.12 -y
  conda activate optbinning
  pip install -e ".[distributed,test,ecos]"   # ecos: CI installs it; test_solvers fails without it
  ```

- **Ignore `.venv/`.** There is a gitignored `.venv/` (Python 3.14.4) in the checkout
  that is **empty** — no numpy, no pytest. It is not the project environment; running
  `.venv/bin/python` produces `ModuleNotFoundError` and wastes a turn.
- Dependency definitions live in `setup.py` (`install_requires` / `extras_require`) and
  are duplicated in `requirements.txt` + `test_requirements.txt`, which is what CI
  installs. **A dependency change must be made in all the places that declare it** —
  grep for the package name before declaring the change done. There is no
  `pyproject.toml`, no `setup.cfg`, no pytest/flake8 config file.
- **Run everything from the repo root.** Several tests write to hard-coded relative
  paths (`tests/results/*.png`); from any other directory they fail on a missing
  directory, which reads as a code failure and is not one.
- No secrets, no `.env`, no network access, no external service. A fresh clone plus the
  env above is a complete setup.

## Key conventions  **[core]**

- **Parameter validation is separate from the estimator.** Each estimator module has a
  module-level `_check_parameters(...)` called from `fit`, raising `TypeError` /
  `ValueError`. New constructor arguments are validated there, not in `__init__`.
- **`_is_fitted` and `_check_is_fitted()`** (`optbinning/binning/base.py`) are the
  fitted-state protocol — not sklearn's `check_is_fitted`. Not fitted raises
  `NotFittedError`.
- **`solver` is a string enum, per estimator:** `"cp"`, `"ls"`, `"mip"` (and
  `mip_solver` in `("bop", "cbc")`). Adding a solver means touching the checker, the
  dispatch in `fit`, and the corresponding `*_cp.py` / `*_mip.py` / `ls.py` model.
- **`transform` returns a *metric*, not bin indices** — `metric="woe"` by default, with
  `metric_special` / `metric_missing` controlling the special-code and missing buckets.
  Changing a default here changes downstream `Scorecard` numbers.
- **`special_codes` accepts an array, a list *or a dict*.** The dict form (named special
  buckets) is easy to miss and is separately validated; handle both.
- **`binning_table` is a property that builds an object** (`binning_statistics.py`), and
  `.build()` / `.plot()` on it are what tests assert against. It is not a DataFrame
  attribute.
- **`status` is the solver's status string**, surfaced verbatim from OR-Tools. Do not
  normalise it — tests assert on `"OPTIMAL"`.
- The Sphinx documentation in `doc/` is upstream's user-facing documentation. Docstrings
  are its source. Changing a public signature means changing the docstring in the same
  commit.

---

## Git workflow  **[core]**

- Feature work happens on **branches in the main checkout** (no worktrees):
  `git checkout -b <name>`.
- Every task ends with the full flow unless told otherwise: commit, push, fast-forward
  merge into `master`, push, delete the local **and** remote branch.
- **Never merge until the test suite has actually finished and reported PASS.** Not on
  an in-flight run, not on a backgrounded one. See *Testing*.
- **There is no `gh` CLI on this machine and no PR workflow.** Do not reach for `gh`,
  and do not propose opening a PR — merge locally and push.
- The remote is `origin` → `https://github.com/dennisdeh/optbinning-slim.git`.
  **There is no `upstream` remote configured.** Syncing with
  `guillermo-navas-palencia/optbinning` requires adding it explicitly first — say so and
  ask before adding a remote. Run `git remote -v` before diagnosing any push/fetch
  failure.
- `.idea/` (PyCharm) is untracked and **not** in `.gitignore`. Never `git add -A` from
  the repo root without checking `git status` first.

## Testing  **[core]**

- Run the suite **in the foreground with a bounded timeout**, from the repo root. Do not
  background the runner or spawn polling loops unless explicitly agreed.

  ```bash
  pytest                                                        # 20 test modules
  flake8 . --count --select=E9,F63,F7,F82 --exclude=.venv --show-source --statistics
  ```

  Both must pass before a merge into `master`. The `flake8` selection above is exactly
  what CI gates on; CI's second, `--exit-zero` flake8 pass is advisory and does not gate.
  **`--exclude=.venv` is not optional locally** — without it flake8 walks the stale
  `.venv/` and reports ~21 `F821`s from vendored pip code that do not exist in this
  repository. CI has no `.venv`, which is why the CI command omits the flag.
- **Report exact pass/fail counts before claiming verification.** "Tests pass" without
  numbers is not a report.
- Binning problems are solver runs, so the suite is not instant. If it exceeds ~10
  minutes, narrow to the relevant module (`pytest tests/test_binning.py`) and **say
  explicitly that you narrowed it** — a silently narrowed run reads as a full one.
- The suite is fully offline: no network, no credentials. Keep it that way. Data
  fixtures are in `tests/data/` (`breast_cancer.csv`, `boston_housing.csv`,
  `breast_cancer.parquet` — the parquet one needs `pyarrow` from the `test` extra).
- Tests **write** PNG and CSV artifacts into `tests/results/`. Some of those files are
  tracked. Check `git status` after a run and do not commit regenerated plot artifacts
  as part of an unrelated change.
- `tdigest` / `pympler` (the `distributed` extra) gate the sketch tests, and `ecos` gates
  `test_binning_piecewise.py::test_solvers`; without them
  those modules fail to import rather than skipping. Install the extras.

## Debugging  **[core]**

- **State the root cause with evidence — a log, a reproducing command, or a failing
  test — before editing code.** A patch without a stated cause is a guess with a diff
  attached.
- **Every fix ships a regression test demonstrated to FAIL against the unfixed code.**
  Stash the fix, run the test, paste the red output. A test written after the fix and
  never seen red proves nothing.
- **Distinguish a bug from solver non-determinism.** OR-Tools can return a different
  optimal solution of equal objective across versions and platforms. Before calling a
  changed split point a regression, check whether the objective/IV is unchanged. Pin
  behaviour on the objective and on `status`, not on incidental split values, unless
  the split value is the property under test.
- **Root-cause discipline:** when correcting a wrong expectation, fixture or literal,
  grep the *whole tree* for the same value before declaring it fixed
  (`rg -n "<value>" tests/`). The binary / continuous / multiclass / 2D estimators are
  near-parallel implementations — the same expectation is very often duplicated in a
  sibling test module, and the same defect very often exists in a sibling source module.
  Fixing one and not the others is the characteristic failure here.
- **A scoped grep answers a scoped question.** "Who references this?" is a whole-tree
  question — including `doc/`, the README, and docstrings.
- Do not treat a prior session's "already fixed" list as an exclusion list. It is a
  point-in-time record, stale by construction. Judge every path on today's source.

---

## Documentation  **[core]**

Two locations, and the split is strict:

- **`reports/`** — *this fork's* development documentation: a small, fixed set of files,
  each answering exactly one **standing** question. Do not add a new top-level file
  without asking. Create the folder and the files below on first use.
- **`doc/`** — upstream's Sphinx *user* documentation. `.rst` + docstrings. It describes
  the library to its users; it is not a place for development notes.

### Where a fact goes

| the fact | the file |
|---|---|
| the code does something other than what it should | `reports/OPEN_ITEMS.md` |
| the code is correct and could be better | `reports/IMPROVEMENT_SUGGESTIONS.md` |
| examined, found correct, not to be re-raised | `reports/DECISIONS.md` |
| what happened in this piece of work | `reports/sessions/YYYY-MM-DD_<slug>.md` |
| how a public class behaves and what its methods branch on | its docstring, and `doc/` |

`OPEN_ITEMS` vs `IMPROVEMENT_SUGGESTIONS` is *"is something wrong?"*, not *"is something
worth doing?"*. A finding that turns out to be by design moves to `DECISIONS.md`
**with its reasoning**, so the next session does not reopen it. Divergences from
upstream — and the reason for each — belong in `DECISIONS.md`; that file is what makes
the next upstream merge tractable.

### Keeping it current

- **Changing a module means updating every document that describes it, in the same
  commit** — its docstrings first, then `doc/` and the README if they name it.
  Documentation that lags the code is worse than none, because it is trusted.
- **Renaming, inverting or deleting a test is a documentation change.** Documents credit
  tests *by name* with pinning a property; a rename silently breaks the credit.
- **Every document carries its vintage.** Each section carries `*Last updated:
  YYYY-MM-DD*`, refreshed when **its** content changes — not when the file is touched
  for something else. Session files are exempt: they are dated by filename.
- **Never write branch or merge state as present tense.** Point-in-time records outlive
  the branches they describe. Write "at the time of writing, nothing was merged", or
  give the date.
- **Update the existing file; do not create a parallel one.** A second document covering
  the same question is how a `reports/` folder reaches 77 files.
- **State a fact once.** If it belongs in two documents, put it where the question is
  answered and link from the other — a fact stated twice goes stale once.
- **Anchor to symbol names, never line numbers**, in code comments too. Every
  `file.py:NNN` in a real repository was stale when audited.
- **Date every measurement.** "measured 2026-08-23: 4.2 s → 1.1 s" stays checkable; a
  bare number does not.
- **End a session by filing what is left.** A session report alone is not filing:
  defects go to `OPEN_ITEMS.md`, enhancements to `IMPROVEMENT_SUGGESTIONS.md`,
  judgements to `DECISIONS.md`.

### Where prose lives

**A code comment says *what* and points; docstrings and `reports/` say *why*, with the
evidence.** Measurements, dated incidents, solver-version quirks and the reasoning
behind a constant belong in the docstring or in `reports/`, not inline.

- **Keep in the code:** what the value is, the one-sentence trap, the pointer.
- **Never delete.** Relocating is the only acceptable way to shorten a comment that
  records a defect or a measurement. A fact with no home is a regression waiting to be
  re-introduced.
- **Do not number steps** (`# 1: validate`) — the numbers drift the moment a step is
  inserted. **Do not restate the next line.**
- Upstream's authorship/copyright header at the top of each module stays. Do not
  rewrite it when editing a file.

## Before investigating, check the docs  **[core]**

Search `reports/` and prior session notes before starting any investigation from
scratch. Reuse and update existing analysis rather than regenerating it — but check its
vintage first. For anything that looks like an upstream bug, also check whether upstream
has already fixed it before writing a patch.

---

## Growing this file

In order of value:

1. **A mechanical check** — a test, a flake8 rule, a CI gate. Every check should exist
   because a defect got through.
2. **A rule in this file**, when a check is impossible or not yet worth writing.
3. **Nothing.** A rule nobody follows is worse than no rule: it trains the reader that
   this file is decoration.

**Keep this file about *how to work here*, not *what the code is*.** Anything the agent
can learn by reading the code belongs in the code. What belongs here is what the code
cannot tell it: which command to trust, what "done" means, what broke last time, and
which plausible-looking action is the wrong one.

**Prune as well as add.** Delete a rule when its check exists, when the subsystem it
guards is gone, or when it has never once been the thing that went wrong.

*Last updated: 2026-08-23*
