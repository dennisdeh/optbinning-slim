# PyPI release automation: GitHub Actions publishes the package

## Ask

Set up the integration between the `optbinning-slim` GitHub repository and the PyPI
package.

## Starting state

`optbinning-slim` 0.22.0 was on PyPI, uploaded by hand on 2026-08-23 (`python -m build`,
`twine upload`) — one sdist and one wheel, no attestations. Nothing connected the two
sides: CI built and `twine check --strict`-ed the distributions on every push but threw
them away as an artifact, the `v0.22.0` tag did nothing, the README's PyPI badge was a
hand-typed literal, and `README.md` documented releasing as a manual `twine upload`. The
2026-08-23 packaging session had named exactly this as the next step.

## What changed

| | |
|---|---|
| `.github/workflows/release.yml` | New. Fires on a `v*` tag push. Four jobs: `check-tag` → `ci` → `pypi` → `github-release`. |
| `.github/workflows/python-package.yml` | Gained a `workflow_call` trigger so the release reuses the matrix and the `package` job rather than duplicating them. No other change. |
| `tests/test_package.py` | Two guards added: the declared version must have a `## X.Y.Z (date)` section in `CHANGELOG.md`, and `release.yml` must keep declaring the `pypi` environment. |
| `README.md` | PyPI badge now reads the live version from shields.io instead of a literal. "Building a release" became "Releasing", describing the tag-push flow and the trusted-publisher binding. Test-count figures re-measured. |
| `CLAUDE.md` | The release rule now says that pushing the tag *is* the release, and that `twine upload` by hand defeats the workflow rather than racing it. |
| `CHANGELOG.md` | A "Release automation" block under Unreleased. |
| `reports/DECISIONS.md` | New entry recording why OIDC over a token, why the CI workflow is reused, why the tag is checked first, and why PyPI is uploaded before the GitHub release is opened. |

The pipeline: `check-tag` rejects a tag disagreeing with `optbinning/_version.py` or
missing its `CHANGELOG.md` section; `ci` runs the full six-cell matrix (a tag push
matches neither branch filter, so it would otherwise run untested) and produces the
`dist` artifact; `pypi` downloads that artifact and uploads it with
`pypa/gh-action-pypi-publish` under trusted publishing; `github-release` opens the
release with `gh`, attaching the same files and using the version's changelog section as
the body.

## Verification

Measured 2026-08-24, Python 3.13.15. The branch was first cut from `master` at d2b5f14 and verified there (1105 passed, 219 s); `master` then moved to 123f353 — the `metric_special` dict work — so the branch was rebased onto it and the whole verification re-run against the merged tree:

- **`pytest`** — **1110 passed, 0 failed**, 26 warnings, 215 s. (The two new tests here are the difference from 123f353's 1108; the README had been carrying 1095, a figure that predated both branches.)
- **`flake8 . --count --select=E9,F63,F7,F82 --exclude=.venv`** — **0**.
- **Both workflow files parse** as YAML; job graph, `needs`, `permissions` and the
  `pypi` environment confirmed by inspection of the parsed tree.
- **The `check-tag` shell logic was run against the real files**: `v0.22.0` → accept;
  `v0.23.0` and `v9.9.9` → reject, naming the `__version__` they disagree with.
- **The changelog extraction was run against the real `CHANGELOG.md`**: 99 lines for
  0.22.0, stopping before the next `## ` heading and containing nothing from
  `Unreleased`.
- **Both new tests were demonstrated red before green.** Setting `__version__` to
  `0.99.0` fails `test_version_has_a_changelog_section`; renaming the environment to
  `pypi-prod` fails `test_release_workflow_keeps_the_names_pypi_trusts`. Restored, all
  four tests in the module pass.
- **`python -m build` + `twine check --strict`** — both artifacts PASSED. The sdist
  contains `CHANGELOG.md` and `tests/`, and zero `.github` entries.
- **`pytest tests/test_package.py` inside the unpacked sdist** — 3 passed, 1 skipped,
  the skip being the workflow guard correctly standing down where `.github` was pruned.

Nothing was published. No tag was pushed, and the workflow has never run.

## Left open

- **The trusted publisher does not exist yet.** It has to be created by hand at
  <https://pypi.org/manage/project/optbinning-slim/settings/publishing/> — owner
  `dennisdeh`, repository `optbinning-slim`, workflow `release.yml`, environment `pypi`
  — before the first tag is pushed. Until then the `pypi` job fails at the upload step
  with `invalid-publisher`. Nothing in the repository can create it or detect that it is
  missing; the four names it binds are pinned in `release.yml`'s header and, for two of
  them, by `test_release_workflow_keeps_the_names_pypi_trusts`.
- **The `pypi` GitHub environment** is created implicitly on the first run. Adding a
  required reviewer to it would turn every release into a manual approval — worth doing,
  but it is a repository setting, not a file.
- The work was done on a worktree branch rather than in the main checkout as `CLAUDE.md` prescribes: at the time it started, the main checkout held uncommitted work from another task on `metric-special-dict`. That work reached `master` first (123f353); this branch was rebased onto it, pushed as `pypi-release-automation`, and fast-forwarded into `master` on 2026-08-24.
- The README's coverage figures (99%, 11,564 statements, 13 uncovered) were **not**
  re-measured here; only the pass count and wall clock were.
