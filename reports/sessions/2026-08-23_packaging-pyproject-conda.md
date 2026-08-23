# Packaging: pyproject.toml, conda environment file, doc/ removed

## Ask

Replace `requirements.txt` with a conda environment file, remove `doc/` and refer to
upstream instead, prepare the package for a PyPI upload without uploading it, bring the
setup up to current practice, and delete `optbinning.egg-info/`.

## What changed

| | |
|---|---|
| `pyproject.toml` | New. PEP 621 metadata, setuptools backend, version read from `optbinning/_version.py`. Dependencies and the `distributed` / `ecos` / `test` extras moved here verbatim; a `dev` extra was added (all extras plus `build` and `twine`). |
| `setup.py` | Deleted, with its `CleanCommand`. |
| `requirements.txt`, `test_requirements.txt` | Deleted. |
| `environment.yml` | New. Creates the `optbinning` conda env: interpreter from `conda-forge`, everything else from `pip install --editable .[dev]`. |
| `MANIFEST.in` | New. Ships `tests/` and `tests/data/` in the sdist, prunes `tests/results/`, `.github/`, `.idea/`, `reports/`. |
| `doc/` | Deleted (57 files, 4.7 MB). The seven images `README.md` embeds moved to `assets/`. |
| `optbinning.egg-info/` | Deleted from the checkout. |
| `README.md` | Installation section rewritten (PyPI name, conda env, extras, release commands); dependency table now points at `pyproject.toml`; a Documentation row explains where the Sphinx docs went; image links are absolute so PyPI renders them. |
| `.github/workflows/python-package.yml` | No longer installs the removed requirements files. New `package` job builds sdist + wheel, runs `twine check --strict`, uploads the artifacts. |
| `CLAUDE.md` | Environment, dependency-declaration, documentation-layout and test-module-count passages brought in line. |

The reasoning behind each choice — the distribution name in particular — is in
[`DECISIONS.md`](../DECISIONS.md).

## Verification

Measured 2026-08-23, Python 3.13.15, in a clean clone of the branch:

- `python -m build` → `optbinning_slim-0.21.0.tar.gz` and
  `optbinning_slim-0.21.0-py3-none-any.whl`.
- `python -m twine check --strict dist/*` → PASSED for both.
- Wheel `top_level.txt` is `optbinning` alone; the wheel carries no tests. The sdist
  carries 28 `tests/` entries including the three data fixtures, and zero
  `tests/results/` entries.
- `pip install -e ".[distributed,test]"` then `import optbinning` → 0.21.0.
- `pytest` → **201 passed, 0 failed**, 9 warnings, 19 test modules. Wall clock was
  796 s, but three pytest processes were sharing the machine; a run of the same suite
  earlier the same day, uncontended, took 189 s. The time is not a measurement of this
  change.
- `flake8 . --count --select=E9,F63,F7,F82` → **0**.

Nothing was uploaded to PyPI. `python -m twine upload dist/*` is the remaining step and
needs a PyPI token; `optbinning-slim` was unregistered when checked on 2026-08-23.

## Note on the run

The first full `pytest` run happened in the shared checkout while another session was
editing `optbinning/binning/*` and `tests/*` on the `fix-open-items` branch; it reported
206 passes, which covered that session's uncommitted work as well. The figures above are
from a clean clone of `packaging-pypi-conda` alone — 201 passes, this branch's own base.
The shared checkout was restored to the other session's state (its branch, its
uncommitted edits, its editable `optbinning` install), and this work was committed to
`packaging-pypi-conda` without moving its `HEAD`. Nothing in `optbinning/` or `tests/`
is touched by this branch, so the two should merge cleanly; `CLAUDE.md`,
`reports/DECISIONS.md` and `reports/IMPROVEMENT_SUGGESTIONS.md` are edited by both and
may not.

## Left open

- The branch was not merged into `master` and not pushed; pushing does not work from an
  agent shell.
- `README.md`'s test-status table still reports 176 passed over 17 modules, figures that
  predate both this branch (201 over 19) and the `fix-open-items` work (209). Refreshing
  it belongs with whichever of the two merges last.
- No PyPI publishing workflow was added. Releasing is the documented manual
  `python -m build && python -m twine upload dist/*`; a release-triggered workflow with
  a trusted publisher would be the next step if the fork is released regularly.
