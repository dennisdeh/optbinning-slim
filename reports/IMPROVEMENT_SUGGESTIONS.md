# Improvement suggestions

The code is correct and could be better. Not defects — those go to
`OPEN_ITEMS.md`.

*Last updated: 2026-08-23*

- **Dependencies are declared three times.** `setup.py` (`install_requires` /
  `extras_require`), `requirements.txt` and `test_requirements.txt` all list
  them, and CI installs the latter two before installing the package. Any
  version change has to be made in every copy. Consolidating on a
  `pyproject.toml` would remove the duplication and is the natural first step of
  the fork's dependency-slimming goal.
- **`ortools>=9.4,<9.12` is an upper-bound pin** that will hold the package back
  as OR-Tools moves; resolved to 9.11.4210 on 2026-08-23. Worth establishing
  what the pin protects against before raising it.
- **The `ecos` extra is required to pass the suite** but is not part of
  `[test]`; `tests/test_binning_piecewise.py::test_solvers` fails without it.
  Either fold `ecos` into `[test]` or make that test skip when ECOS is absent.
- **Stale `.venv/` in the checkout** (Python 3.14.4, no packages). It is
  gitignored so it costs nothing in the repository, but flake8 walks it and
  reports ~21 `F821`s from vendored pip code. Deleting it locally would remove
  the need for `--exclude=.venv`.
- **The CI matrix stops at Python 3.12.** The suite passes on 3.12 with
  numpy 2.5.2 / pandas 3.0.5 (2026-08-23); extending the matrix to 3.13 would
  turn the fork's maintenance goal into something CI enforces.
