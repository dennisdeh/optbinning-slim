# Improvement suggestions

The code is correct and could be better. Not defects — those go to
`OPEN_ITEMS.md`.

*Last updated: 2026-08-23*

- **Dependencies are declared three times.** `setup.py` (`install_requires` /
  `extras_require`), `requirements.txt` and `test_requirements.txt` all list
  them, and CI installs the latter two before installing the package. Any
  version change has to be made in every copy — the 2026-08-23 dependency
  update had to touch all three plus the README. Consolidating on a
  `pyproject.toml` would remove the duplication and is the natural next step of
  the fork's dependency-slimming goal.
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
