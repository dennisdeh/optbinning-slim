"""
Package-level import properties.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import re
import subprocess
import sys

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(code):
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)


def test_ortools_works_after_importing_optbinning():
    # ortools and highspy (a cvxpy dependency) both ship a library with the
    # SONAME libhighs.so.1, built from different HiGHS versions, and only the
    # first one loaded is used. ortools must load first. Importing cvxpy,
    # ropwr or highspy before the ortools-backed modules loads the other one,
    # and then libortools fails on an undefined HiGHS symbol: `import
    # optbinning` itself raises and every solver stops working. Verified
    # 2026-08-23 by prepending `import cvxpy` to optbinning/__init__.py.
    # See reports/OPEN_ITEMS.md.
    result = _run(
        "import optbinning\n"
        "from ortools.sat.python import cp_model\n"
        "model = cp_model.CpModel()\n"
        "v = model.NewIntVar(0, 10, 'v')\n"
        "model.Maximize(v)\n"
        "solver = cp_model.CpSolver()\n"
        "status = solver.Solve(model)\n"
        "assert solver.StatusName(status) == 'OPTIMAL'\n"
        "assert solver.Value(v) == 10\n")

    assert result.returncode == 0, result.stderr


def test_public_api():
    result = _run(
        "import optbinning\n"
        "for name in optbinning.__all__:\n"
        "    assert hasattr(optbinning, name), name\n")

    assert result.returncode == 0, result.stderr


def _declared_version():
    # Read as a literal, the way setuptools and .github/workflows/release.yml
    # both read it. Importing the package would answer a different question.
    text = (ROOT / "optbinning" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "(.*)"$', text, re.M)
    assert match, "no __version__ literal in optbinning/_version.py"
    return match.group(1)


def test_version_has_a_changelog_section():
    # The release workflow refuses a tag whose version has no
    # `## X.Y.Z (YYYY-MM-DD)` section, and takes the GitHub release notes from
    # that section. That check can only fire on the tag push, by which point
    # the version bump is committed and pushed; this one fires on the bump.
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("CHANGELOG.md is not present in this installation")

    version = _declared_version()
    heading = re.compile(r"^## %s( |$)" % re.escape(version), re.M)

    assert heading.search(changelog.read_text(encoding="utf-8")), (
        "CHANGELOG.md has no '## %s (YYYY-MM-DD)' section; pushing tag v%s "
        "would fail the release workflow and open an empty GitHub release"
        % (version, version))


def test_release_workflow_keeps_the_names_pypi_trusts():
    # PyPI issues the upload token only to a run matching the owner, the
    # repository, the workflow *filename* and the environment name registered
    # as the trusted publisher. Renaming either of the last two leaves CI green
    # and breaks the next release with `invalid-publisher`. See
    # reports/DECISIONS.md.
    workflow = ROOT / ".github" / "workflows" / "release.yml"
    if not workflow.exists():
        pytest.skip(".github is pruned from the sdist")

    text = workflow.read_text(encoding="utf-8")

    assert re.search(r"^\s+name: pypi$", text, re.M), (
        "release.yml no longer declares the `pypi` environment that the "
        "trusted publisher on PyPI is registered against")
