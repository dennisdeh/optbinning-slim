"""
Package-level import properties.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import subprocess
import sys


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
