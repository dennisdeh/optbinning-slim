"""
General information routines.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from sklearn.base import BaseEstimator

from ._version import __version__


def print_header():
    header = (
        "optbinning (Version {})\n"
        "Copyright (c) 2019-2025 Guillermo Navas-Palencia, Apache License 2.0"
        "\n".format(__version__))

    print(header)


def print_optional_parameters(dict_default_options, dict_user_options):
    option_format = "    {:<24} {:>15}   * {}\n"
    str_options = "  Begin options\n"
    for key, value in dict_default_options.items():
        user_value = dict_user_options[key]

        if (isinstance(user_value, (list, np.ndarray, dict)) or
                value != user_value):
            user_flag = "U"
        else:
            user_flag = "d"

        if user_value is None:
            user_value = "no"
        elif isinstance(user_value, (list, np.ndarray, dict)):
            user_value = "yes"
        elif isinstance(user_value, BaseEstimator):
            user_value = "yes"

        str_options += option_format.format(key, str(user_value), user_flag)
    str_options += "  End options\n"
    print(str_options)


# Stamped on an MPSolver that holds no solution -- see BinningMIP.solve.
# Reading the objective VALUE of such a solver does not raise, it makes
# OR-Tools log to stderr once per read: "The model has been changed since the
# solution was last computed" when Solve() never ran, "No solution exists.
# MPSolverInterface::result_status_ = ..." when it ran and produced none.
# pywraplp exposes no synchronisation state to ask instead: wall_time() is 0
# after a real BOP solve too (measured 2026-08-24), and Iterations() /
# nodes() / VerifySolution emit the very message they would be asked to
# avoid.
_NO_SOLUTION = "_optbinning_no_solution"

# Stamped on an MPSolver whose Solve() never ran at all -- a zero time budget.
# A strictly smaller set than _NO_SOLUTION, and one that cannot share the
# marker: the BEST BOUND of a solve that ran is readable and worth reporting,
# where the best bound of a solve that did not is neither. Measured
# 2026-08-24 on the 2D MIP of tests/test_binning_solvers.py, BestBound()
# answers a solve stopped at a 1 ms budget silently and with 5.41677614 --
# exactly the objective the same model proves optimal at a 60 s budget --
# and answers an unsolved model with -inf plus the "model has been changed"
# log line.
_SOLVE_SKIPPED = "_optbinning_solve_skipped"


def mark_no_solution(solver):
    """Record on an MPSolver that it holds no solution to report.

    Every ``solve()`` in this package marks the solver on the branch that
    builds a fallback solution -- a skipped solve, a solve that timed out,
    and one that proved the model infeasible or unbounded all end there. The
    first of those is additionally marked by :func:`mark_solve_skipped`,
    which says the stronger thing.

    Parameters
    ----------
    solver : object
        An ``ortools.linear_solver.pywraplp.Solver`` whose model has no
        solution. :func:`solver_statistics` reports a zero objective for it
        instead of reading one that does not exist.
    """
    setattr(solver, _NO_SOLUTION, True)


def mark_solve_skipped(solver):
    """Record on an MPSolver that its ``Solve()`` never ran.

    Called from the branch of each ``solve()`` that declines to solve on a
    zero time budget. A model that was never solved holds no solution, so
    this implies :func:`mark_no_solution` and does not depend on it having
    been called.

    Parameters
    ----------
    solver : object
        An ``ortools.linear_solver.pywraplp.Solver`` that was built but not
        solved. :func:`solver_statistics` reports a zero objective *and* a
        zero best bound for it, reading neither.
    """
    setattr(solver, _SOLVE_SKIPPED, True)
    setattr(solver, _NO_SOLUTION, True)


def solver_statistics(solver_type, solver):
    time_optimizer = None
    d_solver = {}

    if solver_type == "cp":
        d_solver["n_booleans"] = solver.NumBooleans()
        d_solver["n_branches"] = solver.NumBranches()
        d_solver["n_conflicts"] = solver.NumConflicts()
        d_solver["objective"] = int(solver.ObjectiveValue())
        d_solver["best_objective_bound"] = int(solver.BestObjectiveBound())

        time_optimizer = solver.WallTime()

    elif solver_type == "mip":
        d_solver["n_constraints"] = solver.NumConstraints()
        d_solver["n_variables"] = solver.NumVariables()

        if getattr(solver, _NO_SOLUTION, False):
            # No solution, no objective. Zero is what the read would answer
            # anyway, minus the log line it writes on the way.
            d_solver["objective"] = 0.0
        else:
            d_solver["objective"] = solver.Objective().Value()

        if getattr(solver, _SOLVE_SKIPPED, False):
            # A solve that never ran proved no bound either, and the read
            # answers -inf noisily. Reported as zero, which is what the cp
            # branch above gets from CP-SAT for the same unsolved model.
            d_solver["best_bound"] = 0.0
        else:
            # Every other solver has a bound that OR-Tools returns silently,
            # including one that ran out of time before finding any solution
            # -- for which the bound is the only statistic there is.
            d_solver["best_bound"] = solver.Objective().BestBound()

    elif solver_type == "lp":
        d_solver["n_variables"] = solver.n_variables
        d_solver["n_constraints"] = solver.n_constraints
        d_solver["n_iterations"] = solver.n_iterations
        d_solver["objective"] = solver.objective

    return d_solver, time_optimizer


def print_solver_statistics(solver_type, d_solver):
    if solver_type == "cp":
        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of booleans            {:>10}\n"
            "    Number of branches            {:>10}\n"
            "    Number of conflicts           {:>10}\n"
            "    Objective value               {:>10}\n"
            "    Best objective bound          {:>10}\n"
            ).format(solver_type, *d_solver.values())

    elif solver_type == "mip":
        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of variables           {:>10}\n"
            "    Number of constraints         {:>10}\n"
            "    Objective value               {:>10.4f}\n"
            "    Best objective bound          {:>10.4f}\n"
            ).format(solver_type, *d_solver.values())

    elif solver_type == "lp":
        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of variables           {:>10}\n"
            "    Number of constraints         {:>10}\n"
            "    Number of iterations          {:>10}\n"
            "    Objective value               {:>10.4f}\n"
            ).format(solver_type, *d_solver.values())

    print(solver_stats)
