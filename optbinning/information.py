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
# Reading the objective of such a solver does not raise, it makes OR-Tools
# log to stderr once per read: "The model has been changed since the solution
# was last computed" when Solve() never ran, "No solution exists.
# MPSolverInterface::result_status_ = ..." when it ran and produced none.
# pywraplp exposes no synchronisation state to ask instead: wall_time() is 0
# after a real BOP solve too (measured 2026-08-24), and Iterations() /
# nodes() / VerifySolution emit the very message they would be asked to
# avoid.
_NO_SOLUTION = "_optbinning_no_solution"


def mark_no_solution(solver):
    """Record on an MPSolver that it holds no solution to report.

    Every ``solve()`` in this package marks the solver on the branch that
    builds a fallback solution -- a skipped solve, a solve that timed out,
    and one that proved the model infeasible or unbounded all end there.

    Parameters
    ----------
    solver : object
        An ``ortools.linear_solver.pywraplp.Solver`` whose model has no
        solution. :func:`solver_statistics` reports a zero objective for it
        instead of reading one that does not exist.
    """
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
            # No solution, no objective -- and no best bound either: cbc
            # answers that read for an infeasible model with the bound of a
            # relaxation nothing feasible lives in. Reported as zero, which
            # is what the cp branch above gets from CP-SAT for the same
            # unsolved model.
            d_solver["objective"] = 0.0
            d_solver["best_bound"] = 0.0
        else:
            d_solver["objective"] = solver.Objective().Value()
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
