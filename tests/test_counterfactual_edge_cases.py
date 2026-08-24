"""
Counterfactual explanations edge-case and chaos testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np
import pandas as pd

from ortools.linear_solver import pywraplp
from pytest import raises

from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.exceptions import CounterfactualsFoundWarning
from optbinning.exceptions import NotGeneratedError
from optbinning.scorecard import Counterfactual
from optbinning.scorecard.counterfactual.mip import CFMIP
from optbinning.scorecard.counterfactual.model_data import model_data
from optbinning.scorecard.counterfactual.multi_mip import MCFMIP

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# Small synthetic data: the counterfactual MIP is a CBC run per call, so the
# suite keeps the number of features and the number of bins per feature low.
variable_names = ["f0", "f1", "f2", "f3", "f4"]

rng = np.random.RandomState(42)
X = pd.DataFrame(rng.normal(size=(300, 5)), columns=variable_names)

_linear = (1.2 * X["f0"] - 0.9 * X["f1"] + 0.5 * X["f2"] + 0.3 * X["f3"] -
           0.2 * X["f4"])

y_binary = (_linear + rng.normal(scale=0.5, size=300) > 0).astype(int)
y_continuous = (10 + 3 * X["f0"] - 2 * X["f1"] + X["f2"] +
                0.5 * X["f3"]).values

_three_bins = {v: {"max_n_bins": 3} for v in variable_names}
_mixed_bins = {"f0": {"max_n_bins": 4}, "f1": {"max_n_bins": 3},
               "f2": {"max_n_bins": 2}, "f3": {"max_n_bins": 3},
               "f4": {"max_n_bins": 2}}
_user_splits = {v: {"user_splits": [-0.5, 0.5],
                    "user_splits_fixed": [True, True]}
                for v in variable_names}


scorecard_binary = Scorecard(
    binning_process=BinningProcess(variable_names, max_n_prebins=8,
                                   binning_fit_params=_three_bins),
    estimator=LogisticRegression()).fit(X, y_binary)


scorecard_continuous = Scorecard(
    binning_process=BinningProcess(variable_names, max_n_prebins=8,
                                   binning_fit_params=_mixed_bins),
    estimator=LinearRegression()).fit(X, y_continuous)


# Every feature of this scorecard ends up with the same number of candidate
# bins, which is the condition the two test_defect_uniform_* tests exercise.
scorecard_uniform = Scorecard(
    binning_process=BinningProcess(variable_names, max_n_prebins=6,
                                   binning_fit_params=_three_bins),
    estimator=LinearRegression()).fit(X, y_continuous)


scorecard_user_splits = Scorecard(
    binning_process=BinningProcess(variable_names,
                                   binning_fit_params=_user_splits),
    estimator=LogisticRegression()).fit(X, y_binary)


query = X.iloc[[0]]


def _binary():
    return Counterfactual(scorecard_binary).fit(X)


def _continuous():
    return Counterfactual(scorecard_continuous).fit(X)


def _outcomes(cf):
    return cf.display(show_outcome=True)["outcome"].values


def _changed(cf):
    df = cf.display(show_only_changes=True)
    return [set(v for v in df.columns if df[v].values[i] != "-")
            for i in range(df.shape[0])]


def _unbounded_solver():
    solver = pywraplp.Solver(
        "CFMIP", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    solver.Maximize(solver.NumVar(0, np.inf, "v"))

    return solver


def test_generate_type_guards():
    cf = _binary()

    with raises(TypeError):
        cf.generate(query=query.values, y=1, outcome_type="binary", n_cf=1)

    with raises(TypeError):
        cf.generate(query=query, y=None, outcome_type="binary", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="junk", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=2.0)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    max_changes=2.5)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    actionable_features=np.array(["not_a_feature"]))

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    objectives={"proximity": "1"})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    soft_constraints={"diversity_features": 0})

    # min_outcome is a valid hard constraint for probability and continuous
    # outcomes only.
    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    hard_constraints=["min_outcome"])

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    priority_tol="0.5")

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    priority_tol=-0.1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    time_limit=-1)


def test_generate_accepts_hard_constraints_as_tuple_and_ndarray():
    cf = _binary()

    for hard in (("diversity_features",), np.array(["diversity_features"])):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                    max_changes=2, hard_constraints=hard)

        assert cf.status == "OPTIMAL"
        assert cf.display().shape[0] == 2


def test_generate_boolean_target_is_accepted_as_one():
    cf = _binary()

    cf.generate(query=query, y=True, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf.status == "OPTIMAL"
    assert all(_outcomes(cf) > 0.5)


def test_generate_continuous_requires_an_outcome_constraint():
    cf = _continuous()

    with raises(ValueError, match="at least one hard constraint"):
        cf.generate(query=query, y=10, outcome_type="continuous", n_cf=1)

    # A diversity constraint alone does not qualify.
    with raises(ValueError, match='at least one of the hard_constraints'):
        cf.generate(query=query, y=10, outcome_type="continuous", n_cf=2,
                    hard_constraints=["diversity_features"])


def test_generate_binary_negative_outcome():
    cf = _binary()

    cf.generate(query=query, y=0, outcome_type="binary", n_cf=1,
                max_changes=3)

    assert cf.status == "OPTIMAL"
    assert all(_outcomes(cf) <= 0.5)


def test_generate_binary_negative_outcome_multiple():
    cf = _binary()

    cf.generate(query=query, y=0, outcome_type="binary", n_cf=2,
                max_changes=3, hard_constraints=["diversity_values"])

    assert cf.status == "OPTIMAL"

    outcomes = _outcomes(cf)
    assert len(outcomes) == 2
    assert all(outcomes <= 0.5)


def test_generate_probability_max_outcome_and_diff_outcome():
    cf = _binary()

    cf.generate(query=query, y=0.2, outcome_type="probability", n_cf=1,
                max_changes=3, hard_constraints=["max_outcome"],
                soft_constraints={"diff_outcome": 1})

    assert cf.status == "OPTIMAL"
    assert all(_outcomes(cf) <= 0.2 + 1e-2)
    assert "diff_outcome" in cf._optimizer._objectives


def test_generate_probability_multiple_min_outcome():
    cf = _binary()

    cf.generate(query=query, y=0.9, outcome_type="probability", n_cf=2,
                max_changes=3, hard_constraints=["min_outcome"],
                soft_constraints={"diff_outcome": 1,
                                  "diversity_features": 1})

    assert cf.status == "OPTIMAL"

    outcomes = _outcomes(cf)
    assert len(outcomes) == 2
    assert all(outcomes >= 0.9 - 1e-2)

    objectives = cf._optimizer._objectives
    assert set(objectives) == {"proximity", "closeness", "diff_outcome",
                               "diversity_features"}

    # The diversity objectives are stored negated; information() reports
    # their absolute value.
    assert objectives["diversity_features"].solution_value() <= 0

    cf.information(print_level=1)


def test_generate_probability_multiple_max_outcome():
    cf = _binary()

    cf.generate(query=query, y=0.2, outcome_type="probability", n_cf=2,
                max_changes=3,
                hard_constraints=["max_outcome", "diversity_values"])

    assert cf.status == "OPTIMAL"

    outcomes = _outcomes(cf)
    assert len(outcomes) == 2
    assert all(outcomes <= 0.2 + 1e-2)


def test_generate_continuous_max_outcome_and_diff_outcome():
    cf = _continuous()

    target = float(scorecard_continuous.predict(query)[0]) - 2.0

    cf.generate(query=query, y=target, outcome_type="continuous", n_cf=1,
                max_changes=3, hard_constraints=["max_outcome"],
                soft_constraints={"diff_outcome": 1})

    assert cf.status == "OPTIMAL"
    assert all(_outcomes(cf) <= target + 1e-6)


def test_generate_continuous_multiple_min_outcome():
    cf = _continuous()

    target = float(scorecard_continuous.predict(query)[0]) + 3.0

    cf.generate(query=query, y=target, outcome_type="continuous", n_cf=2,
                max_changes=3,
                hard_constraints=["min_outcome", "diversity_values"],
                soft_constraints={"diff_outcome": 1,
                                  "diversity_values": 1})

    assert cf.status == "OPTIMAL"

    outcomes = _outcomes(cf)
    assert len(outcomes) == 2
    assert all(outcomes >= target - 1e-6)

    objectives = cf._optimizer._objectives
    assert "diversity_values" in objectives
    assert objectives["diversity_values"].solution_value() <= 0

    cf.information(print_level=1)


def test_generate_actionable_features():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2, actionable_features=np.array(["f0", "f1"]))

    assert cf.status == "OPTIMAL"
    assert _changed(cf)[0] <= {"f0", "f1"}


def test_generate_actionable_features_multiple():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                max_changes=2, actionable_features=["f0", "f1"],
                hard_constraints=["diversity_features"])

    assert cf.status == "OPTIMAL"

    changed = _changed(cf)
    assert len(changed) == 2
    assert all(c <= {"f0", "f1"} for c in changed)


def test_generate_hierarchical_multiple_diversity_values():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                max_changes=3, method="hierarchical",
                hard_constraints=["diversity_values"])

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 2


def test_generate_hierarchical_explicit_objectives():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2, method="hierarchical",
                objectives={"closeness": 3, "proximity": 1})

    assert cf.status == "OPTIMAL"
    assert set(cf._optimizer._objectives) == {"proximity", "closeness"}
    assert all(_outcomes(cf) > 0.5)


def test_generate_single_objective():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2, objectives={"proximity": 1})

    assert cf.status == "OPTIMAL"
    assert set(cf._optimizer._objectives) == {"proximity"}


def test_generate_hierarchical_infeasible_subproblem_raises():
    cf = _continuous()

    with raises(Exception, match="INFEASIBLE problem in hierarchical model"):
        cf.generate(query=query, y=1e6, outcome_type="continuous", n_cf=1,
                    max_changes=2, method="hierarchical",
                    hard_constraints=["min_outcome"])


def test_generate_hierarchical_infeasible_subproblem_raises_multiple():
    cf = _continuous()

    with raises(Exception, match="INFEASIBLE problem in hierarchical model"):
        cf.generate(query=query, y=1e6, outcome_type="continuous", n_cf=2,
                    max_changes=2, method="hierarchical",
                    hard_constraints=["min_outcome", "diversity_features"])


def test_generate_infeasible_status():
    cf = _continuous()

    cf.generate(query=query, y=1e6, outcome_type="continuous", n_cf=1,
                max_changes=2, hard_constraints=["min_outcome"])

    assert cf.status == "INFEASIBLE"
    assert cf._cfs is None

    with raises(CounterfactualsFoundWarning):
        cf.display()

    # information() must still work when no counterfactual was found.
    cf.information(print_level=0)
    cf.information(print_level=1)


def test_solve_reports_unbounded():
    optimizer = CFMIP("weighted", {"proximity": 1}, 1, [], {}, {}, 0.1, 1, 10)
    optimizer.solver_ = _unbounded_solver()
    optimizer._p = 0
    optimizer._nbins = []
    optimizer._z = {}

    status, solution = optimizer.solve()

    assert status == "UNBOUNDED"
    assert solution is None


def test_multi_solve_reports_unbounded():
    optimizer = MCFMIP(2, "weighted", {"proximity": 1}, 1, [], {}, {}, 0.1, 1,
                       10)
    optimizer.solver_ = _unbounded_solver()
    optimizer._p = 0
    optimizer._nbins = []
    optimizer._z = {}

    status, solution = optimizer.solve()

    assert status == "UNBOUNDED"
    assert solution is None


def test_generate_query_as_dict():
    cf = _binary()

    dict_query = {v: float(query[v].iloc[0]) for v in variable_names}
    cf.generate(query=dict_query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf.status == "OPTIMAL"
    assert cf.display().shape == (1, 5)


def test_generate_query_with_missing_value():
    cf = _binary()

    nan_query = query.copy()
    nan_query["f0"] = np.nan

    cf.generate(query=nan_query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 1


def test_generate_query_with_infinite_value():
    cf = _binary()

    inf_query = query.copy()
    inf_query["f0"] = np.inf

    cf.generate(query=inf_query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 1


def test_generate_multi_row_query_is_rejected():
    cf = _binary()

    with raises(ValueError, match="Length of values"):
        cf.generate(query=X.iloc[[0, 1]], y=1, outcome_type="binary", n_cf=1,
                    max_changes=2)


def test_generate_verbose():
    cf = Counterfactual(scorecard_binary, verbose=True).fit(X)

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2, objectives={"proximity": 1, "closeness": 1})

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 1


def test_special_missing_adds_candidate_bins():
    x = Counterfactual(scorecard_binary).fit(X)._transform_query(query)[0]

    nbins = model_data(scorecard_binary, x, False)[0]
    nbins_special = model_data(scorecard_binary, x, True)[0]

    assert all(ns == n + 2 for n, ns in zip(nbins, nbins_special))

    cf = Counterfactual(scorecard_binary, special_missing=True,
                        n_jobs=2).fit(X)

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf.status == "OPTIMAL"


def test_generate_with_user_splits_scorecard():
    cf = Counterfactual(scorecard_user_splits).fit(X)

    cf.generate(query=query, y=0, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf.status == "OPTIMAL"
    assert all(_outcomes(cf) <= 0.5)

    bins = set(scorecard_user_splits.table()["Bin"].astype(str))
    display = cf.display()
    for v in variable_names:
        value = display[v].values[0]
        if isinstance(value, str):
            assert value in bins


def test_information_without_optimizer():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2)

    # Defensive branch: generate() always stores an optimizer, so the
    # "solver is None" path of information() is only reachable like this.
    cf._optimizer = None
    cf.information(print_level=0)


def test_not_generated_guards():
    cf = _binary()

    with raises(NotGeneratedError):
        cf.status

    with raises(NotGeneratedError):
        cf.display()

    with raises(NotGeneratedError):
        cf.information()


def test_refit_resets_generated_flag():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=2)

    assert cf._is_generated

    cf.fit(X)

    with raises(NotGeneratedError):
        cf.display()


def test_display_show_only_changes_marks_untouched_features():
    cf = _binary()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=1)

    assert cf.status == "OPTIMAL"

    df = cf.display(show_only_changes=True)
    n_changed = sum(df[v].values[0] != "-" for v in variable_names)
    assert n_changed <= 1


def test_defect_uniform_bin_counts_break_counterfactual():
    cf = Counterfactual(scorecard_uniform).fit(X)

    x = cf._transform_query(query)[0]
    nbins = model_data(scorecard_uniform, x, False)[0]

    # Precondition of the defect: every feature offers the same number of
    # candidate bins.
    assert len(set(nbins)) == 1

    target = float(scorecard_uniform.predict(query)[0]) + 0.5

    cf.generate(query=query, y=target, outcome_type="continuous", n_cf=1,
                max_changes=3, hard_constraints=["min_outcome"])

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 1


def test_defect_uniform_bin_counts_break_counterfactual_multiple():
    cf = Counterfactual(scorecard_uniform).fit(X)

    x = cf._transform_query(query)[0]
    nbins = model_data(scorecard_uniform, x, False)[0]

    assert len(set(nbins)) == 1

    target = float(scorecard_uniform.predict(query)[0]) + 0.5

    cf.generate(query=query, y=target, outcome_type="continuous", n_cf=2,
                max_changes=3,
                hard_constraints=["min_outcome", "diversity_features"])

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 2


def test_generate_honours_a_fractional_time_limit():
    # time_limit is documented as an int here and in every sibling estimator,
    # but generate() validates it as any positive number, so a fractional
    # budget reaches the model. pywraplp SetTimeLimit takes an int64, so
    # CFMIP rounds the budget to whole milliseconds instead of handing the
    # float to SWIG -- same contract as optbinning/binning/mip.py.
    for time_limit in (5, 5.0, 2.5):
        cf = _binary()
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    max_changes=2, time_limit=time_limit)

        assert cf.status == "OPTIMAL"
        assert cf.display().shape[0] == 1


def test_generate_multiple_honours_a_fractional_time_limit():
    # MCFMIP.solve is a separate copy of CFMIP.solve, so the same budget has
    # to be pinned on both.
    cf = _binary()
    cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                max_changes=2, hard_constraints=["diversity_features"],
                time_limit=2.5)

    assert cf.status == "OPTIMAL"
    assert cf.display().shape[0] == 2


# ---------------------------------------------------------------------------
# CFMIP.solve / MCFMIP.solve: time limit and solution container
# ---------------------------------------------------------------------------

class _StubVariable:
    def __init__(self, value):
        self._value = value

    def solution_value(self):
        return self._value


class _StubSolver:
    """Stands in for pywraplp.Solver so the time limit handed to OR-Tools is
    observable and no CBC run is needed."""

    def __init__(self, status):
        self._status = status
        self.time_limit_ms = None
        self.n_threads = None

    def SetTimeLimit(self, milliseconds):
        self.time_limit_ms = milliseconds

    def SetNumThreads(self, n_jobs):
        self.n_threads = n_jobs

    def Solve(self):
        return self._status


def _stub_cfmip(status, nbins=(2, 2), time_limit=7):
    optimizer = CFMIP("weighted", {"proximity": 1}, 1, [], {}, {}, 0.1, 2,
                      time_limit)
    optimizer.solver_ = _StubSolver(status)
    optimizer._p = len(nbins)
    optimizer._nbins = list(nbins)
    optimizer._z = {(i, j): _StubVariable(float(j == 0))
                    for i in range(len(nbins)) for j in range(nbins[i])}

    return optimizer


def _stub_mcfmip(status, nbins=(2, 2), K=2, time_limit=7):
    optimizer = MCFMIP(K, "weighted", {"proximity": 1}, 1, [], {}, {}, 0.1, 2,
                       time_limit)
    optimizer.solver_ = _StubSolver(status)
    optimizer._p = len(nbins)
    optimizer._nbins = list(nbins)
    optimizer._z = {(k, i, j): _StubVariable(float(j == 0))
                    for k in range(K)
                    for i in range(len(nbins)) for j in range(nbins[i])}

    return optimizer


def test_cfmip_time_limit_is_rounded_to_int_milliseconds():
    # SetTimeLimit takes int64 milliseconds; time_limit is seconds and
    # generate() accepts any positive number of them.
    for time_limit, milliseconds in ((7, 7000), (2.5, 2500), (0.0006, 1)):
        for optimizer in (_stub_cfmip(pywraplp.Solver.OPTIMAL,
                                      time_limit=time_limit),
                          _stub_mcfmip(pywraplp.Solver.OPTIMAL,
                                       time_limit=time_limit)):
            status, _ = optimizer.solve()

            assert status == "OPTIMAL"
            assert optimizer.solver_.time_limit_ms == milliseconds
            assert isinstance(optimizer.solver_.time_limit_ms, int)
            assert optimizer.solver_.n_threads == 2


def test_cfmip_zero_time_limit_is_not_unlimited():
    # 0 milliseconds is MPSolver's "no limit" sentinel, so a zero budget
    # must not reach SetTimeLimit at all. generate() rejects time_limit <= 0
    # and so never produces one, but the two solve() methods share the
    # contract of optbinning/binning/mip.py, which is reached with a zero
    # budget from OptimalBinning. The stub would answer OPTIMAL if it were
    # solved, so "UNKNOWN" proves the solve was skipped.
    for optimizer in (_stub_cfmip(pywraplp.Solver.OPTIMAL, time_limit=0),
                      _stub_mcfmip(pywraplp.Solver.OPTIMAL, time_limit=0)):
        status, solution = optimizer.solve()

        assert status == "UNKNOWN"
        assert solution is None
        assert optimizer.solver_.time_limit_ms is None
        # The thread count is set whether or not the solve happens.
        assert optimizer.solver_.n_threads == 2


def test_cfmip_sub_millisecond_time_limit_is_clamped_to_one():
    # generate() accepts 0.0004 seconds, which is a real budget and must buy
    # a real solve: rounding it to whole milliseconds gives 0, MPSolver's
    # "no limit" sentinel, so it is clamped up to one millisecond instead --
    # the same contract as optbinning/binning/mip.py.
    for time_limit in (0.0004, 0.0005, 0.0006, 0.001):
        for optimizer in (_stub_cfmip(pywraplp.Solver.OPTIMAL,
                                      time_limit=time_limit),
                          _stub_mcfmip(pywraplp.Solver.OPTIMAL,
                                       time_limit=time_limit)):
            status, solution = optimizer.solve()

            assert status == "OPTIMAL"
            assert optimizer.solver_.time_limit_ms == 1
            assert optimizer.solver_.n_threads == 2


def test_information_after_an_infeasible_generate_is_quiet(capfd):
    # Counterfactual.information calls information.solver_statistics on the
    # MPSolver whenever an optimizer exists, and an infeasible model leaves
    # that solver with no objective to read. OR-Tools does not raise on such
    # a read, it logs to stderr -- "No solution exists" after a solve that
    # produced none, "The model has been changed since the solution was last
    # computed" after no solve at all. Neither may reach the user. Same
    # check as tests/test_binning_solvers.py,
    # assert_no_unsolved_objective_read; the target outcome here is the one
    # test_generate_infeasible_status uses, and n_cf picks CFMIP or MCFMIP.
    for n_cf, hard in ((1, ["min_outcome"]),
                       (2, ["min_outcome", "diversity_features"])):
        cf = _continuous()
        cf.generate(query=query, y=1e6, outcome_type="continuous", n_cf=n_cf,
                    max_changes=2, hard_constraints=hard)
        capfd.readouterr()

        cf.information(print_level=2)
        captured = capfd.readouterr()

        assert cf.status == "INFEASIBLE"
        assert "solution was last computed" not in captured.err
        assert "No solution exists" not in captured.err


def test_cfmip_solution_is_an_object_array_of_boolean_masks():
    # Equal bin counts are the trap: np.array over equal-length arrays
    # builds a rectangular (p, nbins) object array whose rows are object
    # dtype and cannot index anything. The container is built the way
    # binning_2d.py builds its rows -- np.empty(dtype=object), one slot at a
    # time -- so the uniform and the ragged case give the same 1-D object
    # array of boolean masks.
    for nbins in ((2, 2), (2, 3)):
        optimizer = _stub_cfmip(pywraplp.Solver.OPTIMAL, nbins)
        status, solution = optimizer.solve()

        assert status == "OPTIMAL"
        assert isinstance(solution, np.ndarray)
        assert solution.dtype == object
        assert solution.shape == (len(nbins),)

        for i, n in enumerate(nbins):
            assert solution[i].dtype == bool
            assert list(solution[i]) == [True] + [False] * (n - 1)


def test_multi_cfmip_solution_is_an_object_array_per_counterfactual():
    for nbins in ((2, 2), (2, 3)):
        optimizer = _stub_mcfmip(pywraplp.Solver.OPTIMAL, nbins, K=3)
        status, solution = optimizer.solve()

        assert status == "OPTIMAL"
        assert isinstance(solution, np.ndarray)
        assert solution.dtype == object
        assert solution.shape == (3,)

        for k in range(3):
            assert isinstance(solution[k], np.ndarray)
            assert solution[k].dtype == object
            assert solution[k].shape == (len(nbins),)

            for i, n in enumerate(nbins):
                assert solution[k][i].dtype == bool
                assert list(solution[k][i]) == [True] + [False] * (n - 1)
