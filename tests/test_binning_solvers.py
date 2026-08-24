"""
OptimalBinning solver formulation testing: CP-SAT (cp.py), MIP (mip.py) and
the shared model data (model_data.py).

The tests here drive the constraint branches of the two formulations -- bin
size, bin event/non-event counts, the monotonic trend variants, max_pvalue,
min_event_rate_diff, regularization and fixed splits -- plus the scenario
(stochastic) formulation, the multiclass model data, and the time-limit
contract that mip.py shares with the 2D MIP (mip_2d.py) and with
information.solver_statistics.

Split positions are solver artifacts, so the assertions are invariants: the
solver status, the constraint the parameter is supposed to impose, the
ordering of the objective, and the exception type. See CLAUDE.md,
"Distinguish a bug from solver non-determinism".
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import warnings

import numpy as np

from ortools.linear_solver import pywraplp
from pytest import approx, raises
from scipy import stats

from optbinning import ContinuousOptimalBinning2D
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning import OptimalBinning2D
from optbinning.binning.mip import BinningMIP
from optbinning.binning.model_data import continuous_model_data
from optbinning.binning.model_data import model_data
from optbinning.binning.multidimensional.mip_2d import Binning2DMIP
from optbinning.binning.uncertainty import SBOptimalBinning
from sklearn.exceptions import NotFittedError


def build_binary(rates, sizes=40):
    """Build (x, y) whose i-th pre-bin holds ``sizes`` records with event
    rate ``rates[i]``. x takes the value i on the i-th group, so passing
    ``user_splits(len(rates))`` makes every group its own pre-bin and the
    event rate of every pre-bin is exactly the requested one."""
    if not hasattr(sizes, "__len__"):
        sizes = [sizes] * len(rates)

    xs = []
    ys = []
    for i, rate in enumerate(rates):
        n = sizes[i]
        xs.append(np.full(n, float(i)))
        yi = np.zeros(n, dtype=int)
        yi[:int(round(rate * n))] = 1
        ys.append(yi)

    return np.concatenate(xs), np.concatenate(ys)


def user_splits(n_groups):
    return [i + 0.5 for i in range(n_groups - 1)]


def bin_counts(optb):
    """Count / non-event / event of the non-special, non-missing bins."""
    table = optb.binning_table.build()
    return (table["Count"].values[:-3].astype(int),
            table["Non-event"].values[:-3].astype(int),
            table["Event"].values[:-3].astype(int))


def event_rates(optb):
    return optb.binning_table.build()["Event rate"].values[:-3].astype(float)


def iv_of(optb):
    optb.binning_table.build()
    return float(optb.binning_table.iv)


def consecutive_zscores(n_nonevent, n_event):
    """Two-proportion Z statistic of every consecutive pair of bins.

    Re-derived here rather than imported from model_data, so the assertion
    is independent of the implementation it checks."""
    z = []
    for i in range(len(n_event) - 1):
        e1, ne1, e2, ne2 = (n_event[i], n_nonevent[i],
                            n_event[i + 1], n_nonevent[i + 1])
        n1, n2 = e1 + ne1, e2 + ne2
        p1, p2, p = e1 / n1, e2 / n2, (e1 + e2) / (n1 + n2)
        z.append(abs(p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))
    return np.array(z)


def assert_no_unsolved_objective_read(captured):
    """Fail if OR-Tools logged an objective read off a solver holding no
    solution. It does not raise on such a read, it writes to stderr:
    "The model has been changed since the solution was last computed" when
    Solve() never ran, "No solution exists. MPSolverInterface::result_status_
    = ..." when it ran and produced none. The prefix differs by whether
    absl::InitializeLog has run, so the message body is what is matched."""
    assert "solution was last computed" not in captured.err
    assert "No solution exists" not in captured.err


SOLVERS = (("cp", "bop"), ("mip", "bop"), ("mip", "cbc"))

# A peak-shaped event rate: not monotone in either direction, so every
# monotonic trend constrains the problem.
RATES = [0.10, 0.30, 0.50, 0.80, 0.90, 0.60, 0.40, 0.15]
x, y = build_binary(RATES)
splits_all = user_splits(len(RATES))
N_RECORDS = len(x)


# ---------------------------------------------------------------------------
# min / max bin size
# ---------------------------------------------------------------------------

def test_min_bin_size_alone():
    lower = int(np.ceil(0.2 * N_RECORDS))

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, min_bin_size=0.2)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        counts, _, _ = bin_counts(optb)
        assert counts.sum() == N_RECORDS
        assert np.all(counts >= lower)


def test_max_bin_size_alone():
    upper = int(np.ceil(0.15 * N_RECORDS))

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, max_bin_size=0.15)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        counts, _, _ = bin_counts(optb)
        assert counts.sum() == N_RECORDS
        assert np.all(counts <= upper)


def test_min_and_max_bin_size_together():
    # Both bounds set takes the range-constraint branch (an auxiliary integer
    # variable u[i] per pre-bin), not the two one-sided inequalities.
    lower = int(np.ceil(0.2 * N_RECORDS))
    upper = int(np.ceil(0.3 * N_RECORDS))

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, min_bin_size=0.2,
                              max_bin_size=0.3)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        counts, _, _ = bin_counts(optb)
        assert counts.sum() == N_RECORDS
        assert np.all(counts >= lower)
        assert np.all(counts <= upper)


def test_equal_min_and_max_bin_size():
    # bin_size_diff == 0 pins every bin to exactly the same number of
    # records.
    exact = int(np.ceil(0.25 * N_RECORDS))

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, min_bin_size=0.25,
                              max_bin_size=0.25)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        counts, _, _ = bin_counts(optb)
        assert np.all(counts == exact)


# ---------------------------------------------------------------------------
# min / max number of bins
# ---------------------------------------------------------------------------

def test_max_n_bins_alone():
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, max_n_bins=3)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) + 1 <= 3


def test_min_and_max_n_bins_together():
    # Both bounds set uses the auxiliary range variable rather than the two
    # one-sided inequalities.
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, min_n_bins=2,
                              max_n_bins=4)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert 2 <= len(optb.splits) + 1 <= 4


def test_min_n_bins_alone():
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver,
                              monotonic_trend="descending", min_n_bins=3)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) + 1 >= 3

    # A descending trend cannot produce five bins out of this peak-shaped
    # event rate, so the same one-sided constraint makes the model infeasible.
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver,
                              monotonic_trend="descending", min_n_bins=5)
        optb.fit(x, y)

        assert optb.status == "INFEASIBLE"
        assert len(optb.splits) == 0


# ---------------------------------------------------------------------------
# min / max number of events and non-events per bin
# ---------------------------------------------------------------------------

def test_min_max_bin_n_event_and_nonevent():
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, min_bin_n_event=10,
                              max_bin_n_event=80, min_bin_n_nonevent=10,
                              max_bin_n_nonevent=80)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        _, n_nonevent, n_event = bin_counts(optb)
        assert np.all(n_event >= 10) and np.all(n_event <= 80)
        assert np.all(n_nonevent >= 10) and np.all(n_nonevent <= 80)


def test_min_bin_n_event_infeasible():
    # 160 events in total, so no binning can give every bin 1000 of them.
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, min_bin_n_event=1000)
        optb.fit(x, y)

        assert optb.status == "INFEASIBLE"
        assert len(optb.splits) == 0


def test_hellinger_forces_one_event_and_nonevent_per_bin():
    # A pure pre-bin (rate 0 or 1) makes the hellinger / triangular
    # divergences set the implicit min_bin_n_event / min_bin_n_nonevent of 1
    # instead of dropping the bin.
    rates = [0.0, 0.30, 0.50, 0.80, 1.0, 0.60, 0.40, 0.15]
    xh, yh = build_binary(rates)

    for divergence in ("hellinger", "triangular"):
        for solver, mip_solver in SOLVERS:
            optb = OptimalBinning(user_splits=splits_all, solver=solver,
                                  mip_solver=mip_solver,
                                  divergence=divergence)
            optb.fit(xh, yh)

            assert optb.status == "OPTIMAL"
            _, n_nonevent, n_event = bin_counts(optb)
            assert np.all(n_event >= 1)
            assert np.all(n_nonevent >= 1)


# ---------------------------------------------------------------------------
# monotonic trends
# ---------------------------------------------------------------------------

def test_monotonic_ascending_and_descending():
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver,
                              monotonic_trend="ascending")
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"
        rates = event_rates(optb)
        assert np.all(np.diff(rates) >= -1e-9)

        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver,
                              monotonic_trend="descending")
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"
        rates = event_rates(optb)
        assert np.all(np.diff(rates) <= 1e-9)


def test_monotonic_concave_and_convex():
    unconstrained = OptimalBinning(user_splits=splits_all)
    unconstrained.fit(x, y)
    iv_free = iv_of(unconstrained)

    for solver, mip_solver in SOLVERS:
        for trend in ("concave", "convex"):
            optb = OptimalBinning(user_splits=splits_all, solver=solver,
                                  mip_solver=mip_solver,
                                  monotonic_trend=trend)
            optb.fit(x, y)

            assert optb.status == "OPTIMAL"
            assert set(optb.splits) <= set(splits_all)
            assert np.all(np.diff(optb.splits) > 0)
            # A constraint cannot raise the objective.
            assert iv_of(optb) <= iv_free + 1e-9

            rates = event_rates(optb)
            second = np.diff(rates, 2)
            if trend == "concave":
                assert np.all(second <= 1e-9)
            else:
                assert np.all(second >= -1e-9)


def test_monotonic_peak_and_valley():
    # The heuristic trends locate the change point first, so they are run
    # against an event rate of their own shape as well as against the
    # opposite one.
    xv, yv = build_binary([0.90, 0.70, 0.40, 0.15, 0.10, 0.35, 0.60, 0.85])

    for solver, mip_solver in SOLVERS:
        for trend in ("peak", "valley", "peak_heuristic",
                      "valley_heuristic"):
            for xi, yi in ((x, y), (xv, yv)):
                optb = OptimalBinning(user_splits=splits_all, solver=solver,
                                      mip_solver=mip_solver,
                                      monotonic_trend=trend)
                optb.fit(xi, yi)

                assert optb.status == "OPTIMAL"
                rates = event_rates(optb)
                signs = np.sign(np.round(np.diff(rates), 12))
                signs = signs[signs != 0]
                # Unimodal: at most one change of direction.
                assert np.count_nonzero(np.diff(signs)) <= 1

                if trend.startswith("peak"):
                    assert np.all(signs[1:][signs[:-1] < 0] < 0)
                else:
                    assert np.all(signs[1:][signs[:-1] > 0] > 0)


def test_monotonic_heuristics_with_a_non_monotone_leg():
    # peak_heuristic requires the leg before the change point to be
    # ascending and valley_heuristic requires it to be descending; a dip
    # (respectively a bump) inside that leg is what makes the formulation
    # fix the offending pre-bins before solving.
    peak_rates = [0.10, 0.35, 0.25, 0.60, 0.90, 0.55, 0.30, 0.10]
    valley_rates = [0.90, 0.55, 0.70, 0.30, 0.10, 0.40, 0.70, 0.90]

    for trend, rates in (("peak_heuristic", peak_rates),
                         ("valley_heuristic", valley_rates)):
        xh, yh = build_binary(rates)

        for solver, mip_solver in SOLVERS:
            optb = OptimalBinning(user_splits=splits_all, solver=solver,
                                  mip_solver=mip_solver,
                                  monotonic_trend=trend)
            optb.fit(xh, yh)

            assert optb.status == "OPTIMAL"
            binned = event_rates(optb)
            signs = np.sign(np.round(np.diff(binned), 12))
            signs = signs[signs != 0]
            assert np.count_nonzero(np.diff(signs)) <= 1
            if trend == "peak_heuristic":
                assert signs[0] > 0 and signs[-1] < 0
            else:
                assert signs[0] < 0 and signs[-1] > 0


def test_monotonic_trend_none_is_unconstrained():
    free = OptimalBinning(user_splits=splits_all, monotonic_trend=None)
    free.fit(x, y)

    assert free.status == "OPTIMAL"
    for trend in ("ascending", "descending", "concave", "convex", "peak",
                  "valley"):
        optb = OptimalBinning(user_splits=splits_all, monotonic_trend=trend)
        optb.fit(x, y)
        assert iv_of(optb) <= iv_of(free) + 1e-9


# ---------------------------------------------------------------------------
# max_pvalue and min_event_rate_diff
# ---------------------------------------------------------------------------

def test_max_pvalue_consecutive_policy():
    max_pvalue = 0.05
    zscore = stats.norm.ppf(1.0 - max_pvalue / 2)

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, max_pvalue=max_pvalue,
                              max_pvalue_policy="consecutive")
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        _, n_nonevent, n_event = bin_counts(optb)
        assert np.all(consecutive_zscores(n_nonevent, n_event) >= zscore)


def test_max_pvalue_all_policy():
    max_pvalue = 0.05
    zscore = stats.norm.ppf(1.0 - max_pvalue / 2)

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, max_pvalue=max_pvalue,
                              max_pvalue_policy="all")
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        _, n_nonevent, n_event = bin_counts(optb)
        assert np.all(consecutive_zscores(n_nonevent, n_event) >= zscore)


def test_min_event_rate_diff_binary():
    min_diff = 0.15

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver,
                              min_event_rate_diff=min_diff)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        rates = event_rates(optb)
        assert np.all(np.absolute(np.diff(rates)) >= min_diff - 1e-5)


# ---------------------------------------------------------------------------
# divergences
# ---------------------------------------------------------------------------

def test_all_divergences():
    for divergence in ("iv", "js", "hellinger", "triangular"):
        for solver, mip_solver in SOLVERS:
            optb = OptimalBinning(user_splits=splits_all, solver=solver,
                                  mip_solver=mip_solver,
                                  divergence=divergence)
            optb.fit(x, y)

            assert optb.status == "OPTIMAL"
            assert len(optb.splits) >= 1
            assert set(optb.splits) <= set(splits_all)
            assert iv_of(optb) > 0


# ---------------------------------------------------------------------------
# gamma (regularization)
# ---------------------------------------------------------------------------

def test_gamma_reduces_dominating_bins():
    # Two large groups at the ends and six small ones in the middle: the
    # unregularized optimum keeps every split and so has a very uneven bin
    # size distribution.
    sizes = [100, 20, 20, 20, 20, 20, 20, 100]
    xg, yg = build_binary(RATES, sizes)

    for solver, mip_solver in SOLVERS:
        free = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, gamma=0)
        free.fit(xg, yg)
        reg = OptimalBinning(user_splits=splits_all, solver=solver,
                             mip_solver=mip_solver, gamma=5.0)
        reg.fit(xg, yg)

        assert free.status == "OPTIMAL"
        assert reg.status == "OPTIMAL"

        counts_free, _, _ = bin_counts(free)
        counts_reg, _, _ = bin_counts(reg)

        # Regularization cannot improve the divergence, and it is meant to
        # shrink the spread of the bin sizes.
        assert iv_of(reg) <= iv_of(free) + 1e-9
        assert np.ptp(counts_reg) < np.ptp(counts_free)


# ---------------------------------------------------------------------------
# user_splits_fixed
# ---------------------------------------------------------------------------

def test_user_splits_fixed_is_honoured():
    fixed = [False] * len(splits_all)
    fixed[3] = True

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, user_splits_fixed=fixed)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert splits_all[3] in optb.splits


def test_user_splits_fixed_all_true():
    fixed = [True] * len(splits_all)

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, user_splits_fixed=fixed)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert optb.splits == approx(np.asarray(splits_all))


def test_user_splits_fixed_conflicting_with_trend_is_infeasible():
    fixed = [False] * len(splits_all)
    fixed[5] = True
    fixed[6] = True

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver, user_splits_fixed=fixed,
                              monotonic_trend="ascending")
        optb.fit(x, y)

        assert optb.status == "INFEASIBLE"
        assert len(optb.splits) == 0


# ---------------------------------------------------------------------------
# BinningMIP.solve status mapping
# ---------------------------------------------------------------------------

class _StubVariable:
    def __init__(self, value):
        self._value = value

    def solution_value(self):
        return self._value


class _StubSolver:
    """Stands in for pywraplp.Solver so every status branch of
    BinningMIP.solve can be reached: the binning model is always bounded and
    well formed, so ABNORMAL / UNBOUNDED cannot be produced by any input."""

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


def _stub_optimizer(status, n=4):
    optimizer = BinningMIP(None, None, None, None, None, None, None, None,
                           None, 0, None, "consecutive", 0, None, "bop", 7)
    optimizer.solver_ = _StubSolver(status)
    optimizer._n = n
    optimizer._x = {(i, i): _StubVariable(float(i == n - 2))
                    for i in range(n)}
    return optimizer


def test_mip_solve_status_names():
    solved = {pywraplp.Solver.OPTIMAL: "OPTIMAL",
              pywraplp.Solver.FEASIBLE: "FEASIBLE"}
    unsolved = {pywraplp.Solver.ABNORMAL: "ABNORMAL",
                pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
                pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
                pywraplp.Solver.NOT_SOLVED: "UNKNOWN",
                pywraplp.Solver.MODEL_INVALID: "UNKNOWN"}

    for status, name in solved.items():
        optimizer = _stub_optimizer(status)
        status_name, solution = optimizer.solve()

        assert status_name == name
        assert solution.dtype == bool
        assert list(solution) == [False, False, True, False]
        assert optimizer.solver_.time_limit_ms == 7000

    for status, name in unsolved.items():
        optimizer = _stub_optimizer(status)
        status_name, solution = optimizer.solve()

        assert status_name == name
        assert solution.dtype == bool
        # The fallback solution is the single all-in-one bin.
        assert list(solution) == [False, False, False, True]


def test_defect_mip_time_limit_accepts_float():
    # _check_parameters accepts any non-negative number for time_limit, and
    # solver="cp" honours a float one. solver="mip" used to multiply it by
    # 1000 and hand the float straight to SetTimeLimit, which only takes an
    # int64, so fit() died with a SWIG TypeError naming neither time_limit
    # nor the solver.
    optb_cp = OptimalBinning(user_splits=splits_all, solver="cp",
                             time_limit=2.5)
    optb_cp.fit(x, y)
    assert optb_cp.status == "OPTIMAL"

    optb_mip = OptimalBinning(user_splits=splits_all, solver="mip",
                              time_limit=2.5)
    optb_mip.fit(x, y)
    assert optb_mip.status == "OPTIMAL"


def test_defect_time_limit_zero_means_opposite_things():
    # _check_parameters accepts time_limit=0 although its own message says
    # the value must be positive. "cp" reads it as "no time at all" and gives
    # up (UNKNOWN, the fallback single bin), while "mip" used to hand the 0
    # to MPSolver.SetTimeLimit, where 0 is the "no limit" sentinel, and so
    # solved to optimality. The same value must not mean opposite things to
    # the two formulations of the same model. Both statuses are named, so
    # the two backends regressing together cannot pass this.
    statuses = {}
    splits = {}
    for solver in ("cp", "mip"):
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              time_limit=0)
        optb.fit(x, y)
        statuses[solver] = optb.status
        splits[solver] = list(optb.splits)

    assert statuses == {"cp": "UNKNOWN", "mip": "UNKNOWN"}
    assert splits == {"cp": [], "mip": []}


def test_mip_time_limit_is_rounded_to_int_milliseconds():
    # SetTimeLimit takes int64 milliseconds; time_limit is seconds and may be
    # fractional.
    for time_limit, milliseconds in ((7, 7000), (2.5, 2500), (0.0006, 1)):
        optimizer = _stub_optimizer(pywraplp.Solver.OPTIMAL)
        optimizer.time_limit = time_limit
        optimizer.solve()

        assert optimizer.solver_.time_limit_ms == milliseconds
        assert isinstance(optimizer.solver_.time_limit_ms, int)


def test_mip_zero_time_limit_is_not_unlimited():
    # 0 milliseconds is MPSolver's "no limit" sentinel, so a zero budget
    # must not reach SetTimeLimit at all. The stub would answer OPTIMAL if
    # it were solved, so "UNKNOWN" plus the all-in-one-bin fallback proves
    # the solve was skipped.
    optimizer = _stub_optimizer(pywraplp.Solver.OPTIMAL)
    optimizer.time_limit = 0
    status_name, solution = optimizer.solve()

    assert optimizer.solver_.time_limit_ms is None
    assert status_name == "UNKNOWN"
    assert list(solution) == [False, False, False, True]


def test_mip_sub_millisecond_time_limit_is_clamped_to_one():
    # A positive budget is still a budget: only an exact zero skips the
    # solve. Rounding decides nothing here -- int(round(0.0005 * 1000)) is 0
    # under banker's rounding and 0.0006 rounds to 1, so where the skip
    # started depended on which side of half a millisecond the budget fell.
    # Every positive budget now buys at least one whole millisecond of
    # solving.
    for time_limit in (0.0004, 0.0005, 0.0006, 0.001):
        optimizer = _stub_optimizer(pywraplp.Solver.OPTIMAL)
        optimizer.time_limit = time_limit
        status_name, solution = optimizer.solve()

        assert optimizer.solver_.time_limit_ms == 1
        assert status_name == "OPTIMAL"
        assert list(solution) == [False, False, True, False]


def test_multiclass_mip_shares_the_time_limit_domain():
    # MulticlassBinningMIP inherits solve() from BinningMIP, so it inherits
    # both the fractional limit and the zero-budget handling.
    xm, ym = multiclass_data()

    optb = MulticlassOptimalBinning(user_splits=splits_all, solver="mip",
                                    time_limit=2.5)
    optb.fit(xm, ym)
    assert optb.status == "OPTIMAL"

    statuses = {}
    splits = {}
    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(user_splits=splits_all,
                                        solver=solver, time_limit=0)
        optb.fit(xm, ym)
        statuses[solver] = optb.status
        splits[solver] = list(optb.splits)

    assert statuses == {"cp": "UNKNOWN", "mip": "UNKNOWN"}
    assert splits == {"cp": [], "mip": []}


def test_skipped_mip_solve_leaves_the_objective_alone(capfd):
    # information.solver_statistics reads the objective straight off the
    # MPSolver. A solver whose Solve() was skipped has none: asking for it
    # does not raise, it makes OR-Tools log "The model has been changed
    # since the solution was last computed" to stderr, once per read.
    optb = OptimalBinning(user_splits=splits_all, solver="mip", time_limit=0)
    optb.fit(x, y)
    optb.information(print_level=2)
    captured = capfd.readouterr()

    assert optb.status == "UNKNOWN"
    assert_no_unsolved_objective_read(captured)
    assert optb._optimizer["objective"] == 0
    assert optb._optimizer["best_bound"] == 0


def test_infeasible_mip_leaves_the_objective_alone(capfd):
    # A skipped solve is not the only solver with no objective to read: a
    # model that was solved and found INFEASIBLE has none either, and that
    # one is reachable from ordinary parameters. Reading it logs "No
    # solution exists" per read, and cbc answers the best-bound read with
    # the bound of a model that has no feasible solution at all.
    fixed = [False] * len(splits_all)
    fixed[5] = True
    fixed[6] = True

    for mip_solver in ("bop", "cbc"):
        optb = OptimalBinning(user_splits=splits_all, solver="mip",
                              mip_solver=mip_solver, user_splits_fixed=fixed,
                              monotonic_trend="ascending")
        optb.fit(x, y)
        optb.information(print_level=2)
        captured = capfd.readouterr()

        assert optb.status == "INFEASIBLE"
        assert_no_unsolved_objective_read(captured)
        assert optb._optimizer["objective"] == 0
        assert optb._optimizer["best_bound"] == 0


# ---------------------------------------------------------------------------
# the 2D MIP shares the time limit contract
# ---------------------------------------------------------------------------

def build_2d(n=300):
    """A 2D problem whose event rate rises with x + z."""
    rng = np.random.RandomState(0)
    x2 = rng.uniform(size=n)
    z2 = rng.uniform(size=n)
    e2 = x2 + z2 + rng.normal(scale=0.2, size=n)

    return x2, z2, (e2 > 1.0).astype(int), e2


PREBINS_2D = {"max_n_prebins_x": 4, "max_n_prebins_y": 4}


def _stub_2d_optimizer(status, n=3):
    optimizer = Binning2DMIP(None, None, None, None, None, None, 0, 2, 7)
    optimizer.solver_ = _StubSolver(status)
    optimizer._n_rectangles = n
    optimizer._x = {i: _StubVariable(float(i == 0)) for i in range(n)}

    return optimizer


def test_defect_2d_mip_time_limit_accepts_float():
    # Binning2DMIP.solve carried the same unguarded multiplication as
    # BinningMIP.solve, so a fractional time limit died inside OR-Tools with
    # a SWIG TypeError naming neither time_limit nor the solver.
    x2, z2, y2, e2 = build_2d()

    optb = OptimalBinning2D(solver="mip", time_limit=2.5, **PREBINS_2D)
    optb.fit(x2, z2, y2)
    assert optb.status == "OPTIMAL"

    optc = ContinuousOptimalBinning2D(solver="mip", time_limit=2.5,
                                      **PREBINS_2D)
    optc.fit(x2, z2, e2)
    assert optc.status == "OPTIMAL"


def test_defect_2d_time_limit_zero_means_opposite_things():
    # Binning2DCP reads 0 as "no time at all" and gives up; Binning2DMIP
    # used to hand the 0 to MPSolver.SetTimeLimit, where 0 is the "no limit"
    # sentinel, and solved to optimality instead.
    x2, z2, y2, e2 = build_2d()

    statuses = {}
    for solver in ("cp", "mip"):
        optb = OptimalBinning2D(solver=solver, time_limit=0, **PREBINS_2D)
        optb.fit(x2, z2, y2)
        statuses[solver] = (optb.status, len(optb.splits[0]),
                            len(optb.splits[1]))

    assert statuses == {"cp": ("UNKNOWN", 0, 0), "mip": ("UNKNOWN", 0, 0)}

    statuses = {}
    for solver in ("cp", "mip"):
        optc = ContinuousOptimalBinning2D(solver=solver, time_limit=0,
                                          **PREBINS_2D)
        optc.fit(x2, z2, e2)
        statuses[solver] = (optc.status, len(optc.splits[0]),
                            len(optc.splits[1]))

    assert statuses == {"cp": ("UNKNOWN", 0, 0), "mip": ("UNKNOWN", 0, 0)}


def test_2d_mip_time_limit_is_rounded_to_int_milliseconds():
    for time_limit, milliseconds in ((7, 7000), (2.5, 2500), (0.0006, 1)):
        optimizer = _stub_2d_optimizer(pywraplp.Solver.OPTIMAL)
        optimizer.time_limit = time_limit
        status_name, solution = optimizer.solve()

        assert status_name == "OPTIMAL"
        assert optimizer.solver_.time_limit_ms == milliseconds
        assert isinstance(optimizer.solver_.time_limit_ms, int)
        assert list(solution) == [True, False, False]
        assert optimizer.solver_.n_threads == 2


def test_2d_mip_zero_time_limit_is_not_unlimited():
    # The stub would answer OPTIMAL if it were solved, so "UNKNOWN" plus the
    # empty solution proves the solve was skipped. SetNumThreads is not part
    # of the budget and is set either way.
    optimizer = _stub_2d_optimizer(pywraplp.Solver.OPTIMAL)
    optimizer.time_limit = 0
    status_name, solution = optimizer.solve()

    assert status_name == "UNKNOWN"
    assert optimizer.solver_.time_limit_ms is None
    assert not solution.any()
    assert optimizer.solver_.n_threads == 2


def test_2d_mip_sub_millisecond_time_limit_is_clamped_to_one():
    # Same contract as BinningMIP.solve: only an exact zero skips the solve,
    # and a positive budget below a millisecond is clamped up to one rather
    # than rounded down to MPSolver's "no limit" sentinel.
    for time_limit in (0.0004, 0.0005, 0.0006, 0.001):
        optimizer = _stub_2d_optimizer(pywraplp.Solver.OPTIMAL)
        optimizer.time_limit = time_limit
        status_name, solution = optimizer.solve()

        assert status_name == "OPTIMAL"
        assert optimizer.solver_.time_limit_ms == 1
        assert list(solution) == [True, False, False]


def test_2d_mip_without_a_solution_leaves_the_objective_alone(capfd):
    # binning_2d._fit calls solver_statistics itself, so the 2D MIP reaches
    # the same objective read as the 1D one -- with no information() call
    # needed. Both branches that leave the solver without one are named: a
    # zero budget, which skips the solve, and a model no assignment of the
    # 4x4 grid can satisfy, which is solved and found infeasible.
    x2, z2, y2, e2 = build_2d()

    cases = {"UNKNOWN": dict(time_limit=0),
             "INFEASIBLE": dict(min_n_bins=1000)}

    for status, params in cases.items():
        optb = OptimalBinning2D(solver="mip", **params, **PREBINS_2D)
        optb.fit(x2, z2, y2)
        optb.information(print_level=2)
        captured = capfd.readouterr()

        assert optb.status == status
        assert_no_unsolved_objective_read(captured)
        assert optb._optimizer["objective"] == 0
        assert optb._optimizer["best_bound"] == 0


def test_2d_binning_table_at_a_zero_budget_warns_nothing():
    # A zero budget gives up before any rectangle is selected -- for both
    # formulations -- so the table has no bins and no records at all.
    # Building it must stay arithmetic on empty totals rather than a divide
    # by zero: the 1D table is records-gated and warning-free for the same
    # degeneracy, and the 2D one has to match it.
    x2, z2, y2, e2 = build_2d()

    for solver in ("cp", "mip"):
        optb = OptimalBinning2D(solver=solver, time_limit=0, **PREBINS_2D)
        optb.fit(x2, z2, y2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optb.binning_table.build()

        assert optb.status == "UNKNOWN"
        assert [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)] == []


def test_two_prebins_with_higher_order_trends():
    # The concave / convex constraint loops start at the third pre-bin, so a
    # two-bin problem must simply come back unconstrained.
    x2, y2 = build_binary([0.20, 0.80])

    for solver, mip_solver in SOLVERS:
        for trend in ("concave", "convex", "peak", "valley"):
            optb = OptimalBinning(user_splits=[0.5], solver=solver,
                                  mip_solver=mip_solver,
                                  monotonic_trend=trend)
            optb.fit(x2, y2)

            assert optb.status == "OPTIMAL"
            assert optb.splits == approx([0.5])


# ---------------------------------------------------------------------------
# scenario (stochastic) formulation
# ---------------------------------------------------------------------------

def scenarios():
    rng = np.random.RandomState(1)
    i1 = rng.choice(N_RECORDS, 240, replace=False)
    i2 = rng.choice(N_RECORDS, 240, replace=False)
    return [x, x[i1], x[i2]], [y, y[i1], y[i2]]


def test_scenarios_monotonic_trends():
    xs, ys = scenarios()

    for trend in (None, "ascending", "descending", "concave", "convex",
                  "peak", "valley"):
        sboptb = SBOptimalBinning(user_splits=splits_all,
                                  monotonic_trend=trend)
        sboptb.fit(xs, ys)

        assert sboptb.status == "OPTIMAL"
        assert set(sboptb.splits) <= set(splits_all)
        assert np.all(np.diff(sboptb.splits) > 0)

        rates = sboptb.binning_table.build()["Event rate"].values[:-3]
        rates = rates.astype(float)
        if trend == "ascending":
            assert np.all(np.diff(rates) >= -1e-9)
        elif trend == "descending":
            assert np.all(np.diff(rates) <= 1e-9)


def test_scenarios_min_max_n_bins():
    xs, ys = scenarios()

    sboptb = SBOptimalBinning(user_splits=splits_all, min_n_bins=3,
                              max_n_bins=5)
    sboptb.fit(xs, ys)

    assert sboptb.status == "OPTIMAL"
    assert 3 <= len(sboptb.splits) + 1 <= 5


def test_scenarios_min_max_bin_size():
    xs, ys = scenarios()

    sboptb = SBOptimalBinning(user_splits=splits_all, min_bin_size=0.2,
                              max_bin_size=0.4)
    sboptb.fit(xs, ys)

    assert sboptb.status == "OPTIMAL"

    for s, xi in enumerate(xs):
        table = sboptb.binning_table_scenario(s).build()
        counts = table["Count"].values[:-3].astype(int)
        assert np.all(counts >= int(np.ceil(0.2 * len(xi))))
        assert np.all(counts <= int(np.ceil(0.4 * len(xi))))


def test_scenarios_peak_with_bin_bounds():
    # peak / valley allocate the auxiliary change-point variables, and the
    # min/max number of bins allocates the range variable on top of them.
    xs, ys = scenarios()

    sboptb = SBOptimalBinning(user_splits=splits_all, monotonic_trend="peak",
                              min_n_bins=2, max_n_bins=4)
    sboptb.fit(xs, ys)

    assert sboptb.status == "OPTIMAL"
    assert 2 <= len(sboptb.splits) + 1 <= 4


# ---------------------------------------------------------------------------
# multiclass model data
# ---------------------------------------------------------------------------

def multiclass_data():
    props = [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2],
             [0.3, 0.4, 0.3], [0.2, 0.4, 0.4], [0.2, 0.3, 0.5],
             [0.1, 0.3, 0.6], [0.1, 0.2, 0.7]]
    xm = np.repeat(np.arange(len(props)).astype(float), 40)
    ym = np.empty(len(xm), dtype=int)
    for i, p in enumerate(props):
        labels = np.concatenate([np.full(int(round(pi * 40)), k)
                                 for k, pi in enumerate(p)])
        ym[i * 40:(i + 1) * 40] = np.resize(labels, 40)
    return xm, ym


def test_multiclass_max_pvalue():
    xm, ym = multiclass_data()

    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(user_splits=splits_all,
                                        solver=solver, max_pvalue=0.3)
        optb.fit(xm, ym)

        assert optb.status == "OPTIMAL"
        assert set(optb.splits) <= set(splits_all)
        assert np.all(np.diff(optb.splits) > 0)


def test_multiclass_min_event_rate_diff():
    xm, ym = multiclass_data()
    min_diff = 0.05

    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(user_splits=splits_all,
                                        solver=solver,
                                        min_event_rate_diff=min_diff)
        optb.fit(xm, ym)

        assert optb.status == "OPTIMAL"

        table = optb.binning_table.build()
        for column in table.columns:
            if not str(column).startswith("Event_rate_"):
                continue
            rates = table[column].values[:-3].astype(float)
            assert np.all(np.absolute(np.diff(rates)) >= min_diff - 1e-5)


def test_multiclass_max_pvalue_and_min_event_rate_diff():
    xm, ym = multiclass_data()

    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(user_splits=splits_all,
                                        solver=solver, max_pvalue=0.3,
                                        min_event_rate_diff=0.05)
        optb.fit(xm, ym)

        assert optb.status == "OPTIMAL"
        assert set(optb.splits) <= set(splits_all)


# ---------------------------------------------------------------------------
# model_data helpers reached directly
# ---------------------------------------------------------------------------

def test_model_data_return_nonevent_event():
    # return_nonevent_event=True is not used anywhere inside the library; it
    # returns the cumulative non-event / event matrices in place of the
    # min-diff violation indices.
    n_nonevent = np.array([36, 28, 20, 8], dtype=np.int64)
    n_event = np.array([4, 12, 20, 32], dtype=np.int64)

    returned = model_data("iv", n_nonevent, n_event, None, "consecutive", 0,
                          None, True)
    assert len(returned) == 5

    D, V, NE, E, pvalue_violation_indices = returned
    assert pvalue_violation_indices == []
    assert len(D) == len(V) == len(NE) == len(E) == 4

    # Row i holds the statistics of every bin ending at pre-bin i.
    assert E[3] == approx([68, 64, 52, 32])
    assert NE[3] == approx([92, 56, 28, 8])
    assert D[3] == approx(np.asarray(E[3]) / (np.asarray(E[3]) +
                                              np.asarray(NE[3])))
    assert np.all(np.asarray(V[3]) >= 0)

    # Without the flag the fourth item is the (empty) min-diff violations.
    D2, V2, pv2, md2 = model_data("iv", n_nonevent, n_event, None,
                                  "consecutive", 0)
    assert md2 == []
    assert D2[3] == approx(D[3])
    assert V2[3] == approx(V[3])


def test_continuous_model_data_unscaled():
    # continuous_model_data is only ever called with a scale by the library;
    # the unscaled branch returns the raw means and absolute deviations.
    n_records = np.array([10, 10, 10], dtype=np.int64)
    sums = np.array([10.0, 20.0, 60.0])
    ssums = np.array([12.0, 42.0, 362.0])

    U, V, pv, md = continuous_model_data(n_records, sums, ssums, None,
                                         "consecutive", 0, None)

    assert pv == [] and md == []
    assert U[0] == approx([1.0])
    assert U[2] == approx([3.0, 4.0, 6.0])
    total_mean = sums.sum() / n_records.sum()
    assert V[2] == approx(np.absolute(np.asarray(U[2]) - total_mean))
    # Unscaled means keep their floating point type.
    assert np.asarray(U[2]).dtype == np.float64

    U_s, V_s, _, _ = continuous_model_data(n_records, sums, ssums, None,
                                           "consecutive", 0, int(1e6))
    assert np.asarray(U_s[2]).dtype == np.int64
    assert U_s[2] == approx(np.asarray(U[2]) * 1e6)


def test_continuous_model_data_pvalue_and_min_mean_diff():
    # The continuous violation indices come from a Welch t-test on the
    # per-bin mean / std / count, and from the absolute difference of the
    # unscaled means. A violation is a pair the constraint must forbid, so
    # well separated bins produce none.
    n_records = np.array([10, 10, 10], dtype=np.int64)
    separated_sums = np.array([10.0, 20.0, 60.0])
    separated_ssums = np.array([12.0, 42.0, 362.0])

    means = np.array([1.0, 1.05, 1.1])
    close_sums = means * 10
    close_ssums = 10 * (means ** 2 + 1.0)

    for policy in ("consecutive", "all"):
        _, _, pv, md = continuous_model_data(
            n_records, separated_sums, separated_ssums, 0.5, policy, 0.0,
            None)
        assert pv == []
        assert md == []

        _, _, pv, _ = continuous_model_data(
            n_records, close_sums, close_ssums, 0.01, policy, 0.0, None)
        # Means 1.00 / 1.05 / 1.10 with unit variance are indistinguishable.
        assert len(pv) > 0
        assert all(len(pair) == 2 for pair in pv)

    _, _, _, md = continuous_model_data(n_records, separated_sums,
                                        separated_ssums, None,
                                        "consecutive", 0.5, None)
    assert md == []

    _, _, _, md = continuous_model_data(n_records, separated_sums,
                                        separated_ssums, None,
                                        "consecutive", 2.0, None)
    assert len(md) > 0


# ---------------------------------------------------------------------------
# degenerate and hostile inputs
# ---------------------------------------------------------------------------

def test_single_prebin_skips_the_solver():
    xc = np.ones(200)
    yc = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]

    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(solver=solver, mip_solver=mip_solver)
        optb.fit(xc, yc)

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) == 0
        assert optb.transform(xc).shape == (200,)


def test_single_class_target():
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(user_splits=splits_all, solver=solver,
                              mip_solver=mip_solver)
        optb.fit(x, np.ones(N_RECORDS, dtype=int))

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) == 0


def test_two_rows():
    for solver, mip_solver in SOLVERS:
        optb = OptimalBinning(solver=solver, mip_solver=mip_solver)
        optb.fit(np.array([0.0, 1.0]), np.array([0, 1]))

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) == 0


def test_hostile_inputs_raise():
    with raises(ValueError):
        OptimalBinning().fit(np.array([]), np.array([]))

    with raises(ValueError):
        OptimalBinning().fit(x, y[:-1])

    with raises(ValueError):
        OptimalBinning().fit(np.full(100, np.nan),
                             np.r_[np.zeros(50, dtype=int),
                                   np.ones(50, dtype=int)])

    # Magnitudes that overflow the float32 the pre-binning tree uses.
    with raises(ValueError):
        OptimalBinning().fit(x * 1e300, y)


def test_infinite_values_are_binned():
    xi = x.copy()
    xi[0] = np.inf
    xi[1] = -np.inf

    optb = OptimalBinning(user_splits=splits_all)
    optb.fit(xi, y)

    assert optb.status == "OPTIMAL"
    counts, _, _ = bin_counts(optb)
    assert counts.sum() == N_RECORDS


def test_special_codes_as_list_and_as_dict():
    xs = x.copy()
    xs[:20] = -1.0
    xs[20:40] = -2.0

    as_list = OptimalBinning(user_splits=splits_all, special_codes=[-1, -2])
    as_list.fit(xs, y)

    as_dict = OptimalBinning(user_splits=splits_all,
                             special_codes={"a": -1, "b": -2})
    as_dict.fit(xs, y)

    for optb in (as_list, as_dict):
        assert optb.status == "OPTIMAL"
        assert optb.transform(xs).shape == (N_RECORDS,)

    # The named form splits the special bucket in two but bins the rest
    # identically.
    assert as_dict.splits == approx(as_list.splits)

    table_list = as_list.binning_table.build()
    table_dict = as_dict.binning_table.build()
    assert len(table_dict) == len(table_list) + 1
    assert table_dict["Count"].values[-1] == table_list["Count"].values[-1]


def test_unfitted_estimator_raises():
    optb = OptimalBinning(user_splits=splits_all)

    with raises(NotFittedError):
        optb.binning_table
    with raises(NotFittedError):
        optb.splits
    with raises(NotFittedError):
        optb.status
    with raises(NotFittedError):
        optb.information()
    with raises(NotFittedError):
        optb.transform(x)
