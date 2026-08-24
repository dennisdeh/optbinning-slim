"""
OptimalPWBinning and ContinuousOptimalPWBinning edge-case testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import re
import warnings

from contextlib import redirect_stdout
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

from pytest import approx, raises

from optbinning import ContinuousOptimalPWBinning
from optbinning import OptimalBinning
from optbinning import OptimalPWBinning
from optbinning.binning.piecewise.base import _check_parameters
from optbinning.binning.piecewise.binning_information import retrieve_status
from optbinning.binning.piecewise.binning_statistics import PWBinningTable
from optbinning.binning.piecewise.metrics import divergences_asymptotic
from sklearn.exceptions import NotFittedError


# Small synthetic problems: a binning fit is a solver run, so the whole module
# is built on 300-row arrays with a low max_n_prebins rather than on the
# breast-cancer / boston fixtures the other piecewise modules use.
n_samples = 300

x = np.linspace(0, 10, n_samples)
y = (np.random.RandomState(0).uniform(size=n_samples) <
     1.0 / (1.0 + np.exp(-(x - 5)))).astype(int)
y_continuous = 2.0 * x + np.random.RandomState(1).normal(size=n_samples)

# A unimodal event rate: the automatic monotonic trend of a fit on this data
# is "peak", which no other shape in this module produces.
x_peak = np.linspace(-5, 5, n_samples)
y_peak = (np.random.RandomState(0).uniform(size=n_samples) <
          0.9 * np.exp(-x_peak ** 2 / 2.0) + 0.02).astype(int)


def _report(fn, *args, **kwargs):
    """Return everything ``fn`` prints on stdout."""
    with StringIO() as buf, redirect_stdout(buf):
        fn(*args, **kwargs)
        return buf.getvalue()


def _n_significance_rows(report):
    """Number of consecutive-bin rows of an analysis report's test table."""
    block = report.split("Significance tests")[1]
    return len([line for line in block.splitlines()
                if re.match(r"^\s+\d+\s+\d+\s", line)])


class _Regressor:
    """The minimal object ``problem_type="regression"`` accepts."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_params_estimator_regression():
    # ContinuousOptimalPWBinning always passes estimator=None, so the
    # regression branch of the shared checker is only reachable directly.
    params = ContinuousOptimalPWBinning().get_params(deep=False)

    with raises(TypeError):
        _check_parameters(**params, estimator=_Regressor().fit,
                          problem_type="regression")

    with raises(TypeError):
        _check_parameters(**params, estimator=OptimalBinning(),
                          problem_type="regression")

    # An object with both fit and predict is accepted.
    _check_parameters(**params, estimator=_Regressor(),
                      problem_type="regression")


def test_params_continuous_deriv():
    with raises(TypeError):
        OptimalPWBinning(continuous_deriv=1).fit(x, y)

    with raises(TypeError):
        ContinuousOptimalPWBinning(continuous_deriv="yes").fit(
            x, y_continuous)


def test_params_continuous_pw():
    with raises(TypeError):
        ContinuousOptimalPWBinning(name=1).fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(objective="l3").fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(degree=6).fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(degree=-1).fit(x, y_continuous)

    with raises(TypeError):
        ContinuousOptimalPWBinning(continuous="yes").fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(monotonic_trend="upwards").fit(
            x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(monotonic_trend="concave", degree=3).fit(
            x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(solver="cp").fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(h_epsilon=0.5).fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(quantile=1.0).fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(regularization="lasso").fit(
            x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(reg_l1=-1e-8).fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(reg_l2=-1e-8).fit(x, y_continuous)

    with raises(TypeError):
        ContinuousOptimalPWBinning(special_codes="-1").fit(x, y_continuous)

    with raises(ValueError):
        ContinuousOptimalPWBinning(split_digits=-1).fit(x, y_continuous)

    with raises(TypeError):
        ContinuousOptimalPWBinning(random_state=1.5).fit(x, y_continuous)

    with raises(TypeError):
        ContinuousOptimalPWBinning(verbose=None).fit(x, y_continuous)


def test_params_bounds():
    # lb/ub are fit arguments, validated by ropwr rather than by
    # _check_parameters.
    with raises(ValueError):
        OptimalPWBinning(max_n_prebins=6).fit(x, y, lb=1.0, ub=0.0)

    with raises(TypeError):
        OptimalPWBinning(max_n_prebins=6).fit(x, y, lb="0")

    with raises(TypeError):
        ContinuousOptimalPWBinning(max_n_prebins=6).fit(
            x, y_continuous, ub="10")


def test_unfitted():
    optb = OptimalPWBinning()

    with raises(NotFittedError):
        optb.binning_table

    with raises(NotFittedError):
        optb.splits

    with raises(NotFittedError):
        optb.status

    with raises(NotFittedError):
        optb.transform(x)

    with raises(NotFittedError):
        optb.information()

    optbc = ContinuousOptimalPWBinning()

    with raises(NotFittedError):
        optbc.splits

    with raises(NotFittedError):
        optbc.transform(x)


def test_splits_property():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x, y)

    splits = optb.splits
    assert isinstance(splits, np.ndarray)
    assert len(splits) == optb._n_bins - 1
    assert np.all(np.diff(splits) > 0)
    assert x.min() < splits.min() and splits.max() < x.max()

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6).fit(x, y_continuous)
    assert np.all(np.diff(optbc.splits) > 0)


def test_degenerate_inputs_binary():
    with raises(ValueError):
        OptimalPWBinning().fit(np.array([]), np.array([]))

    with raises(ValueError):
        OptimalPWBinning().fit(np.array([1.0]), np.array([1]))

    # A single class is not a binary target.
    with raises(ValueError):
        OptimalPWBinning().fit(x, np.ones(n_samples, dtype=int))

    with raises(ValueError):
        OptimalPWBinning().fit(x, y[:-1])

    with raises(ValueError):
        OptimalPWBinning().fit(np.full(n_samples, np.nan), y)

    with raises(ValueError):
        OptimalPWBinning().fit(np.r_[x[:-1], np.inf], y)

    with raises(ValueError):
        OptimalPWBinning().fit(np.full(n_samples, "a"), y)

    # More than two classes is a multiclass target.
    with raises(ValueError):
        OptimalPWBinning().fit(
            x, np.random.RandomState(2).randint(0, 3, n_samples))


def test_degenerate_inputs_continuous():
    with raises(ValueError):
        ContinuousOptimalPWBinning().fit(np.array([]), np.array([]))

    with raises(ValueError):
        ContinuousOptimalPWBinning().fit(x, y_continuous[:-1])

    with raises(ValueError):
        ContinuousOptimalPWBinning().fit(np.r_[x[:-1], -np.inf], y_continuous)

    # A constant target carries no information: one bin, no splits.
    optb = ContinuousOptimalPWBinning(max_n_prebins=6).fit(
        x, np.ones(n_samples))
    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb.transform(x) == approx(np.ones(n_samples), abs=1e-6)


def test_extreme_magnitudes():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x * 1e12, y)

    assert optb.status == "OPTIMAL"
    assert np.all(np.isfinite(optb.splits))
    assert np.all(np.diff(optb.splits) > 0)


def test_duplicated_values():
    x_dup = np.repeat(np.linspace(0, 10, 20), 15)
    y_dup = (np.random.RandomState(3).uniform(size=n_samples) <
             1.0 / (1.0 + np.exp(-(x_dup - 5)))).astype(int)

    optb = OptimalPWBinning(max_n_prebins=6).fit(x_dup, y_dup)

    assert optb.status == "OPTIMAL"
    assert len(np.unique(optb.splits)) == len(optb.splits)


def test_no_splits_binary():
    # max_n_bins=1 leaves a single bin: the automatic monotonic trend is
    # "undefined" and no trend is imposed on the regression.
    optb = OptimalPWBinning(max_n_prebins=6, max_n_bins=1).fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb._optimizer.monotonic_trend is None

    df = optb.binning_table.build()
    assert list(df["Bin"])[:3] == ["(-inf, inf)", "Special", "Missing"]
    assert df["Count"]["Totals"] == n_samples

    # A single bin means no consecutive pair to test.
    report = _report(optb.binning_table.analysis)
    assert _n_significance_rows(report) == 0
    assert "None" in report.split("Significance tests")[1]
    assert optb.binning_table.quality_score == 0

    x_transform = optb.transform(x, metric="event_rate")
    assert x_transform == approx(np.polyval(optb._c[0, ::-1], x))


def test_no_splits_continuous():
    optb = ContinuousOptimalPWBinning(max_n_prebins=6, max_n_bins=1).fit(
        x, y_continuous)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    df = optb.binning_table.build()
    assert list(df["Bin"])[:3] == ["(-inf, inf)", "Special", "Missing"]

    report = _report(optb.binning_table.analysis)
    assert _n_significance_rows(report) == 0
    assert "None" in report.split("Significance tests")[1]

    assert optb.transform(x) == approx(np.polyval(optb._c[0, ::-1], x))


def test_monotonic_trend_auto_peak():
    optb = OptimalPWBinning(max_n_prebins=8).fit(x_peak, y_peak)

    assert optb.status == "OPTIMAL"
    assert optb._optimizer.monotonic_trend == "peak"

    # A peak is what the fitted event rate actually does.
    x_transform = optb.transform(x_peak, metric="event_rate")
    argmax = np.argmax(x_transform)
    assert 0 < argmax < n_samples - 1
    assert np.all(np.diff(x_transform[:argmax + 1]) >= -1e-8)
    assert np.all(np.diff(x_transform[argmax:]) <= 1e-8)


def test_n_subsamples():
    optb = OptimalPWBinning(max_n_prebins=6, n_subsamples=100,
                            random_state=42, verbose=True).fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb._c.shape == (len(optb.splits) + 1, 2)

    # n_subsamples above the sample count uses every sample.
    optb_all = OptimalPWBinning(max_n_prebins=6, n_subsamples=10 * n_samples,
                                random_state=42).fit(x, y)
    optb_none = OptimalPWBinning(max_n_prebins=6).fit(x, y)
    assert optb_all._c == approx(optb_none._c)


def test_verbose_outlier_detector():
    optb = OptimalPWBinning(max_n_prebins=6, verbose=True,
                            outlier_detector="range").fit(x, y)
    assert optb.status == "OPTIMAL"

    optbc = ContinuousOptimalPWBinning(
        max_n_prebins=6, verbose=True, outlier_detector="zscore",
        outlier_params={"threshold": 3.5}).fit(x, y_continuous)
    assert optbc.status == "OPTIMAL"


def test_information_continuous():
    optb = ContinuousOptimalPWBinning(name="v", max_n_prebins=6).fit(
        x, y_continuous)

    with raises(ValueError):
        optb.information(print_level=-1)

    with raises(ValueError):
        optb.information(print_level=1.5)

    report_0 = _report(optb.information, print_level=0)
    assert "OPTIMAL" in report_0

    report_1 = _report(optb.information, print_level=1)
    assert "Pre-binning statistics" in report_1
    assert "Solver statistics" in report_1

    # ContinuousOptimalPWBinning has no estimator parameter; information()
    # injects one so the options report can name it.
    report_2 = _report(optb.information, print_level=2)
    assert "estimator" in report_2
    assert "Begin options" in report_2


def test_information_solver_statistics_list():
    # A discontinuous regression is solved bin by bin, so the solver stats
    # and the solver status are lists.
    optb = OptimalPWBinning(name="v", max_n_prebins=6, continuous=False).fit(
        x, y)

    assert isinstance(optb._optimizer.stats, list)
    assert optb.status == "OPTIMAL"

    report = _report(optb.information, print_level=1)
    n_variables = sum(info["n_variables"] for info in optb._optimizer.stats)
    n_constraints = sum(
        info["n_constraints"] for info in optb._optimizer.stats)

    assert "Number of variables           {:>10}".format(
        n_variables) in report
    assert "Number of constraints         {:>10}".format(
        n_constraints) in report


def test_information_without_optimizer():
    optb = OptimalPWBinning(name="v", max_n_prebins=6).fit(x, y)

    # Defensive branch: no fit leaves _optimizer unset, so it is stubbed.
    optb._optimizer = None
    report = _report(optb.information, print_level=1)

    assert "Solver statistics" not in report
    assert "Pre-binning statistics" in report
    assert "Solver                              0.00 sec" in report


def test_retrieve_status_list():
    assert retrieve_status(["optimal", "optimal_inaccurate"]) == "OPTIMAL"
    assert retrieve_status(["feasible", "feasible"]) == "FEASIBLE"
    assert retrieve_status(["unbounded", "unbounded"]) == "UNBOUNDED"

    assert retrieve_status(
        ["optimal", "feasible"]) == "OPTIMAL (1/2)FEASIBLE (1/2)"
    assert retrieve_status(
        ["optimal", "unbounded"]) == "OPTIMAL (1/2)UNBOUNDED (1/2)"
    assert retrieve_status(
        ["feasible", "unbounded"]) == "FEASIBLE (1/2)UNBOUNDED (1/2)"
    assert retrieve_status(["optimal", "feasible", "unbounded"]) == (
        "OPTIMAL (1/3)FEASIBLE (1/3)UNBOUNDED (1/3)")


def test_retrieve_status_scalar():
    assert retrieve_status("optimal_inaccurate") == "OPTIMAL"
    assert retrieve_status("feasible") == "FEASIBLE"
    assert retrieve_status("unbounded_inaccurate") == "UNBOUNDED"
    assert retrieve_status("solver_error") is None


def test_divergences_asymptotic_scalar_special():
    # binary_metrics always passes arrays; the scalar branch is only
    # reachable directly.
    event_rate = np.array([0.2, 0.5, 0.8])

    d_array = divergences_asymptotic(
        event_rate, np.array([30.0]), np.array([0.0]), 0.0, 0.0, 100.0, 50.0)
    d_scalar = divergences_asymptotic(
        event_rate, 30.0, 0.0, 0.0, 0.0, 100.0, 50.0)

    assert set(d_scalar) == {"IV (Jeffrey)", "JS (Jensen-Shannon)",
                             "Hellinger", "Triangular"}
    assert all(isinstance(v, float) for v in d_scalar.values())
    for k in d_scalar:
        assert d_scalar[k] == approx(d_array[k])


def test_special_codes_dict_binary():
    x_special = x.copy()
    x_special[:20] = -1
    x_special[20:40] = -2
    special_codes = {"code_a": -1, "code_b": -2}

    optb = OptimalPWBinning(max_n_prebins=6,
                            special_codes=special_codes).fit(x_special, y)

    df = optb.binning_table.build()
    assert optb.binning_table._n_specials == 2
    assert list(df["Bin"])[-4:-1] == ["code_a", "code_b", "Missing"]
    assert df["Count"]["Totals"] == n_samples

    # One empirical value per named special bucket.
    x_transform = optb.transform([-1, -2], metric="event_rate",
                                 metric_special="empirical")
    assert x_transform.shape == (2,)
    assert x_transform[0] != x_transform[1]

    x_woe = optb.transform([-1, -2], metric="woe", metric_special="empirical")
    assert x_woe.shape == (2,)

    # A numeric metric_special overrides every bucket.
    assert optb.transform([-1, -2], metric_special=-7) == approx([-7, -7])


def test_special_codes_dict_continuous():
    x_special = x.copy()
    x_special[:20] = -1
    x_special[20:40] = -2
    x_special[40:50] = np.nan
    special_codes = {"code_a": -1, "code_b": -2}

    optb = ContinuousOptimalPWBinning(
        max_n_prebins=6, special_codes=special_codes).fit(x_special,
                                                          y_continuous)

    df = optb.binning_table.build()
    assert optb.binning_table._n_specials == 2
    assert list(df["Bin"])[-4:-1] == ["code_a", "code_b", "Missing"]
    assert df["Count"]["Totals"] == n_samples

    x_transform = optb.transform([-1, -2, np.nan], metric_special="empirical",
                                 metric_missing="empirical")
    assert x_transform.shape == (3,)
    assert x_transform[0] == approx(y_continuous[:20].mean())
    assert x_transform[1] == approx(y_continuous[20:40].mean())
    assert x_transform[2] == approx(y_continuous[40:50].mean())

    assert optb.transform([-1, -2], metric_special=-7) == approx([-7, -7])


def test_special_codes_list_binary():
    x_special = x.copy()
    y_special = y.copy()
    x_special[:40] = -1
    y_special[:40] = 0
    y_special[:10] = 1

    optb = OptimalPWBinning(max_n_prebins=6,
                            special_codes=[-1]).fit(x_special, y_special)

    df = optb.binning_table.build()
    assert optb.binning_table._n_specials == 1
    assert list(df["Bin"])[-3:-1] == ["Special", "Missing"]
    assert df["Non-event"]["Totals"] + df["Event"]["Totals"] == n_samples

    # The two special values share one bucket, so they share one value.
    x_transform = optb.transform([-1, -1], metric="event_rate",
                                 metric_special="empirical")
    assert x_transform[0] == x_transform[1]


def test_empirical_missing_leaves_clean_values_alone():
    x_missing = x.copy()
    x_missing[:30] = np.nan

    optb = OptimalPWBinning(max_n_prebins=6).fit(x_missing, y)

    # An unbounded fit predicts event rates outside [0, 1], whose WoE is not
    # a number, so the comparison is made on a bounded transform.
    baseline = optb.transform(x_missing, metric="woe", metric_missing=0,
                              lb=1e-6, ub=1 - 1e-6)
    x_transform = optb.transform(x_missing, metric="woe",
                                 metric_missing="empirical", lb=1e-6,
                                 ub=1 - 1e-6)

    # The two calls put different values in the missing bucket by
    # construction, so only the clean part is pinned here; the bucket's own
    # value is pinned by
    # test_defect_empirical_special_metric_is_the_non_event_rate.
    assert x_transform[30:] == approx(baseline[30:])
    assert x_transform.shape == x_missing.shape


def test_transform_metric_guards():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x, y)

    with raises(ValueError):
        optb.transform(x, metric="mean")

    with raises(ValueError):
        optb.transform(x, metric="indices")

    with raises(ValueError):
        optb.transform(x, metric_special="mean")

    with raises(ValueError):
        optb.transform(x, metric_missing="mean")

    with raises(ValueError):
        optb.transform(x, metric_special=[0, 1])

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6).fit(x, y_continuous)

    with raises(ValueError):
        optbc.transform(x, metric_special="mean")

    with raises(ValueError):
        optbc.transform(x, metric_missing="mean")


def test_transform_check_input():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x, y)

    assert optb.transform(x, metric="event_rate", check_input=True) == approx(
        optb.transform(x, metric="event_rate", check_input=False))

    with raises(ValueError):
        optb.transform(np.r_[x[:-1], np.inf], check_input=True)

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6).fit(x, y_continuous)

    assert optbc.transform(x, check_input=True) == approx(
        optbc.transform(x, check_input=False))

    with raises(ValueError):
        optbc.transform(np.r_[x[:-1], np.inf], check_input=True)


def test_transform_bounds_clip():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x, y)

    x_transform = optb.transform(x, metric="event_rate", lb=0.2, ub=0.8)
    assert x_transform.min() >= 0.2
    assert x_transform.max() <= 0.8

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6).fit(x, y_continuous)

    xc_transform = optbc.transform(x, lb=5.0, ub=15.0)
    assert xc_transform.min() >= 5.0
    assert xc_transform.max() <= 15.0


def test_binning_table_build_guards():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x, y)

    with raises(ValueError):
        optb.binning_table.build(show_digits=9)

    with raises(TypeError):
        optb.binning_table.build(add_totals=1)

    df = optb.binning_table.build(show_digits=0, add_totals=False)
    assert "Totals" not in df.index
    assert len(df) == len(optb.splits) + 3

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6).fit(x, y_continuous)

    with raises(ValueError):
        optbc.binning_table.build(show_digits=-1)

    with raises(TypeError):
        optbc.binning_table.build(add_totals="yes")


def test_binning_table_not_built():
    optb = OptimalPWBinning(max_n_prebins=6).fit(x, y)

    with raises(NotFittedError):
        optb.binning_table.plot()

    with raises(NotFittedError):
        optb.binning_table.analysis(print_output=False)

    with raises(NotFittedError):
        optb.binning_table.iv

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6).fit(x, y_continuous)

    with raises(NotFittedError):
        optbc.binning_table.plot()

    with raises(NotFittedError):
        optbc.binning_table.analysis(print_output=False)


def test_binning_table_plot_guards(tmp_path):
    optb = OptimalPWBinning(name="v", max_n_prebins=6).fit(x, y)
    optb.binning_table.build()

    with raises(ValueError):
        optb.binning_table.plot(metric="mean")

    with raises(TypeError):
        optb.binning_table.plot(savefig=1)

    with raises(TypeError):
        optb.binning_table.plot(savefig=str(tmp_path / "p.png"),
                                save_kwargs=1)

    path = tmp_path / "event_rate.png"
    optb.binning_table.plot(metric="event_rate", n_samples=100,
                            savefig=str(path), save_kwargs={"dpi": 50})
    assert path.exists()

    optbc = ContinuousOptimalPWBinning(name="v", max_n_prebins=6).fit(
        x, y_continuous)
    optbc.binning_table.build()

    with raises(TypeError):
        optbc.binning_table.plot(savefig=1)

    path_c = tmp_path / "continuous.png"
    optbc.binning_table.plot(n_samples=100, savefig=str(path_c))
    assert path_c.exists()


def test_binning_table_plot_show(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(1))

    optb = OptimalPWBinning(name="v", max_n_prebins=6).fit(x, y)
    optb.binning_table.build()
    optb.binning_table.plot(n_samples=100)

    optbc = ContinuousOptimalPWBinning(name="v", max_n_prebins=6).fit(
        x, y_continuous)
    optbc.binning_table.build()
    optbc.binning_table.plot(n_samples=100)

    assert len(shown) == 2
    plt.close("all")


def test_binning_table_analysis_print():
    optb = OptimalPWBinning(name="v", max_n_prebins=6).fit(x, y)
    optb.binning_table.build()

    report = _report(optb.binning_table.analysis)

    assert "Binary Binning Table Analysis" in report
    assert "IV (Jeffrey)" in report
    assert "Cramer's V" in report
    assert _n_significance_rows(report) == len(optb.splits)
    assert optb.binning_table.quality_score >= 0


def test_user_splits_fixed():
    user_splits = [2.0, 4.0, 6.0, 8.0]
    user_splits_fixed = [False, True, True, False]

    optb = OptimalPWBinning(max_n_prebins=6, user_splits=user_splits,
                            user_splits_fixed=user_splits_fixed).fit(x, y)

    assert optb.status == "OPTIMAL"
    assert set([4.0, 6.0]).issubset(set(np.round(optb.splits, 8)))
    assert set(np.round(optb.splits, 8)).issubset(set(user_splits))

    optbc = ContinuousOptimalPWBinning(
        max_n_prebins=6, user_splits=user_splits,
        user_splits_fixed=user_splits_fixed).fit(x, y_continuous)

    assert set([4.0, 6.0]).issubset(set(np.round(optbc.splits, 8)))


def test_degree_zero_discontinuous():
    optb = OptimalPWBinning(max_n_prebins=6, degree=0,
                            continuous=False).fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb._c.shape == (len(optb.splits) + 1, 1)

    # A piecewise constant fit takes exactly one value per bin.
    x_transform = optb.transform(x, metric="event_rate")
    assert len(np.unique(np.round(x_transform, 10))) == len(optb.splits) + 1


def test_objectives_and_regularization():
    for kwargs in (dict(objective="l1"),
                   dict(objective="huber", h_epsilon=2.0),
                   dict(objective="quantile", quantile=0.25),
                   dict(regularization="l1", reg_l1=0.1),
                   dict(regularization="l2", reg_l2=0.1),
                   dict(degree=2),
                   dict(degree=3, continuous_deriv=False)):

        optb = OptimalPWBinning(max_n_prebins=6, **kwargs).fit(x, y)

        assert optb.status == "OPTIMAL"
        assert optb._c.shape == (len(optb.splits) + 1,
                                 kwargs.get("degree", 1) + 1)
        assert np.all(np.isfinite(optb._c))


def test_prebinning_methods_and_split_digits():
    for method in ("cart", "quantile", "uniform"):
        optb = OptimalPWBinning(max_n_prebins=6, prebinning_method=method,
                                split_digits=2).fit(x, y)

        assert optb.status == "OPTIMAL"
        assert optb.splits == approx(np.round(optb.splits, 2))


def test_solver_highs_and_direct():
    # "highs" is only accepted for objective="l1", "direct" only for an
    # unconstrained l2 problem.
    optb = OptimalPWBinning(max_n_prebins=6, solver="highs",
                            objective="l1").fit(x, y)
    assert optb.status == "OPTIMAL"

    optb = OptimalPWBinning(max_n_prebins=6, solver="direct",
                            monotonic_trend=None).fit(x, y)
    assert optb.status == "OPTIMAL"

    optbc = ContinuousOptimalPWBinning(max_n_prebins=6, solver="scs").fit(
        x, y_continuous)
    assert optbc.status == "OPTIMAL"


def test_defect_missing_bin_with_both_classes():
    x_missing = x.copy()
    y_missing = y.copy()
    x_missing[:40] = np.nan
    y_missing[:40] = 0
    y_missing[:10] = 1

    # The non-piecewise sibling fits the same data.
    OptimalBinning(dtype="numerical", max_n_prebins=6).fit(
        x_missing, y_missing)

    optb = OptimalPWBinning(max_n_prebins=6).fit(x_missing, y_missing)

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Count"]["Totals"] == n_samples

    # The divergences stay scalars, which is what the crash was about.
    for metric in ("IV (Jeffrey)", "JS (Jensen-Shannon)", "Hellinger",
                   "Triangular"):
        assert isinstance(optb.binning_table.d_metrics[metric], float)


def test_divergences_asymptotic_missing_with_both_classes():
    # A missing bucket carrying both classes contributes one term; summing
    # that term is what keeps the result a scalar.
    event_rate = np.array([0.2, 0.5, 0.8])

    d_pure = divergences_asymptotic(
        event_rate, np.array([0.0]), np.array([0.0]), 30.0, 0.0, 100.0, 50.0)
    d_mixed = divergences_asymptotic(
        event_rate, np.array([0.0]), np.array([0.0]), 30.0, 10.0, 100.0, 50.0)

    assert all(isinstance(v, float) for v in d_mixed.values())

    # The term is added, not dropped.
    for k in d_mixed:
        assert d_mixed[k] > d_pure[k]


def test_defect_empirical_special_metric_is_the_non_event_rate():
    x_special = x.copy()
    y_special = y.copy()
    x_special[:40] = -1
    y_special[:40] = 0
    y_special[:10] = 1

    # The special bucket holds 10 events out of 40 records.
    optb = OptimalPWBinning(max_n_prebins=6,
                            special_codes=[-1]).fit(x_special, y_special)
    optb_ref = OptimalBinning(dtype="numerical", max_n_prebins=6,
                              special_codes=[-1]).fit(x_special, y_special)

    assert optb_ref.transform([-1], metric="event_rate",
                              metric_special="empirical") == approx([0.25])

    assert optb.transform([-1], metric="event_rate",
                          metric_special="empirical") == approx([0.25])

    # The missing bucket takes the same route.
    x_missing = x.copy()
    y_missing = y.copy()
    x_missing[:40] = np.nan
    y_missing[:40] = 0
    y_missing[:10] = 1

    optb_m = OptimalPWBinning(max_n_prebins=6).fit(x_missing, y_missing)
    optb_m_ref = OptimalBinning(dtype="numerical", max_n_prebins=6).fit(
        x_missing, y_missing)

    assert optb_m.transform([np.nan], metric="event_rate",
                            metric_missing="empirical") == approx(
        optb_m_ref.transform([np.nan], metric="event_rate",
                             metric_missing="empirical"))

    # And so does each named bucket of the dict form.
    x_dict = x.copy()
    y_dict = y.copy()
    x_dict[:20] = -1
    x_dict[20:40] = -2
    y_dict[:20] = 0
    y_dict[:5] = 1
    y_dict[20:40] = 1
    y_dict[20:25] = 0
    special_codes = {"code_a": -1, "code_b": -2}

    optb_d = OptimalPWBinning(max_n_prebins=6,
                              special_codes=special_codes).fit(x_dict, y_dict)

    assert optb_d.transform([-1, -2], metric="event_rate",
                            metric_special="empirical") == approx(
        [0.25, 0.75])


def test_defect_analysis_counts_special_bins_as_regular_bins():
    x_special = x.copy()
    x_special[:20] = -1
    x_special[20:40] = -2
    special_codes = {"code_a": -1, "code_b": -2}

    optb = OptimalPWBinning(max_n_prebins=6,
                            special_codes=special_codes).fit(x_special, y)
    optb.binning_table.build()
    report = _report(optb.binning_table.analysis)

    # One row per consecutive pair of non-special bins.
    assert _n_significance_rows(report) == len(optb.splits)

    optbc = ContinuousOptimalPWBinning(
        max_n_prebins=6, special_codes=special_codes).fit(
            x_special, y_continuous)
    optbc.binning_table.build()
    report_c = _report(optbc.binning_table.analysis)

    assert _n_significance_rows(report_c) == len(optbc.splits)


def test_defect_continuous_fit_transform_drops_bounds():
    optb_ft = ContinuousOptimalPWBinning(max_n_prebins=6)
    x_ft = optb_ft.fit_transform(x, y_continuous, lb=5.0, ub=15.0)

    optb = ContinuousOptimalPWBinning(max_n_prebins=6)
    optb.fit(x, y_continuous, lb=5.0, ub=15.0)
    x_t = optb.transform(x, lb=5.0, ub=15.0)

    # The binary sibling passes lb/ub through to fit; this one does not, and
    # sends check_input into the lb argument instead.
    assert optb_ft.binning_table.lb == 5.0
    assert optb_ft.binning_table.ub == 15.0
    assert x_ft == approx(x_t)

    # check_input landed in lb, so even a fit without bounds diverged.
    optb_ci = ContinuousOptimalPWBinning(max_n_prebins=6)
    x_ci = optb_ci.fit_transform(x, y_continuous, check_input=True)

    assert optb_ci.binning_table.lb is None
    assert optb_ci.binning_table.ub is None
    assert x_ci == approx(
        ContinuousOptimalPWBinning(max_n_prebins=6).fit(
            x, y_continuous).transform(x))


def test_defect_pure_special_bucket_event_rate_parity():
    # A special bucket holding only events. Its empirical event rate is 1,
    # and the c0 column of the binning table is exactly the constant the
    # transform uses for that bucket, so the two must agree.
    x_special = x.copy()
    y_special = y.copy()
    x_special[:40] = -1
    y_special[:40] = 1

    optb = OptimalPWBinning(max_n_prebins=6,
                            special_codes=[-1]).fit(x_special, y_special)

    df = optb.binning_table.build()
    row = df[df["Bin"] == "Special"].iloc[0]

    assert row["Non-event"] == 0
    assert row["Event"] == 40
    assert row["c0"] == approx(1.0)

    assert optb.transform([-1], metric="event_rate",
                          metric_special="empirical") == approx([row["c0"]])

    # An all-non-event bucket is the mirror image: rate 0, not "no rate".
    y_nonevent = y.copy()
    y_nonevent[:40] = 0

    optb_n = OptimalPWBinning(max_n_prebins=6,
                              special_codes=[-1]).fit(x_special, y_nonevent)

    df_n = optb_n.binning_table.build()
    assert df_n[df_n["Bin"] == "Special"].iloc[0]["c0"] == approx(0.0)
    assert optb_n.transform([-1], metric="event_rate",
                            metric_special="empirical") == approx([0.0])


def test_defect_empirical_woe_of_a_pure_bucket():
    # A bucket missing one of the two classes has no odds ratio to report:
    # its WoE is 0, the same gate the binning table uses. Converting its
    # event rate directly divides by zero -- the empty missing bucket even
    # raised ZeroDivisionError.
    x_special = x.copy()
    y_special = y.copy()
    x_special[:40] = -1
    y_special[:40] = 1

    optb = OptimalPWBinning(max_n_prebins=6,
                            special_codes=[-1]).fit(x_special, y_special)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        t = optb.transform([-1, np.nan], metric="woe",
                           metric_special="empirical",
                           metric_missing="empirical")

    assert t == approx([0.0, 0.0])


def test_defect_fit_does_not_replace_the_estimator_parameter():
    # fit() must not write a fitted LogisticRegression into the estimator
    # constructor parameter: get_params must survive a fit unchanged.
    optb = OptimalPWBinning(max_n_prebins=6)

    assert optb.get_params()["estimator"] is None

    optb.fit(x, y)

    assert optb.get_params()["estimator"] is None
    assert optb.estimator is None


def test_defect_plot_woe_curve_stays_finite_at_the_bounds(monkeypatch):
    # The plotted WoE curve is the fitted probability run through
    # log((1 - p) / p) + constant, so a p of exactly 0 or 1 makes it
    # infinite. plot() clipped the probability to [0, 1], which permits
    # both: the divide escaped as a RuntimeWarning and matplotlib silently
    # dropped every infinite point, so part of the drawn curve was missing.
    # piecewise/metrics.py::binary_metrics already bounds the same quantity
    # away from 0 and 1 by 1e-8 before scoring it.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    optb = OptimalPWBinning(name="v", max_n_prebins=6).fit(x, y)
    optb.binning_table.build()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        optb.binning_table.plot(metric="woe", n_samples=10000)

    curve = plt.gcf().axes[1].lines[-1].get_ydata()
    plt.close("all")

    assert len(curve) == 10000
    assert np.isfinite(curve).all()


def test_woe_transform_is_finite_on_an_unbounded_fit():
    # The piecewise fit is a regression, so its predicted event rate is not
    # confined to [0, 1] unless lb/ub are passed, and log(1 / p - 1) is not
    # finite outside it. Before the bound, this ordinary step function gave
    # 98 NaN of 300 plus `invalid value encountered in log`, and the NaN
    # travelled through BinningProcess.transform and transform_disk.
    x_step = np.linspace(0, 10, 300)
    y_step = (x_step > 5).astype(int)

    optb = OptimalPWBinning(max_n_prebins=6)
    optb.fit(x_step, y_step)

    # The prediction really does leave [0, 1] here -- otherwise the test
    # would pass for the wrong reason.
    event_rate = optb.transform(x_step, metric="event_rate")
    assert event_rate.min() < 0
    assert event_rate.max() > 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        woe = optb.transform(x_step, metric="woe")

    assert np.isfinite(woe).all()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    # The bound is a safety net for the WoE conversion only: event_rate is
    # the raw prediction the caller asked for and stays unbounded.
    assert np.isfinite(event_rate).all()


def test_caller_supplied_estimator_is_used():
    # The default builds a LogisticRegression; the else branch takes the
    # caller's object. Passing one is the only route into it.
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(C=0.5)
    optb = OptimalPWBinning(estimator=estimator, max_n_prebins=6)
    x_step = np.linspace(0, 10, 300)
    optb.fit(x_step, (x_step > 5).astype(int))

    assert optb.status == "OPTIMAL"
    # Documented in reports/OPEN_ITEMS.md: fit uses the caller's object
    # rather than a clone of it.
    assert optb.estimator is estimator


def test_woe_of_a_missing_bucket_holding_both_classes():
    # metric="woe" with metric_missing="empirical" and a missing bucket that
    # is not pure -- the branch that converts the bucket's empirical event
    # rate to WoE.
    x_m = np.r_[np.linspace(0, 10, 300), np.full(40, np.nan)]
    y_m = np.r_[(np.linspace(0, 10, 300) > 5).astype(int),
                np.array([1] * 15 + [0] * 25)]

    optb = OptimalPWBinning(max_n_prebins=6).fit(x_m, y_m)
    woe = optb.transform(np.array([np.nan]), metric="woe",
                         metric_missing="empirical")

    assert np.isfinite(woe).all()
    assert woe[0] != 0.0

    # A pure missing bucket has no WoE and reports 0, not +/-inf.
    y_pure = np.r_[(np.linspace(0, 10, 300) > 5).astype(int),
                   np.zeros(40, int)]
    optb_pure = OptimalPWBinning(max_n_prebins=6).fit(x_m, y_pure)
    assert optb_pure.transform(np.array([np.nan]), metric="woe",
                               metric_missing="empirical")[0] == 0.0


def test_empty_piecewise_table_reports_zero_shares():
    # t_n_records == 0 has no shares to report. Only reachable on a directly
    # constructed table: a fit always has records.
    table = PWBinningTable(
        name="x", special_codes=None, splits=np.array([]),
        coef=np.array([[0.5]]), n_nonevent=np.array([0, 0, 0]),
        n_event=np.array([0, 0, 0]), min_x=0.0, max_x=1.0,
        d_metrics={"Gini index": 0.0, "IV (Jeffrey)": 0.0,
                   "JS (Jensen-Shannon)": 0.0, "Hellinger": 0.0,
                   "Triangular": 0.0, "KS": 0.0, "Avg precision": 0.0,
                   "Brier score": 0.0})

    df = table.build()

    # Zero shares, not nan, and no RuntimeWarning on the way there.
    assert list(df["Count (%)"]) == [0.0] * len(df)
