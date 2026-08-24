"""
Scorecard, ScorecardRounding and ScorecardMonitoring edge-case testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import io

from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.scorecard import ScorecardMonitoring
from optbinning.scorecard import scorecard as scorecard_module
from optbinning.scorecard.rounding import RoundingMIP

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def _data(n=300, seed=1):
    """Small binary-target frame; two informative numerical variables."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 10, size=n)
    z = 1.5 * x1 - 0.3 * x2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame({"x1": x1, "x2": x2}), y


def _binning_process(**kwargs):
    # Small prebin/bin counts: every fit here is a CP-SAT run.
    return BinningProcess(["x1", "x2"], max_n_prebins=5, max_n_bins=4,
                          **kwargs)


class _CoefOnlyEstimator(BaseEstimator):
    """Exposes coef_ but no intercept_, unlike every sklearn linear model."""
    def fit(self, X, y):
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        return np.asarray(X).sum(axis=1)


class _InfeasibleRoundingMIP:
    """Stands in for RoundingMIP to force the non-optimal solve branch."""
    def build_model(self, df_scorecard):
        self.df_scorecard = df_scorecard

    def solve(self):
        return "INFEASIBLE", None


# ------------------------------------------------------- unfitted estimator

def test_unfitted_accessors_raise():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression())

    with raises(NotFittedError):
        scorecard.table()

    with raises(NotFittedError):
        scorecard.transform(X)

    with raises(NotFittedError):
        scorecard.decision_function(X)

    with raises(NotFittedError):
        scorecard.information()


# --------------------------------------------------- transform / score / df

def test_transform_agrees_with_score():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 100},
                          intercept_based=True).fit(X, y)

    X_head = X.head(10)
    df_points = scorecard.transform(X_head)

    assert isinstance(df_points, pd.DataFrame)
    assert list(df_points.columns) == list(
        scorecard.binning_process_.get_support(names=True))
    assert df_points.shape == (10, 2)

    # score is the row sum of the per-variable points plus the intercept.
    assert df_points.sum(axis=1).values + scorecard.intercept_ == approx(
        scorecard.score(X_head))


def test_transform_special_and_missing_use_empirical_metric():
    X, y = _data()
    X = X.copy()
    X.loc[:20, "x1"] = -9
    X.loc[21:30, "x2"] = np.nan

    scorecard = Scorecard(
        binning_process=_binning_process(special_codes=[-9]),
        estimator=LogisticRegression()).fit(X, y)

    df_points = scorecard.transform(X.head(40))

    # transform always requests the empirical metric, so the special and
    # missing rows of the scorecard table are the ones used.
    table = scorecard.table()
    special_x1 = table[(table.Variable == "x1") &
                       (table.Bin == "Special")].Points.values[0]
    missing_x2 = table[(table.Variable == "x2") &
                       (table.Bin == "Missing")].Points.values[0]

    assert df_points.x1.values[0] == approx(special_x1)
    assert df_points.x2.values[25] == approx(missing_x2)
    assert np.isfinite(df_points.values).all()


def test_decision_function_matches_predict():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression()).fit(X, y)

    scores = scorecard.decision_function(X)

    assert scores.shape == (len(y),)
    assert np.array_equal((scores > 0).astype(int), scorecard.predict(X))


def test_decision_function_estimator_without_it():
    X, y = _data()
    y_continuous = y.astype(float) + np.linspace(0, 1, len(y))

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LinearRegression()).fit(X, y_continuous)

    with raises(AttributeError):
        scorecard.decision_function(X)

    with raises(AttributeError):
        scorecard.predict_proba(X)


def test_score_requires_the_fitted_variables():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression()).fit(X, y)

    # Extra columns and a different column order are both fine.
    X_extra = X.copy()
    X_extra["unused"] = 1.0
    X_extra = X_extra[["unused", "x2", "x1"]]
    assert scorecard.score(X_extra.head()) == approx(
        scorecard.score(X.head()))

    with raises(KeyError):
        scorecard.score(X[["x1"]].head())


# ------------------------------------------------------------- save / load

def test_save_load_round_trip(tmp_path):
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression()).fit(X, y)

    with raises(TypeError):
        scorecard.save(1)

    with raises(TypeError):
        Scorecard.load(tmp_path / "scorecard.pkl")

    path = str(tmp_path / "scorecard.pkl")
    scorecard.save(path)
    loaded = Scorecard.load(path)

    assert isinstance(loaded, Scorecard)
    assert loaded.score(X.head()) == approx(scorecard.score(X.head()))
    assert loaded.table().Points.values == approx(
        scorecard.table().Points.values)


# ------------------------------------------------------------- fit guards

def test_fit_X_must_be_dataframe():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression())

    with raises(TypeError):
        scorecard.fit(X.values, y)


def test_fit_sample_weight_continuous_target():
    X, y = _data()
    y_continuous = y.astype(float) + np.linspace(0, 1, len(y))

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LinearRegression())

    with raises(ValueError):
        scorecard.fit(X, y_continuous, sample_weight=np.ones(len(y)))


def test_fit_sample_weight_binary_target():
    X, y = _data()

    unweighted = Scorecard(binning_process=_binning_process(),
                           estimator=LogisticRegression()).fit(X, y).table()

    unit = Scorecard(binning_process=_binning_process(),
                     estimator=LogisticRegression()).fit(
                         X, y, sample_weight=np.ones(len(y))).table()

    weighted = Scorecard(binning_process=_binning_process(),
                         estimator=LogisticRegression()).fit(
                             X, y, sample_weight=np.where(y == 1, 3.0, 1.0)
                             ).table()

    # Unit weights change nothing; non-uniform weights do.
    assert unit.Points.values == approx(unweighted.Points.values)
    assert not np.allclose(weighted.Points.values, unweighted.Points.values)


def test_fit_degenerate_targets_and_frames():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression())

    # Multiclass target.
    with raises(ValueError):
        scorecard.fit(X, np.arange(len(y)) % 3)

    # Mismatched lengths.
    with raises(ValueError):
        scorecard.fit(X, y[:100])

    # Single class.
    with raises(ValueError):
        scorecard.fit(X, np.zeros(len(y), dtype=int))

    # Empty frame.
    with raises(ValueError):
        scorecard.fit(X.iloc[:0], y[:0])

    # Single row.
    with raises(ValueError):
        scorecard.fit(X.head(1), y[:1])

    # All-missing variable.
    X_nan = X.copy()
    X_nan["x2"] = np.nan
    with raises(ValueError):
        scorecard.fit(X_nan, y)

    # Infinite values.
    X_inf = X.copy()
    X_inf.loc[0, "x1"] = np.inf
    X_inf.loc[1, "x1"] = -np.inf
    with raises(ValueError):
        scorecard.fit(X_inf, y)


def test_fit_constant_variable_gives_a_single_bin():
    X, y = _data()
    X = X.copy()
    X["x2"] = 5.0

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression()).fit(X, y)

    table = scorecard.table()
    x2_bins = table[table.Variable == "x2"].Bin.tolist()

    assert x2_bins == ["(-inf, inf)", "Special", "Missing"]
    assert table[table.Variable == "x2"].Points.values == approx(0, abs=1e-9)


def test_fit_unusual_category_labels():
    X, y = _data()
    rng = np.random.RandomState(3)
    labels = np.array(["a b", "Ω", "1", "", "x/y"], dtype=object)
    X = X.copy()
    X["x2"] = rng.choice(labels, size=len(y))

    binning_process = _binning_process(categorical_variables=["x2"])
    scorecard = Scorecard(binning_process=binning_process,
                          estimator=LogisticRegression()).fit(X, y)

    table = scorecard.table()
    x2_bins = table[table.Variable == "x2"].Bin.tolist()

    assert x2_bins[-2:] == ["Special", "Missing"]

    # Category bins carry the labels themselves, one array per bin; every
    # label -- the empty string included -- lands in exactly one of them.
    binned_labels = sorted(
        str(label) for b in x2_bins[:-2] for label in list(b))
    assert binned_labels == sorted(str(label) for label in labels)
    assert np.isfinite(table.Points.values).all()


def test_fit_user_splits_fixed():
    X, y = _data()
    binning_process = _binning_process(binning_fit_params={
        "x1": {"user_splits": [-0.5, 0.5],
               "user_splits_fixed": [True, True]}})

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=LogisticRegression()).fit(X, y)

    table = scorecard.table()
    x1_bins = table[table.Variable == "x1"].Bin.tolist()

    assert x1_bins == ["(-inf, -0.50)", "[-0.50, 0.50)", "[0.50, inf)",
                       "Special", "Missing"]


def test_estimator_without_intercept_attribute():
    X, y = _data()

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=_CoefOnlyEstimator(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 100}
                          ).fit(X, y)

    table = scorecard.table()
    sc_min, sc_max = table.groupby("Variable").agg(
        {"Points": ["min", "max"]}).sum()

    # intercept defaults to 0 when the estimator does not expose intercept_.
    assert sc_min == approx(0, abs=1e-9)
    assert sc_max == approx(100, rel=1e-6)


# ------------------------------------------------------- special code paths

def test_special_codes_dict_and_list_points():
    X, y = _data()
    X = X.copy()
    X.loc[:20, "x1"] = -9
    X.loc[21:30, "x1"] = -8

    binning_process = _binning_process(special_codes={"s1": -9, "s2": -8})
    scorecard = Scorecard(binning_process=binning_process,
                          estimator=LogisticRegression()
                          ).fit(X, y, metric_special=0.5)

    table = scorecard.table(style="detailed")
    x1 = table[table.Variable == "x1"]
    coef = x1.Coefficient.values[0]

    assert x1.Bin.tolist()[-3:] == ["s1", "s2", "Missing"]
    # One row per named special bucket, all at metric_special * coefficient.
    assert x1.Points.values[-3:-1] == approx([0.5 * coef] * 2)
    assert x1.Points.values[-1] == approx(0.0)

    # The list form produces a single Special row, same rule.
    X_list = X.copy()
    X_list.loc[21:30, "x1"] = -9
    binning_process = _binning_process(special_codes=[-9])
    scorecard = Scorecard(binning_process=binning_process,
                          estimator=LogisticRegression()
                          ).fit(X_list, y, metric_special=0.5)

    table = scorecard.table(style="detailed")
    x1 = table[table.Variable == "x1"]
    coef = x1.Coefficient.values[0]

    assert x1.Bin.tolist()[-2:] == ["Special", "Missing"]
    assert x1.Points.values[-2] == approx(0.5 * coef)


def test_metric_special_missing_empirical():
    X, y = _data()
    X = X.copy()
    X.loc[:20, "x1"] = -9
    X.loc[21:30, "x2"] = np.nan

    binning_process = _binning_process(special_codes=[-9])
    scorecard = Scorecard(binning_process=binning_process,
                          estimator=LogisticRegression()).fit(
                              X, y, metric_special="empirical",
                              metric_missing="empirical")

    table = scorecard.table()
    special_x1 = table[(table.Variable == "x1") &
                       (table.Bin == "Special")].Points.values[0]
    missing_x2 = table[(table.Variable == "x2") &
                       (table.Bin == "Missing")].Points.values[0]

    # "empirical" leaves the WoE-derived points in place instead of
    # overwriting them with metric * coefficient.
    assert special_x1 != 0
    assert missing_x2 != 0
    assert np.isfinite(table.Points.values).all()


def test_metric_special_invalid_string():
    X, y = _data()
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression())

    with raises(ValueError):
        scorecard.fit(X, y, metric_special="not_a_metric")


# ------------------------------------------------------- scaling / rounding

def test_scaling_method_invalid_type():
    X, y = _data()

    with raises(ValueError):
        Scorecard(binning_process=_binning_process(),
                  estimator=LogisticRegression(), scaling_method=5,
                  scaling_method_params={"min": 0, "max": 1}).fit(X, y)

    with raises(ValueError):
        Scorecard(binning_process=_binning_process(),
                  estimator=LogisticRegression(), scaling_method="min_max",
                  scaling_method_params={"min": 0}).fit(X, y)

    with raises(ValueError):
        Scorecard(binning_process=_binning_process(),
                  estimator=LogisticRegression(), scaling_method="pdo_odds",
                  scaling_method_params={"pdo": 0, "odds": 1,
                                         "scorecard_points": 600}).fit(X, y)


def test_scaling_min_max_min_equals_max():
    X, y = _data()

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 100, "max": 100}
                          ).fit(X, y)

    table = scorecard.table()
    sc_min, sc_max = table.groupby("Variable").agg(
        {"Points": ["min", "max"]}).sum()

    # A degenerate range collapses every score onto the same value.
    assert sc_min == approx(100, rel=1e-6)
    assert sc_max == approx(100, rel=1e-6)
    assert scorecard.score(X) == approx(np.full(len(y), 100.0))


def test_reverse_scorecard_pdo_odds_flips_the_ranking():
    X, y = _data()
    odds = 1 / y.mean()
    params = {"pdo": 20, "odds": odds, "scorecard_points": 600}

    forward = Scorecard(binning_process=_binning_process(),
                        estimator=LogisticRegression(),
                        scaling_method="pdo_odds",
                        scaling_method_params=params).fit(X, y)

    reverse = Scorecard(binning_process=_binning_process(),
                        estimator=LogisticRegression(),
                        scaling_method="pdo_odds",
                        scaling_method_params=params,
                        reverse_scorecard=True).fit(X, y)

    score_f = forward.score(X)
    score_r = reverse.score(X)

    assert np.corrcoef(score_f, score_r)[0, 1] == approx(-1.0, rel=1e-9)


def test_intercept_based_rounding_pdo_odds():
    X, y = _data()
    odds = 1 / y.mean()

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="pdo_odds",
                          scaling_method_params={"pdo": 20, "odds": odds,
                                                 "scorecard_points": 600},
                          intercept_based=True, rounding=True).fit(X, y)

    points = scorecard.table().Points.values

    assert points == approx(np.rint(points))
    assert scorecard.intercept_ == approx(np.rint(scorecard.intercept_))
    # intercept-based: the smallest point of every variable is zero.
    assert scorecard.table().groupby("Variable").Points.min().values == (
        approx(0))
    assert scorecard.score(X) == approx(np.rint(scorecard.score(X)))


def test_intercept_based_rounding_min_max():
    X, y = _data()

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 100},
                          intercept_based=True, rounding=True).fit(X, y)

    table = scorecard.table()
    sc_min, sc_max = table.groupby("Variable").agg(
        {"Points": ["min", "max"]}).sum()

    assert table.Points.values == approx(np.rint(table.Points.values))
    assert scorecard.intercept_ == approx(np.rint(scorecard.intercept_))
    assert table.groupby("Variable").Points.min().values == approx(0)
    assert sc_min + scorecard.intercept_ == approx(0, abs=1e-9)
    assert sc_max + scorecard.intercept_ == approx(100, rel=1e-6)


def test_rounding_min_max_narrow_range():
    X, y = _data()

    # A range this narrow leaves the MIP no room: whichever branch is taken,
    # the published points must be integers.
    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 1, "max": 3},
                          rounding=True).fit(X, y)

    points = scorecard.table().Points.values

    assert points == approx(np.rint(points))
    assert np.isfinite(scorecard.score(X)).all()


def test_rounding_min_max_mip_failure_falls_back(monkeypatch):
    X, y = _data()

    unrounded = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 100}
                          ).fit(X, y).table()

    monkeypatch.setattr(scorecard_module, "RoundingMIP",
                        _InfeasibleRoundingMIP)

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 100},
                          rounding=True, verbose=True)

    with redirect_stdout(io.StringIO()):
        scorecard.fit(X, y)

    # Back-up method: nearest integer of the unrounded points.
    assert scorecard.table().Points.values == approx(
        np.rint(unrounded.Points.values))


def test_rounding_mip_failure_with_intercept_based(monkeypatch):
    X, y = _data()

    monkeypatch.setattr(scorecard_module, "RoundingMIP",
                        _InfeasibleRoundingMIP)

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 100},
                          intercept_based=True, rounding=True).fit(X, y)

    points = scorecard.table().Points.values

    assert points == approx(np.rint(points))
    assert scorecard.intercept_ == approx(np.rint(scorecard.intercept_))


# --------------------------------------------------------------- RoundingMIP

def test_rounding_mip_infeasible():
    # No integer lies inside [0.4, 0.6], so the single-variable model cannot
    # place its points at all.
    df_scorecard = pd.DataFrame({"Variable": ["v", "v"],
                                 "Points": [0.4, 0.6]})

    round_mip = RoundingMIP()
    round_mip.build_model(df_scorecard)
    status, solution = round_mip.solve()

    assert status == "INFEASIBLE"
    assert solution is None


def test_rounding_mip_optimal_solution_shape():
    df_scorecard = pd.DataFrame(
        {"Variable": ["a", "a", "b", "b"],
         "Points": [0.0, 5.7, 0.3, 4.0]})

    round_mip = RoundingMIP()
    round_mip.build_model(df_scorecard)
    status, solution = round_mip.solve()

    assert status == "OPTIMAL"
    assert len(solution) == len(df_scorecard)
    assert np.asarray(solution) == approx(np.rint(solution))


def test_rounding_mip_status_contract_under_time_limit():
    # A time limit is the only way the solver stops without an answer; the
    # contract is that a solution is returned exactly for OPTIMAL/FEASIBLE.
    rng = np.random.RandomState(0)
    rows = [("v{}".format(i), rng.uniform(-50, 50))
            for i in range(60) for _ in range(10)]
    df_scorecard = pd.DataFrame(rows, columns=["Variable", "Points"])

    round_mip = RoundingMIP()
    round_mip.build_model(df_scorecard)
    round_mip.solver_.SetTimeLimit(1)
    status, solution = round_mip.solve()

    assert status in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED",
                      "ABNORMAL", "UNKNOWN")
    assert (solution is not None) == (status in ("OPTIMAL", "FEASIBLE"))


def test_defect_rounding_mip_min_score_not_guaranteed():
    # Variable "a" reaches far below the others; unless every variable's
    # minimum is attained after rounding, the min_b of "b" and "c" can sit
    # below their own points and the equality constraint is satisfied without
    # the rounded points honouring the minimum score.
    df_scorecard = pd.DataFrame(
        {"Variable": ["a", "a", "b", "b", "c", "c"],
         "Points": [-10.0, 0.0, 1.6, 5.0, 1.6, 5.0]})

    round_mip = RoundingMIP()
    round_mip.build_model(df_scorecard)
    status, solution = round_mip.solve()

    assert status == "OPTIMAL"

    rounded = df_scorecard.assign(Points=solution)
    min_point = np.rint(df_scorecard.groupby("Variable").Points.min().sum())
    max_point = np.rint(df_scorecard.groupby("Variable").Points.max().sum())

    assert rounded.groupby("Variable").Points.max().sum() == approx(max_point)
    assert rounded.groupby("Variable").Points.min().sum() == approx(min_point)


def test_rounding_mip_tied_points_reach_the_endpoints():
    # Variable "a" has two equal points, and the endpoints are only reachable
    # if its rounded minimum and maximum are free to differ: breaking the tie
    # towards a single bin makes the model infeasible instead.
    df_scorecard = pd.DataFrame({"Variable": ["a", "a", "b", "b"],
                                 "Points": [1.5, 1.5, 0.2, 3.7]})

    round_mip = RoundingMIP()
    round_mip.build_model(df_scorecard)
    status, solution = round_mip.solve()

    assert status == "OPTIMAL"

    rounded = df_scorecard.assign(Points=solution)
    min_point = np.rint(df_scorecard.groupby("Variable").Points.min().sum())
    max_point = np.rint(df_scorecard.groupby("Variable").Points.max().sum())

    assert rounded.groupby("Variable").Points.max().sum() == approx(max_point)
    assert rounded.groupby("Variable").Points.min().sum() == approx(min_point)


def test_rounding_min_max_endpoints_are_attained():
    # End-to-end: the MIP rounding must land the scorecard on the requested
    # [min, max] range, not merely inside it.
    X, y = _data()

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression(),
                          scaling_method="min_max",
                          scaling_method_params={"min": 0, "max": 300},
                          rounding=True).fit(X, y)

    table = scorecard.table()
    sc_min, sc_max = table.groupby("Variable").agg(
        {"Points": ["min", "max"]}).sum()

    assert table.Points.values == approx(np.rint(table.Points.values))
    assert sc_min == approx(0, abs=1e-9)
    assert sc_max == approx(300, rel=1e-6)


def test_defect_min_max_scaling_of_a_degenerate_scorecard():
    # Every variable in a single bin => smax == smin. The scaling has no
    # score range to map onto [min, max], and used to divide by zero and
    # publish an all-nan scorecard from a successful fit.
    _, y = _data()
    X = pd.DataFrame({"x1": np.full(len(y), 3.0)})

    scorecard = Scorecard(
        binning_process=BinningProcess(["x1"], max_n_prebins=5, max_n_bins=4),
        estimator=LogisticRegression(), scaling_method="min_max",
        scaling_method_params={"min": 0, "max": 100})

    with raises(ValueError):
        scorecard.fit(X, y)


def test_pdo_odds_scaling_of_a_degenerate_scorecard():
    # The sibling scaling method has no data-dependent denominator, so the
    # same degenerate scorecard is scaled to constant, finite points.
    _, y = _data()
    X = pd.DataFrame({"x1": np.full(len(y), 3.0)})
    odds = 1 / y.mean()

    scorecard = Scorecard(
        binning_process=BinningProcess(["x1"], max_n_prebins=5, max_n_bins=4),
        estimator=LogisticRegression(), scaling_method="pdo_odds",
        scaling_method_params={"pdo": 20, "odds": odds,
                               "scorecard_points": 600}).fit(X, y)

    points = scorecard.table().Points.values

    assert np.isfinite(points).all()


# --------------------------------------------------------------- monitoring

def test_monitoring_psi_plot_show(monkeypatch):
    X, y = _data(n=400)
    X_actual, y_actual = _data(n=300, seed=7)

    scorecard = Scorecard(binning_process=_binning_process(),
                          estimator=LogisticRegression()).fit(X, y)

    monitoring = ScorecardMonitoring(scorecard=scorecard, psi_method="cart",
                                     psi_n_bins=4).fit(X_actual, y_actual,
                                                       X, y)

    # savefig=None shows the figure, which blocks on an interactive backend
    # until the window is closed, so the display call is stubbed out here.
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(1))

    monitoring.psi_plot()
    plt.close("all")

    assert shown == [1]
