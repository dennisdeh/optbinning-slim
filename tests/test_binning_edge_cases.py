"""
OptimalBinning edge-case and chaos testing.

Exercises the guard clauses, degenerate inputs and rarely-taken branches of
``optbinning/binning/binning.py``, ``optbinning/binning/transformations.py``
and ``optbinning/binning/preprocessing.py``. Everything here fits small
synthetic arrays so the whole module stays fast; expectations are pinned on
invariants (status, ordering, dtype, exception type and message) rather than
on solver artifacts.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import warnings

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import ContinuousOptimalBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning.binning.preprocessing import categorical_cutoff
from optbinning.binning.preprocessing import categorical_transform
from optbinning.binning.preprocessing import (
    preprocessing_user_splits_categorical)
from optbinning.binning.preprocessing import split_data
from optbinning.binning.preprocessing import split_data_scenarios
from optbinning.binning.transformations import transform_event_rate_to_woe
from optbinning.binning.transformations import transform_woe_to_event_rate
from sklearn.exceptions import NotFittedError


def _logistic_data(n=300, seed=42):
    """Noisy separable-ish binary problem: small, fast, and not degenerate."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    p = 1. / (1. + np.exp(-(x - 5.)))
    y = (rng.uniform(size=n) < p).astype(int)
    return x, y


def _peak_data(n=400, seed=7):
    """Event rate peaks in the middle -- non-monotonic on purpose."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    p = np.exp(-((x - 5.) ** 2) / 4.) * 0.85 + 0.05
    y = (rng.uniform(size=n) < p).astype(int)
    return x, y


def _pure_tail_data():
    """Logistic data whose two tails are pure, so prebins get removed."""
    x, y = _logistic_data()
    y = y.copy()
    y[x > 8.5] = 1
    y[x < 1.5] = 0
    return x, y


def _categorical_data(seed=3):
    rng = np.random.RandomState(seed)
    cats = np.array(['A'] * 120 + ['B'] * 100 + ['C'] * 60 + ['D'] * 15 +
                    ['E'] * 5)
    rng.shuffle(cats)
    rates = {'A': 0.2, 'B': 0.5, 'C': 0.8, 'D': 0.3, 'E': 0.9}
    y = np.array([1 if rng.uniform() < rates[c] else 0 for c in cats])
    return cats, y


x, y = _logistic_data()


# ---------------------------------------------------------------------------
# transformations.py -- the WoE / event-rate conversion pair
# ---------------------------------------------------------------------------

def test_transform_woe_event_rate_round_trip():
    n_nonevent, n_event = 151, 149
    event_rate = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    woe = transform_event_rate_to_woe(event_rate, n_nonevent, n_event)
    back = transform_woe_to_event_rate(woe, n_nonevent, n_event)

    assert back == approx(event_rate, rel=1e-12)

    # WoE is decreasing in the event rate.
    assert np.all(np.diff(woe) < 0)

    # A scalar in gives a scalar out.
    scalar = transform_woe_to_event_rate(0.0, n_nonevent, n_event)
    assert scalar == approx(n_event / (n_event + n_nonevent), rel=1e-12)


# ---------------------------------------------------------------------------
# transformations.py -- guard clauses
# ---------------------------------------------------------------------------

def test_transform_invalid_metric():
    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    with raises(ValueError, match="Invalid value for metric"):
        optb.transform(x, metric="mean")

    with raises(ValueError, match="Invalid value for metric"):
        optb.transform(x, metric=None)


def test_transform_invalid_metric_special():
    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    with raises(ValueError, match="Invalid value for metric_special"):
        optb.transform(x, metric_special="mean")

    # A dict is accepted, but only with numeric values.
    with raises(ValueError, match="Invalid value for metric_special key"):
        optb.transform(x, metric_special={"a": "not-a-number"})

    with raises(ValueError, match="Invalid value for metric_special"):
        optb.transform(x, metric_special=[0, 1])


def test_transform_invalid_metric_missing():
    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    with raises(ValueError, match="Invalid value for metric_missing"):
        optb.transform(x, metric_missing="mean")

    # Unlike metric_special, a dict is not an allowed metric_missing.
    with raises(ValueError, match="Invalid value for metric_missing"):
        optb.transform(x, metric_missing={"a": 0})


def test_transform_invalid_show_digits():
    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    for bad in (-1, 9, 2.5, "2"):
        with raises(ValueError, match="show_digits must be an integer"):
            optb.transform(x, metric="bins", show_digits=bad)

    # The boundaries are allowed.
    for good in (0, 8):
        optb.transform(x[:10], metric="bins", show_digits=good)


def test_transform_invalid_cat_unknown():
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical", cat_unknown=1.5,
                          max_n_prebins=6)
    optb.fit(cats, yc)

    with raises(ValueError, match="cat_unknown must be string"):
        optb.transform(cats[:5], metric="bins")

    with raises(ValueError, match="cat_unknown must be an integer"):
        optb.transform(cats[:5], metric="indices")

    optb = OptimalBinning(dtype="categorical", cat_unknown="?",
                          max_n_prebins=6)
    optb.fit(cats, yc)

    with raises(ValueError, match="cat_unknown must be numeric"):
        optb.transform(cats[:5], metric="woe")


def test_transform_check_input():
    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    # check_input=True routes through sklearn's check_array, which allows NaN
    # but rejects infinities.
    xt = x.copy()
    xt[0] = np.nan
    assert optb.transform(xt, check_input=True).shape == xt.shape

    xt[1] = np.inf
    with raises(ValueError, match="infinity"):
        optb.transform(xt, check_input=True)


# ---------------------------------------------------------------------------
# transformations.py -- special_codes as a list and as a dict
# ---------------------------------------------------------------------------

def test_transform_special_codes_dict_every_metric():
    xs = x.copy()
    xs[:10] = -1.
    xs[10:20] = -2.
    xs[20:30] = np.nan

    special_codes = {"neg_one": -1., "neg_two_three": [-2., -3.]}
    optb = OptimalBinning(special_codes=special_codes, max_n_prebins=8,
                          max_n_bins=4)
    optb.fit(xs, y)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    # One row per bin, then one row per named special, then Missing, then
    # Totals.
    assert list(table["Bin"])[-4:-1] == ["neg_one", "neg_two_three",
                                         "Missing"]

    probe = np.array([-1., -2., np.nan, xs[100]])

    # "empirical" sends each named special group to its own statistic.
    woe = optb.transform(probe, metric="woe", metric_special="empirical",
                         metric_missing="empirical")
    assert woe[0] != woe[1]

    er = optb.transform(probe, metric="event_rate",
                        metric_special="empirical",
                        metric_missing="empirical")
    assert np.all((0. <= er) & (er <= 1.))

    idx = optb.transform(probe, metric="indices",
                         metric_special="empirical",
                         metric_missing="empirical")
    n_bins = len(optb.splits) + 1
    assert idx.dtype == np.dtype(int)
    assert list(idx[:3]) == [n_bins, n_bins + 1, n_bins + 2]

    bins = optb.transform(probe, metric="bins", metric_special="empirical",
                          metric_missing="empirical")
    assert list(bins[:3]) == ["neg_one", "neg_two_three", "Missing"]

    # A numeric metric_special/metric_missing overrides every group.
    for metric in ("woe", "event_rate"):
        out = optb.transform(probe, metric=metric, metric_special=-99.,
                             metric_missing=-88.)
        assert out[0] == -99. and out[1] == -99. and out[2] == -88.

    out = optb.transform(probe, metric="indices", metric_special=-99,
                         metric_missing=-88)
    assert list(out[:3]) == [-99, -99, -88]

    # metric="bins" ignores numeric metric_special/metric_missing.
    out = optb.transform(probe, metric="bins", metric_special=-99,
                         metric_missing=-88)
    assert list(out[:3]) == ["neg_one", "neg_two_three", "Missing"]


def test_transform_special_codes_list_every_metric():
    xs = x.copy()
    xs[:10] = -1.
    xs[10:20] = -2.
    xs[20:30] = np.nan

    optb = OptimalBinning(special_codes=[-1., -2.], max_n_prebins=8,
                          max_n_bins=4)
    optb.fit(xs, y)

    probe = np.array([-1., -2., np.nan, xs[100]])
    n_bins = len(optb.splits) + 1

    # The list form collapses every special into a single "Special" bucket.
    bins = optb.transform(probe, metric="bins", metric_special="empirical",
                          metric_missing="empirical")
    assert list(bins[:3]) == ["Special", "Special", "Missing"]

    idx = optb.transform(probe, metric="indices",
                         metric_special="empirical",
                         metric_missing="empirical")
    assert list(idx[:3]) == [n_bins, n_bins, n_bins + 1]

    woe = optb.transform(probe, metric="woe", metric_special="empirical",
                         metric_missing="empirical")
    assert woe[0] == woe[1]

    out = optb.transform(probe, metric="event_rate", metric_special=0.5,
                         metric_missing=0.25)
    assert list(out[:3]) == [0.5, 0.5, 0.25]


def test_transform_special_codes_ndarray():
    xs = x.copy()
    xs[:10] = -1.

    optb = OptimalBinning(special_codes=np.array([-1.]), max_n_prebins=6,
                          max_n_bins=4)
    optb.fit(xs, y)

    out = optb.transform(np.array([-1.]), metric="event_rate",
                         metric_special="empirical")
    assert 0. <= out[0] <= 1.


def test_transform_special_codes_dict_of_ndarrays():
    xs = x.copy()
    xs[:10] = -1.
    xs[10:20] = -2.

    optb = OptimalBinning(special_codes={"neg": np.array([-1., -2.])},
                          max_n_prebins=6, max_n_bins=4)
    optb.fit(xs, y)

    out = optb.transform(np.array([-1., -2.]), metric="bins")
    assert list(out) == ["neg", "neg"]


# ---------------------------------------------------------------------------
# transformations.py -- multiclass and continuous entry points
# ---------------------------------------------------------------------------

def _multiclass_data(n=400, seed=13):
    """Three classes whose proportions shift with x but never vanish."""
    rng = np.random.RandomState(seed)
    xm = rng.uniform(0, 10, n)
    w = np.stack([10. - xm, 5. - np.abs(xm - 5.) * 0.8, xm], axis=1) + 1.
    w /= w.sum(axis=1, keepdims=True)
    ym = (rng.uniform(size=n)[:, None] > np.cumsum(w, axis=1)).sum(axis=1)
    return xm, ym


def test_multiclass_transform_metrics():
    xm, ym = _multiclass_data()

    optb = MulticlassOptimalBinning(max_n_prebins=8, max_n_bins=4)
    optb.fit(xm, ym)
    assert optb.status == "OPTIMAL"
    assert len(optb.splits) >= 1

    with raises(ValueError, match="Invalid value for metric"):
        optb.transform(xm, metric="woe")

    mean_woe = optb.transform(xm, metric="mean_woe")
    weighted = optb.transform(xm, metric="weighted_mean_woe")
    assert mean_woe.shape == xm.shape
    assert weighted.shape == xm.shape
    # Both take one value per bin, but they are different averages of the
    # same one-vs-all WoEs, so they must not coincide.
    assert len(np.unique(mean_woe)) == len(optb.splits) + 1
    assert len(np.unique(weighted)) == len(optb.splits) + 1
    assert not np.allclose(mean_woe, weighted)

    idx = optb.transform(xm, metric="indices")
    assert idx.dtype == np.dtype(int)
    assert set(np.unique(idx)) == set(range(len(optb.splits) + 1))
    # Indices increase with x.
    order = np.argsort(xm)
    assert np.all(np.diff(idx[order]) >= 0)

    bins = optb.transform(xm, metric="bins")
    assert bins.dtype == np.dtype(object)
    assert all(b.startswith(("(", "[")) for b in bins)

    # check_input=True still accepts the same array.
    assert optb.transform(xm, metric="mean_woe",
                          check_input=True).shape == xm.shape


def test_multiclass_transform_guards():
    xm, ym = _multiclass_data()

    optb = MulticlassOptimalBinning(max_n_prebins=8, max_n_bins=4)
    optb.fit(xm, ym)

    with raises(ValueError, match="Invalid value for metric_special"):
        optb.transform(xm, metric_special="mean")

    with raises(ValueError, match="Invalid value for metric_missing"):
        optb.transform(xm, metric_missing="mean")

    with raises(ValueError, match="show_digits must be an integer"):
        optb.transform(xm, metric="bins", show_digits=-1)


def test_continuous_transform_metrics():
    rng = np.random.RandomState(5)
    xc = rng.uniform(0, 10, 300)
    yc = 2. * xc + rng.normal(size=300)

    optb = ContinuousOptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(xc, yc)
    assert optb.status == "OPTIMAL"

    with raises(ValueError, match="Invalid value for metric"):
        optb.transform(xc, metric="woe")

    means = optb.transform(xc, metric="mean")
    assert means.shape == xc.shape
    # The target is increasing in x, so bin means come out ordered.
    order = np.argsort(xc)
    assert np.all(np.diff(means[order]) >= 0)

    assert optb.transform(xc, metric="mean",
                          check_input=True).shape == xc.shape

    idx = optb.transform(xc, metric="indices")
    assert idx.dtype == np.dtype(int)

    bins = optb.transform(xc, metric="bins", show_digits=4)
    assert bins.dtype == np.dtype(object)


def test_continuous_transform_constant_target():
    xc = np.linspace(0, 10, 200)
    yc = np.zeros(200)

    optb = ContinuousOptimalBinning(max_n_prebins=6)
    optb.fit(xc, yc)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    # No split: every clean record lands in bin 0 with the constant mean.
    assert np.all(optb.transform(xc, metric="indices") == 0)
    assert np.all(optb.transform(xc, metric="mean") == 0.)
    assert set(optb.transform(xc, metric="bins")) == {"(-inf, inf)"}


def test_continuous_categorical_transform():
    rng = np.random.RandomState(17)
    cats = np.array(['A'] * 100 + ['B'] * 100 + ['C'] * 100)
    yc = np.repeat([1., 5., 9.], 100) + rng.normal(size=300)

    optb = ContinuousOptimalBinning(dtype="categorical", max_n_prebins=6)
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"

    means = optb.transform(np.array(['A', 'B', 'C']), metric="mean")
    assert np.all(np.diff(means) > 0)

    # An unseen category falls back to the overall mean.
    unknown = optb.transform(np.array(['Z']), metric="mean")[0]
    assert unknown == approx(yc.mean(), rel=1e-9)

    idx = optb.transform(np.array(['A', 'B', 'C']), metric="indices")
    assert sorted(idx) == [0, 1, 2]

    bins = optb.transform(np.array(['A', 'Z']), metric="bins")
    assert bins.dtype == np.dtype(object)
    assert bins[1] == "unknown"


# ---------------------------------------------------------------------------
# Degenerate fits: no split survives pre-binning
# ---------------------------------------------------------------------------

def test_single_class_target_yields_no_splits():
    xd = np.linspace(0, 10, 200)
    yd = np.zeros(200, dtype=int)

    optb = OptimalBinning(max_n_prebins=6)
    optb.fit(xd, yd)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb._optimizer is None

    # transform with no splits puts every clean record in bin 0.
    assert np.all(optb.transform(xd, metric="indices") == 0)
    assert np.all(optb.transform(xd, metric="woe") == 0.)
    assert set(optb.transform(xd, metric="bins")) == {"(-inf, inf)"}


def test_perfectly_separable_target_yields_no_splits():
    xd = np.linspace(0, 10, 200)
    yd = (xd > 5.).astype(int)

    optb = OptimalBinning(max_n_prebins=6)
    optb.fit(xd, yd)

    # Every prebin is pure, so refinement removes all of them.
    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb._n_prebins == 0


def test_constant_x_yields_no_splits():
    xd = np.full(100, 3.)
    yd = np.array([0, 1] * 50)

    optb = OptimalBinning(max_n_prebins=6)
    optb.fit(xd, yd)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0


def test_single_row():
    optb = OptimalBinning(max_n_prebins=4)
    optb.fit(np.array([1.]), np.array([1]))

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb.transform(np.array([1.]), metric="indices")[0] == 0


def test_information_on_degenerate_fit(capsys):
    xd = np.linspace(0, 10, 200)
    yd = np.zeros(200, dtype=int)

    optb = OptimalBinning(max_n_prebins=6)
    optb.fit(xd, yd)

    # The solver never ran, so information() must fall back to no optimizer.
    assert optb._optimizer is None
    optb.information(print_level=0)
    optb.information(print_level=1)
    optb.information(print_level=2)

    out = capsys.readouterr().out
    assert "optbinning" in out

    with raises(ValueError, match="print_level must be an integer"):
        optb.information(print_level=-1)


def test_verbose_degenerate_fit(caplog):
    xd = np.linspace(0, 10, 200)
    yd = np.zeros(200, dtype=int)

    optb = OptimalBinning(max_n_prebins=6, verbose=True)
    optb.fit(xd, yd)

    assert optb.status == "OPTIMAL"
    assert "solver not run" in caplog.text


# ---------------------------------------------------------------------------
# Chaos: bad shapes, dtypes and unfitted access
# ---------------------------------------------------------------------------

def test_unfitted_access():
    optb = OptimalBinning()

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
    with raises(NotFittedError):
        optb.to_dict()


def test_mismatched_lengths():
    optb = OptimalBinning()

    with raises(ValueError, match="inconsistent numbers of samples"):
        optb.fit(np.arange(10.), np.array([0, 1] * 4), check_input=True)

    # Without check_input the mismatch still fails, just later and louder.
    with raises(ValueError):
        optb.fit(np.arange(10.), np.array([0, 1] * 4))


def test_infinite_values_rejected():
    xi = np.linspace(0, 10, 100)
    yi = (xi > 5).astype(int)
    yi[3] = 1
    yi[80] = 0
    xi = xi.copy()
    xi[0] = -np.inf
    xi[-1] = np.inf

    optb = OptimalBinning(max_n_prebins=5)
    with raises(ValueError, match="infinity"):
        optb.fit(xi, yi, check_input=True)


def test_all_missing_x():
    optb = OptimalBinning()
    with raises(ValueError, match="0 sample"):
        optb.fit(np.full(20, np.nan), np.array([0, 1] * 10))


def test_extreme_magnitudes():
    rng = np.random.RandomState(21)
    xd = rng.uniform(0, 10, 200) * 1e12
    p = 1. / (1. + np.exp(-(xd / 1e12 - 5.)))
    yd = (rng.uniform(size=200) < p).astype(int)

    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(xd, yd)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)
    optb.binning_table.build()
    assert np.isfinite(optb.binning_table.iv)


def test_heavily_duplicated_values():
    xd = np.repeat(np.array([1., 2., 3.]), 100)
    rng = np.random.RandomState(31)
    rates = {1.: 0.1, 2.: 0.5, 3.: 0.9}
    yd = np.array([1 if rng.uniform() < rates[v] else 0 for v in xd])

    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(xd, yd)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)
    # Splits fall strictly inside the observed range.
    assert np.all((optb.splits > xd.min()) & (optb.splits <= xd.max()))


# ---------------------------------------------------------------------------
# binning.py -- parameter guards not covered elsewhere
# ---------------------------------------------------------------------------

def test_special_codes_empty_dict_rejected():
    optb = OptimalBinning(special_codes={})
    with raises(ValueError, match="special_codes empty"):
        optb.fit(x, y)

    optb = OptimalBinning(special_codes="not-a-container")
    with raises(TypeError, match="special_codes must be a dict, list"):
        optb.fit(x, y)

    # An empty list, unlike an empty dict, is accepted and means "no special".
    optb = OptimalBinning(special_codes=[], max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)
    assert optb.status == "OPTIMAL"


def test_cat_unknown_type_guard():
    optb = OptimalBinning(cat_unknown=[1, 2])
    with raises(TypeError, match="cat_unknown must be a number or string"):
        optb.fit(x, y)


def test_user_splits_type_guard():
    optb = OptimalBinning(user_splits=(1, 2, 3))
    with raises(TypeError, match="user_splits must be a list or numpy"):
        optb.fit(x, y)


def test_split_digits():
    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4, split_digits=1)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx(np.round(optb.splits, 1), rel=1e-12)

    optb = OptimalBinning(split_digits=9)
    with raises(ValueError, match="split_digits must be an integer"):
        optb.fit(x, y)


def test_min_max_bin_size_fractions():
    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4, min_bin_size=0.15,
                          max_bin_size=0.6)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    counts = np.array(table["Count"][:len(optb.splits) + 1])
    assert counts.min() >= np.ceil(0.15 * len(x))
    assert counts.max() <= np.ceil(0.6 * len(x))


def test_min_max_bin_n_event_nonevent():
    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4,
                          min_bin_n_event=10, max_bin_n_event=200,
                          min_bin_n_nonevent=10, max_bin_n_nonevent=200)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    n_bins = len(optb.splits) + 1
    assert np.array(table["Event"][:n_bins]).min() >= 10
    assert np.array(table["Non-event"][:n_bins]).min() >= 10


def test_time_limit_and_gamma():
    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4, gamma=0.5,
                          time_limit=30)
    optb.fit(x, y)
    assert optb.status == "OPTIMAL"

    with raises(ValueError, match="time_limit must be a positive value"):
        OptimalBinning(time_limit=-1).fit(x, y)

    with raises(TypeError, match="verbose must be a boolean"):
        OptimalBinning(verbose="yes").fit(x, y)


# ---------------------------------------------------------------------------
# binning.py -- divergence and monotonic-trend branches
# ---------------------------------------------------------------------------

def test_hellinger_triangular_with_pure_prebins():
    xd, yd = _pure_tail_data()

    for divergence in ("hellinger", "triangular"):
        # min_bin_n_* left at None: the flag forces them up to 1.
        optb = OptimalBinning(divergence=divergence, max_n_prebins=8,
                              max_n_bins=4)
        optb.fit(xd, yd)

        assert optb.status == "OPTIMAL"
        # A pure prebin sets the flag instead of being refined away.
        assert optb._flag_min_n_event_nonevent is True
        assert optb.min_bin_n_nonevent is None
        assert optb.min_bin_n_event is None

        table = optb.binning_table.build()
        n_bins = len(optb.splits) + 1
        assert np.array(table["Event"][:n_bins]).min() >= 1
        assert np.array(table["Non-event"][:n_bins]).min() >= 1

        # An explicit min is raised to at least 1, i.e. kept as given here.
        optb2 = OptimalBinning(divergence=divergence, max_n_prebins=8,
                               max_n_bins=4, min_bin_n_nonevent=3,
                               min_bin_n_event=3)
        optb2.fit(xd, yd)

        assert optb2.status == "OPTIMAL"
        assert optb2._flag_min_n_event_nonevent is True

        table2 = optb2.binning_table.build()
        n_bins2 = len(optb2.splits) + 1
        assert np.array(table2["Event"][:n_bins2]).min() >= 3
        assert np.array(table2["Non-event"][:n_bins2]).min() >= 3


def test_peak_and_valley_heuristic(caplog):
    xp, yp = _peak_data()

    optb = OptimalBinning(monotonic_trend="peak_heuristic", max_n_prebins=8,
                          max_n_bins=5, verbose=True)
    optb.fit(xp, yp)

    assert optb.status == "OPTIMAL"
    assert "trend change position" in caplog.text

    table = optb.binning_table.build()
    n_bins = len(optb.splits) + 1
    event_rate = np.array(table["Event rate"][:n_bins])
    peak = int(np.argmax(event_rate))
    assert np.all(np.diff(event_rate[:peak + 1]) >= 0)
    assert np.all(np.diff(event_rate[peak:]) <= 0)

    optb = OptimalBinning(monotonic_trend="valley_heuristic", max_n_prebins=8,
                          max_n_bins=5)
    optb.fit(xp, yp)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    n_bins = len(optb.splits) + 1
    event_rate = np.array(table["Event rate"][:n_bins])
    valley = int(np.argmin(event_rate))
    assert np.all(np.diff(event_rate[:valley + 1]) <= 0)
    assert np.all(np.diff(event_rate[valley:]) >= 0)


def test_monotonic_trend_none_verbose(caplog):
    optb = OptimalBinning(monotonic_trend=None, max_n_prebins=8, max_n_bins=4,
                          verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert "monotonic trend not set" in caplog.text


def test_categorical_monotonic_trend_is_forced_ascending():
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical", monotonic_trend="descending",
                          max_n_prebins=6, max_n_bins=4)
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"

    # For a categorical variable any non-None trend becomes "ascending",
    # because the categories are ordered by event rate first.
    table = optb.binning_table.build()
    n_bins = len(optb.splits)
    event_rate = np.array(table["Event rate"][:n_bins])
    assert np.all(np.diff(event_rate) >= 0)


def test_verbose_with_sample_weight(caplog):
    sample_weight = np.full(len(x), 2.)

    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4, verbose=True)
    optb.fit(x, y, sample_weight=sample_weight)

    assert optb.status == "OPTIMAL"
    assert "Weighted samples: 600" in caplog.text
    assert optb._n_samples == len(x)
    assert optb._n_samples_weighted == 2 * len(x)


# ---------------------------------------------------------------------------
# binning.py -- categorical "others" bin
# ---------------------------------------------------------------------------

def test_cat_cutoff_builds_others_bin():
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical", cat_cutoff=0.1,
                          max_n_prebins=6)
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"
    # 'D' (15) and 'E' (5) are below 10% of 300 and move to the others bin.
    assert set(optb._cat_others) == {"D", "E"}
    assert optb._n_nonevent_cat_others + optb._n_event_cat_others == 20

    table = optb.binning_table.build()
    assert int(table["Count"][len(optb.splits) - 1]) == 20

    # An unseen category maps to the unknown index.
    assert optb.transform(np.array(["Z"]), metric="indices")[0] == -1


def test_cat_cutoff_moving_everything_raises():
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical", cat_cutoff=0.99)
    with raises(ValueError, match="All categories moved to others"):
        optb.fit(cats, yc)

    # Same message straight from the helper.
    with raises(ValueError, match="All categories moved to others"):
        categorical_cutoff(cats, yc, cutoff=0.99)


def test_categorical_transform_order_matches_event_rate():
    cats, yc = _categorical_data()

    sorted_categories, nominal = categorical_transform(cats, yc)

    rates = pd.Series(yc).groupby(cats).mean()
    assert list(sorted_categories) == list(rates.sort_values().index)
    assert nominal.min() == 0
    assert nominal.max() == len(sorted_categories) - 1


def test_user_splits_categorical_repeated_category():
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical",
                          user_splits=[["A", "B"], ["B", "C"]])
    with raises(ValueError, match="Category B is repeated"):
        optb.fit(cats, yc)

    with raises(ValueError, match="Category B is repeated"):
        preprocessing_user_splits_categorical(
            [["A", "B"], ["B", "C"]], cats, yc)


def test_user_splits_categorical_leftovers_go_to_others():
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical",
                          user_splits=[["A"], ["B"], ["C"]], max_n_bins=3)
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"
    # 'D' and 'E' were not named in user_splits, so they become others.
    assert set(optb._cat_others) == {"D", "E"}


# ---------------------------------------------------------------------------
# binning.py -- user_splits / user_splits_fixed interaction with refinement
# ---------------------------------------------------------------------------

def test_user_splits_fixed_survives_prebin_removal():
    xd, yd = _pure_tail_data()
    user_splits = [1., 3., 5., 7., 9.]

    optb = OptimalBinning(user_splits=user_splits,
                          user_splits_fixed=[False, True, False, False, False],
                          max_n_bins=4)
    optb.fit(xd, yd)

    assert optb.status == "OPTIMAL"
    # The pure tails removed 1.0 and 9.0; the fixed split 3.0 stays, and the
    # boolean mask was shrunk in step with the split list. The shrinking
    # happens on the private working copies -- the constructor parameters
    # have to survive the fit so the estimator can be refitted.
    assert 3. in optb.splits
    assert len(optb._user_splits) == len(optb._user_splits_fixed)
    assert list(optb._user_splits) == [3., 5., 7.]
    assert list(optb._user_splits_fixed) == [True, False, False]

    assert optb.user_splits is user_splits
    assert user_splits == [1., 3., 5., 7., 9.]


def test_user_splits_fixed_removed_by_refinement_raises():
    xd, yd = _pure_tail_data()
    user_splits = [1., 3., 5., 7., 9.]

    optb = OptimalBinning(user_splits=user_splits,
                          user_splits_fixed=[True, False, False, False, False],
                          max_n_bins=4)

    with raises(ValueError, match="are removed because produce pure prebins"):
        optb.fit(xd, yd)


def test_user_splits_unsorted_is_sorted():
    optb = OptimalBinning(user_splits=[7., 3., 5.], max_n_bins=4)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)


# ---------------------------------------------------------------------------
# binning.py -- json helpers
# ---------------------------------------------------------------------------

def test_to_json_without_path():
    optb = OptimalBinning(max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    with raises(ValueError, match="Specify the path for the json file"):
        optb.to_json(None)


# ---------------------------------------------------------------------------
# preprocessing.py -- split_data guards, reached by calling it directly
# ---------------------------------------------------------------------------

_XS = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
_YS = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])


def test_split_data_outlier_guards():
    with raises(ValueError, match="Invalid value for outlier_detector"):
        split_data("numerical", _XS, _YS, outlier_detector="iqr")

    with raises(TypeError, match="outlier_params must be a dict or None"):
        split_data("numerical", _XS, _YS, outlier_detector="range",
                   outlier_params=[1, 2])


def test_split_data_fix_bounds_guards():
    with raises(ValueError, match="fix_lb must be a number"):
        split_data("numerical", _XS, _YS, fix_lb="3")

    with raises(ValueError, match="fix_ub must be a number"):
        split_data("numerical", _XS, _YS, fix_ub="7")

    with raises(ValueError, match=r"fix_lb must be <= fix_ub"):
        split_data("numerical", _XS, _YS, fix_lb=7., fix_ub=3.)


def test_split_data_fix_lb_only():
    x_clean = split_data("numerical", _XS, _YS, fix_lb=3.)[0]
    assert list(x_clean) == [3., 4., 5., 6., 7., 8., 9., 10.]


def test_split_data_fix_ub_only():
    x_clean = split_data("numerical", _XS, _YS, fix_ub=7.)[0]
    assert list(x_clean) == [1., 2., 3., 4., 5., 6., 7.]


def test_split_data_check_input():
    with raises(ValueError, match="inconsistent numbers of samples"):
        split_data("numerical", _XS, _YS[:5], check_input=True)

    # check_input allows NaN in x but not in y.
    xn = _XS.copy()
    xn[0] = np.nan
    x_clean, y_clean = split_data("numerical", xn, _YS, check_input=True)[:2]
    assert len(x_clean) == 9

    yn = _YS.astype(float)
    yn[0] = np.nan
    with raises(ValueError, match="NaN"):
        split_data("numerical", _XS, yn, check_input=True)


def test_split_data_class_weight():
    balanced = split_data("numerical", _XS, _YS, class_weight="balanced")
    sw_clean = balanced[9]

    # Six events and four non-events, so the minority class is up-weighted.
    assert sw_clean[_YS == 0].min() > sw_clean[_YS == 1].max()
    assert sw_clean.sum() == approx(len(_XS), rel=1e-12)

    explicit = split_data("numerical", _XS, _YS,
                          class_weight={0: 1., 1: 3.})
    sw_explicit = explicit[9]
    assert list(np.unique(sw_explicit)) == [1., 3.]


def test_split_data_special_codes_dict_scalar_and_list():
    xs = np.concatenate([_XS, [-1., -2., -3.]])
    ys = np.concatenate([_YS, [0, 1, 0]])

    out = split_data("numerical", xs, ys,
                     special_codes={"a": -1., "b": [-2., -3.]})
    x_clean, _, _, _, x_special = out[:5]

    assert set(x_special) == {-1., -2., -3.}
    assert set(x_clean) == set(_XS)

    # The list form gives the same partition.
    out_list = split_data("numerical", xs, ys, special_codes=[-1., -2., -3.])
    assert set(out_list[4]) == {-1., -2., -3.}


def test_split_data_scenarios_weights():
    X = [_XS, _XS[:6]]
    Y = [_YS, _YS[:6]]

    out = split_data_scenarios(X, Y, weights=[0.75, 0.25], special_codes=None,
                               check_input=True)
    w = out[-1]

    assert len(w) == len(_XS) + 6
    assert w[:len(_XS)] == [0.75] * len(_XS)
    assert w[len(_XS):] == [0.25] * 6

    out_none = split_data_scenarios(X, Y, weights=None, special_codes=None,
                                    check_input=True)
    assert out_none[-1] is None


def test_continuous_yquantile_outlier_detector():
    rng = np.random.RandomState(42)
    xc = rng.uniform(0, 10, 300)
    yc = 2. * xc + rng.normal(size=300)
    yc[:5] += 200.

    optb = ContinuousOptimalBinning(outlier_detector="yquantile",
                                    max_n_prebins=8, max_n_bins=4)
    optb.fit(xc, yc)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)

    optb = ContinuousOptimalBinning(outlier_detector="yquantile",
                                    outlier_params={"n_bins": 5},
                                    max_n_prebins=8, max_n_bins=4)
    optb.fit(xc, yc)

    assert optb.status == "OPTIMAL"


# ---------------------------------------------------------------------------
# Defects: kept red on purpose. See the accompanying report.
# ---------------------------------------------------------------------------

def test_defect_user_splits_empty_list():
    """``user_splits=[]`` means "no split points", not an error.

    ``_check_parameters`` accepts an empty list and ``_fit`` has a dedicated
    ``if not n_splits:`` branch for it. That branch used to skip
    ``_prebinning_refinement``, leaving ``_n_nonevent_special`` and friends
    ``None`` so ``bin_info`` raised ``TypeError``; it now delegates to
    ``_prebinning_refinement``, which early-returns the empty split arrays and
    computes the special and missing counts on the way. The result is the same
    single bin ``user_splits=None`` gives on the same data.
    """
    optb_none = OptimalBinning(user_splits=None, max_n_prebins=6)
    optb_none.fit(x, y)
    assert optb_none.status == "OPTIMAL"

    optb = OptimalBinning(user_splits=[], max_n_prebins=6)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    # the special and missing counts are what the empty branch used to leave
    # unset, so pin them: one clean bin plus an empty special and missing row
    xm = x.copy()
    xm[:5] = -1.
    xm[5:8] = np.nan

    optb = OptimalBinning(user_splits=[], special_codes=[-1.],
                          max_n_prebins=6)
    optb.fit(xm, y)

    df = optb.binning_table.build()

    assert optb.status == "OPTIMAL"
    assert list(df["Bin"])[:-1] == ["(-inf, inf)", "Special", "Missing"]
    assert list(df["Count"])[:-1] == [len(x) - 8, 5, 3]


def test_metric_special_covers_every_named_special_group():
    """``metric_special`` is documented as a number or "empirical".

    With dict-form ``special_codes`` each named group is a bin of its own, so
    "empirical" gives each one its own WoE while a single numeric value
    overrides all of them. The per-key dict form is neither documented nor
    honoured by ``_apply_transform``; only its values are validated.
    """
    xs = x.copy()
    xs[:10] = -1.
    xs[10:20] = -2.

    special_codes = {"neg_one": -1., "neg_two": -2.}
    optb = OptimalBinning(special_codes=special_codes, max_n_prebins=8,
                          max_n_bins=4)
    optb.fit(xs, y)

    xt = np.array([-1., -2.])

    empirical = optb.transform(xt, metric="woe", metric_special="empirical")
    assert empirical[0] != empirical[1]

    assert list(optb.transform(xt, metric="woe", metric_special=-9.)) == \
        [-9., -9.]

    # the named groups keep their own bin under "bins" and "indices"
    assert list(optb.transform(xt, metric="bins")) == ["neg_one", "neg_two"]
    assert list(optb.transform(xt, metric="indices",
                               metric_special="empirical")) == [4, 5]


def test_defect_split_data_ignores_fix_ub_when_fix_lb_given():
    """``split_data`` drops ``fix_ub`` whenever ``fix_lb`` is also given.

    The bound validation explicitly supports both being supplied -- it raises
    when ``fix_lb > fix_ub`` -- and the docstring documents both. But the
    ``if fix_lb is not None: ... elif fix_ub is not None: ...`` chain applies
    only the first, and the ``else`` that combines them is unreachable.
    """
    x_clean = split_data("numerical", _XS, _YS, fix_lb=3., fix_ub=7.)[0]

    assert list(x_clean) == [3., 4., 5., 6., 7.]


def test_defect_binning_table_build_single_class_target():
    """A single-class target is degenerate but legal, all the way to build().

    ``OptimalBinning.fit`` accepts it and reports ``status == "OPTIMAL"``;
    ``BinningTable.build`` used to mask every bin out of the divergence
    computation and hand ``jeffrey`` an empty array, which sklearn's
    ``check_array`` refuses. It now reports zeros for the divergence metrics,
    which are undefined without both classes.
    """
    xd = np.linspace(0, 10, 200)
    yd = np.zeros(200, dtype=int)

    optb = OptimalBinning(max_n_prebins=6)
    optb.fit(xd, yd)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()

    assert optb.binning_table.iv == 0.
    assert list(table["Bin"])[0] == "(-inf, inf)"


# ---------------------------------------------------------------------------
# Chaos: input containers, target dtypes, and the verbose log paths
# ---------------------------------------------------------------------------

def test_input_containers_and_target_dtypes():
    reference = OptimalBinning(max_n_prebins=8, max_n_bins=4)
    reference.fit(x, y)

    for xi, yi in ((list(x), list(y)),
                   (pd.Series(x), pd.Series(y)),
                   (x, y.astype(bool)),
                   (x, y.astype(float))):
        optb = OptimalBinning(max_n_prebins=8, max_n_bins=4)
        optb.fit(xi, yi)

        assert optb.status == "OPTIMAL"
        assert optb.splits == approx(reference.splits, rel=1e-12)


def test_nonbinary_target_treats_every_nonzero_as_event():
    yd = y + (x > 8.).astype(int)  # values 0, 1 and 2
    assert set(np.unique(yd)) == {0, 1, 2}

    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4)
    optb.fit(x, yd)

    assert optb.status == "OPTIMAL"

    # The binary estimator splits on y == 0, so 2 counts as an event and the
    # fit matches the binarised target exactly.
    optb_binary = OptimalBinning(max_n_prebins=8, max_n_bins=4)
    optb_binary.fit(x, (yd != 0).astype(int))

    assert optb.splits == approx(optb_binary.splits, rel=1e-12)
    optb.binning_table.build()
    optb_binary.binning_table.build()
    assert optb.binning_table.iv == approx(optb_binary.binning_table.iv,
                                           rel=1e-12)


def test_string_x_with_numerical_dtype_rejected():
    optb = OptimalBinning(max_n_prebins=6)
    with raises(ValueError, match="could not convert string to float"):
        optb.fit(np.array(['a', 'b'] * 150), y)


def test_fit_transform_matches_fit_then_transform():
    optb = OptimalBinning(max_n_prebins=8, max_n_bins=4)
    fitted = optb.fit_transform(x, y, metric="indices", metric_special=-1,
                                metric_missing=-2, show_digits=3)

    optb2 = OptimalBinning(max_n_prebins=8, max_n_bins=4)
    optb2.fit(x, y)
    expected = optb2.transform(x, metric="indices", metric_special=-1,
                               metric_missing=-2, show_digits=3)

    assert list(fitted) == list(expected)


def test_verbose_outlier_and_categorical_paths(caplog):
    optb = OptimalBinning(outlier_detector="zscore", max_n_prebins=8,
                          max_n_bins=4, verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert "number of outlier samples" in caplog.text

    caplog.clear()
    cats, yc = _categorical_data()
    optb = OptimalBinning(dtype="categorical", cat_cutoff=0.1,
                          max_n_prebins=6, verbose=True)
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"
    assert "number of others samples" in caplog.text
    assert "number of categories" in caplog.text


def test_categorical_user_splits_refinement_drops_pure_bin(caplog):
    rng = np.random.RandomState(3)
    cats = np.array(['A'] * 80 + ['B'] * 80 + ['C'] * 80 + ['D'] * 40 +
                    ['E'] * 20)
    rng.shuffle(cats)
    # 'E' is pure, so its prebin is removed by refinement.
    rates = {'A': 0.2, 'B': 0.5, 'C': 0.8, 'D': 0.3, 'E': 1.0}
    yc = np.array([1 if rng.uniform() < rates[c] else 0 for c in cats])

    optb = OptimalBinning(dtype="categorical",
                          user_splits=[['A'], ['B'], ['C'], ['D'], ['E']],
                          max_n_bins=4, verbose=True)
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"
    assert optb._n_refinements == 1
    assert "user splits supplied: 5" in caplog.text
    assert "number prebins removed: 1" in caplog.text

    flat = [c for split in optb.splits for c in split]
    assert 'E' not in flat


def test_auto_heuristic_trend(caplog):
    xp, yp = _peak_data()

    optb = OptimalBinning(monotonic_trend="auto_heuristic", max_n_prebins=8,
                          max_n_bins=5, verbose=True)
    optb.fit(xp, yp)

    assert optb.status == "OPTIMAL"
    assert "classifier predicts" in caplog.text

    table = optb.binning_table.build()
    n_bins = len(optb.splits) + 1
    event_rate = np.array(table["Event rate"][:n_bins])
    peak = int(np.argmax(event_rate))
    assert np.all(np.diff(event_rate[:peak + 1]) >= 0)
    assert np.all(np.diff(event_rate[peak:]) <= 0)


def test_mip_solver_matches_cp_objective():
    optb_cp = OptimalBinning(solver="cp", max_n_prebins=8, max_n_bins=4)
    optb_cp.fit(x, y)

    for mip_solver in ("bop", "cbc"):
        optb_mip = OptimalBinning(solver="mip", mip_solver=mip_solver,
                                  max_n_prebins=8, max_n_bins=4)
        optb_mip.fit(x, y)

        assert optb_mip.status == "OPTIMAL"

        optb_mip.binning_table.build()
        optb_cp.binning_table.build()
        # Equal-objective optima may differ in the split positions chosen, so
        # pin the objective rather than the splits.
        assert optb_mip.binning_table.iv == approx(optb_cp.binning_table.iv,
                                                   rel=1e-6)

    with raises(ValueError, match="Invalid value for mip_solver"):
        OptimalBinning(solver="mip", mip_solver="glpk").fit(x, y)


def test_auto_heuristic_predicts_valley(caplog):
    rng = np.random.RandomState(7)
    xv = rng.uniform(0, 10, 400)
    p = 1. - (np.exp(-((xv - 5.) ** 2) / 4.) * 0.85 + 0.05)
    yv = (rng.uniform(size=400) < p).astype(int)

    optb = OptimalBinning(monotonic_trend="auto_heuristic", max_n_prebins=8,
                          max_n_bins=5, verbose=True)
    optb.fit(xv, yv)

    assert optb.status == "OPTIMAL"
    assert "predicts valley_heuristic" in caplog.text

    table = optb.binning_table.build()
    n_bins = len(optb.splits) + 1
    event_rate = np.array(table["Event rate"][:n_bins])
    valley = int(np.argmin(event_rate))
    assert np.all(np.diff(event_rate[:valley + 1]) <= 0)
    assert np.all(np.diff(event_rate[valley:]) >= 0)


def test_multiclass_single_class_target():
    xm = np.linspace(0, 10, 200)
    ym = np.zeros(200, dtype=int)

    optb = MulticlassOptimalBinning(max_n_prebins=6)
    optb.fit(xm, ym)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    # No splits: bin 0 for everything, and a WoE of exactly zero.
    assert np.all(optb.transform(xm, metric="indices") == 0)
    assert np.all(optb.transform(xm, metric="mean_woe") == 0.)
    assert np.all(optb.transform(xm, metric="weighted_mean_woe") == 0.)

    # Unlike the binary table, the multiclass one builds on this input.
    optb.binning_table.build()


def test_outlier_detectors_drop_samples():
    xo = x.copy()
    xo[:5] = 1e6

    baseline = OptimalBinning(max_n_prebins=8, max_n_bins=4)
    baseline.fit(xo, y)

    for detector, params in (("range", {"method": "ETI"}),
                             ("zscore", {"threshold": 3.0})):
        optb = OptimalBinning(outlier_detector=detector,
                              outlier_params=params, max_n_prebins=8,
                              max_n_bins=4)
        optb.fit(xo, y)

        assert optb.status == "OPTIMAL"
        # The outliers are gone, so no split can sit above the clean range.
        assert optb.splits.max() < 1e6

    with raises(ValueError, match="Invalid value for outlier_detector"):
        OptimalBinning(outlier_detector="yquantile").fit(x, y)

    with raises(TypeError, match="outlier_params must be a dict or None"):
        OptimalBinning(outlier_detector="range",
                       outlier_params=["method"]).fit(x, y)


# ---------------------------------------------------------------------------
# The degenerate-input contract: a single-class target is legal, and every
# reported number has to be honest about it.
# ---------------------------------------------------------------------------

def _single_class_fit(label):
    """Fit a one-class target with a populated special bucket."""
    xs = np.linspace(0., 10., 300)
    xs[:10] = -99.
    ys = np.full(300, label, dtype=int)

    optb = OptimalBinning(name="v", special_codes=[-99.], max_n_prebins=6)
    optb.fit(xs, ys)

    return optb


def test_single_class_target_event_rate_is_gated_on_records():
    """An all-event bin reports an event rate of 1, not 0.

    ``build`` gated the event rate on the same mask as WoE and IV -- the bins
    holding both classes -- so a target carrying one class reported 0.0
    everywhere, including for a bin that is 100% event.
    """
    optb = _single_class_fit(1)
    table = optb.binning_table
    df = table.build()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    # clean bin, special bin: both are pure event and both hold records
    assert list(df["Count"])[:3] == [290, 10, 0]
    assert list(df["Event rate"])[:3] == [1., 1., 0.]
    # ... and the totals row agrees
    assert df.loc["Totals", "Event rate"] == 1.

    # WoE / IV / JS stay gated on mixedness and are undefined here
    assert list(df["WoE"])[:3] == [0., 0., 0.]
    assert list(df["IV"])[:3] == [0., 0., 0.]
    assert list(df["JS"])[:3] == [0., 0., 0.]
    assert table.iv == 0.
    assert table.js == 0.


def test_single_class_target_all_nonevent_event_rate_stays_zero():
    """The mirror case: an all-non-event target reports 0, and only 0."""
    optb = _single_class_fit(0)
    df = optb.binning_table.build()

    assert list(df["Count"])[:3] == [290, 10, 0]
    assert list(df["Event rate"])[:3] == [0., 0., 0.]
    assert df.loc["Totals", "Event rate"] == 0.


def test_single_class_target_gini_is_zero_not_nan():
    """``gini`` divides by ``te * tne``, which a single-class table makes 0.

    Its own ``n <= 1`` branch already answers 0 for the same degeneracy, so
    the multi-bin path must not answer nan.
    """
    optb = _single_class_fit(1)
    table = optb.binning_table
    table.build()

    # two non-empty bins, so the n <= 1 shortcut is not what produces the 0
    assert np.count_nonzero(table.n_event + table.n_nonevent) == 2
    assert table.gini == 0.


def test_single_class_target_analysis_emits_no_runtime_warning():
    """Clause 4: the divides are guarded, not silenced with np.errstate."""
    optb = _single_class_fit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        optb.binning_table.build()
        optb.binning_table.analysis(print_output=False)

    table = optb.binning_table
    assert table.gini == 0.
    assert table.iv == 0.
    assert table.js == 0.
    assert table.quality_score == 0.
    assert table._ks == 0.
    assert table._hellinger == 0.
    assert table._triangular == 0.


def test_categorical_user_splits_single_prebin_keeps_its_bin():
    """One categorical user split is one bin, and it must survive.

    ``_compute_prebins`` lays categorical ``user_splits`` out as
    ``n_bins == n_splits`` rather than ``n_splits + 1``. The degenerate branch
    of ``_fit_optimizer`` ignored that and selected nothing, so the estimator
    reported OPTIMAL and then built a table with no bin at all.
    """
    xc = np.array(["a", "b", "c"] * 100, dtype=object)
    yc = np.array([0, 1] * 150)

    optb = OptimalBinning(dtype="categorical",
                          user_splits=np.array([["a", "b", "c"]],
                                               dtype=object))
    optb.fit(xc, yc)

    assert optb.status == "OPTIMAL"
    assert [sorted(b) for b in optb.splits] == [["a", "b", "c"]]

    df = optb.binning_table.build()

    bins = list(df["Bin"])[:-1]
    assert sorted(bins[0]) == ["a", "b", "c"]
    assert bins[1:] == ["Special", "Missing"]
    assert list(df["Count"])[:-1] == [300, 0, 0]
    assert list(df["Non-event"])[:-1] == [150, 0, 0]
    assert list(df["Event"])[:-1] == [150, 0, 0]

    assert np.all(optb.transform(xc, metric="indices") == 0)


def test_refit_does_not_mutate_user_splits_fixed():
    """``fit`` must leave its constructor parameters alone.

    ``_fit`` reordered ``user_splits_fixed`` into ``self.user_splits_fixed``
    as a numpy array, whose elements are ``numpy.bool_``. ``_check_parameters``
    tests ``isinstance(s, bool)``, so the *second* ``fit`` on the same
    estimator was rejected with "user_splits_fixed must be list of boolean".
    """
    user_splits = [3., 7., 5.]
    user_splits_fixed = [False, True, False]

    optb = OptimalBinning(user_splits=user_splits,
                          user_splits_fixed=user_splits_fixed)
    optb.fit(x, y)

    first_splits = optb.splits.copy()
    first_iv = optb.binning_table.build()["IV"].values.copy()

    # the whole point: a second fit on the same estimator
    optb.fit(x, y)

    assert np.array_equal(optb.splits, first_splits)
    assert optb.binning_table.build()["IV"].values == approx(first_iv)

    # the caller's own objects came back untouched
    assert optb.user_splits is user_splits
    assert optb.user_splits_fixed is user_splits_fixed
    assert user_splits == [3., 7., 5.]
    assert user_splits_fixed == [False, True, False]

    # the reordered working copy is private, and it is the sorted one
    assert list(optb._user_splits_fixed) == [False, False, True]


def test_refit_after_a_pure_prebin_is_removed():
    """The sharper case: pre-binning *prunes* the user splits.

    ``_compute_prebins`` drops the splits whose prebin turns out pure. It used
    to write the pruned arrays back over ``self.user_splits`` and
    ``self.user_splits_fixed``, so a refit silently used fewer splits -- and
    was rejected outright, the pruned ``user_splits_fixed`` being a numpy
    array of ``numpy.bool_``.
    """
    xu = np.concatenate([np.linspace(0., 1.9, 60),
                         np.linspace(2., 6.9, 120),
                         np.linspace(7., 10., 120)])
    yu = np.zeros(300, dtype=int)
    yu[60:] = np.array([0, 1] * 60 + [1, 0] * 60)

    user_splits = [2., 7.]
    user_splits_fixed = [False, False]

    optb = OptimalBinning(user_splits=user_splits,
                          user_splits_fixed=user_splits_fixed)
    optb.fit(xu, yu)

    # the first prebin is pure non-event, so one user split is pruned
    assert optb._n_refinements == 1

    # the constructor parameters are untouched by that pruning ...
    assert optb.user_splits is user_splits
    assert optb.user_splits_fixed is user_splits_fixed
    assert user_splits == [2., 7.]
    assert user_splits_fixed == [False, False]

    # ... so the estimator can be refitted
    first_splits = optb.splits.copy()
    optb.fit(xu, yu)

    assert np.array_equal(optb.splits, first_splits)

    # the pruning happened, on the private working copies
    assert list(optb._user_splits) == [7.]
    assert list(optb._user_splits_fixed) == [False]


def test_empty_user_splits_categorical_fits_a_single_bin():
    """``user_splits=[]`` is "no split points" whatever the dtype is.

    The empty branch of ``_fit`` delegated straight to
    ``_prebinning_refinement``, which skips the categorical preprocessing that
    builds ``categories``. ``bin_categorical`` then took its user-splits
    layout, where ``n_bins == len(splits) == 0``, and produced no bin label at
    all: ``build()`` died with "All arrays must be of the same length" and
    ``transform`` returned nan for every record. An empty split set now means
    the single bin ``user_splits=None`` gives when no split survives.
    """
    cats, yc = _categorical_data()

    optb = OptimalBinning(dtype="categorical", user_splits=[])
    optb.fit(cats, yc)

    assert optb.status == "OPTIMAL"

    # one bin, holding every category
    assert len(optb.splits) == 1
    assert sorted(optb.splits[0]) == ["A", "B", "C", "D", "E"]

    df = optb.binning_table.build()

    assert list(df["Bin"])[1:-1] == ["Special", "Missing"]
    assert list(df["Count"])[:-1] == [len(cats), 0, 0]
    assert list(df["Event"])[:-1] == [int(yc.sum()), 0, 0]

    # a single bin carries the whole population, so its WoE is exactly 0
    assert optb.transform(cats) == approx(np.zeros(len(cats)))


def test_user_splits_fixed_error_names_the_split_it_removed():
    """The message must name the fixed split, not the one at its old index.

    ``user_splits`` is sorted before pre-binning and ``_user_splits_fixed`` is
    reordered with it, but ``_user_splits`` kept the caller's original order.
    ``_compute_prebins`` then indexed the unsorted list with a mask in sorted
    order and named whatever split happened to sit at that position.
    """
    xd, yd = _pure_tail_data()

    # 1.0 is the split the pure left tail removes, and it is the fixed one
    optb = OptimalBinning(user_splits=[9., 1., 3., 5., 7.],
                          user_splits_fixed=[False, True, False, False, False],
                          max_n_bins=4)

    with raises(ValueError, match=r"Fixed user_splits \[1\.\] are removed"):
        optb.fit(xd, yd)


def test_json_round_trip_special_codes_ndarray(tmp_path):
    """``special_codes`` may be an ndarray, and ndarrays are not JSON.

    ``to_dict`` passed ``special_codes`` through raw while every other array
    attribute went through ``_json_value``, so ``to_json`` raised
    ``TypeError: Object of type ndarray is not JSON serializable``. The dict
    form has the same problem one level down, in its values.
    """
    xs = x.copy()
    xs[:5] = -1.
    xs[5:10] = -2.

    for special_codes in (np.array([-1., -2.]),
                          {"a": np.array([-1.]), "b": [-2.]}):
        optb = OptimalBinning(special_codes=special_codes, max_n_prebins=6)
        optb.fit(xs, y)

        path = str(tmp_path / "special_codes.json")
        optb.to_json(path)

        optb_json = OptimalBinning()
        optb_json.read_json(path)

        assert optb_json.transform(xs) == approx(optb.transform(xs),
                                                 rel=1e-12)
