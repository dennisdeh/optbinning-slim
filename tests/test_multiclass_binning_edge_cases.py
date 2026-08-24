"""
MulticlassOptimalBinning edge-case and chaos testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import ContinuousOptimalBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from sklearn.exceptions import NotFittedError


# Small synthetic problems keep the suite fast: every fit below is a CP/MIP
# solver run, and the wine dataset used by tests/test_multiclass_binning.py is
# not needed to reach any of these branches.
def _make_multiclass(n=300, seed=0, centers=(2.0, 5.0, 8.0), width=8.0):
    """Three classes whose probability peaks at a different value of x."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    p = np.vstack([np.exp(-((x - c) ** 2) / width) for c in centers]).T
    p /= p.sum(axis=1, keepdims=True)
    u = rng.uniform(size=n)
    y = (u[:, None] > p.cumsum(axis=1)).sum(axis=1)
    return x, y


def _make_valley_multiclass(n=300, seed=2):
    """Class 0 has a valley-shaped event rate, class 1 a peak-shaped one.

    Chosen so that ``auto_monotonic`` predicts "valley" for class 0 and "peak"
    for class 1, which is what drives the ``"auto_heuristic"`` conversion to
    "valley_heuristic" / "peak_heuristic".
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    p0 = 0.15 + 0.7 * np.abs(x - 5.0) / 5.0
    p1 = 0.5 * np.exp(-((x - 2.5) ** 2) / 6.0)
    p2 = 0.5 * np.exp(-((x - 7.5) ** 2) / 6.0)
    p = np.vstack([p0, p1, p2]).T
    p /= p.sum(axis=1, keepdims=True)
    u = rng.uniform(size=n)
    y = (u[:, None] > p.cumsum(axis=1)).sum(axis=1)
    return x, y


x, y = _make_multiclass()
x_valley, y_valley = _make_valley_multiclass()

# Small pre-binning keeps every solver run well under a second.
PREBIN = dict(max_n_prebins=10, min_prebin_size=0.05)


def _event_rates(optb):
    """Per-class event rate of the optimal bins, special/missing excluded."""
    table = optb.binning_table.build()
    n_bins = len(optb.splits) + 1
    cols = [c for c in table.columns if c.startswith("Event_rate_")]
    return np.asarray(table[cols].values[:n_bins], dtype=float)


# ---------------------------------------------------------------------------
# _check_parameters guards not exercised by tests/test_multiclass_binning.py
# ---------------------------------------------------------------------------

def test_params_min_event_rate_diff():
    with raises(ValueError, match="min_event_rate_diff must be in"):
        MulticlassOptimalBinning(min_event_rate_diff=1.5).fit(x, y)

    with raises(ValueError, match="min_event_rate_diff must be in"):
        MulticlassOptimalBinning(min_event_rate_diff=-0.1).fit(x, y)

    with raises(ValueError, match="min_event_rate_diff must be in"):
        MulticlassOptimalBinning(min_event_rate_diff="0.1").fit(x, y)

    # The closed interval [0, 1] is accepted at both ends.
    for value in (0, 1.0):
        optb = MulticlassOptimalBinning(min_event_rate_diff=value, **PREBIN)
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"


def test_params_outlier_detector():
    with raises(ValueError, match="Invalid value for outlier_detector"):
        MulticlassOptimalBinning(outlier_detector="new_detector").fit(x, y)

    with raises(TypeError, match="outlier_params must be a dict or None"):
        MulticlassOptimalBinning(outlier_detector="range",
                                 outlier_params=[0.1]).fit(x, y)

    # outlier_params is only validated when outlier_detector is set.
    optb = MulticlassOptimalBinning(outlier_params=[0.1], **PREBIN)
    optb.fit(x, y)
    assert optb.status == "OPTIMAL"


def test_params_special_codes_empty_dict():
    with raises(ValueError, match="special_codes empty"):
        MulticlassOptimalBinning(special_codes={}).fit(x, y)

    with raises(TypeError, match="special_codes must be a dit, list or"):
        MulticlassOptimalBinning(special_codes=1).fit(x, y)


def test_params_user_splits_type():
    with raises(TypeError,
                match="user_splits must be a list or numpy.ndarray"):
        MulticlassOptimalBinning(user_splits="2.0").fit(x, y)

    with raises(ValueError, match="user_splits must be provided"):
        MulticlassOptimalBinning(user_splits_fixed=[True]).fit(x, y)


def test_params_monotonic_trend_list_with_none_entries():
    # None is an allowed member of the per-class list.
    optb = MulticlassOptimalBinning(monotonic_trend=[None, None, None],
                                    **PREBIN)
    optb.fit(x, y)
    assert optb.status == "OPTIMAL"

    with raises(ValueError, match="Invalid value for monotonic trend"):
        MulticlassOptimalBinning(monotonic_trend=["ascending", 3]).fit(x, y)


# ---------------------------------------------------------------------------
# verbose branches
# ---------------------------------------------------------------------------

def test_verbose_outlier_detector():
    optb = MulticlassOptimalBinning(outlier_detector="range", verbose=True,
                                    **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) >= 1


def test_verbose_user_splits():
    optb = MulticlassOptimalBinning(user_splits=[2.0, 4.0, 6.0, 8.0],
                                    verbose=True, **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert set(optb.splits).issubset({2.0, 4.0, 6.0, 8.0})


def test_verbose_no_bins_after_prebinning():
    # A constant x yields no pre-binning splits at all, so the optimizer is
    # never run and the estimator short-circuits to a single bin.
    optb = MulticlassOptimalBinning(verbose=True)
    optb.fit(np.ones(120), np.arange(120) % 3)

    assert optb.status == "OPTIMAL"
    assert optb.splits.size == 0

    table = optb.binning_table.build()
    # one bin + special + missing + totals
    assert len(table) == 4
    assert table["Count"].values[0] == 120

    assert optb.transform(np.ones(5)) == approx(np.zeros(5))


def test_verbose_monotonic_trend_none():
    optb = MulticlassOptimalBinning(monotonic_trend=None, verbose=True,
                                    **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    # The unconstrained problem keeps every pre-binning split here.
    assert len(optb.splits) == optb._n_prebins - 1


# ---------------------------------------------------------------------------
# class-count and pre-binning refinement guards
# ---------------------------------------------------------------------------

def test_max_number_of_classes():
    rng = np.random.RandomState(1)

    x_many = rng.uniform(0, 10, 303)
    y_many = np.arange(303) % 101
    with raises(ValueError, match="Maximum number of classes exceeded"):
        MulticlassOptimalBinning().fit(x_many, y_many)

    # 100 classes is the documented maximum and must still fit.
    x_100 = rng.uniform(0, 10, 300)
    y_100 = np.arange(300) % 100
    optb = MulticlassOptimalBinning()
    optb.fit(x_100, y_100)
    assert optb.status == "OPTIMAL"
    assert len(optb.classes) == 100


def test_split_digits():
    optb = MulticlassOptimalBinning(split_digits=2, **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits)
    assert optb.splits == approx(np.round(optb.splits, 2))

    optb0 = MulticlassOptimalBinning(split_digits=0, **PREBIN)
    optb0.fit(x, y)
    assert optb0.splits == approx(np.round(optb0.splits, 0))


def test_prebinning_method_quantile_and_uniform():
    for method in ("quantile", "uniform"):
        optb = MulticlassOptimalBinning(prebinning_method=method, **PREBIN)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert np.all(np.diff(optb.splits) > 0)


def test_prebinning_kwargs():
    optb = MulticlassOptimalBinning(max_depth=1, **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    # A depth-1 CART produces a single pre-binning split.
    assert optb._n_prebins <= 2


# ---------------------------------------------------------------------------
# monotonic_trend as a list, one entry per class
# ---------------------------------------------------------------------------

def test_monotonic_trend_list_wrong_length():
    with raises(ValueError,
                match="List of monotonic trends must be of size n_classes"):
        MulticlassOptimalBinning(monotonic_trend=["auto", "auto"],
                                 **PREBIN).fit(x, y)


def test_monotonic_trend_list_mixes_auto_modes_and_fixed_trends():
    optb = MulticlassOptimalBinning(
        monotonic_trend=["auto", "auto_asc_desc", "descending"],
        verbose=True, **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"

    # class 2 was pinned descending, whatever the classifier predicted.
    event_rate = _event_rates(optb)[:, 2]
    assert np.all(np.diff(event_rate) <= 1e-9)


def test_monotonic_trend_auto_heuristic_converts_peak_and_valley():
    # On this data auto_monotonic predicts "valley" for class 0 and "peak" for
    # class 1, so "auto_heuristic" converts both to their *_heuristic variant
    # and computes a trend change point for each.
    optb = MulticlassOptimalBinning(
        monotonic_trend=["auto_heuristic", "auto_heuristic", "ascending"],
        verbose=True, **PREBIN)
    optb.fit(x_valley, y_valley)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)

    event_rate = _event_rates(optb)[:, 2]
    assert np.all(np.diff(event_rate) >= -1e-9)


def test_monotonic_trend_auto_heuristic_scalar_matches_the_list_form():
    """A scalar ``"auto_heuristic"`` converts to the ``*_heuristic`` trends.

    The scalar branch appended ``auto_monotonic``'s answer verbatim, so
    "peak" and "valley" were enforced as hard shape constraints and
    ``trend_changes`` stayed ``[None] * n_classes`` -- unlike the list form
    two branches below, and unlike ``OptimalBinning``. On this data the two
    forms disagreed: scalar returned [4.17294836, 5.49579668] and the list
    [5.05666447, 5.49579668] (measured 2026-08-24).
    """
    optb_scalar = MulticlassOptimalBinning(monotonic_trend="auto_heuristic",
                                           verbose=True, **PREBIN)
    optb_scalar.fit(x_valley, y_valley)

    optb_list = MulticlassOptimalBinning(
        monotonic_trend=["auto_heuristic"] * 3, **PREBIN)
    optb_list.fit(x_valley, y_valley)

    assert optb_scalar.status == "OPTIMAL"
    assert optb_list.status == "OPTIMAL"
    assert optb_scalar.splits == approx(optb_list.splits)


def test_monotonic_trend_ascending_descending_cp_and_mip():
    trend = ["descending", None, "ascending"]

    results = []
    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(solver=solver, monotonic_trend=trend,
                                        **PREBIN)
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"

        event_rate = _event_rates(optb)
        assert np.all(np.diff(event_rate[:, 0]) <= 1e-9)
        assert np.all(np.diff(event_rate[:, 2]) >= -1e-9)
        results.append(optb.splits)

    assert results[0] == approx(results[1])


def test_monotonic_trend_peak_and_valley_cp_and_mip():
    for trend in (["peak"] * 3, ["valley"] * 3):
        splits = []
        for solver in ("cp", "mip"):
            optb = MulticlassOptimalBinning(solver=solver,
                                            monotonic_trend=trend, **PREBIN)
            optb.fit(x, y)
            assert optb.status == "OPTIMAL"
            splits.append(optb.splits)

        assert splits[0] == approx(splits[1])


# ---------------------------------------------------------------------------
# solver / model options
# ---------------------------------------------------------------------------

def test_mip_solver_cbc_matches_bop():
    optb_bop = MulticlassOptimalBinning(solver="mip", mip_solver="bop",
                                        **PREBIN)
    optb_bop.fit(x, y)

    optb_cbc = MulticlassOptimalBinning(solver="mip", mip_solver="cbc",
                                        **PREBIN)
    optb_cbc.fit(x, y)

    assert optb_bop.status == "OPTIMAL"
    assert optb_cbc.status == "OPTIMAL"
    assert optb_cbc.splits == approx(optb_bop.splits)


def test_min_max_n_bins_cp_and_mip():
    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(solver=solver, min_n_bins=2,
                                        max_n_bins=3, **PREBIN)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert 2 <= len(optb.splits) + 1 <= 3


def test_max_pvalue_policies():
    optb_all = MulticlassOptimalBinning(max_pvalue=0.05,
                                        max_pvalue_policy="all", **PREBIN)
    optb_all.fit(x, y)

    optb_consecutive = MulticlassOptimalBinning(
        max_pvalue=0.05, max_pvalue_policy="consecutive", **PREBIN)
    optb_consecutive.fit(x, y)

    optb_free = MulticlassOptimalBinning(**PREBIN)
    optb_free.fit(x, y)

    for optb in (optb_all, optb_consecutive, optb_free):
        assert optb.status == "OPTIMAL"

    # "all" is the strictest policy, "consecutive" a relaxation of it.
    assert len(optb_all.splits) <= len(optb_consecutive.splits)
    assert len(optb_consecutive.splits) <= len(optb_free.splits)


def test_min_event_rate_diff_reduces_bins():
    optb_free = MulticlassOptimalBinning(**PREBIN)
    optb_free.fit(x, y)

    optb_diff = MulticlassOptimalBinning(min_event_rate_diff=0.1, **PREBIN)
    optb_diff.fit(x, y)

    assert optb_diff.status == "OPTIMAL"
    assert len(optb_diff.splits) <= len(optb_free.splits)


def test_time_limit_zero_still_returns_a_status():
    optb = MulticlassOptimalBinning(time_limit=0, **PREBIN)
    optb.fit(x, y)

    assert isinstance(optb.status, str)
    assert optb.status


# ---------------------------------------------------------------------------
# user_splits
# ---------------------------------------------------------------------------

def test_user_splits_ndarray_unsorted():
    user_splits = np.array([6.0, 2.0, 4.0])
    optb = MulticlassOptimalBinning(user_splits=user_splits, **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)
    assert set(optb.splits).issubset({2.0, 4.0, 6.0})


def test_user_splits_non_finite():
    with raises(ValueError, match="Input contains NaN"):
        MulticlassOptimalBinning(user_splits=[1.0, np.nan]).fit(x, y)

    with raises(ValueError, match="infinity"):
        MulticlassOptimalBinning(user_splits=[1.0, np.inf]).fit(x, y)


def test_empty_user_splits_fits_a_single_bin():
    """``user_splits=[]`` means "no split points", not an error.

    ``_fit`` had no empty-split branch at all, so the empty list reached
    ``check_array`` and came back as "Found array with 0 sample(s) (shape=(0,))
    while a minimum of 1 is required". ``_check_parameters`` accepts the empty
    list and ``_prebinning_refinement`` already early-returns the empty
    counts, so the fit now delegates to it, as OptimalBinning does.
    """
    optb = MulticlassOptimalBinning(user_splits=[], **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    df = optb.binning_table.build()
    # One bin covering everything, plus the special, missing and totals rows.
    assert len(df) == 4
    assert df["Bin"].values[0] == "(-inf, inf)"
    assert df["Count"].values[0] == len(x)

    for cl in range(3):
        assert df["Event_{}".format(cl)].values[0] == np.count_nonzero(y == cl)
        assert df["Event_rate_{}".format(cl)].values[0] == approx(
            np.mean(y == cl), rel=1e-12)

    # The counts _prebinning_refinement owns are populated, not left unset.
    assert list(optb._n_event_missing) == [0, 0, 0]


def test_user_splits_fixed_all_false():
    user_splits = [2.0, 4.0, 6.0, 8.0]
    optb = MulticlassOptimalBinning(user_splits=user_splits,
                                    user_splits_fixed=[False] * 4, **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert set(optb.splits).issubset(set(user_splits))


def test_user_splits_fixed_are_kept():
    user_splits = [2.0, 4.0, 6.0, 8.0]
    optb = MulticlassOptimalBinning(
        user_splits=user_splits, user_splits_fixed=[False, True, True, False],
        **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert 4.0 in optb.splits
    assert 6.0 in optb.splits


def test_refit_does_not_mutate_user_splits_fixed():
    """``fit`` must leave its constructor parameters alone.

    ``_fit`` stored ``np.asarray(user_splits_fixed)[sorted_idx]`` back on the
    public attribute, so the second ``fit`` handed numpy ``bool_`` to
    ``_check_parameters`` and was rejected with "user_splits_fixed must be
    list of boolean".
    """
    user_splits = [6.0, 2.0, 4.0]
    user_splits_fixed = [False, False, True]

    optb = MulticlassOptimalBinning(user_splits=list(user_splits),
                                    user_splits_fixed=user_splits_fixed,
                                    **PREBIN)
    optb.fit(x, y)
    first_splits = optb.splits

    assert optb.user_splits_fixed is user_splits_fixed
    assert list(optb.user_splits) == user_splits
    # The private copies carry the sort and the refinement; 4.0 was flagged.
    assert list(optb._user_splits) == [4.0, 6.0]
    assert list(optb._user_splits_fixed) == [True, False]

    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx(first_splits)
    assert 4.0 in optb.splits


def test_user_splits_outside_range_of_x():
    optb = MulticlassOptimalBinning(user_splits=[-100.0, 5.0, 100.0], **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    # The empty tail prebins are pure and get merged away by the refinement.
    assert optb._n_refinements >= 1
    assert 100.0 not in optb.splits


# ---------------------------------------------------------------------------
# special codes
# ---------------------------------------------------------------------------

def test_special_codes_list_and_dict_agree_on_the_clean_bins():
    x_special = x.copy()
    x_special[:20] = -1.0
    x_special[20:35] = -2.0
    x_special[35:45] = np.nan

    optb_list = MulticlassOptimalBinning(special_codes=[-1, -2], **PREBIN)
    optb_list.fit(x_special, y)

    optb_dict = MulticlassOptimalBinning(
        special_codes={"minus_one": -1, "minus_two": [-2]}, **PREBIN)
    optb_dict.fit(x_special, y)

    assert optb_list.splits == approx(optb_dict.splits)

    table_list = optb_list.binning_table.build()
    table_dict = optb_dict.binning_table.build()

    # The dict form splits the single "Special" row into one row per key.
    assert len(table_dict) == len(table_list) + 1
    assert "minus_one" in list(table_dict["Bin"])
    assert "minus_two" in list(table_dict["Bin"])
    assert table_dict["Count"].values[-4] == 20
    assert table_dict["Count"].values[-3] == 15


def test_special_bucket_missing_a_class():
    x_special = x.copy()
    x_special[np.where(y == 0)[0][:15]] = -9.0

    optb = MulticlassOptimalBinning(special_codes=[-9], **PREBIN)
    optb.fit(x_special, y)

    table = optb.binning_table.build()
    special = table[table["Bin"] == "Special"]
    assert special["Event_0"].values[0] == 15
    assert special["Event_1"].values[0] == 0
    assert special["Event_2"].values[0] == 0
    assert special["Event_rate_1"].values[0] == approx(0.0)

    # A class absent from the special bucket contributes zero WoE, not nan.
    transformed = optb.transform([-9.0], metric_special="empirical")
    assert np.isfinite(transformed).all()


def test_special_codes_empty_list_is_accepted():
    optb = MulticlassOptimalBinning(special_codes=[], **PREBIN)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    table = optb.binning_table.build()
    assert table[table["Bin"] == "Special"]["Count"].values[0] == 0


# ---------------------------------------------------------------------------
# degenerate and hostile inputs
# ---------------------------------------------------------------------------

def test_empty_input():
    with raises(ValueError):
        MulticlassOptimalBinning().fit(np.array([]), np.array([]))


def test_mismatched_lengths():
    with raises(ValueError, match="could not be broadcast|inconsistent"):
        MulticlassOptimalBinning().fit(x[:100], y)


def test_all_missing():
    with raises(ValueError, match="0 sample"):
        MulticlassOptimalBinning().fit(np.full(120, np.nan),
                                       np.arange(120) % 3)


def test_single_row():
    optb = MulticlassOptimalBinning()
    optb.fit(np.array([1.0]), np.array([0]))

    assert optb.status == "OPTIMAL"
    assert optb.splits.size == 0
    assert optb.classes == approx([0])


def test_single_class_target():
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, np.zeros(len(x), dtype=int))

    assert optb.status == "OPTIMAL"
    assert optb.classes == approx([0])
    # Every pre-bin is pure, so the refinement removes all of them.
    assert optb.splits.size == 0
    assert optb.transform(x) == approx(np.zeros(len(x)))


def test_infinite_values_are_rejected_by_prebinning():
    x_inf = x.copy()
    x_inf[0] = np.inf
    x_inf[1] = -np.inf

    with raises(ValueError, match="infinity"):
        MulticlassOptimalBinning(**PREBIN).fit(x_inf, y)


def test_extreme_magnitudes():
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y)

    optb_scaled = MulticlassOptimalBinning(**PREBIN)
    optb_scaled.fit(x * 1e12, y)

    assert optb_scaled.status == "OPTIMAL"
    assert len(optb_scaled.splits) == len(optb.splits)
    assert optb_scaled.splits / 1e12 == approx(optb.splits, rel=1e-6)


def test_many_duplicated_values():
    x_tied = np.round(x)

    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x_tied, y)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)
    # Splits fall between the observed integers.
    assert np.all(optb.splits != np.round(optb.splits))


def test_two_class_target():
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, (y > 0).astype(int))

    assert optb.status == "OPTIMAL"
    assert optb.classes == approx([0, 1])
    assert optb._n_event.shape[1] == 2

    # One-vs-all on two classes is symmetric: the WoE columns are opposites,
    # so their mean is zero in every bin.
    assert optb.transform(x) == approx(np.zeros(len(x)), abs=1e-12)


def test_string_class_labels():
    y_str = np.array(["high", "low", "mid"])[y]

    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y_str)

    assert optb.status == "OPTIMAL"
    assert list(optb.classes) == ["high", "low", "mid"]

    table = optb.binning_table.build()
    assert "Event_high" in table.columns
    assert "Event_rate_mid" in table.columns


def test_non_numerical_x_after_preprocessing():
    x_str = pd.Series(["a", "b", "c"] * (len(y) // 3)).values

    with raises(ValueError, match="must be numerical"):
        MulticlassOptimalBinning(name="x").fit(x_str, y[:len(x_str)])


def test_check_input():
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y, check_input=True)

    assert optb.status == "OPTIMAL"
    assert optb.transform(x, check_input=True) == approx(optb.transform(x))


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def test_transform_metrics():
    optb = MulticlassOptimalBinning(special_codes=[-1], **PREBIN)
    optb.fit(x, y)

    probe = np.array([-1.0, np.nan, 0.0, 5.0, 1e6])

    indices = optb.transform(probe, metric="indices")
    assert indices.dtype.kind in "iu"
    assert indices[-1] == len(optb.splits)

    bins = optb.transform(probe, metric="bins")
    assert bins[0] == "Special"
    assert bins[1] == "Missing"
    assert bins[-1].endswith("inf)")

    mean_woe = optb.transform(probe, metric="mean_woe")
    weighted = optb.transform(probe, metric="weighted_mean_woe")
    assert mean_woe.shape == weighted.shape == probe.shape
    assert np.isfinite(mean_woe).all()
    assert np.isfinite(weighted).all()

    # Constant metric values for the special and missing buckets.
    assert mean_woe[0] == approx(0.0)
    assert mean_woe[1] == approx(0.0)

    filled = optb.transform(probe, metric_special=-1.5, metric_missing=-2.5)
    assert filled[0] == approx(-1.5)
    assert filled[1] == approx(-2.5)


def test_transform_invalid_arguments():
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y)

    with raises(ValueError, match="Invalid value for metric"):
        optb.transform(x, metric="woe")

    with raises(ValueError):
        optb.transform(x, metric_special="mean")

    with raises(ValueError):
        optb.transform(x, show_digits=9)


def test_transform_before_fit():
    optb = MulticlassOptimalBinning()

    with raises(NotFittedError):
        optb.transform(x)


def test_fit_transform_equals_fit_then_transform():
    optb = MulticlassOptimalBinning(**PREBIN)
    transformed = optb.fit_transform(x, y, metric="weighted_mean_woe")

    optb2 = MulticlassOptimalBinning(**PREBIN)
    optb2.fit(x, y)

    assert transformed == approx(
        optb2.transform(x, metric="weighted_mean_woe"))


# ---------------------------------------------------------------------------
# fitted-state protocol, information and persistence
# ---------------------------------------------------------------------------

def test_unfitted_access():
    optb = MulticlassOptimalBinning()

    for accessor in ("binning_table", "classes", "splits", "status"):
        with raises(NotFittedError):
            getattr(optb, accessor)

    with raises(NotFittedError):
        optb.information()

    with raises(NotFittedError):
        optb.to_json("unused.json")


def test_information_print_levels(capsys):
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y)

    for print_level in (0, 1, 2):
        optb.information(print_level=print_level)

    captured = capsys.readouterr().out
    assert "optbinning" in captured
    assert "OPTIMAL" in captured

    with raises(ValueError, match="print_level must be an integer >= 0"):
        optb.information(print_level=-1)

    with raises(ValueError, match="print_level must be an integer >= 0"):
        optb.information(print_level=1.5)


def test_information_print_level_2_reports_mip_solver(capsys):
    optb = MulticlassOptimalBinning(solver="mip", **PREBIN)
    optb.fit(x, y)
    optb.information(print_level=2)

    captured = capsys.readouterr().out
    assert "Solver statistics" in captured
    assert "mip" in captured


def test_json_path_none():
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y)

    with raises(ValueError, match="Specify the path for the json file"):
        optb.to_json(None)

    with raises(ValueError, match="Specify the path for the json file"):
        MulticlassOptimalBinning().read_json(None)


def test_json_round_trip_with_special_codes_dict(tmp_path):
    x_special = x.copy()
    x_special[:20] = -1.0

    optb = MulticlassOptimalBinning(name="v", special_codes={"neg": -1},
                                    **PREBIN)
    optb.fit(x_special, y)

    path = str(tmp_path / "multiclass.json")
    optb.to_json(path)

    optb_json = MulticlassOptimalBinning()
    optb_json.read_json(path)

    assert optb_json.splits == approx(optb.splits, rel=1e-12)
    assert optb_json.transform(x_special) == approx(
        optb.transform(x_special), rel=1e-12)


def test_binning_table_plot(tmp_path):
    optb = MulticlassOptimalBinning(**PREBIN)
    optb.fit(x, y)
    optb.binning_table.build()

    path = str(tmp_path / "multiclass_edge_cases.png")
    optb.binning_table.plot(show_bin_labels=True, savefig=path)

    with raises(TypeError, match="add_special must be a boolean"):
        optb.binning_table.plot(add_special=1)

    with raises(TypeError, match="show_bin_labels must be a boolean"):
        optb.binning_table.plot(show_bin_labels="yes")


# ---------------------------------------------------------------------------
# defects
# ---------------------------------------------------------------------------

def test_defect_min_max_bin_size_not_supported():
    """min_bin_size / max_bin_size are documented and validated, but the
    multiclass model passes a 2-D n_records array to the inherited
    add_constraint_min_max_bin_size, which expects one record count per prebin.

    The binary and continuous siblings both honour the same parameters.
    """
    optb_binary = OptimalBinning(min_bin_size=0.1, max_bin_size=0.5, **PREBIN)
    optb_binary.fit(x, (y == 1).astype(int))
    assert optb_binary.status == "OPTIMAL"

    optb_continuous = ContinuousOptimalBinning(min_bin_size=0.1,
                                               max_bin_size=0.5, **PREBIN)
    optb_continuous.fit(x, y.astype(float))
    assert optb_continuous.status == "OPTIMAL"

    # Every combination is attempted so the report names all of them rather
    # than stopping at the first one.
    failures = []
    for solver in ("cp", "mip"):
        for kwargs in (dict(min_bin_size=0.1),
                       dict(max_bin_size=0.5),
                       dict(min_bin_size=0.1, max_bin_size=0.5)):
            optb = MulticlassOptimalBinning(solver=solver, **kwargs, **PREBIN)
            try:
                optb.fit(x, y)

                assert optb.status == "OPTIMAL"

                counts = optb.binning_table.build()["Count"].values[
                    :len(optb.splits) + 1]
                if "min_bin_size" in kwargs:
                    assert counts.min() >= np.ceil(0.1 * len(x))
                if "max_bin_size" in kwargs:
                    assert counts.max() <= np.ceil(0.5 * len(x))
            except Exception as exc:
                failures.append("solver={} {} -> {}: {}".format(
                    solver, kwargs, type(exc).__name__,
                    str(exc).splitlines()[0][:80]))

    assert not failures, "\n".join(failures)


def test_defect_monotonic_trend_convex_concave_ignored():
    """"convex" and "concave" are documented members of the per-class
    monotonic_trend list and pass validation, but MulticlassBinningCP /
    MulticlassBinningMIP have no branch for them, so no constraint is added.

    The binary sibling OptimalBinning enforces both.
    """
    optb_binary = OptimalBinning(monotonic_trend="convex", **PREBIN)
    optb_binary.fit(x, (y == 1).astype(int))
    binary_rate = optb_binary.binning_table.build()["Event rate"].values[
        :len(optb_binary.splits) + 1]
    assert np.all(np.diff(np.asarray(binary_rate, dtype=float), 2) >= -1e-9)

    for solver in ("cp", "mip"):
        optb = MulticlassOptimalBinning(solver=solver,
                                        monotonic_trend=["convex"] * 3,
                                        **PREBIN)
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"

        event_rate = _event_rates(optb)
        for c in range(3):
            assert np.all(np.diff(event_rate[:, c], 2) >= -1e-9), (
                "class {} event rate is not convex".format(c))

        optb_concave = MulticlassOptimalBinning(
            solver=solver, monotonic_trend=["concave"] * 3, **PREBIN)
        optb_concave.fit(x, y)
        event_rate = _event_rates(optb_concave)
        for c in range(3):
            assert np.all(np.diff(event_rate[:, c], 2) <= 1e-9), (
                "class {} event rate is not concave".format(c))


def test_defect_monotonic_trend_peak_valley_heuristic_ignored():
    """"peak_heuristic" / "valley_heuristic" reach the model as members of the
    monotonic_trend list, but the model tests `self.monotonic_trend ==
    "peak_heuristic"` -- the whole list against a string -- so the branch is
    never taken and no constraint is added.

    _fit_optimizer computes trend_changes for exactly these values and hands
    them to build_model, where they are then unused.
    """
    optb_free = MulticlassOptimalBinning(monotonic_trend=None, **PREBIN)
    optb_free.fit(x, y)
    free_rate = _event_rates(optb_free)

    # The unconstrained optimum is not peak-shaped for any class here, so a
    # peak constraint of any flavour has to change the answer.
    for c in range(3):
        diffs = np.sign(np.diff(free_rate[:, c]))
        assert np.count_nonzero(diffs[1:] != diffs[:-1]) >= 2

    for solver in ("cp", "mip"):
        for trend in ("peak_heuristic", "valley_heuristic"):
            optb = MulticlassOptimalBinning(solver=solver,
                                            monotonic_trend=[trend] * 3,
                                            **PREBIN)
            optb.fit(x, y)
            assert optb.status == "OPTIMAL"

            unchanged = (len(optb.splits) == len(optb_free.splits) and
                         np.allclose(optb.splits, optb_free.splits))
            assert not unchanged, (
                "{} left the unconstrained solution untouched under "
                "solver={}".format(trend, solver))

            # The constraint makes every per-class event rate unimodal: one
            # run up and one run down around the trend change point.
            rate = _event_rates(optb)
            for c in range(3):
                signs = np.sign(np.round(np.diff(rate[:, c]), 12))
                signs = signs[signs != 0]
                assert np.count_nonzero(signs[1:] != signs[:-1]) <= 1, (
                    "class {} event rate is not unimodal under {} "
                    "solver={}".format(c, trend, solver))


def test_json_round_trip_special_codes_ndarray(tmp_path):
    """``special_codes`` may be an ndarray, and ndarrays are not JSON.

    ``to_json`` wrote ``table.special_codes`` raw while every other attribute
    was converted, so it raised ``TypeError: Object of type ndarray is not
    JSON serializable``. The dict form has the same problem in its values.
    """
    x_special = x.copy()
    x_special[:20] = -1.0
    x_special[20:30] = -2.0

    for special_codes in (np.array([-1., -2.]),
                          {"a": np.array([-1.]), "b": [-2.]}):
        optb = MulticlassOptimalBinning(name="v",
                                        special_codes=special_codes,
                                        **PREBIN)
        optb.fit(x_special, y)

        path = str(tmp_path / "multiclass_special_codes.json")
        optb.to_json(path)

        optb_json = MulticlassOptimalBinning()
        optb_json.read_json(path)

        assert optb_json.transform(x_special) == approx(
            optb.transform(x_special), rel=1e-12)
