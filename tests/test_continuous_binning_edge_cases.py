"""
ContinuousOptimalBinning edge cases and chaos testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import warnings

import numpy as np

from pytest import approx, raises

from optbinning import ContinuousOptimalBinning
from sklearn.exceptions import NotFittedError


# Small synthetic instances: the estimator runs a CP-SAT solver per fit, so the
# whole module stays fast only if the arrays stay small and max_n_prebins low.
n = 300
rng = np.random.RandomState(42)

x = np.linspace(0, 10, n)
y = 2.0 * x + rng.normal(0, 0.5, n)

# A clean peak: mean(y) rises to x = 5 and falls afterwards.
n_peak = 400
rng_peak = np.random.RandomState(3)
x_peak = np.linspace(0, 10, n_peak)
y_peak = -((x_peak - 5.0) ** 2) + rng_peak.normal(0, 0.3, n_peak)
y_valley = ((x_peak - 5.0) ** 2) + rng_peak.normal(0, 0.3, n_peak)

# Three well separated categories plus one rare category.
rng_cat = np.random.RandomState(7)
x_cat = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100 + ["rare"] * 5)
y_cat = np.concatenate([rng_cat.normal(1, .2, 100), rng_cat.normal(5, .2, 100),
                        rng_cat.normal(9, .2, 100), rng_cat.normal(50, .2, 5)])


def bin_means(optb):
    """Mean per optimal bin, without the special / missing / totals rows."""
    return optb.binning_table.build()["Mean"].values[:-3]


def bin_counts(optb):
    """Count per optimal bin, without the special / missing / totals rows."""
    return optb.binning_table.build()["Count"].values[:-3]


def test_params_gamma():
    with raises(ValueError):
        optb = ContinuousOptimalBinning(gamma=-1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(gamma="0.5")
        optb.fit(x, y)


def test_params_outlier_detector():
    with raises(ValueError):
        optb = ContinuousOptimalBinning(outlier_detector="isolation_forest")
        optb.fit(x, y)

    with raises(TypeError):
        optb = ContinuousOptimalBinning(outlier_detector="range",
                                        outlier_params=[0.5])
        optb.fit(x, y)


def test_params_cat_unknown():
    with raises(TypeError):
        optb = ContinuousOptimalBinning(cat_unknown=[1, 2])
        optb.fit(x, y)

    # A number and a string are both accepted.
    for cat_unknown in (-1.0, "unknown"):
        optb = ContinuousOptimalBinning(dtype="categorical",
                                        cat_unknown=cat_unknown)
        optb.fit(x_cat, y_cat)
        assert optb.status == "OPTIMAL"


def test_params_special_codes_empty_dict():
    with raises(ValueError):
        optb = ContinuousOptimalBinning(special_codes={})
        optb.fit(x, y)


def test_params_min_prebin_size_zero():
    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_prebin_size=0.0)
        optb.fit(x, y)


def test_unfitted_access():
    optb = ContinuousOptimalBinning()

    with raises(NotFittedError):
        optb.splits

    with raises(NotFittedError):
        optb.status

    with raises(NotFittedError):
        optb.binning_table

    with raises(NotFittedError):
        optb.information()

    with raises(NotFittedError):
        optb.transform(x)

    with raises(NotFittedError):
        optb.to_json("continuous_binning.json")


def test_json_path_none():
    optb = ContinuousOptimalBinning()

    # The path guard runs before the fitted-state guard.
    with raises(ValueError):
        optb.to_json(None)

    with raises(ValueError):
        optb.read_json(None)


def test_constant_x_single_bin():
    x_const = np.full(n, 3.0)

    optb = ContinuousOptimalBinning()
    optb.fit(x_const, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    df = optb.binning_table.build()
    assert df["Bin"].values[0] == "(-inf, inf)"
    assert df["Count"].values[0] == n
    assert df["Mean"].values[0] == approx(y.mean(), rel=1e-9)

    # Every record, in range or not, maps to the single bin mean.
    assert optb.transform([3.0, -99.0, 1e6]) == approx(
        [y.mean()] * 3, rel=1e-9)


def test_single_row():
    optb = ContinuousOptimalBinning()
    optb.fit(np.array([1.0]), np.array([7.0]))

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb.binning_table.build()["Count"].values[0] == 1
    assert optb.transform([1.0]) == approx([7.0], rel=1e-9)


def test_single_unique_value_categorical():
    x_one = np.array(["only"] * 200)
    y_one = np.linspace(0, 1, 200)

    optb = ContinuousOptimalBinning(dtype="categorical")
    optb.fit(x_one, y_one)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].values[0] == 200
    assert df["Mean"].values[0] == approx(y_one.mean(), rel=1e-9)

    # An unseen category falls back to the overall mean target.
    assert optb.transform(np.array(["only", "unseen"])) == approx(
        [y_one.mean()] * 2, rel=1e-9)
    assert list(optb.transform(np.array(["only", "unseen"]),
                               metric="indices")) == [0, -1]


def test_constant_y():
    optb = ContinuousOptimalBinning(max_n_prebins=6)
    optb.fit(x, np.full(n, 3.0))

    assert optb.status == "OPTIMAL"
    assert np.all(bin_means(optb) == approx(3.0, rel=1e-12))


def test_all_nan_x_rejected():
    with raises(ValueError):
        optb = ContinuousOptimalBinning()
        optb.fit(np.full(n, np.nan), y)


def test_all_nan_y_rejected():
    with raises(ValueError):
        optb = ContinuousOptimalBinning()
        optb.fit(x, np.full(n, np.nan))


def test_empty_arrays_rejected():
    with raises(ValueError):
        optb = ContinuousOptimalBinning()
        optb.fit(np.array([]), np.array([]))


def test_mismatched_lengths_rejected():
    with raises(ValueError):
        optb = ContinuousOptimalBinning()
        optb.fit(x, y[:10], check_input=True)

    # Without check_input the mismatch still raises, from the mask broadcast.
    with raises(ValueError):
        optb = ContinuousOptimalBinning()
        optb.fit(x, y[:10])


def test_infinite_values_rejected():
    x_inf = x.copy()
    x_inf[0] = np.inf

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_n_prebins=5)
        optb.fit(x_inf, y)

    y_inf = y.copy()
    y_inf[0] = -np.inf

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_n_prebins=5)
        optb.fit(x, y_inf, check_input=True)


def test_duplicated_values():
    # Only three distinct values of x, each repeated 100 times.
    x_dup = np.repeat([1.0, 2.0, 3.0], 100)
    y_dup = np.repeat([10.0, 20.0, 30.0], 100) + rng.normal(0, .1, 300)

    optb = ContinuousOptimalBinning(max_n_prebins=6)
    optb.fit(x_dup, y_dup)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) <= 2
    means = bin_means(optb)
    assert np.all(np.diff(means) > 0)


def test_missing_statistics():
    x_missing = x.copy()
    x_missing[:20] = np.nan

    optb = ContinuousOptimalBinning(max_n_prebins=6)
    optb.fit(x_missing, y)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    row = df[df["Bin"] == "Missing"].iloc[0]

    assert row["Count"] == 20
    assert row["Sum"] == approx(y[:20].sum(), rel=1e-9)
    assert row["Std"] == approx(np.std(y[:20]), rel=1e-9)
    assert row["Min"] == approx(y[:20].min(), rel=1e-9)
    assert row["Max"] == approx(y[:20].max(), rel=1e-9)


def test_special_codes_list_and_dict():
    x_special = x.copy()
    x_special[:15] = -1.0
    x_special[15:25] = -2.0

    optb_list = ContinuousOptimalBinning(special_codes=[-1.0, -2.0],
                                         max_n_prebins=6)
    optb_list.fit(x_special, y)

    df_list = optb_list.binning_table.build()
    assert df_list[df_list["Bin"] == "Special"]["Count"].values[0] == 25

    optb_dict = ContinuousOptimalBinning(
        special_codes={"first": -1.0, "second": [-2.0]}, max_n_prebins=6)
    optb_dict.fit(x_special, y)

    df_dict = optb_dict.binning_table.build()
    assert df_dict[df_dict["Bin"] == "first"]["Count"].values[0] == 15
    assert df_dict[df_dict["Bin"] == "second"]["Count"].values[0] == 10

    # The dict form only names the buckets; the clean bins are unchanged.
    assert optb_dict.splits == approx(optb_list.splits, rel=1e-12)


def test_transform_metric_special_missing():
    x_mix = x.copy()
    x_mix[:10] = -1.0
    x_mix[10:20] = np.nan

    optb = ContinuousOptimalBinning(special_codes=[-1.0], max_n_prebins=6)
    optb.fit(x_mix, y)

    x_new = np.array([-1.0, np.nan, 5.0])

    numeric = optb.transform(x_new, metric_special=1.5, metric_missing=2.5)
    assert numeric[0] == approx(1.5, rel=1e-12)
    assert numeric[1] == approx(2.5, rel=1e-12)

    empirical = optb.transform(x_new, metric_special="empirical",
                               metric_missing="empirical")
    assert empirical[0] == approx(y[:10].mean(), rel=1e-9)
    assert empirical[1] == approx(y[10:20].mean(), rel=1e-9)
    assert empirical[2] == approx(numeric[2], rel=1e-12)

    with raises(ValueError):
        optb.transform(x_new, metric="event_rate")


def test_split_digits():
    optb = ContinuousOptimalBinning(split_digits=1, max_n_prebins=6)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx(np.round(optb.splits, 1), rel=1e-12)


def test_min_max_bin_size():
    optb = ContinuousOptimalBinning(min_bin_size=0.15, max_bin_size=0.4,
                                    max_n_prebins=10)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"

    counts = bin_counts(optb)
    assert counts.min() >= int(np.ceil(0.15 * n))
    assert counts.max() <= int(np.ceil(0.4 * n))


def test_min_max_n_bins():
    optb = ContinuousOptimalBinning(min_n_bins=2, max_n_bins=3,
                                    max_n_prebins=10)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert 2 <= len(optb.splits) + 1 <= 3


def test_sample_weight():
    sample_weight = np.full(n, 2.0)

    optb = ContinuousOptimalBinning(max_n_prebins=6)
    optb.fit(x, y, sample_weight=sample_weight)

    assert optb.status == "OPTIMAL"
    assert optb._n_samples == n
    assert optb._n_samples_weighted == approx(2 * n, rel=1e-12)
    assert bin_counts(optb).sum() == 2 * n


def test_outlier_detectors():
    x_outlier = np.concatenate([x, np.array([500.0, 600.0, -400.0])])
    y_outlier = np.concatenate([y, np.array([1.0, 2.0, 3.0])])

    for detector in ("range", "zscore", "yquantile"):
        optb = ContinuousOptimalBinning(outlier_detector=detector,
                                        max_n_prebins=6)
        optb.fit(x_outlier, y_outlier)

        assert optb.status == "OPTIMAL"
        # Outliers are dropped entirely: they land in no bin at all.
        assert bin_counts(optb).sum() < len(x_outlier)


def test_outlier_params():
    x_outlier = np.concatenate([x, np.array([500.0, 600.0, -400.0])])
    y_outlier = np.concatenate([y, np.array([1.0, 2.0, 3.0])])

    optb = ContinuousOptimalBinning(
        outlier_detector="range", outlier_params={"interval_length": 0.8},
        max_n_prebins=6)
    optb.fit(x_outlier, y_outlier)

    assert optb.status == "OPTIMAL"
    assert bin_counts(optb).sum() < len(x_outlier)


def test_prebinning_methods():
    for method in ("cart", "quantile", "uniform"):
        optb = ContinuousOptimalBinning(prebinning_method=method,
                                        max_n_prebins=6)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"
        assert np.all(np.diff(optb.splits) > 0)


def test_monotonic_trend_concave():
    optb = ContinuousOptimalBinning(monotonic_trend="concave",
                                    max_n_prebins=8)
    optb.fit(x_peak, y_peak)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(bin_means(optb), 2) <= 1e-9)


def test_monotonic_trend_convex():
    optb = ContinuousOptimalBinning(monotonic_trend="convex",
                                    max_n_prebins=8)
    optb.fit(x_peak, y_valley)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(bin_means(optb), 2) >= -1e-9)


def test_monotonic_trend_peak():
    optb = ContinuousOptimalBinning(monotonic_trend="peak", max_n_prebins=8)
    optb.fit(x_peak, y_peak)

    assert optb.status == "OPTIMAL"

    signs = np.sign(np.diff(bin_means(optb)))
    # A peak: the mean never turns back up once it has started to fall.
    assert np.all(np.diff(signs) <= 0)
    assert -1 in signs and 1 in signs


def test_monotonic_trend_valley():
    optb = ContinuousOptimalBinning(monotonic_trend="valley", max_n_prebins=8)
    optb.fit(x_peak, y_valley)

    assert optb.status == "OPTIMAL"

    signs = np.sign(np.diff(bin_means(optb)))
    assert np.all(np.diff(signs) >= 0)
    assert -1 in signs and 1 in signs


def test_monotonic_trend_peak_heuristic():
    optb = ContinuousOptimalBinning(monotonic_trend="peak_heuristic",
                                    max_n_prebins=8)
    optb.fit(x_peak, y_peak)

    assert optb.status == "OPTIMAL"

    signs = np.sign(np.diff(bin_means(optb)))
    assert np.all(np.diff(signs) <= 0)
    assert -1 in signs and 1 in signs


def test_monotonic_trend_valley_heuristic():
    optb = ContinuousOptimalBinning(monotonic_trend="valley_heuristic",
                                    max_n_prebins=8)
    optb.fit(x_peak, y_valley)

    assert optb.status == "OPTIMAL"

    signs = np.sign(np.diff(bin_means(optb)))
    assert np.all(np.diff(signs) >= 0)
    assert -1 in signs and 1 in signs


def test_monotonic_trend_auto_heuristic_selects_peak():
    optb_auto = ContinuousOptimalBinning(monotonic_trend="auto_heuristic",
                                         max_n_prebins=8)
    optb_auto.fit(x_peak, y_peak)

    optb_peak = ContinuousOptimalBinning(monotonic_trend="peak_heuristic",
                                         max_n_prebins=8)
    optb_peak.fit(x_peak, y_peak)

    assert optb_auto.status == "OPTIMAL"
    # "auto_heuristic" recognises the peak and rewrites itself to
    # "peak_heuristic", so both fits agree.
    assert optb_auto.splits == approx(optb_peak.splits, rel=1e-12)


def test_monotonic_trend_none():
    optb = ContinuousOptimalBinning(monotonic_trend=None, max_n_prebins=8)
    optb.fit(x_peak, y_peak)

    assert optb.status == "OPTIMAL"
    # Unconstrained: the peak shape survives.
    assert np.diff(bin_means(optb)).min() < 0


def test_categorical_monotonic_trend_is_ascending():
    # Whatever trend is asked for, a categorical fit sorts the bins ascending.
    optb = ContinuousOptimalBinning(dtype="categorical",
                                    monotonic_trend="descending")
    optb.fit(x_cat, y_cat)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(bin_means(optb)) > 0)


def test_categorical_cat_cutoff_others(capsys):
    optb = ContinuousOptimalBinning(dtype="categorical", cat_cutoff=0.05)
    optb.fit(x_cat, y_cat)

    assert optb.status == "OPTIMAL"
    assert list(optb._cat_others) == ["rare"]

    df = optb.binning_table.build()
    others = df.iloc[len(df) - 4]

    assert others["Count"] == 5
    assert others["Sum"] == approx(y_cat[-5:].sum(), rel=1e-9)
    assert others["Std"] == approx(np.std(y_cat[-5:]), rel=1e-9)
    assert others["Min"] == approx(y_cat[-5:].min(), rel=1e-9)
    assert others["Max"] == approx(y_cat[-5:].max(), rel=1e-9)


def test_user_splits_out_of_range_refinement():
    user_splits = [-50.0, 2.0, 5.0, 8.0, 100.0]

    optb = ContinuousOptimalBinning(user_splits=user_splits,
                                    monotonic_trend="ascending")
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    # The two splits outside the range of x produce empty prebins and are
    # dropped by the refinement pass.
    assert optb._n_refinements == 1
    assert optb._n_prebins == 4
    assert set(optb.splits).issubset({2.0, 5.0, 8.0})


def test_user_splits_fixed_survives_refinement():
    user_splits = [-50.0, 2.0, 5.0, 8.0, 100.0]

    optb = ContinuousOptimalBinning(user_splits=user_splits,
                                    user_splits_fixed=[False] * 5,
                                    monotonic_trend="ascending")
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    # The refinement shrinks its own copies; the two constructor parameters
    # come out of the fit exactly as they went in, so a refit still validates.
    assert list(optb.user_splits) == user_splits
    assert list(optb.user_splits_fixed) == [False] * 5
    assert list(optb._user_splits) == [2.0, 5.0, 8.0]
    assert list(optb._user_splits_fixed) == [False, False, False]


def test_user_splits_fixed_removed_raises():
    user_splits = [-50.0, 2.0, 5.0, 8.0, 100.0]
    user_splits_fixed = [True, False, False, False, False]

    optb = ContinuousOptimalBinning(user_splits=user_splits,
                                    user_splits_fixed=user_splits_fixed)

    with raises(ValueError, match=r"Fixed user_splits \[-50\.\]"):
        optb.fit(x, y)

    # The message names the split by its position in the *sorted* splits, so
    # an unsorted user_splits must not shift it: the fixed -50.0 is the one
    # removed, whatever position it was passed in.
    optb = ContinuousOptimalBinning(
        user_splits=[100.0, -50.0, 2.0, 5.0, 8.0],
        user_splits_fixed=[False, True, False, False, False])

    with raises(ValueError, match=r"Fixed user_splits \[-50\.\]"):
        optb.fit(x, y)


def test_user_splits_all_removed():
    # Every split is below min(x), so every prebin but the last is empty.
    optb = ContinuousOptimalBinning(user_splits=[-100.0, -50.0],
                                    user_splits_fixed=[False, False])
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert list(optb.user_splits) == [-100.0, -50.0]
    assert list(optb._user_splits) == []

    df = optb.binning_table.build()
    assert df["Bin"].values[0] == "(-inf, inf)"
    assert df["Count"].values[0] == n


def test_categorical_user_splits_empty_group():
    # "zz" is not in x_cat, so that prebin holds no records; "c" is in x_cat
    # but in no user split, so it goes to cat_others.
    user_splits = np.array([["a"], ["b"], ["zz"]], dtype=object)

    optb = ContinuousOptimalBinning(dtype="categorical",
                                    user_splits=user_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optb.fit(x_cat[:300], y_cat[:300])

    assert optb.status == "OPTIMAL"
    assert optb._n_refinements == 1
    assert list(optb._cat_others) == ["c"]

    assert bin_counts(optb).sum() == 300


def test_max_pvalue_and_min_mean_diff_together():
    optb = ContinuousOptimalBinning(max_pvalue=0.05, max_pvalue_policy="all",
                                    min_mean_diff=1.0, monotonic_trend=None,
                                    max_n_prebins=8)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"

    means = bin_means(optb)
    assert np.all(np.abs(np.diff(means)) >= 1.0)


def test_information_print_levels(capsys):
    optb = ContinuousOptimalBinning(max_n_prebins=6)
    optb.fit(x, y)

    for print_level in (0, 1, 2):
        optb.information(print_level=print_level)

    out = capsys.readouterr().out
    assert "Status" in out
    assert "OPTIMAL" in out
    assert "Solver statistics" in out
    assert "Objective value" in out

    with raises(ValueError):
        optb.information(print_level=-1)


def test_information_without_solver_run(capsys):
    # A degenerate fit never builds a model, so there are no solver statistics.
    optb = ContinuousOptimalBinning()
    optb.fit(np.full(n, 3.0), y)
    optb.information(print_level=2)

    out = capsys.readouterr().out
    assert "OPTIMAL" in out
    assert "Solver statistics" not in out


def test_verbose_numerical(capsys):
    optb = ContinuousOptimalBinning(max_n_prebins=6, monotonic_trend=None,
                                    outlier_detector="zscore", verbose=True)
    optb.fit(x, y, sample_weight=np.full(n, 2.0))

    assert optb.status == "OPTIMAL"


def test_verbose_categorical(capsys):
    optb = ContinuousOptimalBinning(dtype="categorical", cat_cutoff=0.05,
                                    verbose=True)
    optb.fit(x_cat, y_cat)

    assert optb.status == "OPTIMAL"


def test_verbose_user_splits_refinement(capsys):
    optb = ContinuousOptimalBinning(user_splits=[-50.0, 2.0, 5.0, 100.0],
                                    verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb._n_refinements == 1


def test_verbose_no_solver_run(capsys):
    optb = ContinuousOptimalBinning(verbose=True)
    optb.fit(np.full(n, 3.0), y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0


def test_defect_stray_debug_print_on_categorical_others(capsys):
    """`_prebinning_refinement` prints the dtypes of y_others / sw_others."""
    optb = ContinuousOptimalBinning(dtype="categorical", cat_cutoff=0.05)
    optb.fit(x_cat, y_cat)

    assert capsys.readouterr().out == ""

    # The same others block is reached through user_splits, not only cat_cutoff
    optb = ContinuousOptimalBinning(
        dtype="categorical",
        user_splits=np.array([["a"], ["b"]], dtype=object))
    optb.fit(x_cat[:300], y_cat[:300])

    assert capsys.readouterr().out == ""


def test_defect_gamma_is_not_a_regularization():
    """`gamma` must penalise the objective and reach the solution.

    The dominating-bin penalty enters the objective with a minus sign, so a
    larger gamma can only lower the reported objective, and a large enough
    gamma must collapse the fit to the single bin that carries no penalty.
    """
    rng_g = np.random.RandomState(5)
    n_g = 500
    x_g = np.sort(rng_g.uniform(0, 10, n_g))
    y_g = 0.1 * x_g + rng_g.normal(0, 0.05, n_g)

    def fit(gamma):
        optb = ContinuousOptimalBinning(gamma=gamma, max_n_prebins=10,
                                        monotonic_trend="ascending")
        optb.fit(x_g, y_g)
        assert optb.status == "OPTIMAL"
        return optb

    optb0 = fit(0)
    objective0 = optb0._optimizer["objective"]

    # gamma subtracts a penalty from the objective, so it can only lower it.
    for gamma in (1, 10, 100):
        assert fit(gamma)._optimizer["objective"] <= objective0

    # And it is documented to reduce dominating bins: a strong gamma leaves
    # the one bin whose size spread -- and therefore whose penalty -- is zero.
    assert len(optb0.splits) > 0
    optb_strong = fit(100)
    assert len(optb_strong.splits) == 0
    assert bin_counts(optb_strong)[0] == n_g


def test_empty_user_splits_fits_a_single_bin():
    """``user_splits=[]`` means "no split points", not an error.

    The empty branch of ``_fit`` used to bind only ``splits``, ``n_records``,
    ``sums`` and ``stds`` and then call ``_fit_optimizer`` with five
    arguments, dying on ``UnboundLocalError: cannot access local variable
    'ssums'``; ``_n_records_special`` and ``_sum_missing`` were left at None
    because ``_prebinning_refinement`` -- the only place they are computed --
    was skipped. It now delegates to ``_prebinning_refinement``, which
    early-returns the eight empty arrays, exactly as ``OptimalBinning`` does.
    """
    optb = ContinuousOptimalBinning(user_splits=[])
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    df = optb.binning_table.build()
    # One bin covering everything, plus the special, missing and totals rows.
    assert len(df) == 4
    assert df["Bin"].values[0] == "(-inf, inf)"
    assert df["Count"].values[0] == n
    assert df["Mean"].values[0] == approx(y.mean(), rel=1e-9)

    # The aggregates _prebinning_refinement owns are populated, not None.
    assert optb._n_records_special == 0
    assert optb._n_records_missing == 0
    assert optb._sum_missing == 0


def test_defect_categorical_single_user_split_group():
    """A one-group categorical fit reports OPTIMAL but has no bins."""
    user_splits = np.array([["a", "b", "c"]], dtype=object)

    optb = ContinuousOptimalBinning(dtype="categorical",
                                    user_splits=user_splits)
    optb.fit(x_cat[:300], y_cat[:300])

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].values[0] == 300
    assert df["Mean"].values[0] == approx(y_cat[:300].mean(), rel=1e-9)


def test_defect_categorical_user_splits_collapsing_to_one_group():
    """Two groups collapse to one when a group names an absent category.

    ``preprocessing_user_splits_categorical`` averages the target over each
    group to order them, so a group no record falls into is averaged over an
    empty selection: numpy warns "Mean of empty slice" and returns nan. The
    nan is harmless -- ``np.argsort`` puts it last, and the empty prebin is
    then dropped by the refinement -- and the assertion below proves it, by
    fitting the same data without the absent group and getting the same
    table. The warning is suppressed exactly as in
    ``test_categorical_user_splits_empty_group`` above; guarding the mean
    belongs in preprocessing.py.
    """
    user_splits = np.array([["a", "b", "c"], ["absent"]], dtype=object)

    optb = ContinuousOptimalBinning(dtype="categorical",
                                    user_splits=user_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optb.fit(x_cat[:300], y_cat[:300])

    assert optb.status == "OPTIMAL"
    assert optb._n_refinements == 1

    df = optb.binning_table.build()
    assert df["Count"].values[0] == 300
    assert df["Mean"].values[0] == approx(y_cat[:300].mean(), rel=1e-9)

    # The absent group changes nothing but the refinement count.
    optb_present = ContinuousOptimalBinning(
        dtype="categorical", user_splits=np.array([["a", "b", "c"]],
                                                  dtype=object))
    optb_present.fit(x_cat[:300], y_cat[:300])

    assert optb_present._n_refinements == 0
    assert optb_present.binning_table.build()["Mean"].values[0] == approx(
        df["Mean"].values[0], rel=1e-12)


def test_monotonic_trend_descending():
    optb = ContinuousOptimalBinning(monotonic_trend="descending",
                                    max_n_prebins=8)
    optb.fit(x_peak, y_peak)

    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(bin_means(optb)) < 0)


def test_monotonic_trend_auto_heuristic_selects_valley():
    optb_auto = ContinuousOptimalBinning(monotonic_trend="auto_heuristic",
                                         max_n_prebins=8)
    optb_auto.fit(x_peak, y_valley)

    optb_valley = ContinuousOptimalBinning(monotonic_trend="valley_heuristic",
                                           max_n_prebins=8)
    optb_valley.fit(x_peak, y_valley)

    assert optb_auto.status == "OPTIMAL"
    assert optb_auto.splits == approx(optb_valley.splits, rel=1e-12)


def test_user_splits_unsorted_with_fixed():
    # user_splits are sorted internally; user_splits_fixed must follow.
    optb = ContinuousOptimalBinning(user_splits=[8.0, 2.0, 5.0],
                                    user_splits_fixed=[True, False, False],
                                    monotonic_trend="ascending")
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert list(optb.user_splits_fixed) == [True, False, False]
    assert list(optb._user_splits_fixed) == [False, False, True]
    # The fixed split survives into the solution.
    assert 8.0 in list(optb.splits)


def test_refit_does_not_mutate_user_splits_fixed():
    """``fit`` must leave its constructor parameters alone.

    ``_fit`` stored ``np.asarray(user_splits_fixed)[sorted_idx]`` back on the
    public attribute, so the second ``fit`` handed numpy ``bool_`` to
    ``_check_parameters`` and was rejected with "user_splits_fixed must be
    list of boolean". Same class of defect as the ``MDLP.fit`` reset in
    reports/DECISIONS.md.
    """
    user_splits_fixed = [True, False, False]
    optb = ContinuousOptimalBinning(user_splits=[8.0, 2.0, 5.0],
                                    user_splits_fixed=user_splits_fixed,
                                    monotonic_trend="ascending")
    optb.fit(x, y)
    first_splits = optb.splits

    assert optb.user_splits_fixed is user_splits_fixed
    assert all(isinstance(s, bool) for s in optb.user_splits_fixed)

    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx(first_splits)
    assert 8.0 in list(optb.splits)


def test_min_mean_diff_larger_than_target_range():
    # No two bins can differ by 1000 in mean, so a single bin is optimal.
    optb = ContinuousOptimalBinning(min_mean_diff=1000, monotonic_trend=None,
                                    max_n_prebins=8)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert bin_counts(optb)[0] == n


def test_time_limit_zero():
    optb = ContinuousOptimalBinning(time_limit=0, max_n_prebins=20)
    optb.fit(x, y)

    # No time to prove anything, but the fit must still complete cleanly.
    assert optb.status in ("OPTIMAL", "FEASIBLE", "UNKNOWN")
    assert optb._is_fitted


def test_list_and_series_input():
    import pandas as pd

    optb_array = ContinuousOptimalBinning(max_n_prebins=5)
    optb_array.fit(x, y)

    optb_list = ContinuousOptimalBinning(max_n_prebins=5)
    optb_list.fit(list(x), list(y))

    optb_series = ContinuousOptimalBinning(max_n_prebins=5)
    optb_series.fit(pd.Series(x), pd.Series(y))

    assert optb_list.splits == approx(optb_array.splits, rel=1e-12)
    assert optb_series.splits == approx(optb_array.splits, rel=1e-12)


def test_every_record_special_rejected():
    with raises(ValueError):
        optb = ContinuousOptimalBinning(special_codes=list(x))
        optb.fit(x, y)


def test_numerical_dtype_with_strings_rejected():
    with raises(ValueError):
        optb = ContinuousOptimalBinning()
        optb.fit(np.array(["a", "b"] * (n // 2)), y)


def test_max_n_prebins_minimum():
    optb = ContinuousOptimalBinning(max_n_prebins=2)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) <= 1


def test_transform_bins_and_show_digits():
    optb = ContinuousOptimalBinning(max_n_prebins=4)
    optb.fit(x, y)

    bins = optb.transform([-1.0, 1e6], metric="bins", show_digits=4)
    assert bins[0].startswith("(-inf,")
    assert bins[1].endswith("inf)")

    indices = optb.transform([-1.0, 1e6], metric="indices")
    assert indices[0] == 0
    assert indices[1] == len(optb.splits)

    with raises(ValueError):
        optb.transform(x, metric="bins", show_digits=9)


def test_fit_transform_indices():
    optb = ContinuousOptimalBinning(max_n_prebins=4)
    x_transform = optb.fit_transform(x, y, metric="indices")

    assert x_transform.min() == 0
    assert x_transform.max() == len(optb.splits)
    # x is sorted, so the assigned indices must be non-decreasing.
    assert np.all(np.diff(x_transform) >= 0)


def test_sample_weight_with_zeros():
    sample_weight = np.ones(n)
    sample_weight[:100] = 0.0

    optb = ContinuousOptimalBinning(max_n_prebins=5)
    optb.fit(x, y, sample_weight=sample_weight)

    assert optb.status == "OPTIMAL"
    assert bin_counts(optb).sum() == n - 100
