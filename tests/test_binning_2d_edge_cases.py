"""
OptimalBinning2D and ContinuousOptimalBinning2D edge-case testing.

Small synthetic grids that reach the branches the main 2D suites leave
untouched: the whole ``solver="mip"`` formulation, ``strategy="cart"``,
one-sided monotonic trends and bin-count bounds, the special-code and
missing masks in the preprocessing and transformation helpers, and the
guards of the 2D binning tables.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytest import approx, mark, raises

from optbinning import ContinuousOptimalBinning2D
from optbinning import OptimalBinning2D
from sklearn.exceptions import NotFittedError


# A small, well-separated synthetic grid. 2D fits are the slowest in the
# package, so every test in this module keeps the prebinning grid at 5x5 or
# below and the sample at a few hundred rows.
n_samples = 300

rng = np.random.RandomState(0)
x = rng.uniform(0, 10, n_samples)
y = rng.uniform(0, 10, n_samples)
_p = 1.0 / (1.0 + np.exp(-(0.6 * x + 0.4 * y - 5.0)))
z = (rng.uniform(size=n_samples) < _p).astype(int)
zc = 0.6 * x + 0.4 * y + rng.normal(0, 1, n_samples)

# Keyword bundle shared by every fit below.
prebins = {"max_n_prebins_x": 4, "max_n_prebins_y": 4}

# String labels for the categorical dtype paths.
xs = np.array(["a", "b", "c", "d"])[np.digitize(x, [2.5, 5.0, 7.5])]
ys = np.array(["p", "q", "r"])[np.digitize(y, [3.3, 6.6])]

# A XOR target: no axis carries the signal on its own, and the CART partition
# it induces contains single-class rectangles.
z_xor = ((x > 5) ^ (y > 5)).astype(int)


def _capture_logs(module):
    """Return (handler, messages) attached to a module logger.

    ``Logger`` sets ``propagate = False``, so pytest's ``caplog`` never sees
    these records; attach directly to the module logger instead.
    """
    messages = []

    class _Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Handler()
    logger = logging.getLogger(module)
    logger.addHandler(handler)
    return handler, messages, logger


# ---------------------------------------------------------------------------
# preprocessing_2d: check_input, non-numeric masks, special codes
# ---------------------------------------------------------------------------

def test_fit_check_input_rejects_inconsistent_length():
    optb = OptimalBinning2D(**prebins)

    with raises(ValueError, match="inconsistent numbers of samples"):
        optb.fit(x, y[:-1], z, check_input=True)


def test_fit_check_input_rejects_infinite_x():
    x_inf = x.copy()
    x_inf[:3] = np.inf

    optb = OptimalBinning2D(**prebins)

    with raises(ValueError, match="infinity"):
        optb.fit(x_inf, y, z, check_input=True)


def test_fit_check_input_rejects_nan_target():
    z_nan = z.astype(float)
    z_nan[:3] = np.nan

    optb = OptimalBinning2D(**prebins)

    with raises(ValueError, match="NaN"):
        optb.fit(x, y, z_nan, check_input=True)


def test_fit_check_input_accepts_clean_data():
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z, check_input=True)

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build().shape[1] == 10


def test_fit_check_input_continuous():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc, check_input=True)

    assert optb.status == "OPTIMAL"


def test_string_categorical_x_and_y():
    # Non-numeric x and y take the pandas null mask in split_data_2d rather
    # than np.isnan, and the categorical bin formatter in the binning table.
    optb = OptimalBinning2D(dtype_x="categorical", dtype_y="categorical",
                            **prebins)
    optb.fit(xs, ys, z)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()

    # Every clean bin lists categories, and the two trailing rows are the
    # special and missing buckets.
    assert list(df["Bin x"])[-3:-1] == ["Special", "Missing"]
    assert set(np.concatenate([b for b in df["Bin x"][:-3]])) <= set("abcd")

    splits_x, splits_y = optb.splits
    assert set(np.concatenate(splits_x)) <= set("abcd")
    assert set(np.concatenate(splits_y)) <= set("pqr")


def test_string_categorical_transform():
    optb = OptimalBinning2D(dtype_x="categorical", dtype_y="categorical",
                            **prebins)
    optb.fit(xs, ys, z)

    z_woe = optb.transform(xs, ys, metric="woe")
    z_er = optb.transform(xs, ys, metric="event_rate")

    assert z_woe.shape == (n_samples,)
    assert np.all((z_er >= 0) & (z_er <= 1))
    # WoE and event rate agree on which bin a record lands in.
    assert len(np.unique(z_woe)) == len(np.unique(z_er))


def test_special_codes_x_and_y_lists():
    x_sp = x.copy()
    x_sp[:20] = -1.0
    y_sp = y.copy()
    y_sp[20:35] = -2.0

    optb = OptimalBinning2D(special_codes_x=[-1.0], special_codes_y=[-2.0],
                            **prebins)
    optb.fit(x_sp, y_sp, z)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    # 20 + 15 rows are routed to the single Special bucket, none to Missing.
    assert df["Count"].values[-3] == 35
    assert df["Count"].values[-2] == 0
    assert df["Count"].values[-1] == n_samples


def test_special_codes_single_element_ndarray():
    x_sp = x.copy()
    x_sp[:20] = -1.0

    optb = OptimalBinning2D(special_codes_x=np.array([-1.0]), **prebins)
    optb.fit(x_sp, y, z)

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Count"].values[-3] == 20


def test_missing_values_in_x_and_target():
    x_nan = x.copy()
    x_nan[:10] = np.nan
    y_nan = y.copy()
    y_nan[10:16] = np.nan

    optb = OptimalBinning2D(**prebins)
    optb.fit(x_nan, y_nan, z)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].values[-2] == 16


# ---------------------------------------------------------------------------
# transformations_2d: metric_special / metric_missing, indices and bins
# ---------------------------------------------------------------------------

def test_transform_metric_special_variants():
    x_sp = x.copy()
    x_sp[:20] = -1.0

    optb = OptimalBinning2D(special_codes_x=[-1.0], **prebins)
    optb.fit(x_sp, y, z)

    special = slice(0, 20)

    z_empirical = optb.transform(x_sp, y, metric="woe",
                                 metric_special="empirical")
    empirical = np.unique(z_empirical[special])
    assert len(empirical) == 1

    z_value = optb.transform(x_sp, y, metric="woe", metric_special=0.5)
    assert np.unique(z_value[special]) == approx([0.5])

    z_indices = optb.transform(x_sp, y, metric="indices", metric_special=1)
    assert np.unique(z_indices[special]) == approx([1])

    # A non-integer metric_special under metric="indices" falls back to the
    # index of the special bin itself, which is n_bins.
    z_indices = optb.transform(x_sp, y, metric="indices",
                               metric_special="empirical")
    n_bins = len(optb.splits[0])
    assert np.unique(z_indices[special]) == approx([n_bins])

    z_bins = optb.transform(x_sp, y, metric="bins")
    assert list(np.unique(z_bins[special])) == ["Special"]


def test_transform_metric_missing_variants():
    x_nan = x.copy()
    x_nan[:10] = np.nan

    optb = OptimalBinning2D(**prebins)
    optb.fit(x_nan, y, z)

    missing = slice(0, 10)

    z_empirical = optb.transform(x_nan, y, metric="event_rate",
                                 metric_missing="empirical")
    assert len(np.unique(z_empirical[missing])) == 1

    z_value = optb.transform(x_nan, y, metric="woe", metric_missing=-3.5)
    assert np.unique(z_value[missing]) == approx([-3.5])

    z_bins = optb.transform(x_nan, y, metric="bins")
    assert list(np.unique(z_bins[missing])) == ["Missing"]

    n_bins = len(optb.splits[0])
    z_indices = optb.transform(x_nan, y, metric="indices",
                               metric_missing="empirical")
    assert np.unique(z_indices[missing]) == approx([n_bins + 1])


def test_continuous_transform_indices_and_bins():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)

    n_bins = len(optb.splits[0])

    z_indices = optb.transform(x, y, metric="indices")
    assert z_indices.dtype == np.dtype(int)
    assert z_indices.min() >= 0
    assert z_indices.max() < n_bins

    z_bins = optb.transform(x, y, metric="bins")
    assert z_bins.dtype == object
    assert all(r"$\cup$" in b for b in z_bins)


def test_continuous_transform_special_and_missing():
    x_sp = x.copy()
    x_sp[:15] = -1.0
    y_nan = y.copy()
    y_nan[15:25] = np.nan

    optb = ContinuousOptimalBinning2D(special_codes_x=[-1.0], **prebins)
    optb.fit(x_sp, y_nan, zc)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].values[-3] == 15
    assert df["Count"].values[-2] == 10
    # Non-empty special and missing buckets carry a target std.
    assert optb.binning_table.stds[-2] > 0
    assert optb.binning_table.stds[-1] > 0

    z_special = optb.transform(x_sp, y_nan, metric="mean",
                               metric_special="empirical")
    assert len(np.unique(z_special[:15])) == 1

    z_bins = optb.transform(x_sp, y_nan, metric="bins")
    assert list(np.unique(z_bins[:15])) == ["Special"]
    assert list(np.unique(z_bins[15:25])) == ["Missing"]


def test_transform_invalid_metric_continuous():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)

    with raises(ValueError, match="Invalid value for metric"):
        optb.transform(x, y, metric="woe")


def test_transform_rejects_bad_show_digits():
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    # _check_show_digits raises ValueError for both an out-of-range integer
    # and a non-integer.
    with raises(ValueError, match="show_digits must be an integer"):
        optb.transform(x, y, metric="bins", show_digits=9)

    with raises(ValueError, match="show_digits must be an integer"):
        optb.transform(x, y, metric="bins", show_digits="2")


# ---------------------------------------------------------------------------
# mip_2d and cp_2d: one-sided bin bounds, one-sided trends, infeasibility
# ---------------------------------------------------------------------------

def test_min_n_bins_only():
    for solver in ("mip", "cp"):
        optb = OptimalBinning2D(solver=solver, min_n_bins=4, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        assert len(optb.splits[0]) >= 4


def test_max_n_bins_only():
    for solver in ("mip", "cp"):
        optb = OptimalBinning2D(solver=solver, max_n_bins=3, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        assert len(optb.splits[0]) <= 3


def test_continuous_min_max_n_bins_only():
    for solver in ("mip", "cp"):
        optb = ContinuousOptimalBinning2D(solver=solver, min_n_bins=4,
                                          **prebins)
        optb.fit(x, y, zc)
        assert len(optb.splits[0]) >= 4

        optb = ContinuousOptimalBinning2D(solver=solver, max_n_bins=3,
                                          **prebins)
        optb.fit(x, y, zc)
        assert len(optb.splits[0]) <= 3


def test_monotonic_trend_x_only():
    ivs = []
    for solver in ("mip", "cp"):
        optb = OptimalBinning2D(solver=solver, monotonic_trend_x="ascending",
                                **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        optb.binning_table.build()
        ivs.append(optb.binning_table.iv)

    # Both formulations describe the same problem, so they agree on the
    # objective even where they pick a different optimal solution.
    assert ivs[0] == approx(ivs[1], rel=1e-6)


def test_monotonic_trend_y_only():
    ivs = []
    for solver in ("mip", "cp"):
        optb = OptimalBinning2D(solver=solver, monotonic_trend_y="descending",
                                **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        optb.binning_table.build()
        ivs.append(optb.binning_table.iv)

    assert ivs[0] == approx(ivs[1], rel=1e-6)


def test_continuous_monotonic_trend_one_sided():
    for solver in ("mip", "cp"):
        optb = ContinuousOptimalBinning2D(
            solver=solver, monotonic_trend_x="descending", **prebins)
        optb.fit(x, y, zc)
        assert optb.status == "OPTIMAL"

        optb = ContinuousOptimalBinning2D(
            solver=solver, monotonic_trend_y="ascending", **prebins)
        optb.fit(x, y, zc)
        assert optb.status == "OPTIMAL"


def test_min_n_bins_above_grid_is_infeasible():
    # No assignment of the 4x4 prebinning grid yields 1000 bins, so both
    # formulations report infeasibility and return an all-False solution
    # rather than raising.
    for solver in ("mip", "cp"):
        optb = OptimalBinning2D(solver=solver, min_n_bins=1000, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "INFEASIBLE"
        assert optb._solution.dtype == np.dtype(bool)
        assert not optb._solution.any()


def test_continuous_min_n_bins_above_grid_is_infeasible():
    for solver in ("mip", "cp"):
        optb = ContinuousOptimalBinning2D(solver=solver, min_n_bins=1000,
                                          **prebins)
        optb.fit(x, y, zc)

        assert optb.status == "INFEASIBLE"
        assert not optb._solution.any()


def test_n_jobs():
    # n_jobs > 1 switches the CP solver from linearization to parallel
    # search workers; joblib resolves -1 to every core.
    for n_jobs in (2, -1):
        optb = OptimalBinning2D(solver="cp", n_jobs=n_jobs, **prebins)
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

    optb = OptimalBinning2D(solver="mip", n_jobs=2, **prebins)
    optb.fit(x, y, z)
    assert optb.status == "OPTIMAL"


# ---------------------------------------------------------------------------
# model_data_2d: bin size and bin count bounds, divergences
# ---------------------------------------------------------------------------

def test_bin_count_and_size_bounds():
    optb = OptimalBinning2D(min_bin_size=0.02, max_bin_size=0.6,
                            min_bin_n_event=5, max_bin_n_event=120,
                            min_bin_n_nonevent=5, max_bin_n_nonevent=120,
                            **prebins)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build(add_totals=False)
    clean = df.iloc[:-2]

    assert clean["Count"].min() >= np.ceil(0.02 * n_samples)
    assert clean["Count"].max() <= np.ceil(0.6 * n_samples)
    assert clean["Event"].min() >= 5
    assert clean["Event"].max() <= 120
    assert clean["Non-event"].min() >= 5
    assert clean["Non-event"].max() <= 120


def test_bin_event_count_bounds_only():
    # With no size bounds set, the event and non-event caps are the filters
    # that reject the large rectangles.
    optb = OptimalBinning2D(max_bin_n_event=60, max_bin_n_nonevent=60,
                            **prebins)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    clean = optb.binning_table.build(add_totals=False).iloc[:-2]
    assert clean["Event"].max() <= 60
    assert clean["Non-event"].max() <= 60


def test_continuous_bin_size_bounds():
    optb = ContinuousOptimalBinning2D(min_bin_size=0.05, max_bin_size=0.5,
                                      **prebins)
    optb.fit(x, y, zc)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build(add_totals=False)
    clean = df.iloc[:-2]

    assert clean["Count"].min() >= np.ceil(0.05 * n_samples)
    assert clean["Count"].max() <= np.ceil(0.5 * n_samples)


def test_divergences():
    ivs = {}
    for divergence in ("iv", "js", "hellinger", "triangular"):
        optb = OptimalBinning2D(divergence=divergence, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        ivs[divergence] = optb.binning_table.iv

    # The binning table always reports the Jeffrey IV; a different divergence
    # only changes what the solver maximises, so the reported IV cannot beat
    # the one the "iv" objective attains.
    for divergence in ("js", "hellinger", "triangular"):
        assert ivs[divergence] <= ivs["iv"] + 1e-9


# ---------------------------------------------------------------------------
# model_data_cart_2d: strategy="cart" under every option
# ---------------------------------------------------------------------------

def test_cart_solvers():
    ivs = []
    for solver in ("mip", "cp"):
        optb = OptimalBinning2D(strategy="cart", solver=solver, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        optb.binning_table.build()
        ivs.append(optb.binning_table.iv)

    assert ivs[0] == approx(ivs[1], rel=1e-6)


def test_cart_bin_size_bounds():
    optb = OptimalBinning2D(strategy="cart", min_bin_size=0.15,
                            max_bin_size=0.8, **prebins)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    clean = optb.binning_table.build(add_totals=False).iloc[:-2]
    assert clean["Count"].min() >= np.ceil(0.15 * n_samples)
    assert clean["Count"].max() <= np.ceil(0.8 * n_samples)


def test_cart_bin_count_floors():
    optb = OptimalBinning2D(strategy="cart", min_bin_n_event=40,
                            min_bin_n_nonevent=40, **prebins)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    clean = optb.binning_table.build(add_totals=False).iloc[:-2]
    assert clean["Event"].min() >= 40
    assert clean["Non-event"].min() >= 40


def test_cart_bin_count_bounds():
    # Both per-bin count caps drop rectangles inside model_data_cart, and a
    # cover of the remainder still exists: the caps bind, the fit does not
    # fail. The free grid reaches the same answer.
    bounds = {"max_bin_n_event": 70, "max_bin_n_nonevent": 70,
              "min_bin_n_event": 5, "min_bin_n_nonevent": 5}

    for strategy in ("cart", "grid"):
        optb = OptimalBinning2D(strategy=strategy, **bounds, **prebins)
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        clean = optb.binning_table.build(add_totals=False).iloc[:-2]
        assert clean["Event"].max() <= 70
        assert clean["Non-event"].max() <= 70
        assert clean["Event"].min() >= 5
        assert clean["Non-event"].min() >= 5


def test_cart_bin_size_cap_can_be_infeasible():
    # A cart bin is a union of two or more of the tree's leaves, never a
    # single one, so a size cap that only a single grid cell can meet has no
    # admissible cover -- while the free grid, which may bin one cell on its
    # own, satisfies it.
    optb_cart = OptimalBinning2D(strategy="cart", max_bin_size=0.12,
                                 **prebins)
    optb_cart.fit(x, y, z)
    assert optb_cart.status == "INFEASIBLE"

    optb_grid = OptimalBinning2D(strategy="grid", max_bin_size=0.12,
                                 **prebins)
    optb_grid.fit(x, y, z)
    assert optb_grid.status == "OPTIMAL"

    clean = optb_grid.binning_table.build(add_totals=False).iloc[:-2]
    assert clean["Count"].max() <= np.ceil(0.12 * n_samples)


def test_cart_single_class_rectangles():
    # A XOR target makes whole cart rectangles single-class; those carry no
    # divergence and are dropped before the model is built.
    optb = OptimalBinning2D(strategy="cart", max_n_prebins_x=5,
                            max_n_prebins_y=5)
    optb.fit(x, y, z_xor)

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    iv_cart = optb.binning_table.iv

    optb_grid = OptimalBinning2D(max_n_prebins_x=5, max_n_prebins_y=5)
    optb_grid.fit(x, y, z_xor)
    optb_grid.binning_table.build()

    # cart searches a subset of the rectangles the free grid searches, so it
    # cannot beat it.
    assert 0 < iv_cart <= optb_grid.binning_table.iv + 1e-9


def test_cart_monotonic_trends():
    for kwargs in ({"monotonic_trend_x": "ascending"},
                   {"monotonic_trend_y": "descending"},
                   {"monotonic_trend_x": "descending",
                    "monotonic_trend_y": "descending"}):
        optb = OptimalBinning2D(strategy="cart", **kwargs, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        optb.binning_table.build()
        assert optb.binning_table.iv >= 0


def test_cart_divergences():
    for divergence in ("js", "hellinger", "triangular"):
        optb = OptimalBinning2D(strategy="cart", divergence=divergence,
                                **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"


def test_continuous_cart_solvers():
    ivs = []
    for solver in ("mip", "cp"):
        optb = ContinuousOptimalBinning2D(strategy="cart", solver=solver,
                                          **prebins)
        optb.fit(x, y, zc)

        assert optb.status == "OPTIMAL"
        optb.binning_table.build()
        ivs.append(optb.binning_table.iv)

    assert ivs[0] == approx(ivs[1], rel=1e-6)


def test_continuous_cart_bin_size_bounds():
    optb = ContinuousOptimalBinning2D(strategy="cart", min_bin_size=0.15,
                                      max_bin_size=0.5, **prebins)
    optb.fit(x, y, zc)

    assert optb.status == "OPTIMAL"

    clean = optb.binning_table.build(add_totals=False).iloc[:-2]
    assert clean["Count"].min() >= np.ceil(0.15 * n_samples)
    assert clean["Count"].max() <= np.ceil(0.5 * n_samples)


def test_continuous_cart_monotonic_trends():
    for kwargs in ({"monotonic_trend_x": "ascending"},
                   {"monotonic_trend_y": "descending"},
                   {"monotonic_trend_x": "ascending",
                    "monotonic_trend_y": "ascending"}):
        optb = ContinuousOptimalBinning2D(strategy="cart", **kwargs, **prebins)
        optb.fit(x, y, zc)

        assert optb.status == "OPTIMAL"


# ---------------------------------------------------------------------------
# prebinning methods and other documented-but-untested keywords
# ---------------------------------------------------------------------------

def test_prebinning_methods():
    for method in ("cart", "mdlp", "quantile", "uniform"):
        optb = OptimalBinning2D(prebinning_method=method, **prebins)
        optb.fit(x, y, z)

        assert optb.status == "OPTIMAL"
        assert len(optb.splits[0]) >= 1


def test_continuous_prebinning_methods():
    for method in ("cart", "quantile", "uniform"):
        optb = ContinuousOptimalBinning2D(prebinning_method=method, **prebins)
        optb.fit(x, y, zc)

        assert optb.status == "OPTIMAL"


def test_min_event_rate_diff():
    optb = OptimalBinning2D(min_event_rate_diff_x=0.2,
                            min_event_rate_diff_y=0.2,
                            monotonic_trend_x="ascending",
                            monotonic_trend_y="ascending", **prebins)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"


def test_continuous_min_mean_diff():
    optb = ContinuousOptimalBinning2D(min_mean_diff_x=1.0, min_mean_diff_y=1.0,
                                      monotonic_trend_x="ascending",
                                      monotonic_trend_y="ascending", **prebins)
    optb.fit(x, y, zc)

    assert optb.status == "OPTIMAL"


def test_time_limit_is_accepted():
    optb = OptimalBinning2D(time_limit=1, **prebins)
    optb.fit(x, y, z)

    assert optb.status in ("OPTIMAL", "FEASIBLE")


def test_gamma_regularization_mip():
    optb = ContinuousOptimalBinning2D(gamma=600, min_bin_size=0.05,
                                      solver="mip", **prebins)
    optb.fit(x, y, zc)

    assert optb.status == "OPTIMAL"

    optb_plain = ContinuousOptimalBinning2D(min_bin_size=0.05, solver="mip",
                                            **prebins)
    optb_plain.fit(x, y, zc)

    # Regularization cannot increase the number of bins.
    assert len(optb.splits[0]) <= len(optb_plain.splits[0])


# ---------------------------------------------------------------------------
# binning_statistics_2d: build, plot and analysis guards
# ---------------------------------------------------------------------------

def test_build_show_bin_xy():
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    df = optb.binning_table.build(show_bin_xy=True)

    assert "Bin" in df.columns
    assert "Bin x" not in df.columns
    assert list(df["Bin"])[-3:-1] == ["Special", "Missing"]
    assert df["Count"].values[-1] == n_samples
    assert all(r"$\cup$" in b for b in df["Bin"][:-3])


def test_build_show_bin_xy_without_totals():
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    df = optb.binning_table.build(show_bin_xy=True, add_totals=False)
    assert "Totals" not in df.index


def test_continuous_build_show_bin_xy():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)

    df = optb.binning_table.build(show_bin_xy=True)

    assert "Bin" in df.columns
    assert "Bin x" not in df.columns
    assert df["Count"].values[-1] == n_samples
    assert df["Sum"].values[-1] == approx(zc.sum(), rel=1e-9)


def test_continuous_build_rejects_bad_show_bin_xy():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)

    with raises(TypeError, match="show_bin_xy"):
        optb.binning_table.build(show_bin_xy=1)


def test_plot_savefig_guards(tmp_path):
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)
    optb.binning_table.build()

    with raises(TypeError, match="savefig must be a string path"):
        optb.binning_table.plot(savefig=1)

    with raises(TypeError, match="save_kwargs must be a dictionary"):
        optb.binning_table.plot(savefig=str(tmp_path / "a.png"),
                                save_kwargs=1)

    path = tmp_path / "binning_2d_edge.png"
    optb.binning_table.plot(savefig=str(path), save_kwargs={"dpi": 50})
    assert path.exists()


def test_plot_show(monkeypatch):
    # savefig=None displays the figure; stub plt.show so it cannot block.
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **kw: shown.append(True))

    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)
    optb.binning_table.build()
    optb.binning_table.plot(metric="event_rate")

    assert shown == [True]
    plt.close("all")


def test_plot_before_build_raises():
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    with raises(NotFittedError, match="not built yet"):
        optb.binning_table.plot()


def test_continuous_plot_savefig_guard(tmp_path):
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)
    optb.binning_table.build()

    with raises(TypeError, match="savefig must be a string path"):
        optb.binning_table.plot(savefig=1)

    path = tmp_path / "continuous_binning_2d_edge.png"
    optb.binning_table.plot(savefig=str(path))
    assert path.exists()


def test_continuous_plot_show(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **kw: shown.append(True))

    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)
    optb.binning_table.build()
    optb.binning_table.plot()

    assert shown == [True]
    plt.close("all")


def test_continuous_plot_before_build_raises():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)

    with raises(NotFittedError, match="not built yet"):
        optb.binning_table.plot()


def test_analysis_fisher():
    optb = OptimalBinning2D(max_n_bins=4, **prebins)
    optb.fit(x, y, z)
    optb.binning_table.build()
    optb.binning_table.analysis(pvalue_test="fisher", n_samples=20,
                                print_output=False)

    assert 0.0 <= optb.binning_table.quality_score <= 1.0


def test_analysis_single_bin():
    # A single optimal bin leaves no consecutive pair to test, so Cramer's V
    # falls back to 0 and the significance table is empty.
    optb = OptimalBinning2D(max_n_bins=1, **prebins)
    optb.fit(x, y, z)

    df = optb.binning_table.build()
    assert len(df) == 4

    optb.binning_table.analysis(print_output=False)
    assert optb.binning_table.iv == approx(0.0)
    assert optb.binning_table.quality_score == approx(0.0)


def test_continuous_analysis_single_bin():
    optb = ContinuousOptimalBinning2D(max_n_bins=1, **prebins)
    optb.fit(x, y, zc)
    optb.binning_table.build()
    optb.binning_table.analysis(print_output=False)

    assert optb.binning_table.iv == approx(0.0)


def test_continuous_analysis_zero_total_mean():
    # The normalized WoE divides by the total mean; an exactly zero mean
    # takes the unnormalized branch instead.
    z_zero = np.round(zc)
    z_zero[0] -= z_zero.sum()

    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, z_zero)
    optb.binning_table.build()

    assert optb.binning_table._t_mean == 0.0

    optb.binning_table.analysis(print_output=False)
    assert optb.binning_table.quality_score >= 0.0


# ---------------------------------------------------------------------------
# unfitted access and degenerate inputs
# ---------------------------------------------------------------------------

def test_unfitted_access():
    for optb in (OptimalBinning2D(), ContinuousOptimalBinning2D()):
        with raises(NotFittedError):
            optb.binning_table
        with raises(NotFittedError):
            optb.splits
        with raises(NotFittedError):
            optb.status
        with raises(NotFittedError):
            optb.information()
        with raises(NotFittedError):
            optb.transform(x, y)


def test_constant_x_only():
    # A single distinct value on one axis is not degenerate: the other axis
    # still splits, and the fit succeeds.
    optb = OptimalBinning2D(**prebins)
    optb.fit(np.ones(n_samples), y, z)

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Count"].values[-1] == n_samples


def test_duplicated_and_extreme_values():
    x_dup = np.repeat(np.arange(10.0), n_samples // 10)
    y_big = y * 1e12

    optb = OptimalBinning2D(**prebins)
    optb.fit(x_dup, y_big, z)

    assert optb.status == "OPTIMAL"

    splits_x, splits_y = optb.splits
    for bins in (splits_x, splits_y):
        for lo, hi in bins:
            assert lo < hi


def test_two_row_grid():
    # min_prebin_size at its maximum leaves at most two prebins per axis.
    optb = OptimalBinning2D(min_prebin_size_x=0.5, min_prebin_size_y=0.5,
                            **prebins)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits[0]) >= 1


def test_accepts_lists_and_pandas_series():
    optb_array = OptimalBinning2D(**prebins)
    optb_array.fit(x, y, z)
    optb_array.binning_table.build()

    optb_list = OptimalBinning2D(**prebins)
    optb_list.fit(list(x), list(y), list(z))
    optb_list.binning_table.build()

    optb_series = OptimalBinning2D(**prebins)
    optb_series.fit(pd.Series(x), pd.Series(y), pd.Series(z))
    optb_series.binning_table.build()

    assert optb_list.binning_table.iv == approx(optb_array.binning_table.iv)
    assert optb_series.binning_table.iv == approx(optb_array.binning_table.iv)


def test_transform_on_a_shorter_array():
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    assert optb.transform(x[:10], y[:10]).shape == (10,)

    # Records that were never missing at fit time still route to the missing
    # bucket at transform time.
    z_nan = optb.transform(np.full(5, np.nan), y[:5])
    assert z_nan == approx([0.0] * 5)


def test_infinite_values_are_rejected_without_check_input():
    x_inf = x.copy()
    x_inf[:3] = np.inf

    optb = OptimalBinning2D(**prebins)

    # check_input=False skips sklearn's validation in split_data_2d, but the
    # prebinning tree rejects the same data a step later.
    with raises(ValueError, match="infinity"):
        optb.fit(x_inf, y, z)


def test_verbose_categorical_and_cart():
    handler, messages, logger = _capture_logs(
        "optbinning.binning.multidimensional.binning_2d")
    try:
        optb = OptimalBinning2D(verbose=True, strategy="cart",
                                dtype_x="categorical", dtype_y="categorical",
                                **prebins)
        optb.fit(xs, ys, z)
    finally:
        logger.removeHandler(handler)

    assert optb.status == "OPTIMAL"
    assert "Pre-processing: number of categories in x: 4" in messages
    assert "Pre-processing: number of categories in y: 3" in messages
    assert "Prebinning: applying strategy cart..." in messages


def test_continuous_verbose_categorical_and_cart():
    handler, messages, logger = _capture_logs(
        "optbinning.binning.multidimensional.continuous_binning_2d")
    try:
        optb = ContinuousOptimalBinning2D(
            verbose=True, strategy="cart", dtype_x="categorical",
            dtype_y="categorical", **prebins)
        optb.fit(xs, ys, zc)
    finally:
        logger.removeHandler(handler)

    assert optb.status == "OPTIMAL"
    assert "Pre-processing: number of categories in x: 4" in messages
    assert "Pre-processing: number of categories in y: 3" in messages
    assert "Prebinning: applying strategy cart..." in messages


# ---------------------------------------------------------------------------
# Defects. Each of the tests below asserts the behaviour the docstring, the
# parameter validation or the 1D sibling promises, and is red against the
# source as it stands.
# ---------------------------------------------------------------------------

def test_defect_verbose_logs_trend_x_under_trend_y():
    # _fit_optimizer formats self.monotonic_trend_x into the "monotonic trend
    # y" message, in both binning_2d.py and continuous_binning_2d.py.
    handler, messages, logger = _capture_logs(
        "optbinning.binning.multidimensional.binning_2d")
    try:
        optb = OptimalBinning2D(verbose=True, monotonic_trend_x="ascending",
                                monotonic_trend_y="descending", **prebins)
        optb.fit(x, y, z)
    finally:
        logger.removeHandler(handler)

    assert "Optimizer: monotonic trend x set to ascending." in messages
    assert "Optimizer: monotonic trend y set to descending." in messages


def test_defect_continuous_verbose_logs_trend_x_under_trend_y():
    handler, messages, logger = _capture_logs(
        "optbinning.binning.multidimensional.continuous_binning_2d")
    try:
        optb = ContinuousOptimalBinning2D(
            verbose=True, monotonic_trend_x="ascending",
            monotonic_trend_y="descending", **prebins)
        optb.fit(x, y, zc)
    finally:
        logger.removeHandler(handler)

    assert "Optimizer: monotonic trend y set to descending." in messages


def test_defect_analysis_before_build():
    # All three 1D binning tables guard analysis() with _check_is_built and
    # raise NotFittedError; the two 2D tables reach for self._paths_x and
    # raise AttributeError instead.
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    with raises(NotFittedError, match="not built yet"):
        optb.binning_table.analysis(print_output=False)


def test_defect_continuous_analysis_before_build():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, zc)

    with raises(NotFittedError, match="not built yet"):
        optb.binning_table.analysis(print_output=False)


def test_defect_special_codes_ndarray_transform():
    # _check_parameters accepts "a list or numpy.ndarray" and fit() honours
    # it, but _apply_transform tests `if special_codes_x or special_codes_y`,
    # which raises on any ndarray holding more than one code.
    x_sp = x.copy()
    x_sp[:20] = -1.0
    x_sp[20:30] = -2.0

    optb = OptimalBinning2D(special_codes_x=np.array([-1.0, -2.0]), **prebins)
    optb.fit(x_sp, y, z)

    z_transform = optb.transform(x_sp, y, metric="woe", metric_special=0.25)
    assert np.unique(z_transform[:30]) == approx([0.25])


def test_defect_continuous_special_codes_ndarray_transform():
    x_sp = x.copy()
    x_sp[:20] = -1.0
    x_sp[20:30] = -2.0

    optb = ContinuousOptimalBinning2D(special_codes_x=np.array([-1.0, -2.0]),
                                      **prebins)
    optb.fit(x_sp, y, zc)

    z_transform = optb.transform(x_sp, y, metric="mean", metric_special=0.25)
    assert np.unique(z_transform[:30]) == approx([0.25])


def test_defect_constant_x_and_y():
    # np.array(rows, dtype=object) is rectangular when every rectangle spans
    # the same number of grid cells -- as it does for the 1x1 grid a pair of
    # constant variables produces -- so `P[r] = i` indexes with an object
    # array and raises IndexError. OptimalBinning and ContinuousOptimalBinning
    # both fit a constant x and report OPTIMAL with no splits.
    optb = OptimalBinning2D(**prebins)
    optb.fit(np.ones(n_samples), np.ones(n_samples), z)

    assert optb.status == "OPTIMAL"


def test_defect_continuous_constant_x_and_y():
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(np.ones(n_samples), np.ones(n_samples), zc)

    assert optb.status == "OPTIMAL"


@mark.parametrize("cls, target", [("binary", "z"), ("continuous", "zc")])
def test_defect_cart_on_a_single_cell_grid(cls, target):
    # Two defects on one input. strategy="cart" bounds its tree with
    # clf_nodes = n_splits_x * n_splits_y, which is zero as soon as *one*
    # axis carries no split -- here because both variables are constant --
    # and sklearn rejects max_leaf_nodes=0 outright. Past that, a one-cell
    # grid leaves the tree with no split at all, and get_rectangles walks a
    # parent node that is not there. The grid strategy fits the same input
    # (test_defect_constant_x_and_y).
    estimator = OptimalBinning2D if cls == "binary" else \
        ContinuousOptimalBinning2D
    zt = z if target == "z" else zc

    optb = estimator(strategy="cart", **prebins)
    optb.fit(np.ones(n_samples), np.ones(n_samples), zt)

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Count"].iloc[0] == n_samples


def test_defect_cart_with_one_unsplit_axis():
    # The same clf_nodes zero on a grid that is not degenerate at all: y is
    # constant, x is not, so the grid is 4x1 and the tree does have a split
    # to make on x. Flooring the product at 2 got past sklearn's
    # max_leaf_nodes bound but left the tree with exactly two leaves, and
    # model_data_cart only admits a rectangle that merges two or more of
    # them -- so the only candidate was their union, the whole grid: one
    # bin, zero IV, status OPTIMAL and nothing discretised. The budget is
    # the grid's cell count, four here, which buys two bins. It cannot buy
    # more than one on a grid of three cells or fewer, whatever the budget:
    # see test_defect_cart_leaf_budget_on_a_small_grid.
    optb = OptimalBinning2D(strategy="cart", **prebins)
    optb.fit(x, np.ones(n_samples), z)

    assert optb.status == "OPTIMAL"
    assert optb._n_prebins > 1

    df = optb.binning_table.build()
    assert len(optb.splits[0]) > 1
    assert df["Count"].iloc[0] < n_samples
    assert df.loc["Totals", "IV"] > 0

    # The grid strategy searches every rectangle, so it separates x too.
    optb_grid = OptimalBinning2D(**prebins)
    optb_grid.fit(x, np.ones(n_samples), z)

    assert optb_grid.binning_table.build().loc["Totals", "IV"] > 0


def test_defect_continuous_cart_with_one_unsplit_axis():
    # The continuous twin of the leaf budget above: the same expression
    # sits in ContinuousOptimalBinning2D._fit, feeding a
    # DecisionTreeRegressor.
    optb = ContinuousOptimalBinning2D(strategy="cart", **prebins)
    optb.fit(x, np.ones(n_samples), zc)

    assert optb.status == "OPTIMAL"
    assert optb._n_prebins > 1

    df = optb.binning_table.build()
    assert len(optb.splits[0]) > 1
    assert df["Count"].iloc[0] < n_samples
    assert df.loc["Totals", "IV"] > 0


@mark.parametrize("caps", [(2, 3), (3, 2), (4, 2), (2, 4)])
def test_defect_cart_leaf_budget_on_a_small_grid(caps):
    # Nothing is degenerate here: both axes are informative and the grid has
    # six to eight cells. But upstream's clf_nodes = n_splits_x * n_splits_y
    # is 2 or 3 at these prebinning caps, and a cart bin merges two or more
    # of the tree's leaves -- so b bins cost 2b leaves and a budget under
    # four can only return their union, the whole grid: one bin, zero IV,
    # status OPTIMAL and nothing discretised. Flooring the product at 2
    # covered only the caps that drive it below 2. The budget is the finest
    # partition the grid admits, its cell count.
    cx, cy = caps
    caps_kw = {"max_n_prebins_x": cx, "max_n_prebins_y": cy}

    optb = OptimalBinning2D(strategy="cart", **caps_kw)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].iloc[0] < n_samples
    assert df.loc["Totals", "IV"] > 0

    # cart searches a subset of the rectangles the free grid searches, so it
    # cannot beat it -- but it must not be left at the trivial cover either.
    optb_grid = OptimalBinning2D(**caps_kw)
    optb_grid.fit(x, y, z)
    optb_grid.binning_table.build()

    assert 0 < optb.binning_table.iv <= optb_grid.binning_table.iv + 1e-9


@mark.parametrize("caps", [(2, 3), (3, 2), (4, 2), (2, 4)])
def test_defect_continuous_cart_leaf_budget_on_a_small_grid(caps):
    # The continuous twin: the same expression feeds a DecisionTreeRegressor
    # in ContinuousOptimalBinning2D._fit.
    cx, cy = caps
    caps_kw = {"max_n_prebins_x": cx, "max_n_prebins_y": cy}

    optb = ContinuousOptimalBinning2D(strategy="cart", **caps_kw)
    optb.fit(x, y, zc)

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].iloc[0] < n_samples
    assert optb.binning_table.iv > 0


def test_defect_cart_one_rectangle():
    # The same object-array bug on a perfectly ordinary input: a target that
    # depends on y alone leaves the cart strategy with a single admissible
    # rectangle, so np.array(rows, dtype=object) is rectangular again. The
    # grid strategy fits the same data.
    z_y = (y > 6).astype(int)

    optb_grid = OptimalBinning2D(max_n_prebins_x=5, max_n_prebins_y=5)
    optb_grid.fit(x, y, z_y)
    assert optb_grid.status == "OPTIMAL"

    optb = OptimalBinning2D(strategy="cart", max_n_prebins_x=5,
                            max_n_prebins_y=5)
    optb.fit(x, y, z_y)

    assert optb.status == "OPTIMAL"


@mark.parametrize("value, rate", [(0, 0.0), (1, 1.0)])
def test_defect_single_class_target(value, rate):
    # A target carrying a single class is degenerate but legal: the 1D
    # OptimalBinning fits it, reports OPTIMAL and builds a one-bin table.
    # Every 2D rectangle is pure in that case, and model_data keeps only
    # mixed rectangles, so the whole model came out empty and fit() died --
    # first on sklearn's "Found array with 0 sample(s)" from jeffrey(), then
    # on a ValueError naming the target. The optimizer now returns the whole
    # grid as the single bin, exactly as the 1D estimator does.
    z_single = np.full(n_samples, value, dtype=int)

    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z_single)

    assert optb.status == "OPTIMAL"

    splits_x, splits_y = optb.splits
    assert len(splits_x) == 1
    assert splits_x[0] == approx([-np.inf, np.inf])
    assert splits_y[0] == approx([-np.inf, np.inf])

    table = optb.binning_table
    df = table.build()

    assert df["Count"][0] == n_samples
    # The event rate is gated on records, not on mixedness: an all-event bin
    # reports 1.0, an all-non-event bin 0.0. Never 0.0 for both.
    assert df["Event rate"][0] == rate
    assert df["Event"][0] == n_samples * value
    assert df["Non-event"][0] == n_samples * (1 - value)

    # WoE, IV and JS stay gated on mixedness, so they are zero.
    assert df["WoE"][0] == 0.0
    assert df["IV"][0] == 0.0
    assert df["JS"][0] == 0.0
    assert df.loc["Totals", "IV"] == 0.0
    assert df.loc["Totals", "JS"] == 0.0
    assert df.loc["Totals", "Event rate"] == rate

    assert table.iv == 0.0
    assert table.js == 0.0
    assert table.hellinger == 0.0
    assert table.triangular == 0.0
    assert table.ks == 0.0
    # gini is 0, not nan: metrics.gini already returns 0 for a single
    # populated bin.
    assert table.gini == 0.0

    # transform still maps every record onto the single bin, and answers
    # what the table answers: transform_binary_target in
    # optbinning/binning/multidimensional/transformations_2d.py splits the
    # record gate from the mixedness gate the same way build() does.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.all(optb.transform(x, y, metric="woe") == 0.0)
        assert np.all(optb.transform(x, y, metric="event_rate") == rate)


@mark.parametrize("value", [0, 1])
def test_defect_single_class_target_no_warning(value, tmp_path):
    # A degenerate target must not leak a RuntimeWarning out of the divides
    # in build(), plot() and analysis(): they are guarded, not silenced. The
    # WoE matrix plot() draws is all zeros, like the WoE column, instead of
    # the -inf that log(1 / 1 - 1) gives.
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, np.full(n_samples, value, dtype=int))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        optb.binning_table.build()
        optb.binning_table.plot(
            metric="woe", savefig=str(tmp_path / "single_class.png"))
        optb.binning_table.analysis(print_output=False)

    assert np.all(optb.binning_table._W == 0.0)
    assert optb.binning_table.quality_score == 0.0


@mark.parametrize("strategy", ["grid", "cart"])
def test_defect_single_class_target_strategies(strategy):
    # The guard sits ahead of the strategy branch, so model_data_cart never
    # sees the empty model either.
    optb = OptimalBinning2D(strategy=strategy, **prebins)
    optb.fit(x, y, np.ones(n_samples, dtype=int))

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Event rate"][0] == 1.0


def test_defect_single_class_target_unsupervised_prebinning():
    # "quantile" prebinning ignores the target, so the grid is several cells
    # wide even though every one of its cells is pure. The 1D estimator
    # merges the pure prebins away and reports a single bin; so must this
    # one.
    optb = OptimalBinning2D(prebinning_method="quantile", **prebins)
    optb.fit(x, y, np.ones(n_samples, dtype=int))

    assert optb.status == "OPTIMAL"
    assert optb._n_prebins > 1

    df = optb.binning_table.build()
    assert len(optb.splits[0]) == 1
    assert df["Count"][0] == n_samples
    assert df["Event rate"][0] == 1.0
    assert df.loc["Totals", "IV"] == 0.0


def test_single_class_target_is_logged():
    # The solver is skipped, so verbose says why rather than reporting a run
    # that did not happen.
    handler, messages, logger = _capture_logs(
        "optbinning.binning.multidimensional.binning_2d")
    try:
        optb = OptimalBinning2D(verbose=True, **prebins)
        optb.fit(x, y, np.ones(n_samples, dtype=int))
    finally:
        logger.removeHandler(handler)

    assert optb.status == "OPTIMAL"
    assert "Optimizer: target contains a single class." in messages
    assert "Optimizer: solver not run." in messages


def test_continuous_constant_target_is_a_single_bin():
    # The continuous analogue of a single-class target. This is a pin, not a
    # regression test: continuous_model_data drops only *empty* rectangles,
    # so a constant target has never been degenerate for it. It exists so the
    # binary guard above is not later copied onto a path that does not need
    # it.
    optb = ContinuousOptimalBinning2D(**prebins)
    optb.fit(x, y, np.full(n_samples, 3.0))

    assert optb.status == "OPTIMAL"

    df = optb.binning_table.build()
    assert df["Count"].iloc[0] == n_samples
    assert df["Mean"].iloc[0] == approx(3.0)


def test_defect_no_admissible_rectangle():
    optb = OptimalBinning2D(min_bin_n_event=100000, **prebins)

    with raises(ValueError, match="No bin candidate") as excinfo:
        optb.fit(x, y, z)

    # The bin bounds are the only cause left. A single-class target is
    # answered by _fit_optimizer before model_data runs
    # (test_defect_single_class_target), and with both classes present the
    # whole grid is always a mixed rectangle -- so the message must stop
    # sending the user to look at the target.
    assert "both classes" not in str(excinfo.value)
    assert "min_bin_n_event" in str(excinfo.value)


def test_defect_cart_no_admissible_rectangle():
    optb = OptimalBinning2D(strategy="cart", min_bin_n_event=100000,
                            **prebins)

    with raises(ValueError, match="No bin candidate") as excinfo:
        optb.fit(x, y, z)

    assert "both classes" not in str(excinfo.value)
    assert "min_bin_n_event" in str(excinfo.value)


def test_defect_split_digits_is_ignored():
    # "If split_digits is set to 0, the split points are integers." The 1D
    # estimators round in _fit_prebinning; neither 2D estimator reads
    # self.split_digits anywhere outside _check_parameters.
    optb = OptimalBinning2D(split_digits=0, **prebins)
    optb.fit(x, y, z)

    splits = np.unique(np.concatenate(optb.splits[0]))
    finite = splits[np.isfinite(splits)]

    assert len(finite)
    assert finite == approx(np.round(finite))


def test_defect_continuous_split_digits_is_ignored():
    optb = ContinuousOptimalBinning2D(split_digits=0, **prebins)
    optb.fit(x, y, zc)

    splits = np.unique(np.concatenate(optb.splits[0]))
    finite = splits[np.isfinite(splits)]

    assert len(finite)
    assert finite == approx(np.round(finite))


def test_defect_split_digits_leaves_categorical_alone():
    # A categorical axis reaches the prebinning as the ordinal encoding of
    # its categories, so rounding it would regroup categories rather than
    # shorten a split value. split_digits must not touch it. This is a pin,
    # not a regression test: it has never been red, and it exists so the
    # rounding above is not later widened to every dtype.
    kwargs = dict(dtype_x="categorical", dtype_y="categorical", **prebins)

    optb = OptimalBinning2D(**kwargs)
    optb.fit(xs, ys, z)

    optb_digits = OptimalBinning2D(split_digits=0, **kwargs)
    optb_digits.fit(xs, ys, z)

    assert optb_digits.status == "OPTIMAL"
    assert len(optb_digits.splits[0]) == len(optb.splits[0])


def test_continuous_prebinning_matrix_counts_are_exact_integers():
    # Why R is an int matrix and what that costs. np.count_nonzero returns an
    # exact record count, so int carries everything float64 carried; the
    # gamma test below is what needs it to be int.
    optb = ContinuousOptimalBinning2D(**prebins)
    empty = np.array([])

    R, S, SS = optb._prebinning_matrices(
        np.array([3.0, 6.0]), np.array([5.0]), x, y, zc,
        empty, empty, empty, empty, empty, empty)

    assert R.dtype.kind == "i"
    assert R.sum() == n_samples
    assert S.sum() == approx(zc.sum())


@mark.parametrize("strategy", ["grid", "cart"])
def test_defect_continuous_gamma_with_cp_solver(strategy):
    # gamma is documented and validated on ContinuousOptimalBinning2D, but
    # _prebinning_matrices built a float record-count matrix where the binary
    # estimator builds an int one, so continuous_model_data returned a float
    # n_records and Binning2DCP multiplied it into a CP-SAT linear
    # constraint. solver="mip" fits, and the binary OptimalBinning2D fits
    # under both solvers.
    optb = ContinuousOptimalBinning2D(gamma=600, min_bin_size=0.05,
                                      solver="cp", strategy=strategy,
                                      **prebins)
    optb.fit(x, y, zc)

    unregularized = ContinuousOptimalBinning2D(
        min_bin_size=0.05, solver="cp", strategy=strategy, **prebins)
    unregularized.fit(x, y, zc)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits[0]) <= len(unregularized.splits[0])


def test_defect_woe_matrix_on_a_pure_clean_grid():
    # The WoE *matrix* is a per-cell quantity but was gated on the table
    # totals: it is computed from the per-cell event-rate matrix D, while
    # the guard tested t_n_event and t_n_nonevent. _prebinning_matrices
    # builds D from the clean grid alone -- the specials and the missing
    # bucket are counted separately -- so a fit whose clean records all
    # carry the same class while the missing bucket carries both leaves the
    # totals mixed, the guard silent, D == 1.0 and log(1 / 1 - 1) leaking
    # `divide by zero encountered in log` out of build(), with _W at -inf.
    xm = x.copy()
    ym = y.copy()
    zm = np.ones(n_samples, dtype=int)

    xm[:40] = np.nan
    ym[:40] = np.nan
    zm[:40] = np.tile([0, 1], 20)

    optb = OptimalBinning2D(**prebins)
    optb.fit(xm, ym, zm)

    assert optb.status == "OPTIMAL"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        df = optb.binning_table.build()

    # The clean grid is one all-event cell, so its WoE is zero -- the same
    # rule the WoE column already applied to it. The totals are not
    # degenerate: the Missing bucket carries both classes and its IV is
    # what the table reports.
    assert np.all(np.isfinite(optb.binning_table._W))
    assert np.all(optb.binning_table._W == 0.0)
    assert df["Event rate"][0] == 1.0
    assert df["WoE"][0] == 0.0
    assert df["IV"][0] == 0.0
    assert df["IV"][2] > 0
    assert df.loc["Totals", "IV"] > 0


def test_woe_matrix_matches_the_woe_column():
    # A pin, not a regression test: it has never been red. The per-cell
    # gate added above must leave a mixed grid exactly as it was, so every
    # cell of _W still carries log(1 / D - 1) + log(t_event / t_nonevent).
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    table = optb.binning_table
    df = table.build()

    n_event = df["Event"].iloc[:-3].sum()
    n_nonevent = df["Non-event"].iloc[:-3].sum()
    constant = np.log(n_event / n_nonevent)

    D = table.D
    expected = np.log(1 / D - 1) + constant

    assert np.all((D > 0) & (D < 1))
    assert table._W == approx(expected)

    # Every cell of a bin carries that bin's WoE.
    woe = df["WoE"].iloc[:-3].to_numpy(dtype=float)
    assert set(np.round(table._W.ravel(), 12)) == set(np.round(woe, 12))


def test_defect_refit_single_class_reports_the_fit_that_happened(capsys):
    # The degenerate branch of _fit_optimizer skips the solver, so it must
    # also drop the previous fit's solver record. It set _time_solver and
    # left _optimizer, _time_optimizer and _time_model_data alone, so
    # information() printed the discarded run's objective and split the new
    # (near-zero) solver time against the old model-generation time, giving
    # a negative percentage.
    optb = OptimalBinning2D(**prebins)
    optb.fit(x, y, z)

    assert optb._optimizer is not None

    optb.fit(x, y, np.ones(n_samples, dtype=int))

    assert optb.status == "OPTIMAL"
    assert optb._optimizer is None
    assert optb._time_optimizer is None
    assert optb._time_model_data == 0.

    optb.information(print_level=2)
    out = capsys.readouterr().out

    assert "Solver statistics" not in out
    assert "Objective value" not in out
    assert "model generation" not in out


@mark.parametrize("solver", ["cp", "mip"])
def test_defect_build_without_a_solution_no_warning(solver):
    # time_limit=0 is no budget at all: both solvers report UNKNOWN and
    # select no rectangle, so every count in the table is zero and
    # t_n_records with it. build() divided by it twice -- t_n_event /
    # t_n_records and n_records / t_n_records -- and leaked `invalid value
    # encountered in divide`. Clause 4 of the degenerate-input contract:
    # guard the divides, do not silence them.
    optb = OptimalBinning2D(solver=solver, time_limit=0, **prebins)
    optb.fit(x, y, z)

    assert optb.status == "UNKNOWN"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = optb.binning_table.build()

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    assert df.loc["Totals", "Count"] == 0
    assert df.loc["Totals", "Event rate"] == 0.0
    assert df.loc["Totals", "Count (%)"] == 0.0
    assert df["Count (%)"].iloc[0] == 0.0
    assert np.all(np.isfinite(optb.binning_table._W))


@mark.parametrize("solver", ["cp", "mip"])
def test_defect_continuous_build_without_a_solution_no_warning(solver):
    # The same two divides in ContinuousBinningTable2D.build: t_sum /
    # t_n_records and n_records / t_n_records.
    optb = ContinuousOptimalBinning2D(solver=solver, time_limit=0, **prebins)
    optb.fit(x, y, zc)

    assert optb.status == "UNKNOWN"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = optb.binning_table.build()

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    assert df.loc["Totals", "Count"] == 0
    assert df.loc["Totals", "Mean"] == 0.0
    assert df.loc["Totals", "Count (%)"] == 0.0
    assert df["Count (%)"].iloc[0] == 0.0
