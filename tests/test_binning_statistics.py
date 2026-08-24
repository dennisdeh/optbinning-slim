"""
Binning table statistics testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pytest import approx, raises

from optbinning import ContinuousOptimalBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning.binning.binning_statistics import BinningTable
from optbinning.binning.binning_statistics import ContinuousBinningTable
from optbinning.binning.binning_statistics import MulticlassBinningTable
from optbinning.binning.binning_statistics import _bin_str_label_format
from optbinning.binning.binning_statistics import _check_build_parameters
from optbinning.binning.binning_statistics import bin_categorical
from optbinning.binning.binning_statistics import bin_info
from optbinning.binning.binning_statistics import bin_str_format
from optbinning.binning.binning_statistics import continuous_bin_info
from optbinning.binning.binning_statistics import multiclass_bin_info
from optbinning.binning.binning_statistics import target_info
from optbinning.binning.binning_statistics import target_info_samples
from optbinning.binning.binning_statistics import target_info_special
from optbinning.binning.binning_statistics import (
    target_info_special_continuous)
from optbinning.binning.binning_statistics import (
    target_info_special_multiclass)
from optbinning.binning.metrics import gini
from sklearn.exceptions import NotFittedError


SPECIAL_CODES = {"s_a": -9, "s_b": -8}


def _numerical_data():
    rng = np.random.RandomState(0)
    n = 300
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    x[:10] = -9.0
    x[10:20] = -8.0
    x[20:25] = np.nan

    return x, y


def _categorical_data():
    rng = np.random.RandomState(0)
    x = np.array(["A"] * 100 + ["B"] * 80 + ["C"] * 60 + ["D"] * 40 +
                 ["rare1"] * 8 + ["rare2"] * 7 + ["rare3"] * 5, dtype=object)
    rates = {"A": 0.1, "B": 0.3, "C": 0.5, "D": 0.7,
             "rare1": 0.5, "rare2": 0.5, "rare3": 0.5}
    means = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0,
             "rare1": 2.5, "rare2": 2.5, "rare3": 2.5}

    y = np.array([rng.binomial(1, rates[c]) for c in x])
    yc = np.array([means[c] + rng.normal(0, 0.3) for c in x])

    return x, y, yc


@functools.lru_cache(maxsize=None)
def _fit_binary_specials():
    """Numerical binary fit whose special bucket is a *dict* of two codes."""
    x, y = _numerical_data()
    optb = OptimalBinning(name="numerical", special_codes=SPECIAL_CODES,
                          max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    return optb


@functools.lru_cache(maxsize=None)
def _fit_binary_cat_others():
    """Categorical binary fit with an 'others' bin (cat_cutoff)."""
    x, y, _ = _categorical_data()
    optb = OptimalBinning(name="categorical", dtype="categorical",
                          cat_cutoff=0.05, max_n_prebins=6)
    optb.fit(x, y)

    return optb


@functools.lru_cache(maxsize=None)
def _fit_continuous_cat_others():
    x, _, yc = _categorical_data()
    optb = ContinuousOptimalBinning(name="categorical", dtype="categorical",
                                    cat_cutoff=0.05, max_n_prebins=6)
    optb.fit(x, yc)

    return optb


@functools.lru_cache(maxsize=None)
def _fit_continuous_numerical():
    rng = np.random.RandomState(0)
    n = 300
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    optb = ContinuousOptimalBinning(name="numerical", max_n_prebins=6,
                                    max_n_bins=4)
    optb.fit(x, y)

    return optb


@functools.lru_cache(maxsize=None)
def _fit_multiclass_specials():
    rng = np.random.RandomState(0)
    n = 300
    x = rng.normal(0, 1, n)
    y = np.digitize(x, [-0.5, 0.5])
    x[:10] = -9.0
    x[10:20] = -8.0
    optb = MulticlassOptimalBinning(name="multiclass",
                                    special_codes=SPECIAL_CODES,
                                    max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)

    return optb


def _binary_table(special_codes=None, min_x=None, max_x=None):
    """A three-bin binary table built without running a solver."""
    n_specials = len(special_codes) if isinstance(special_codes, dict) else 1

    n_nonevent = np.array([90, 60, 30] + [20] * n_specials + [5])
    n_event = np.array([10, 40, 70] + [10] * n_specials + [5])

    return BinningTable("x", "numerical", special_codes,
                        np.array([1.0, 2.0]), n_nonevent, n_event,
                        min_x=min_x, max_x=max_x)


def _continuous_table(special_codes=None, min_x=None, max_x=None,
                      dtype="numerical", sums=None):
    n_specials = len(special_codes) if isinstance(special_codes, dict) else 1
    n_extra = n_specials + 1

    n_records = np.array([100, 100, 100] + [10] * n_extra)
    if sums is None:
        sums = np.array([100.0, 200.0, 300.0] + [15.0] * n_extra)
    stds = np.array([1.0, 1.0, 1.0] + [1.0] * n_extra)
    min_target = np.array([0.0, 1.0, 2.0] + [1.0] * n_extra)
    max_target = np.array([2.0, 3.0, 4.0] + [2.0] * n_extra)
    n_zeros = np.array([1, 0, 0] + [0] * n_extra)

    return ContinuousBinningTable(
        "x", dtype, special_codes, np.array([1.0, 2.0]), n_records, sums,
        stds, min_target, max_target, n_zeros, min_x=min_x, max_x=max_x)


def _multiclass_table(special_codes=None):
    n_specials = len(special_codes) if isinstance(special_codes, dict) else 1
    n_extra = n_specials + 1

    n_event = np.array([[80, 10, 10], [30, 50, 20], [10, 20, 70]] +
                       [[5, 5, 5]] * n_extra)

    return MulticlassBinningTable("x", special_codes, np.array([1.0, 2.0]),
                                  n_event, [0, 1, 2])


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------

def test_bin_str_format_show_digits():
    bins = np.array([-np.inf, 0.123456, np.inf])

    # show_digits=None falls back to two digits
    assert bin_str_format(bins, None) == bin_str_format(bins, 2)
    assert bin_str_format(bins, 2) == ["(-inf, 0.12)", "[0.12, inf)"]
    assert bin_str_format(bins, 8) == ["(-inf, 0.12345600)",
                                       "[0.12345600, inf)"]

    # only the -inf edge gets an open parenthesis
    assert bin_str_format(np.array([0.0, 1.0]), 0) == ["[0, 1)"]

    # a single edge produces no bin at all
    assert bin_str_format(np.array([-np.inf]), 2) == []


def test_bin_categorical_cat_others():
    categories = np.array(["a", "b", "c", "d"], dtype=object)

    bins = bin_categorical(np.array([2.0]), categories, [], None)
    assert [list(b) for b in bins] == [["a", "b"], ["c", "d"]]

    # the others' bin is appended last
    bins = bin_categorical(np.array([2.0]), categories, ["x", "y"], None)
    assert [list(b) for b in bins] == [["a", "b"], ["c", "d"], ["x", "y"]]


def test_bin_categorical_user_splits_flattens():
    # with user_splits the categories are themselves groups, and the groups
    # of a bin are flattened into a single list
    categories = np.array([["a", "b"], ["c"], ["d"]], dtype=object)

    bins = bin_categorical(np.array([1.0, 2.0]), categories, [],
                           np.array([1]))
    assert bins == [["a", "b", "c"], ["d"]]

    bins = bin_categorical(np.array([1.0, 2.0]), categories, ["z"],
                           np.array([1]))
    assert bins == [["a", "b", "c"], ["d"], ["z"]]


def test_target_info_empty_and_no_sample_weight():
    assert target_info(np.array([])) == (0, 0)
    assert target_info(np.array([0, 0, 1, 1, 1])) == (2, 3)
    assert target_info(np.array([0, 0, 1, 1, 1]), cl=1) == (3, 2)

    # an empty y short-circuits before the weights are looked at
    assert target_info_samples(np.array([]), np.array([])) == (0, 0)

    # an empty weight vector falls back to unweighted counting
    y = np.array([0, 0, 1, 1, 1])
    assert target_info_samples(y, np.array([])) == target_info(y)

    # and weights are summed, not counted
    sw = np.array([2.0, 2.0, 3.0, 3.0, 3.0])
    assert target_info_samples(y, sw) == (4.0, 9.0)


def test_target_info_special_dict():
    special_codes = {"s_a": -9, "s_b": [-8, -7], "s_c": -6}
    x = np.array([-9, -9, -9, -8, -7])
    y = np.array([0, 1, 1, 0, 1])
    sw = np.ones(5)

    n_nonevent, n_event = target_info_special(special_codes, x, y, sw)

    # one entry per named special, in dict order; "s_c" matches nothing
    assert n_nonevent == [1.0, 1.0, 0]
    assert n_event == [2.0, 1.0, 0]

    # the array form collapses every special into a single bucket
    assert target_info_special([-9, -8, -7], x, y, sw) == (2.0, 3.0)


def test_target_info_special_multiclass_dict():
    special_codes = {"s_a": -9, "s_b": [-8, -7], "s_c": -6}
    x = np.array([-9, -9, -9, -8, -7])
    y = np.array([0, 1, 2, 1, 2])

    n_event = target_info_special_multiclass(special_codes, x, y, [0, 1, 2])

    assert n_event == [[1, 1, 1], [0, 1, 1], [0, 0, 0]]

    # the array form is a single bucket
    assert target_info_special_multiclass([-9], x, y, [0, 1, 2]) == [1, 2, 2]


def test_target_info_special_continuous_dict():
    special_codes = {"s_a": -9, "s_b": -8}
    x = np.array([-9.0, -9.0, -9.0])
    y = np.array([1.0, 2.0, 3.0])
    sw = np.ones(3)

    (n_records, sums, n_zeros, stds, min_target,
     max_target) = target_info_special_continuous(special_codes, x, y, sw)

    assert n_records == approx([3.0, 0.0])
    assert sums == approx([6.0, 0.0])
    # an unmatched code contributes an empty slice, so it counts no zeros
    assert n_zeros == [0, 0]
    assert stds == approx([np.std([1.0, 2.0, 3.0]), 0.0])
    assert min_target == approx([1.0, 0.0])
    assert max_target == approx([3.0, 0.0])


def test_bin_info_empty_solution():
    # an empty solution collapses everything into the first prebin
    nev, ev = bin_info(np.array([]), np.array([7]), np.array([3]),
                       1, 2, 4, 5, 0, 0, [])
    assert list(nev) == [7, 4, 1]
    assert list(ev) == [3, 5, 2]


def test_bin_info_unselected_prebins_are_accumulated():
    # a prebin whose split is not selected is merged into the next one
    merged, _ = bin_info(np.array([False, True]), np.array([5, 6]),
                         np.array([1, 2]), 0, 0, 0, 0, 0, 0, [])
    single, _ = bin_info(np.array([True]), np.array([11]), np.array([3]),
                         0, 0, 0, 0, 0, 0, [])
    assert list(merged) == list(single)


def test_bin_info_cat_others_and_dict_specials():
    solution = np.array([True, True])
    n_nonevent = np.array([5, 6])
    n_event = np.array([1, 2])

    # array-form specials: a single special bucket
    nev, ev = bin_info(solution, n_nonevent, n_event, 3, 4, 7, 8, 0, 0, [])
    assert list(nev) == [5, 6, 7, 3]
    assert list(ev) == [1, 2, 8, 4]

    # dict-form specials: one bucket per named special, and an others' bin
    nev, ev = bin_info(solution, n_nonevent, n_event, 3, 4, [7, 9], [8, 10],
                       11, 12, ["o"])
    assert list(nev) == [5, 6, 11, 7, 9, 3]
    assert list(ev) == [1, 2, 12, 8, 10, 4]
    assert nev.dtype == np.int64 and ev.dtype == np.int64


def test_multiclass_bin_info_dict_specials():
    solution = np.array([True, True])
    n_event = np.array([[5, 6], [1, 2]])

    ev = multiclass_bin_info(solution, 2, n_event, np.array([0, 0]),
                             [[1, 1], [2, 2]])
    assert ev.tolist() == [[5, 6], [1, 2], [1, 1], [2, 2], [0, 0]]
    assert ev.dtype == np.int64


def test_multiclass_bin_info_array_specials():
    # a prebin whose split is not selected is merged into the next one
    ev = multiclass_bin_info(np.array([False, True]), 2,
                             np.array([[5, 6], [1, 2]]), np.array([0, 0]),
                             [3, 4])
    assert ev.tolist() == [[6, 8], [3, 4], [0, 0]]


def test_target_info_special_continuous_array_form():
    x = np.array([-9.0, -9.0, -8.0])
    y = np.array([1.0, 2.0, 3.0])

    # weights scale the target before the statistics are taken
    res = target_info_special_continuous([-9, -8], x, y, np.array([2.0] * 3))
    n_records, sums, n_zeros, stds, min_target, max_target = res
    assert n_records == approx(6.0)
    assert sums == approx(12.0)
    assert n_zeros == 0
    assert stds == approx(np.std([2.0, 4.0, 6.0]))
    assert min_target == approx(2.0)
    assert max_target == approx(6.0)

    # an empty special slice reports None rather than NaN
    res = target_info_special_continuous([-9], np.array([]), np.array([]),
                                         np.array([]))
    assert res[3] is None and res[4] is None and res[5] is None


def test_continuous_bin_info_empty_solution():
    # an empty solution means a single bin holding everything
    r, s, st, min_t, max_t, z = continuous_bin_info(
        np.array([]), 100, 250.0, 700.0, 1.5, 0.0, 5.0, 3,
        4, 8.0, 0.5, 1.0, 3.0, 1,
        [2, 3], [4.0, 9.0], [0.5, 0.5], [1.0, 2.0], [3.0, 4.0], [0, 1],
        6, 12.0, 0.25, 1.0, 3.0, 2, ["other"])

    # bins, others, one row per named special, missing
    assert list(r) == [100, 6, 2, 3, 4]
    assert list(s) == approx([250.0, 12.0, 4.0, 9.0, 8.0])
    assert list(z) == [3, 2, 0, 1, 1]
    assert list(st) == approx([1.5, 0.25, 0.5, 0.5, 0.5])
    assert list(min_t) == approx([0.0, 1.0, 1.0, 2.0, 1.0])
    assert list(max_t) == approx([5.0, 3.0, 3.0, 4.0, 3.0])
    assert r.dtype == np.int64 and s.dtype == np.float64


def test_check_build_parameters():
    with raises(ValueError, match="show_digits must be an integer"):
        _check_build_parameters(-1, True)

    with raises(ValueError, match="show_digits must be an integer"):
        _check_build_parameters(9, True)

    with raises(ValueError, match="show_digits must be an integer"):
        _check_build_parameters(2.0, True)

    with raises(TypeError, match="add_totals must be a boolean"):
        _check_build_parameters(2, 1)

    # bounds are inclusive
    _check_build_parameters(0, True)
    _check_build_parameters(8, False)


def test_bin_str_label_format_truncates():
    labels = _bin_str_label_format(["a" * 30, "b" * 27, 12345])

    assert labels[0] == "a" * 27 + "..."
    assert labels[1] == "b" * 27
    assert labels[2] == "12345"

    assert _bin_str_label_format(["abcdef"], max_length=3) == ["abc..."]


# ---------------------------------------------------------------------------
# BinningTable
# ---------------------------------------------------------------------------

def test_binning_table_not_built():
    table = _binary_table()

    for name in ("gini", "iv", "js", "hellinger", "triangular", "ks"):
        with raises(NotFittedError, match="BinningTable"):
            getattr(table, name)

    with raises(NotFittedError, match="not analyzed yet"):
        table.quality_score

    with raises(NotFittedError, match="not built yet"):
        table.plot()

    with raises(NotFittedError, match="not built yet"):
        table.analysis(print_output=False)


def test_binning_table_build_parameter_guards():
    table = _binary_table()

    with raises(ValueError):
        table.build(show_digits=-1)

    with raises(TypeError):
        table.build(add_totals="yes")

    df = table.build(show_digits=0, add_totals=False)
    assert len(df) == 5
    assert df["Bin"].tolist()[:3] == ["(-inf, 1)", "[1, 2)", "[2, inf)"]

    df = table.build(add_totals=True)
    assert df.index[-1] == "Totals"
    assert df.loc["Totals", "Count"] == table.n_event.sum() + \
        table.n_nonevent.sum()


def test_binning_table_special_codes_dict():
    optb = _fit_binary_specials()
    table = optb.binning_table
    df = table.build()

    assert optb.status == "OPTIMAL"
    # one row per named special instead of a single "Special" row
    assert table._n_specials == 2
    # the last row is "Totals"
    assert df["Bin"].tolist()[-4:-1] == ["s_a", "s_b", "Missing"]
    assert df["Count"].tolist()[-4:-2] == [10, 10]

    # the array form collapses them
    x, y = _numerical_data()
    optb_list = OptimalBinning(name="numerical", special_codes=[-9, -8],
                               max_n_prebins=6, max_n_bins=4)
    optb_list.fit(x, y)
    df_list = optb_list.binning_table.build()

    assert optb_list.binning_table._n_specials == 1
    assert df_list["Bin"].tolist()[-3:-1] == ["Special", "Missing"]
    assert df_list["Count"].tolist()[-3] == 20

    # the same records, differently partitioned: naming the specials splits
    # one bin into two, which cannot lower the Jeffrey divergence
    assert df_list["Count"].tolist()[-1] == df["Count"].tolist()[-1]
    assert table.iv >= optb_list.binning_table.iv
    assert table.js >= optb_list.binning_table.js


def test_binning_table_divergence_properties():
    table = _binary_table()
    table.build()

    assert table.iv > 0
    assert 0 <= table.js <= np.log(2)
    assert table.hellinger > 0
    assert table.triangular > 0
    assert 0 <= table.ks <= 1
    assert -1 <= table.gini <= 1

    # a near-separable table drives every divergence towards its maximum
    sharp = BinningTable("x", "numerical", None, np.array([1.0]),
                         np.array([99, 1, 0, 0]), np.array([1, 99, 0, 0]))
    sharp.build()
    assert sharp.ks == approx(0.98, rel=1e-12)
    assert sharp.gini > 0.97
    assert sharp.iv > table.iv
    assert sharp.js > table.js


def test_binning_table_plot_parameter_guards():
    table = _binary_table()
    table.build()

    with raises(TypeError, match="add_special must be a boolean"):
        table.plot(add_special=1)

    with raises(TypeError, match="add_missing must be a boolean"):
        table.plot(add_missing="no")

    with raises(ValueError, match="Invalid value for style"):
        table.plot(style="real")

    with raises(TypeError, match="show_bin_labels must be a boolean"):
        table.plot(show_bin_labels=1)

    with raises(TypeError, match="figsize argument must be a tuple"):
        table.plot(figsize=[6, 4])

    # style="actual" needs the real x range
    with raises(ValueError, match="min_x and max_x must be provided"):
        table.plot(style="actual")

    # ... and a numerical variable
    cat = BinningTable("c", "categorical", None, np.array([1.0, 2.0]),
                       np.array([90, 60, 30, 20, 5]),
                       np.array([10, 40, 70, 10, 5]),
                       categories=np.array(["a", "b", "c"], dtype=object))
    cat.build()
    with raises(ValueError, match="dtype must be numerical"):
        cat.plot(style="actual")


def test_binning_table_plot_metric_iv(tmp_path):
    table = _binary_table()
    table.build()

    table.plot(metric="iv", figsize=(6, 4),
               savefig=str(tmp_path / "iv.png"))
    table.plot(metric="event_rate", savefig=str(tmp_path / "event_rate.png"))

    with raises(ValueError, match="Invalid value for metric"):
        table.plot(metric="mean")

    plt.close("all")


def test_binning_table_plot_savefig_guards(monkeypatch, tmp_path):
    table = _binary_table()
    table.build()

    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(1))

    table.plot()
    assert shown == [1]

    with raises(TypeError, match="savefig must be a string path"):
        table.plot(savefig=tmp_path / "p.png")

    with raises(TypeError, match="save_kwargs must be a dictionary"):
        table.plot(savefig=str(tmp_path / "p.png"), save_kwargs="dpi=10")

    path = tmp_path / "kwargs.png"
    table.plot(savefig=str(path), save_kwargs={"dpi": 50})
    assert path.exists()

    plt.close("all")


def test_binning_table_plot_cat_others(tmp_path):
    optb = _fit_binary_cat_others()
    table = optb.binning_table
    table.build()

    assert len(table.cat_others) == 3

    table.plot(savefig=str(tmp_path / "others.png"))
    table.plot(add_special=False, add_missing=False,
               savefig=str(tmp_path / "others_no_sm.png"))
    table.plot(show_bin_labels=True,
               savefig=str(tmp_path / "others_labels.png"))

    plt.close("all")


def test_binning_table_plot_show_bin_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    table = _binary_table()
    table.build()

    table.plot(show_bin_labels=True)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels == ["(-inf, 1.00)", "[1.00, 2.00)", "[2.00, inf)",
                      "Special", "Missing"]

    plt.close("all")
    table.plot(show_bin_labels=True, add_missing=False)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels[-1] == "Special"

    plt.close("all")
    table.plot(show_bin_labels=True, add_special=False)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels[-1] == "Missing"

    # style="actual" and show_bin_labels are mutually exclusive
    with raises(ValueError, match="show_bin_labels"):
        table.plot(style="actual", show_bin_labels=True)

    plt.close("all")


def test_binning_table_plot_style_actual_with_dict_specials(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    optb = _fit_binary_specials()
    table = optb.binning_table
    table.build()

    # style="actual" drops every special bucket, not a hard-coded two
    table.plot(style="actual")
    ax1 = plt.gcf().axes[0]
    assert len(ax1.patches) == 2 * (len(optb.splits) + 1)
    assert ax1.get_xlabel() == "x"

    plt.close("all")


def test_binning_table_analysis_guards():
    table = _binary_table()
    table.build()

    with raises(ValueError, match="Invalid value for pvalue_test"):
        table.analysis(pvalue_test="ttest", print_output=False)

    with raises(ValueError, match="n_samples must be a positive integer"):
        table.analysis(n_samples=0, print_output=False)

    with raises(ValueError, match="n_samples must be a positive integer"):
        table.analysis(n_samples=1.5, print_output=False)


def test_binning_table_analysis_fisher(capsys):
    table = _binary_table()
    table.build()

    table.analysis(pvalue_test="fisher", n_samples=10)
    out = capsys.readouterr().out

    # the Fisher test reports an odd ratio, not a t-statistic
    assert "odd ratio" in out
    assert "t-statistic" not in out
    assert 0 <= table.quality_score <= 1


def test_binning_table_analysis_cat_others(capsys):
    optb = _fit_binary_cat_others()
    table = optb.binning_table
    table.build()
    table.analysis(n_samples=10)

    out = capsys.readouterr().out
    # 4 categorical bins plus an others' bin: the others' bin is excluded
    # from the significance tests, leaving three consecutive pairs
    assert "Bin A" in out
    assert 0 <= table.quality_score <= 1
    assert table._n_specials == 1


def test_binning_table_analysis_single_bin(capsys):
    table = BinningTable("x", "numerical", None, np.array([]),
                         np.array([90, 5, 5]), np.array([10, 5, 5]))
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    # no consecutive pair of bins to test
    assert "None" in out
    assert 0 <= table.quality_score <= 1


def test_binning_table_degenerate_single_class():
    # every record is an event: WoE and IV are defined to be zero, which is
    # the limit of build()'s own "a pure bin contributes zero divergence"
    # convention rather than an error
    table = BinningTable("x", "numerical", None, np.array([1.0, 2.0]),
                         np.array([0, 0, 0, 0, 0]),
                         np.array([50, 30, 20, 5, 5]))

    df = table.build()

    assert list(df["WoE"])[:-1] == [0.0] * 5
    assert list(df["IV"])[:-1] == [0.0] * 5
    assert table.iv == 0.
    assert table.js == 0.
    # the empty-distribution limit keeps KS finite instead of NaN
    assert table._ks == 0.
    assert table._hellinger == 0.
    assert table._triangular == 0.

    # the event rate is gated on records, not on mixedness: every one of
    # these bins is 100% event and has to say so
    assert list(df["Event rate"])[:-1] == [1.0] * 5
    assert df.loc["Totals", "Event rate"] == 1.0

    # gini divides by te * tne, which is 0 here; its own n <= 1 branch
    # already answers 0 for the same degeneracy
    assert table.gini == 0.


def test_binning_table_degenerate_single_class_all_nonevent():
    # the mirror case: no event anywhere, so every event rate is 0
    table = BinningTable("x", "numerical", None, np.array([1.0, 2.0]),
                         np.array([50, 30, 20, 5, 5]),
                         np.array([0, 0, 0, 0, 0]))

    df = table.build()

    assert list(df["Event rate"])[:-1] == [0.0] * 5
    assert df.loc["Totals", "Event rate"] == 0.0
    assert table.gini == 0.
    assert table.iv == 0.


def test_binning_table_mixed_and_pure_bins_report_their_own_rate():
    # bins 0 and 2 hold both classes; bin 1 is pure event and bin 3 pure
    # non-event. Only the mixed ones get a WoE, but all four get a rate.
    table = BinningTable("x", "numerical", None, np.array([1., 2., 3.]),
                         np.array([80, 0, 40, 25, 0, 0]),
                         np.array([20, 30, 60, 0, 0, 0]))

    df = table.build()

    assert list(df["Event rate"])[:4] == [0.2, 1.0, 0.6, 0.0]
    assert list(df["WoE"])[1] == 0.
    assert list(df["WoE"])[3] == 0.
    assert list(df["WoE"])[0] != 0.
    assert list(df["WoE"])[2] != 0.


def test_binning_table_empty_counts_report_zeros():
    # nothing to divide by: t_n_records is 0, and neither the totals rate nor
    # the record shares may come back nan
    table = BinningTable("x", "numerical", None, np.array([]),
                         np.array([0, 0, 0]), np.array([0, 0, 0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        df = table.build()

    assert list(df["Count (%)"])[:-1] == [0.0, 0.0, 0.0]
    assert list(df["Event rate"])[:-1] == [0.0, 0.0, 0.0]
    assert df.loc["Totals", "Event rate"] == 0.0
    assert table.gini == 0.


def test_gini_single_class_over_several_bins_is_zero():
    # more than one non-empty bin, so the n <= 1 shortcut does not fire; the
    # normalising area te * tne is still 0 and the answer must not be nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        assert gini(np.array([50, 30, 20]), np.array([0, 0, 0])) == 0
        assert gini(np.array([0, 0, 0]), np.array([50, 30, 20])) == 0

    # the degenerate branch it has to agree with
    assert gini(np.array([50]), np.array([0])) == 0

    # a mixed table is untouched
    assert gini(np.array([10, 90]), np.array([90, 10])) == approx(0.8)


# ---------------------------------------------------------------------------
# MulticlassBinningTable
# ---------------------------------------------------------------------------

def test_multiclass_table_not_built():
    table = _multiclass_table()

    with raises(NotFittedError, match="MulticlassBinningTable"):
        table.js

    with raises(NotFittedError, match="not analyzed yet"):
        table.quality_score

    with raises(NotFittedError, match="not built yet"):
        table.plot()


def test_multiclass_table_special_codes_dict():
    optb = _fit_multiclass_specials()
    table = optb.binning_table
    df = table.build()

    assert optb.status == "OPTIMAL"
    assert table._n_specials == 2
    assert df["Bin"].tolist()[-4:-1] == ["s_a", "s_b", "Missing"]
    assert 0 <= table.js <= np.log(len(table.classes))


def test_multiclass_table_build_guards():
    table = _multiclass_table()

    with raises(ValueError):
        table.build(show_digits=100)

    with raises(TypeError):
        table.build(add_totals=None)

    df = table.build(add_totals=False)
    assert len(df) == 5
    assert df["Count"].sum() == table.n_event.sum()

    df = table.build()
    assert df.loc["Totals", "Count"] == table.n_event.sum()


def test_multiclass_table_plot_guards():
    table = _multiclass_table()
    table.build()

    with raises(TypeError, match="add_special must be a boolean"):
        table.plot(add_special=0)

    with raises(TypeError, match="add_missing must be a boolean"):
        table.plot(add_missing=0)

    with raises(TypeError, match="show_bin_labels must be a boolean"):
        table.plot(show_bin_labels="yes")

    with raises(TypeError, match="figsize argument must be a tuple"):
        table.plot(figsize=[6, 4])

    with raises(TypeError, match="savefig must be a string path"):
        table.plot(savefig=1)


def test_multiclass_table_plot(monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    table = _multiclass_table()
    table.build()

    table.plot()
    assert len(plt.gcf().axes) == 2

    plt.close("all")
    table.plot(show_bin_labels=True, figsize=(7, 5))
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels == ["(-inf, 1.00)", "[1.00, 2.00)", "[2.00, inf)",
                      "Special", "Missing"]

    plt.close("all")
    table.plot(show_bin_labels=True, add_special=False)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels[-1] == "Missing"

    plt.close("all")
    table.plot(show_bin_labels=True, add_missing=False)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels[-1] == "Special"

    plt.close("all")
    table.plot(add_special=False, add_missing=False,
               savefig=str(tmp_path / "mc.png"))
    assert (tmp_path / "mc.png").exists()

    plt.close("all")


def test_multiclass_table_analysis_single_bin(capsys):
    n_event = np.array([[80, 10, 10], [5, 5, 5], [5, 5, 5]])
    table = MulticlassBinningTable("x", None, np.array([]), n_event, [0, 1, 2])
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    # one bin leaves no consecutive pair, so Cramer's V is reported as zero
    assert "None" in out
    cramer = [ln for ln in out.splitlines() if "Cramer's V" in ln][0]
    assert float(cramer.split()[-1]) == 0.0


def test_multiclass_table_analysis(capsys):
    table = _multiclass_table()
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    assert "Multiclass Binning Table Analysis" in out
    assert out.count("Class") == len(table.classes)
    assert 0 <= table.quality_score <= 1

    table.analysis(print_output=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# ContinuousBinningTable
# ---------------------------------------------------------------------------

def test_continuous_table_not_built():
    table = _continuous_table()

    with raises(NotFittedError, match="ContinuousBinningTable"):
        table.iv

    with raises(NotFittedError, match="ContinuousBinningTable"):
        table.woe

    with raises(NotFittedError, match="not analyzed yet"):
        table.quality_score

    with raises(NotFittedError, match="not built yet"):
        table.plot()


def test_continuous_table_build_guards():
    table = _continuous_table()

    with raises(ValueError):
        table.build(show_digits=-3)

    with raises(TypeError):
        table.build(add_totals=1)

    df = table.build(add_totals=False)
    assert len(df) == 5
    assert df["Mean"].tolist()[:3] == approx([1.0, 2.0, 3.0])

    df = table.build()
    assert df.loc["Totals", "Count"] == table.n_records.sum()
    assert table.iv >= 0
    assert table.woe >= 0


def test_continuous_table_special_codes_dict():
    rng = np.random.RandomState(0)
    n = 300
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    x[:10] = -9.0
    x[10:20] = -8.0

    optb = ContinuousOptimalBinning(name="numerical",
                                    special_codes=SPECIAL_CODES,
                                    max_n_prebins=6, max_n_bins=4)
    optb.fit(x, y)
    table = optb.binning_table
    df = table.build()

    assert optb.status == "OPTIMAL"
    assert table._n_specials == 2
    assert df["Bin"].tolist()[-4:-1] == ["s_a", "s_b", "Missing"]
    assert df["Count"].tolist()[-4:-2] == [10, 10]


def test_continuous_table_plot_guards():
    table = _continuous_table()
    table.build()

    with raises(TypeError, match="add_special must be a boolean"):
        table.plot(add_special=1)

    with raises(TypeError, match="add_missing must be a boolean"):
        table.plot(add_missing=1)

    with raises(ValueError, match="Invalid value for style"):
        table.plot(style="raw")

    with raises(TypeError, match="show_bin_labels must be a boolean"):
        table.plot(show_bin_labels=1)

    with raises(TypeError, match="figsize argument must be a tuple"):
        table.plot(figsize=[6, 4])

    with raises(ValueError, match="Invalid value for metric"):
        table.plot(metric="event_rate")

    with raises(ValueError, match="min_x and max_x must be provided"):
        table.plot(style="actual")

    cat = _continuous_table(dtype="categorical")
    cat.categories = np.array(["a", "b", "c"], dtype=object)
    cat.build()
    with raises(ValueError, match='dtype must be numerical'):
        cat.plot(style="actual")


def test_continuous_table_plot_metrics(tmp_path):
    table = _continuous_table()
    table.build()

    for metric in ("mean", "woe", "iv"):
        table.plot(metric=metric, figsize=(6, 4),
                   savefig=str(tmp_path / "{}.png".format(metric)))
        assert (tmp_path / "{}.png".format(metric)).exists()

    with raises(TypeError, match="savefig must be a string path"):
        table.plot(savefig=1.0)

    plt.close("all")


def test_continuous_table_plot_style_actual(monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    optb = _fit_continuous_numerical()
    table = optb.binning_table
    table.build()

    table.plot(style="actual")
    ax1, ax2 = plt.gcf().axes
    # one bar per bin, positioned on the real x scale
    assert len(ax1.patches) == len(optb.splits) + 1
    assert ax1.get_xlabel() == "x"

    plt.close("all")
    table.plot(style="actual", metric="woe",
               savefig=str(tmp_path / "actual_woe.png"))
    assert (tmp_path / "actual_woe.png").exists()

    plt.close("all")


def test_continuous_table_plot_show_bin_labels(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    table = _continuous_table()
    table.build()

    table.plot(show_bin_labels=True)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels == ["(-inf, 1.00)", "[1.00, 2.00)", "[2.00, inf)",
                      "Special", "Missing"]

    plt.close("all")
    table.plot(show_bin_labels=True, add_special=False)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels[-1] == "Missing"

    plt.close("all")
    table.plot(show_bin_labels=True, add_missing=False)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert labels[-1] == "Special"

    with raises(ValueError, match="show_bin_labels"):
        table.plot(style="actual", show_bin_labels=True)

    plt.close("all")


def test_continuous_table_plot_cat_others(monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    optb = _fit_continuous_cat_others()
    table = optb.binning_table
    table.build()

    assert len(table.cat_others) == 3

    table.plot(savefig=str(tmp_path / "cont_others.png"))
    plt.close("all")

    table.plot(show_bin_labels=True)
    labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    # long category groups are truncated to 30 characters
    assert all(len(lb) <= 30 for lb in labels)

    plt.close("all")


def test_continuous_table_analysis_cat_others(capsys):
    optb = _fit_continuous_cat_others()
    table = optb.binning_table
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    assert "Continuous Binning Table Analysis" in out
    assert 0 <= table.quality_score <= 1


def test_continuous_table_analysis_single_bin(capsys):
    rng = np.random.RandomState(0)
    x = np.ones(200)
    y = rng.normal(0, 1, 200)

    optb = ContinuousOptimalBinning(name="constant", max_n_prebins=5)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    # a single bin leaves no consecutive pair to test
    assert "None" in out
    assert table.iv == approx(0.0, abs=1e-12)


def test_continuous_table_analysis_zero_total_mean(capsys):
    # a target that sums to exactly zero makes the normalised WoE fall back
    # to the raw WoE rather than dividing by zero
    sums = np.array([-100.0, 0.0, 100.0, 0.0, 0.0])
    table = _continuous_table(sums=sums)
    table.build()

    assert table._t_mean == 0.0

    table.analysis()
    out = capsys.readouterr().out

    woe_line = [ln for ln in out.splitlines() if "WoE (normalized)" in ln][0]
    assert float(woe_line.split()[-1]) == approx(table.woe, rel=1e-8)
    assert 0 <= table.quality_score <= 1


# ---------------------------------------------------------------------------
# defects
# ---------------------------------------------------------------------------

def test_defect_continuous_special_codes_dict_absent_from_data():
    """A named special code that matches no record crashes the fit.

    ``target_info_special_continuous`` sets ``std_special``/
    ``min_target_special``/``max_target_special`` to ``None`` when the special
    slice is empty, then unconditionally appends to them in the loop over the
    dict entries. The binary and multiclass siblings return zero counts for an
    unmatched code.
    """
    rng = np.random.RandomState(0)
    n = 200
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)

    # the binary sibling copes
    optb_bin = OptimalBinning(name="v", special_codes={"s": -999},
                              max_n_prebins=5)
    optb_bin.fit(x, (y > 0).astype(int))
    assert optb_bin.status == "OPTIMAL"

    optb = ContinuousOptimalBinning(name="v", special_codes={"s": -999},
                                    max_n_prebins=5)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Count"].tolist()[-2] == 0


def test_defect_bin_labels_keep_a_special_when_add_special_is_false(
        monkeypatch):
    """``plot(add_special=False, show_bin_labels=True)`` mislabels the bars.

    The bars are dropped with ``pop(-2)`` once per named special, but the
    labels are sliced with a hard-coded ``bin_str[:-2] + [bin_str[-1]]``. With
    a dict of two special codes one special label survives, so every label
    from the specials onwards names the wrong bar.
    """
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    optb = _fit_binary_specials()
    table = optb.binning_table
    table.build()

    plt.close("all")
    table.plot(add_special=False, show_bin_labels=True)

    ax1 = plt.gcf().axes[0]
    labels = [t.get_text() for t in ax1.get_xticklabels()]
    n_bars = len(ax1.patches) // 2

    assert labels[-1] == "Missing"
    assert len(labels) == n_bars

    plt.close("all")


def test_defect_monotonic_trend_includes_a_special_bin(capsys):
    """``analysis`` reports the trend over ``self._event_rate[:-2]``.

    With a dict of two special codes that slice keeps the first special bin,
    so a strictly ascending set of real bins is reported as a "peak" trend.
    The continuous and multiclass siblings slice the same way.
    """
    n_nonevent = np.array([90, 70, 50, 30, 50, 50, 10])
    n_event = np.array([10, 30, 50, 70, 1, 50, 10])

    table = BinningTable("x", "numerical", SPECIAL_CODES,
                         np.array([1.0, 2.0, 3.0]), n_nonevent, n_event)
    table.build()
    table.analysis(n_samples=10)

    out = capsys.readouterr().out
    trend = [ln for ln in out.splitlines() if "Monotonic trend" in ln][0]

    assert "ascending" in trend


def test_defect_show_bin_labels_message_names_the_wrong_style():
    """The guard fires on ``style="actual"`` but the message names it too.

    ``show_bin_labels`` is only honoured by ``style="bin"``, which is what the
    message should say. Both ``BinningTable`` and ``ContinuousBinningTable``
    carry the inverted text.
    """
    table = _binary_table()
    table.build()

    with raises(ValueError, match='show_bin_labels only supported when '
                                  'style="bin"'):
        table.plot(style="actual", show_bin_labels=True)

    # the continuous sibling carries the same guard and the same message
    ctable = _continuous_table()
    ctable.build()

    with raises(ValueError, match='show_bin_labels only supported when '
                                  'style="bin"'):
        ctable.plot(style="actual", show_bin_labels=True)


def test_target_info_special_continuous_dict_all_codes_absent():
    # every named code misses: the helper still answers one zero per code,
    # like its binary and multiclass siblings, instead of raising
    res = target_info_special_continuous({"a": -9, "b": -8}, np.array([]),
                                         np.array([]), np.array([]))

    assert [list(r) for r in res] == [[0.0, 0.0], [0.0, 0.0], [0, 0],
                                      [0, 0], [0, 0], [0, 0]]


def _labels_and_bars(table, **kwargs):
    plt.close("all")
    table.plot(**kwargs)
    ax1 = plt.gcf().axes[0]
    labels = [t.get_text() for t in ax1.get_xticklabels()]
    # the three tables draw a different number of stacked series, so count
    # the bars of one series rather than every patch
    n_bars = len(ax1.containers[0])
    plt.close("all")

    return labels, n_bars


def test_bin_labels_track_the_bars_for_every_table(monkeypatch):
    # one x-tick label per drawn bar, for all three tables, with a dict of
    # two special codes and with and without the missing bar
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    tables = [_binary_table(SPECIAL_CODES), _continuous_table(SPECIAL_CODES),
              _multiclass_table(SPECIAL_CODES)]

    for table in tables:
        table.build()

        labels, n_bars = _labels_and_bars(
            table, add_special=False, show_bin_labels=True)
        assert len(labels) == n_bars
        assert labels[-1] == "Missing"
        assert not set(SPECIAL_CODES) & set(labels)

        labels, n_bars = _labels_and_bars(
            table, add_special=False, add_missing=False,
            show_bin_labels=True)
        assert len(labels) == n_bars
        assert not set(SPECIAL_CODES) & set(labels)


def test_monotonic_trend_excludes_specials_continuous(capsys):
    # the continuous sibling reported the trend over the real bins plus the
    # first named special
    sums = np.array([100.0, 200.0, 300.0, 15.0, 15.0, 15.0])
    table = _continuous_table(SPECIAL_CODES, sums=sums)
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    trend = [ln for ln in out.splitlines() if "Monotonic trend" in ln][0]

    assert "ascending" in trend


def test_monotonic_trend_excludes_specials_multiclass(capsys):
    table = _multiclass_table(SPECIAL_CODES)
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    # one line per class, and none of them may see a special bin
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("Class")]

    assert len(lines) == 3
    assert lines[0].split()[-1] == "descending"
    assert lines[2].split()[-1] == "ascending"


def test_monotonic_trend_excludes_the_others_bin(capsys):
    # the cat_others half of the same defect: reachable with array-form
    # special codes, because [:-2] keeps the "others" bin
    optb = _fit_binary_cat_others()
    table = optb.binning_table
    table.build()
    table.analysis()

    out = capsys.readouterr().out
    trend = [ln for ln in out.splitlines() if "Monotonic trend" in ln][0]

    assert "ascending" in trend


def test_binning_table_build_pure_bins_without_a_pure_target():
    # no bin holds both an event and a non-event, yet both totals are
    # non-zero: the divergence metrics still get an empty distribution
    table = BinningTable("x", "numerical", None, np.array([1.0]),
                         np.array([0, 40, 0, 0]), np.array([60, 0, 0, 0]))

    df = table.build()

    assert table.iv == 0.
    assert table.js == 0.
    assert list(df["WoE"])[:-1] == [0.0] * 4
