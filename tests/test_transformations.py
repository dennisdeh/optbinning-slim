"""
Binning transformations testing.

Exercises ``optbinning.binning.transformations`` and
``optbinning.binning.piecewise.transformations`` through the estimators
that reach them, with the three ``special_codes`` container types the
parameter checkers accept: list, numpy.ndarray and dict.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx

from optbinning import BinningProcess
from optbinning import ContinuousOptimalBinning
from optbinning import ContinuousOptimalPWBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning import OptimalPWBinning
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from tests.datasets import load_boston


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target

data_continuous = load_boston()
df_continuous = pd.DataFrame(data_continuous.data,
                             columns=data_continuous.feature_names)

variable_continuous = "LSTAT"
x_continuous = df_continuous[variable_continuous].values
y_continuous = data_continuous.target

data_multiclass = load_wine()
df_multiclass = pd.DataFrame(data_multiclass.data,
                             columns=data_multiclass.feature_names)

variable_multiclass = "ash"
x_multiclass = df_multiclass[variable_multiclass].values
y_multiclass = data_multiclass.target


# Two sentinel values planted in the data so that both special buckets
# are populated. -1 and -2 lie outside the range of every variable used.
SPECIAL_A = -1.0
SPECIAL_B = -2.0

SPECIAL_LIST = [SPECIAL_A, SPECIAL_B]
SPECIAL_ARRAY = np.array([SPECIAL_A, SPECIAL_B])
SPECIAL_ARRAY_ONE = np.array([SPECIAL_A])
SPECIAL_DICT = {"code_a": SPECIAL_A, "code_b": SPECIAL_B}

# Bucket A gets 8 events and 2 non-events, bucket B 3 events and 7
# non-events, so each special bucket carries both classes and the two
# have distinguishable empirical event rates.
_idx_event = np.flatnonzero(y == 1)
_idx_nonevent = np.flatnonzero(y == 0)

IDX_A = np.concatenate([_idx_event[:8], _idx_nonevent[:2]])
IDX_B = np.concatenate([_idx_event[8:11], _idx_nonevent[2:9]])

EVENT_RATE_A = 0.8
EVENT_RATE_B = 0.3


def _plant(values):
    """Return a copy of ``values`` with both sentinels planted in it."""
    planted = np.asarray(values, dtype=float).copy()
    planted[IDX_A] = SPECIAL_A
    planted[IDX_B] = SPECIAL_B
    return planted


x_special = _plant(x)

# Boston and wine are shorter than breast cancer, so IDX_A / IDX_B do not
# apply to them; plant on their own leading index range instead.
_idx_a_short = np.arange(10)
_idx_b_short = np.arange(10, 20)


def _plant_short(values):
    planted = np.asarray(values, dtype=float).copy()
    planted[_idx_a_short] = SPECIAL_A
    planted[_idx_b_short] = SPECIAL_B
    return planted


x_continuous_special = _plant_short(x_continuous)
x_multiclass_special = _plant_short(x_multiclass)


def test_special_codes_ndarray_binary():
    optb_list = OptimalBinning(special_codes=SPECIAL_LIST)
    optb_list.fit(x_special, y)
    t_list = optb_list.transform(x_special)

    optb_array = OptimalBinning(special_codes=SPECIAL_ARRAY)
    optb_array.fit(x_special, y)
    t_array = optb_array.transform(x_special)

    assert optb_array.status == "OPTIMAL"
    assert t_array == approx(t_list, rel=1e-9)


def test_special_codes_ndarray_binary_metrics():
    optb_list = OptimalBinning(special_codes=SPECIAL_LIST)
    optb_list.fit(x_special, y)

    optb_array = OptimalBinning(special_codes=SPECIAL_ARRAY)
    optb_array.fit(x_special, y)

    for metric in ("woe", "event_rate", "indices"):
        t_list = optb_list.transform(x_special, metric=metric,
                                     metric_special="empirical")
        t_array = optb_array.transform(x_special, metric=metric,
                                       metric_special="empirical")
        assert t_array == approx(t_list, rel=1e-9)

    t_list = optb_list.transform(x_special, metric="bins")
    t_array = optb_array.transform(x_special, metric="bins")
    assert list(t_array) == list(t_list)
    assert set(t_array[IDX_A]) == {"Special"}


def test_special_codes_ndarray_binary_empirical():
    optb = OptimalBinning(special_codes=SPECIAL_ARRAY)
    optb.fit(x_special, y)

    t = optb.transform(x_special, metric="event_rate",
                       metric_special="empirical")

    # One unnamed special bin pools both codes.
    pooled = np.concatenate([IDX_A, IDX_B])
    assert t[pooled] == approx(np.full(len(pooled), y[pooled].mean()),
                               rel=1e-9)


def test_special_codes_ndarray_binary_metric_special_value():
    optb = OptimalBinning(special_codes=SPECIAL_ARRAY)
    optb.fit(x_special, y)

    t = optb.transform(x_special, metric="woe", metric_special=0.5)

    assert t[IDX_A] == approx(np.full(len(IDX_A), 0.5), rel=1e-9)
    assert t[IDX_B] == approx(np.full(len(IDX_B), 0.5), rel=1e-9)


def test_special_codes_ndarray_single_element_binary():
    # A one-element ndarray is legal too, and only SPECIAL_A is special.
    optb = OptimalBinning(special_codes=SPECIAL_ARRAY_ONE)
    optb.fit(x_special, y)

    t = optb.transform(x_special, metric="bins",
                       metric_special="empirical")

    assert set(t[IDX_A]) == {"Special"}
    assert "Special" not in set(t[IDX_B])


def test_special_codes_empty_list_binary():
    optb = OptimalBinning(special_codes=[])
    optb.fit(x_special, y)
    t_empty = optb.transform(x_special, metric="indices")

    optb_none = OptimalBinning(special_codes=None)
    optb_none.fit(x_special, y)
    t_none = optb_none.transform(x_special, metric="indices")

    assert t_empty == approx(t_none)


def test_special_codes_dict_binary():
    optb = OptimalBinning(special_codes=SPECIAL_DICT)
    optb.fit(x_special, y)

    t = optb.transform(x_special, metric="bins")
    assert set(t[IDX_A]) == {"code_a"}
    assert set(t[IDX_B]) == {"code_b"}

    t_er = optb.transform(x_special, metric="event_rate",
                          metric_special="empirical")
    assert t_er[IDX_A] == approx(np.full(len(IDX_A), EVENT_RATE_A), rel=1e-9)
    assert t_er[IDX_B] == approx(np.full(len(IDX_B), EVENT_RATE_B), rel=1e-9)


def test_special_codes_ndarray_continuous():
    optb_list = ContinuousOptimalBinning(special_codes=SPECIAL_LIST)
    optb_list.fit(x_continuous_special, y_continuous)
    t_list = optb_list.transform(x_continuous_special)

    optb_array = ContinuousOptimalBinning(special_codes=SPECIAL_ARRAY)
    optb_array.fit(x_continuous_special, y_continuous)
    t_array = optb_array.transform(x_continuous_special)

    assert optb_array.status == "OPTIMAL"
    assert t_array == approx(t_list, rel=1e-9)

    t_bins = optb_array.transform(x_continuous_special, metric="bins")
    assert set(t_bins[:20]) == {"Special"}


def test_special_codes_dict_continuous():
    optb = ContinuousOptimalBinning(special_codes=SPECIAL_DICT)
    optb.fit(x_continuous_special, y_continuous)

    t = optb.transform(x_continuous_special, metric="bins")
    assert set(t[_idx_a_short]) == {"code_a"}
    assert set(t[_idx_b_short]) == {"code_b"}

    t_mean = optb.transform(x_continuous_special, metric="mean",
                            metric_special="empirical")
    assert t_mean[0] == approx(y_continuous[_idx_a_short].mean(), rel=1e-9)
    assert t_mean[10] == approx(y_continuous[_idx_b_short].mean(), rel=1e-9)


def test_special_codes_ndarray_multiclass():
    optb_list = MulticlassOptimalBinning(special_codes=SPECIAL_LIST)
    optb_list.fit(x_multiclass_special, y_multiclass)
    t_list = optb_list.transform(x_multiclass_special)

    optb_array = MulticlassOptimalBinning(special_codes=SPECIAL_ARRAY)
    optb_array.fit(x_multiclass_special, y_multiclass)
    t_array = optb_array.transform(x_multiclass_special)

    assert optb_array.status == "OPTIMAL"
    assert t_array == approx(t_list, rel=1e-9)

    t_bins = optb_array.transform(x_multiclass_special, metric="bins")
    assert set(t_bins[:20]) == {"Special"}


def test_special_codes_dict_multiclass():
    optb = MulticlassOptimalBinning(special_codes=SPECIAL_DICT)
    optb.fit(x_multiclass_special, y_multiclass)

    t = optb.transform(x_multiclass_special, metric="bins")
    assert set(t[_idx_a_short]) == {"code_a"}
    assert set(t[_idx_b_short]) == {"code_b"}


def test_special_codes_ndarray_piecewise_binary():
    optb_list = OptimalPWBinning(special_codes=SPECIAL_LIST)
    optb_list.fit(x_special, y)

    optb_array = OptimalPWBinning(special_codes=SPECIAL_ARRAY)
    optb_array.fit(x_special, y)

    t_list = optb_list.transform(x_special, metric="event_rate")
    t_array = optb_array.transform(x_special, metric="event_rate")
    assert t_array == approx(t_list, rel=1e-9)
    assert not np.isnan(t_array).any()

    # metric="woe" without lb/ub can produce NaN wherever the piecewise
    # event-rate prediction leaves [0, 1]; that is unrelated to
    # special_codes and happens identically for both container types.
    t_list = optb_list.transform(x_special, metric="woe", lb=1e-8, ub=1 - 1e-8)
    t_array = optb_array.transform(x_special, metric="woe", lb=1e-8,
                                   ub=1 - 1e-8)
    assert t_array == approx(t_list, rel=1e-9)
    assert not np.isnan(t_array).any()


def test_special_codes_dict_piecewise_binary():
    optb = OptimalPWBinning(special_codes=SPECIAL_DICT)
    optb.fit(x_special, y)

    t = optb.transform(x_special, metric="event_rate",
                       metric_special="empirical")

    assert t[IDX_A] == approx(np.full(len(IDX_A), EVENT_RATE_A), rel=1e-9)
    assert t[IDX_B] == approx(np.full(len(IDX_B), EVENT_RATE_B), rel=1e-9)


def test_special_codes_ndarray_piecewise_continuous():
    optb_list = ContinuousOptimalPWBinning(special_codes=SPECIAL_LIST)
    optb_list.fit(x_continuous_special, y_continuous)
    t_list = optb_list.transform(x_continuous_special)

    optb_array = ContinuousOptimalPWBinning(special_codes=SPECIAL_ARRAY)
    optb_array.fit(x_continuous_special, y_continuous)
    t_array = optb_array.transform(x_continuous_special)

    assert t_array == approx(t_list, rel=1e-9)


def test_special_codes_dict_piecewise_continuous():
    optb = ContinuousOptimalPWBinning(special_codes=SPECIAL_DICT)
    optb.fit(x_continuous_special, y_continuous)

    t = optb.transform(x_continuous_special, metric_special="empirical")

    assert t[0] == approx(y_continuous[_idx_a_short].mean(), rel=1e-9)
    assert t[10] == approx(y_continuous[_idx_b_short].mean(), rel=1e-9)


def test_special_codes_ndarray_binning_process():
    variable_names = ["mean radius", "mean texture"]
    df_special = df[variable_names].copy()
    for name in variable_names:
        df_special[name] = _plant(df_special[name].values)

    process_list = BinningProcess(variable_names, special_codes=SPECIAL_LIST)
    process_list.fit(df_special, y)
    t_list = process_list.transform(df_special)

    process_array = BinningProcess(variable_names, special_codes=SPECIAL_ARRAY)
    process_array.fit(df_special, y)
    t_array = process_array.transform(df_special)

    assert t_array.values == approx(t_list.values, rel=1e-9)
