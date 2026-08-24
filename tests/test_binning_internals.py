"""
Binning internals testing: automatic monotonic trend selection, the divergence
and quality-score metrics, the pre-binning wrapper, the outlier detector base
class and the information printers.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises
from scipy.spatial import ConvexHull, QhullError
from sklearn.exceptions import NotFittedError

from optbinning import MDLP
from optbinning import OptimalBinning
from optbinning.binning import auto_monotonic as am
from optbinning.binning import metrics as mt
from optbinning.binning.binning_information import print_binning_information
from optbinning.binning.outlier import ModifiedZScoreDetector
from optbinning.binning.outlier import OutlierDetector
from optbinning.binning.outlier import RangeDetector
from optbinning.binning.outlier import YQuantileDetector
from optbinning.binning.prebinning import PreBinning
from optbinning.information import print_optional_parameters
from optbinning.information import print_solver_statistics
from optbinning.information import solver_statistics
from optbinning.options import continuous_optimal_binning_2d_default_options
from optbinning.options import continuous_optimal_binning_default_options
from optbinning.options import multiclass_optimal_binning_default_options
from optbinning.options import optimal_binning_2d_default_options
from optbinning.options import optimal_binning_default_options
from optbinning.options import sboptimal_binning_default_options


def _binary_sample(n=400, beta=2.0, seed=0):
    """Small synthetic binary problem with a strictly increasing event rate."""
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-beta * x))).astype(int)
    return x, y


def _counts(event_rate, n_per_bin=100.0):
    """Event / non-event counts realising a given event rate per pre-bin."""
    n_records = np.full(len(event_rate), n_per_bin)
    n_event = event_rate * n_records
    return n_records - n_event, n_event


# ---------------------------------------------------------------------------
# auto_monotonic: shape descriptors
# ---------------------------------------------------------------------------

def test_n_peaks_valleys():
    assert am.n_peaks_valleys(np.array([1., 2., 3.])) == 0
    assert am.n_peaks_valleys(np.array([3., 2., 1.])) == 0
    assert am.n_peaks_valleys(np.array([1., 3., 2.])) == 1
    assert am.n_peaks_valleys(np.array([1., 3., 2., 4., 1.])) == 3

    # a constant array never changes direction, and neither do the degenerate
    # arrays that carry no difference at all
    assert am.n_peaks_valleys(np.array([2., 2., 2.])) == 0
    assert am.n_peaks_valleys(np.array([1.])) == 0
    assert am.n_peaks_valleys(np.array([])) == 0


def test_peak_valley_trend_change_heuristic():
    x = np.array([0.1, 0.5, 0.9, 0.2])

    assert am.peak_valley_trend_change_heuristic(x, "peak_heuristic") == 2
    assert am.peak_valley_trend_change_heuristic(x, "valley_heuristic") == 0

    # anything that is not "peak_heuristic" takes the argmin branch
    assert am.peak_valley_trend_change_heuristic(x, "valley") == 0


def test_extreme_points_area_degenerate():
    # fewer than three points leave no interior, so the area is zero
    assert am.extreme_points_area(np.array([0.1, 0.9])) == 0
    assert am.extreme_points_area(np.array([0.5])) == 0

    # a constant array has no rectangle to normalise by
    with np.errstate(invalid="ignore"):
        assert np.isnan(am.extreme_points_area(np.array([0.3] * 4)))

    # an empty array has no extreme points at all: numpy refuses first
    with raises(ValueError):
        am.extreme_points_area(np.array([]))


def test_extreme_points_area_shapes():
    # a monotone series covers less of its bounding rectangle than a peak
    monotone = am.extreme_points_area(np.array([0.1, 0.3, 0.5, 0.9]))
    peak = am.extreme_points_area(np.array([0.1, 0.9, 0.2]))

    assert 0 <= monotone <= 1
    assert 0 <= peak <= 1
    assert monotone < peak


def test_auto_monotonic_data_convex_hull_is_zero_when_degenerate():
    # collinear event rates: qhull cannot build a hull from a flat simplex,
    # and the bare except must fall back to 0 rather than propagate
    n_nonevent, n_event = _counts(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    event_rate = n_event / (n_nonevent + n_event)
    points = np.column_stack([np.arange(len(event_rate)), event_rate])

    with raises(QhullError):
        ConvexHull(points)

    dict_data = am.auto_monotonic_data(n_nonevent, n_event)
    assert dict_data["p_convex_hull"] == 0
    assert dict_data["n_prebins"] == 5

    # same fallback on the continuous side
    n_records = np.full(5, 100.0)
    sums = np.array([1., 2., 3., 4., 5.]) * n_records
    dict_data = am.auto_monotonic_data_continuous(n_records, sums)
    assert dict_data["p_convex_hull"] == 0


def test_auto_monotonic_data_two_prebins():
    # a hull needs three points, so two pre-bins skip the computation
    n_nonevent, n_event = _counts(np.array([0.1, 0.6]))

    dict_data = am.auto_monotonic_data(n_nonevent, n_event)
    assert dict_data["n_prebins"] == 2
    assert dict_data["p_convex_hull"] == 0
    assert dict_data["p_area"] == 0

    dict_data = am.auto_monotonic_data_continuous(
        np.array([100., 100.]), np.array([100., 500.]))
    assert dict_data["n_prebins"] == 2
    assert dict_data["p_convex_hull"] == 0
    assert dict_data["p_area"] == 0


def test_auto_monotonic_data_descriptors():
    event_rate = np.array([0.05, 0.2, 0.5, 0.9, 0.55, 0.25, 0.08])
    n_nonevent, n_event = _counts(event_rate)

    dict_data = am.auto_monotonic_data(n_nonevent, n_event)

    assert dict_data["n_prebins"] == 7
    assert dict_data["n_trend_changes"] == 1
    assert dict_data["p_trend_changes"] == approx(1 / 7)
    assert dict_data["pos_min"] == 0
    assert dict_data["pos_max"] == 3
    assert dict_data["p_bins_min_left"] == 0
    assert dict_data["p_bins_min_right"] == 1
    assert dict_data["p_records_min_left"] == 0
    assert dict_data["p_records_max_left"] == approx(3 / 7)
    assert dict_data["p_records_max_right"] == approx(3 / 7)
    assert 0 < dict_data["p_convex_hull"] <= 1

    # the continuous descriptors are computed from the mean, and agree with
    # the binary ones when the mean is the event rate
    n_records = n_nonevent + n_event
    dict_cont = am.auto_monotonic_data_continuous(n_records, n_event)
    for key, value in dict_data.items():
        assert dict_cont[key] == approx(value), key


# ---------------------------------------------------------------------------
# auto_monotonic: decision trees
# ---------------------------------------------------------------------------

def test_auto_monotonic_decision_leaves():
    # (lr_sense, p_records_min_left, p_records_min_right, p_records_max_left,
    #  p_records_max_right, p_area, p_convex_hull) -> trend. One case per leaf
    # of the tree.
    cases = [
        ((0, 0.5, 0.005, 0.5, 0.5, 0.10, 0.10), "descending"),
        ((0, 0.5, 0.020, 0.5, 0.5, 0.10, 0.10), "valley"),
        ((0, 0.5, 0.500, 0.5, 0.5, 0.10, 0.10), "descending"),
        ((0, 0.5, 0.500, 0.5, 0.1, 0.10, 0.90), "valley"),
        ((0, 0.5, 0.500, 0.5, 0.9, 0.10, 0.90), "descending"),
        ((1, 0.5, 0.500, 0.5, 0.5, 0.10, 0.50), "ascending"),
        ((0, 0.5, 0.010, 0.500, 0.5, 0.50, 0.10), "descending"),
        ((0, 0.5, 0.010, 0.010, 0.5, 0.50, 0.50), "valley"),
        ((0, 0.5, 0.010, 0.050, 0.5, 0.50, 0.50), "descending"),
        ((0, 0.5, 0.010, 0.500, 0.5, 0.50, 0.50), "peak"),
        ((0, 0.01, 0.500, 0.500, 0.01, 0.50, 0.50), "ascending"),
        ((0, 0.01, 0.500, 0.100, 0.50, 0.50, 0.50), "ascending"),
        ((0, 0.01, 0.500, 0.500, 0.50, 0.50, 0.50), "peak"),
        ((0, 0.50, 0.500, 0.500, 0.50, 0.50, 0.50), "valley"),
        ((0, 0.90, 0.500, 0.500, 0.10, 0.50, 0.50), "valley"),
        ((0, 0.90, 0.500, 0.500, 0.50, 0.50, 0.50), "peak"),
    ]

    for args, expected in cases:
        assert am.auto_monotonic_decision(*args) == expected, args


def test_auto_monotonic_asc_desc_decision_leaves():
    # (p_trend_changes, lr_sense, p_records_min_left, p_records_min_right,
    #  p_records_max_left, p_records_max_right, p_area, p_convex_hull)
    cases = [
        ((0.1, 0, 0.50, 0.5, 0.50, 0.010, 0.10, 0.50), "ascending"),
        ((0.1, 0, 0.50, 0.5, 0.50, 0.500, 0.10, 0.50), "descending"),
        ((0.1, 0, 0.50, 0.5, 0.50, 0.500, 0.90, 0.10), "ascending"),
        ((0.1, 0, 0.50, 0.5, 0.50, 0.500, 0.90, 0.90), "descending"),
        ((0.1, 1, 0.50, 0.5, 0.010, 0.500, 0.10, 0.50), "descending"),
        ((0.1, 1, 0.50, 0.5, 0.500, 0.500, 0.10, 0.50), "ascending"),
        ((0.1, 1, 0.50, 0.5, 0.500, 0.500, 0.90, 0.50), "ascending"),
        ((0.1, 1, 0.50, 0.5, 0.900, 0.500, 0.90, 0.50), "descending"),
        ((0.1, 1, 0.81, 0.5, 0.500, 0.500, 0.10, 0.50), "descending"),
        ((0.1, 1, 0.90, 0.5, 0.500, 0.500, 0.10, 0.50), "ascending"),
        ((0.9, 1, 0.90, 0.5, 0.500, 0.500, 0.10, 0.50), "descending"),
    ]

    for args, expected in cases:
        assert am.auto_monotonic_asc_desc_decision(*args) == expected, args


def test_auto_monotonic_decisions_are_total():
    # fuzz the decision trees: every combination of descriptors must land on a
    # named trend, never on the implicit None of a fallen-through if-chain
    rng = np.random.RandomState(42)

    for _ in range(500):
        p = rng.uniform(size=7)
        lr_sense = int(rng.randint(2))

        trend = am.auto_monotonic_decision(
            lr_sense, p[0], p[1], p[2], p[3], p[4], p[5])
        assert trend in ("ascending", "descending", "peak", "valley")

        trend = am.auto_monotonic_asc_desc_decision(
            p[6], lr_sense, p[0], p[1], p[2], p[3], p[4], p[5])
        assert trend in ("ascending", "descending")


def test_auto_monotonic_shapes():
    ascending = np.array([0.02, 0.05, 0.12, 0.30, 0.55, 0.80, 0.92])
    peak = np.array([0.05, 0.2, 0.5, 0.9, 0.55, 0.25, 0.08])

    n_nonevent, n_event = _counts(ascending)
    for auto_mode in ("auto", "auto_heuristic", "auto_asc_desc"):
        assert am.auto_monotonic(n_nonevent, n_event, auto_mode) == "ascending"

    n_nonevent, n_event = _counts(ascending[::-1].copy())
    for auto_mode in ("auto", "auto_heuristic", "auto_asc_desc"):
        assert am.auto_monotonic(
            n_nonevent, n_event, auto_mode) == "descending"

    n_nonevent, n_event = _counts(peak)
    assert am.auto_monotonic(n_nonevent, n_event, "auto") == "peak"
    assert am.auto_monotonic(n_nonevent, n_event, "auto_heuristic") == "peak"
    # the asc/desc tree has no peak leaf: it must still answer
    assert am.auto_monotonic(n_nonevent, n_event, "auto_asc_desc") in (
        "ascending", "descending")

    n_nonevent, n_event = _counts(1 - peak)
    assert am.auto_monotonic(n_nonevent, n_event, "auto") == "valley"


def test_auto_monotonic_continuous_shapes():
    n_records = np.full(7, 100.0)

    means = np.array([1., 2., 4., 8., 16., 32., 70.])
    for auto_mode in ("auto", "auto_heuristic", "auto_asc_desc"):
        assert am.auto_monotonic_continuous(
            n_records, means * n_records, auto_mode) == "ascending"

    means = np.array([1., 3., 9., 20., 8., 3., 1.])
    assert am.auto_monotonic_continuous(
        n_records, means * n_records, "auto") == "peak"


def test_auto_monotonic_unknown_mode_returns_none():
    # the estimators validate monotonic_trend before they get here, so the
    # helper simply falls through
    n_nonevent, n_event = _counts(np.array([0.1, 0.3, 0.7]))
    assert am.auto_monotonic(n_nonevent, n_event, "auto_new") is None
    assert am.auto_monotonic_continuous(
        np.array([1., 2., 3.]), np.array([1., 4., 9.]), "auto_new") is None


# ---------------------------------------------------------------------------
# auto_monotonic: trend classification
# ---------------------------------------------------------------------------

def test_is_peak_is_valley():
    assert am._is_peak(np.array([0.1, 0.9, 0.2]))
    assert not am._is_peak(np.array([0.9, 0.1, 0.8]))
    assert am._is_valley(np.array([0.9, 0.1, 0.8]))
    assert not am._is_valley(np.array([0.1, 0.9, 0.2]))

    # neither a peak nor a valley: two direction changes
    zigzag = np.array([0., 1., 0., 1.])
    assert not am._is_peak(zigzag)
    assert not am._is_valley(zigzag)

    # a constant array satisfies both definitions vacuously
    constant = np.array([1., 1., 1.])
    assert am._is_peak(constant)
    assert am._is_valley(constant)


def test_is_convex_is_concave():
    assert am._is_convex(np.array([1., 0., 1.]))
    assert not am._is_convex(np.array([0., 1., 0.]))
    assert am._is_concave(np.array([0., 1., 0.]))
    assert not am._is_concave(np.array([1., 0., 1.]))

    # fewer than three points: the loop never runs, so both hold
    assert am._is_convex(np.array([1., 0.]))
    assert am._is_concave(np.array([1., 0.]))
    assert am._is_convex(np.array([1.]))
    assert am._is_concave(np.array([1.]))


def test_type_of_monotonic_trend():
    assert am.type_of_monotonic_trend(np.array([0.5])) == "undefined"
    assert am.type_of_monotonic_trend(np.array([0.1, 0.2, 0.3])) == "ascending"
    assert am.type_of_monotonic_trend(
        np.array([0.3, 0.2, 0.1])) == "descending"
    assert am.type_of_monotonic_trend(np.array([0.1, 0.5, 0.2])) == (
        "peak (concave)")
    assert am.type_of_monotonic_trend(
        np.array([0.1, 0.15, 0.9, 0.2])) == "peak"
    assert am.type_of_monotonic_trend(np.array([0.5, 0.1, 0.4])) == (
        "valley (convex)")
    assert am.type_of_monotonic_trend(
        np.array([0.9, 0.85, 0.1, 0.8])) == "valley"

    # two direction changes and no single extreme: not classifiable
    assert am.type_of_monotonic_trend(
        np.array([0., 1., 0., 1.])) == "no monotonic"

    # a constant series has no descent, so it reads as ascending
    assert am.type_of_monotonic_trend(np.array([0.2, 0.2, 0.2])) == "ascending"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_entropy():
    assert mt.entropy([0.5, 0.5]) == approx(np.log(2))
    assert mt.entropy([1.0, 0.0]) == approx(0)
    assert mt.entropy(np.array([0.25] * 4)) == approx(np.log(4))


def test_check_x_y_guards():
    with raises(ValueError):
        mt.gini([1, 2, 3], [1, 2])

    with raises(ValueError):
        mt.gini([1., np.inf], [1., 2.])

    with raises(ValueError):
        mt.jensen_shannon([np.nan, 1.], [1., 1.])

    with raises(ValueError):
        mt.hellinger([1., 2.], [np.inf, 2.])


def test_gini_degenerate():
    # a single bin cannot discriminate
    assert mt.gini([10], [20]) == 0

    # empty bins are masked out, and what is left is a single bin
    assert mt.gini([0, 10], [0, 20]) == 0

    # every bin empty: nothing survives the mask
    assert mt.gini([0, 0], [0, 0]) == 0

    # identical event rates in both bins
    assert mt.gini([10, 10], [10, 10]) == approx(0)

    # perfect separation
    assert mt.gini([0, 50], [50, 0]) == approx(1)
    assert mt.gini([50, 0], [0, 50]) == approx(1)


def test_divergences_identical_distributions():
    p = np.array([0.25, 0.25, 0.5])

    for divergence in (mt.kullback_leibler, mt.jeffrey, mt.jensen_shannon,
                       mt.hellinger, mt.triangular):
        assert divergence(p, p, return_sum=True) == approx(0)

        elementwise = divergence(p, p)
        assert isinstance(elementwise, np.ndarray)
        assert elementwise.shape == p.shape
        assert elementwise == approx(np.zeros(3))


def test_divergences_elementwise_sums_to_return_sum():
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])

    for divergence in (mt.kullback_leibler, mt.jeffrey, mt.jensen_shannon,
                       mt.hellinger, mt.triangular):
        elementwise = divergence(p, q)
        assert elementwise.sum() == approx(divergence(p, q, return_sum=True))

    # these three are non-negative termwise; jeffrey and jensen-shannon are
    # symmetric, kullback-leibler is not
    assert np.all(mt.hellinger(p, q) >= 0)
    assert np.all(mt.triangular(p, q) >= 0)
    assert mt.jeffrey(p, q, True) == approx(mt.jeffrey(q, p, True))
    assert mt.jensen_shannon(p, q, True) == approx(
        mt.jensen_shannon(q, p, True))
    assert mt.kullback_leibler(p, q, True) != approx(
        mt.kullback_leibler(q, p, True))


def test_jensen_shannon_multivariate():
    X = np.array([[0.2, 0.3], [0.3, 0.3], [0.5, 0.4]])

    js = mt.jensen_shannon_multivariate(X)
    assert js > 0

    # the default is a uniform weighting
    assert js == approx(mt.jensen_shannon_multivariate(X, [0.5, 0.5]))
    assert js == approx(
        mt.jensen_shannon_multivariate(X, np.array([0.5, 0.5])))

    # identical distributions carry no divergence
    assert mt.jensen_shannon_multivariate(
        np.array([[0.5, 0.5], [0.5, 0.5]])) == approx(0)

    # a single distribution is compared with itself
    assert mt.jensen_shannon_multivariate(
        np.array([[0.3], [0.7]])) == approx(0)


def test_jensen_shannon_multivariate_guards():
    X = np.array([[0.2, 0.3], [0.3, 0.3], [0.5, 0.4]])

    with raises(ValueError):
        mt.jensen_shannon_multivariate(np.array([0.4, 0.6]))

    with raises(ValueError):
        mt.jensen_shannon_multivariate(X, [0.3, 0.3, 0.4])

    with raises(ValueError):
        mt.jensen_shannon_multivariate(X, [0.5, 0.6])


def test_defect_jensen_shannon_multivariate_array_like():
    # documented as array-like, and np.asarray is called -- but only after the
    # guard has already asked a list for its ndim
    X = [[0.2, 0.3], [0.3, 0.3], [0.5, 0.4]]

    assert mt.jensen_shannon_multivariate(X) == approx(
        mt.jensen_shannon_multivariate(np.asarray(X)))

    # a 1-D list reaches the documented guard rather than AttributeError
    with raises(ValueError):
        mt.jensen_shannon_multivariate([0.4, 0.6])


def test_metrics_test_proportions():
    statistic, pvalue = mt.test_proportions(50, 50, 50, 50)
    assert statistic == approx(0)
    assert pvalue == approx(1)

    # z = (0.9 - 0.1) / sqrt(0.5 * 0.5 * (1 / 100 + 1 / 100))
    statistic, pvalue = mt.test_proportions(90, 10, 10, 90)
    assert statistic == approx(0.8 / np.sqrt(0.005))
    assert 0 <= pvalue <= 1
    assert pvalue < 1e-20

    # the statistic is an absolute value, so the test is symmetric
    assert mt.test_proportions(10, 90, 90, 10) == approx(
        mt.test_proportions(90, 10, 10, 90))

    # more separation is more significant
    assert mt.test_proportions(60, 40, 40, 60)[1] > pvalue


def test_metrics_test_proportions_degenerate():
    # no events at all on either side: the pooled variance is zero and the
    # statistic is undefined rather than an exception
    with np.errstate(invalid="ignore"):
        statistic, pvalue = mt.test_proportions(0, 50, 0, 50)

    assert np.isnan(statistic)
    assert np.isnan(pvalue)


def test_hhi():
    # a single bin holds the whole exposure
    assert mt.hhi([1.0]) == approx(1)
    assert mt.hhi([1.0], normalized=True) == 1

    # a uniform split is perfectly homogeneous once normalized
    assert mt.hhi([0.5, 0.5], normalized=True) == approx(0)
    assert mt.hhi([0.25] * 4, normalized=True) == approx(0)

    assert mt.hhi([0.9, 0.1]) == approx(0.82)
    assert 0 <= mt.hhi([0.9, 0.1], normalized=True) <= 1
    assert mt.hhi(np.array([0.9, 0.1])) == approx(mt.hhi([0.9, 0.1]))


def test_binning_quality_score():
    # no p-values at all: the empty product is 1
    assert mt.binning_quality_score(0, [], 0) == approx(0)

    score = mt.binning_quality_score(0.4, [0.01, 0.02], 0.1)
    assert 0 <= score <= 1

    # a perfectly homogeneous binning scores nothing
    assert mt.binning_quality_score(0.4, [0.01], 1.0) == approx(0)

    # a certain pairwise p-value cancels the score too
    assert mt.binning_quality_score(0.4, [1.0], 0.1) == approx(0)


def test_multiclass_binning_quality_score():
    assert mt.multiclass_binning_quality_score(0, 3, [], 0) == approx(0)

    score = mt.multiclass_binning_quality_score(0.3, 3, [0.01], 0.1)
    assert score == approx(
        mt.binning_quality_score(0.3 / np.log(3), [0.01], 0.1))


def test_continuous_binning_quality_score():
    # rwoe == 0 short-circuits, and a ratio below 1 is clipped rather than
    # allowed to go negative
    assert mt.continuous_binning_quality_score(0, [0.1], 0.2) == approx(0)
    assert mt.continuous_binning_quality_score(0.5, [0.1], 0.2) == approx(0)
    assert mt.continuous_binning_quality_score(1, [0.1], 0.2) == approx(0)

    assert mt.continuous_binning_quality_score(4, [], 0) == approx(0.75)
    assert mt.continuous_binning_quality_score(4, [0.5], 0.5) == approx(0.1875)


# ---------------------------------------------------------------------------
# prebinning
# ---------------------------------------------------------------------------

def test_prebinning_params():
    x, y = _binary_sample(200)

    with raises(ValueError):
        PreBinning("classification", "new_method", 4, 2).fit(x, y)

    with raises(ValueError):
        PreBinning("clustering", "cart", 4, 2).fit(x, y)

    # mdlp is a supervised binary-classification method only
    with raises(ValueError):
        PreBinning("regression", "mdlp", 4, 2).fit(x, y)


def test_prebinning_splits_before_fit():
    assert PreBinning("classification", "cart", 4, 2).splits is None


def test_prebinning_unsupervised_methods():
    x, y = _binary_sample(200)

    for method in ("uniform", "quantile"):
        prebinning = PreBinning("classification", method, 4, 2).fit(x, y)
        splits = prebinning.splits

        # n_bins bins means n_bins - 1 interior edges
        assert len(splits) == 3
        assert np.all(np.diff(splits) > 0)
        assert splits.min() > x.min()
        assert splits.max() < x.max()

    # uniform edges are equally spaced, quantile edges are not
    uniform = PreBinning("classification", "uniform", 4, 2).fit(x, y).splits
    quantile = PreBinning("classification", "quantile", 4, 2).fit(x, y).splits
    assert np.diff(uniform)[0] == approx(np.diff(uniform)[1])
    assert not np.allclose(uniform, quantile)

    # neither method looks at the target
    no_target = PreBinning("classification", "uniform", 4, 2).fit(x, None)
    regression = PreBinning("regression", "uniform", 4, 2).fit(x, x)
    assert np.array_equal(uniform, no_target.splits)
    assert np.array_equal(uniform, regression.splits)


def test_prebinning_unsupervised_constant_x():
    y = np.array([0, 1] * 15)
    x = np.ones(30)

    for method in ("uniform", "quantile"):
        prebinning = PreBinning("classification", method, 4, 2).fit(x, y)
        assert len(prebinning.splits) == 0


def test_prebinning_kwargs_reach_the_estimator():
    x, y = _binary_sample(200)

    # kwargs are forwarded verbatim, so an unknown one is the discretizer's
    # error, not a silently ignored argument
    with raises(TypeError):
        PreBinning("classification", "uniform", 4, 2, not_a_param=1).fit(x, y)

    with raises(TypeError):
        PreBinning("classification", "cart", 4, 2, not_a_param=1).fit(x, y)

    with raises(TypeError):
        PreBinning("classification", "mdlp", 4, 2, not_a_param=1).fit(x, y)

    # and they win over the values the wrapper computes
    overridden = PreBinning("classification", "quantile", 4, 2,
                            strategy="uniform").fit(x, y).splits
    uniform = PreBinning("classification", "uniform", 4, 2).fit(x, y).splits
    assert np.array_equal(overridden, uniform)


def test_prebinning_cart_and_mdlp():
    x, y = _binary_sample(300)

    cart = PreBinning("classification", "cart", 4, 20).fit(x, y).splits
    assert len(cart) <= 3
    assert np.all(np.diff(cart) > 0)

    mdlp = PreBinning("classification", "mdlp", 4, 20).fit(x, y).splits
    assert np.all(np.diff(mdlp) > 0)

    # a degenerate target gives nothing to split on
    assert len(PreBinning("classification", "cart", 4, 20).fit(
        x, np.zeros(300, dtype=int)).splits) == 0
    assert len(PreBinning("classification", "mdlp", 4, 20).fit(
        x, np.zeros(300, dtype=int)).splits) == 0


def test_prebinning_cart_regression():
    x, _ = _binary_sample(300)
    y = 2.0 * x + 1.0

    splits = PreBinning("regression", "cart", 4, 20).fit(x, y).splits

    assert len(splits) <= 3
    assert np.all(np.diff(splits) > 0)

    # a constant target gives the regressor nothing to reduce
    assert len(PreBinning("regression", "cart", 4, 20).fit(
        x, np.zeros(300)).splits) == 0


def test_prebinning_cart_sample_weight():
    x, y = _binary_sample(300)

    unweighted = PreBinning("classification", "cart", 4, 20).fit(x, y).splits
    weighted = PreBinning("classification", "cart", 4, 20).fit(
        x, y, sample_weight=np.ones(300)).splits

    assert np.all(np.diff(weighted) > 0)
    # the same problem, expressed through min_weight_fraction_leaf
    assert len(weighted) == len(unweighted)


# ---------------------------------------------------------------------------
# mdlp
# ---------------------------------------------------------------------------

def test_mdlp_no_candidate_split():
    y = np.array([0, 1] * 15)

    # a single distinct value offers no midpoint to cut at
    assert len(MDLP().fit(np.ones(30), y).splits) == 0

    # and neither does a single row
    assert len(MDLP().fit(np.array([1.0]), np.array([1])).splits) == 0

    # two distinct values do, when the target separates them
    x = np.array([0.0] * 15 + [1.0] * 15)
    splits = MDLP().fit(x, np.array([0] * 15 + [1] * 15)).splits
    assert len(splits) == 1
    assert splits[0] == approx(0.5)


# ---------------------------------------------------------------------------
# outlier detectors
# ---------------------------------------------------------------------------

def test_outlier_detector_base_unfitted():
    detector = OutlierDetector()

    assert detector._support is None
    assert detector._is_fitted is False

    with raises(NotFittedError):
        detector.get_support()

    # the base class is abstract: it has no _fit of its own
    with raises(AttributeError):
        detector.fit(np.arange(10.))


def test_defect_outlier_detectors_unfitted_get_support():
    # get_support documents a NotFittedError, and formats the message with the
    # subclass name -- but no concrete detector initialises _is_fitted
    for detector_class in (RangeDetector, ModifiedZScoreDetector,
                           YQuantileDetector):
        with raises(NotFittedError):
            detector_class().get_support()


def test_outlier_get_support_indices():
    x = np.append(np.arange(1.0, 21.0), 500.0)

    detector = ModifiedZScoreDetector().fit(x)
    mask = detector.get_support()
    indices = detector.get_support(indices=True)

    assert mask.dtype == np.dtype(bool)
    assert mask.shape == x.shape
    assert np.array_equal(np.where(mask)[0], indices)
    assert list(indices) == [20]


# ---------------------------------------------------------------------------
# information printers
# ---------------------------------------------------------------------------

def test_print_optional_parameters(capsys):
    dict_default_options = {"a": None, "b": 5, "c": None, "d": None,
                            "e": None, "f": None}
    dict_user_options = {"a": [1, 2], "b": 5, "c": np.array([1.0]),
                         "d": {"key": 1}, "e": None,
                         "f": RangeDetector()}

    print_optional_parameters(dict_default_options, dict_user_options)
    out = capsys.readouterr().out

    lines = {line.split()[0]: line for line in out.splitlines()
             if line.startswith("    ") and not line.startswith("  End")}

    # a list, an array and a dict are reported as user-supplied, whatever
    # their content
    for key in ("a", "c", "d"):
        assert lines[key].split()[1:] == ["yes", "*", "U"]

    # an estimator is reported the same way, but a default value is not
    assert lines["f"].split()[1:] == ["yes", "*", "U"]
    assert lines["b"].split()[1:] == ["5", "*", "d"]
    assert lines["e"].split()[1:] == ["no", "*", "d"]

    assert "Begin options" in out
    assert "End options" in out


def test_solver_statistics_lp():
    class _LPSolver:
        n_variables = 5
        n_constraints = 3
        n_iterations = 7
        objective = 1.5

    d_solver, time_optimizer = solver_statistics("lp", _LPSolver())

    assert time_optimizer is None
    assert list(d_solver) == ["n_variables", "n_constraints", "n_iterations",
                              "objective"]
    assert d_solver["objective"] == approx(1.5)


def test_solver_statistics_unknown_type():
    d_solver, time_optimizer = solver_statistics("new_solver", None)

    assert d_solver == {}
    assert time_optimizer is None


def test_print_solver_statistics(capsys):
    print_solver_statistics("lp", {"n_variables": 5, "n_constraints": 3,
                                   "n_iterations": 7, "objective": 1.5})
    out = capsys.readouterr().out
    assert "Number of iterations" in out
    assert "1.5000" in out

    print_solver_statistics("mip", {"n_variables": 5, "n_constraints": 3,
                                    "objective": 1.5, "best_bound": 1.25})
    out = capsys.readouterr().out
    assert "Number of constraints" in out
    assert "Number of iterations" not in out

    print_solver_statistics("cp", {"n_booleans": 1, "n_branches": 2,
                                   "n_conflicts": 3, "objective": 4,
                                   "best_objective_bound": 5})
    out = capsys.readouterr().out
    assert "Number of booleans" in out


def test_print_binning_information_all_binning_types(capsys):
    binning_types = {
        "optimalbinning": optimal_binning_default_options,
        "multiclassoptimalbinning": multiclass_optimal_binning_default_options,
        "continuousoptimalbinning":
            continuous_optimal_binning_default_options,
        "sboptimalbinning": sboptimal_binning_default_options,
        "optimalbinning2d": optimal_binning_2d_default_options,
        "continuousoptimalbinning2d":
            continuous_optimal_binning_2d_default_options,
    }

    for binning_type, default_options in binning_types.items():
        print_binning_information(
            binning_type, 2, "x", "OPTIMAL", "cp", None, 1.0, 0.1, 0.2, 0.3,
            0.15, 0.4, 10, 1, dict(default_options))

        out = capsys.readouterr().out

        assert "Begin options" in out
        for key in default_options:
            assert key in out, (binning_type, key)

        # print_level 2 also prints everything print_level 1 prints
        assert "Pre-binning statistics" in out
        assert "Timing" in out


def test_print_binning_information_print_levels(capsys):
    options = dict(optimal_binning_default_options)
    d_solver = {"n_booleans": 1, "n_branches": 2, "n_conflicts": 3,
                "objective": 4, "best_objective_bound": 5}

    print_binning_information("optimalbinning", 0, "x", "OPTIMAL", "cp",
                              d_solver, 1.0, 0.1, 0.2, 0.3, 0.15, 0.4, 10, 1,
                              options)
    out = capsys.readouterr().out
    assert "Time    :" in out
    assert "Pre-binning statistics" not in out
    assert "Begin options" not in out

    print_binning_information("optimalbinning", 1, "", "OPTIMAL", "cp",
                              d_solver, 1.0, 0.1, 0.2, 0.3, 0.15, 0.4, 10, 1,
                              options)
    out = capsys.readouterr().out
    # an empty name is reported as UNKNOWN
    assert "Name    : UNKNOWN" in out
    assert "Solver statistics" in out
    # a cp solver splits the solver time into model generation and optimizer
    assert "model generation" in out
    assert "Begin options" not in out

    # no solver: no solver statistics, and the short timing block
    print_binning_information("optimalbinning", 1, "x", "OPTIMAL", "cp", None,
                              1.0, 0.1, 0.2, 0.3, 0.15, 0.4, 10, 1, options)
    out = capsys.readouterr().out
    assert "Solver statistics" not in out
    assert "model generation" not in out
    assert "Timing" in out

    # an unsolved problem reports no timing at all
    print_binning_information("optimalbinning", 1, "x", "INFEASIBLE", "cp",
                              d_solver, 1.0, 0.1, 0.2, 0.3, 0.15, 0.4, 10, 1,
                              options)
    out = capsys.readouterr().out
    assert "Pre-binning statistics" in out
    assert "Timing" not in out


# ---------------------------------------------------------------------------
# integration: the internals as the estimators reach them
# ---------------------------------------------------------------------------

def test_auto_monotonic_trend_integration():
    x, y = _binary_sample(400)

    for monotonic_trend in ("auto", "auto_asc_desc", "auto_heuristic"):
        optb = OptimalBinning(monotonic_trend=monotonic_trend,
                              prebinning_method="quantile", max_n_prebins=8)
        optb.fit(x, y)

        assert optb.status == "OPTIMAL"

        table = optb.binning_table.build()
        event_rate = table["Event rate"].values[:len(optb.splits) + 1]

        # the event rate of this sample increases with x, so every auto mode
        # must settle on an ascending binning
        assert np.all(np.diff(event_rate.astype(float)) >= 0), monotonic_trend


def test_information_reports_user_options(capsys):
    x, y = _binary_sample(200)

    optb = OptimalBinning(name="x", prebinning_method="quantile",
                          max_n_prebins=6, special_codes=[-9])
    optb.fit(x, y)
    optb.information(print_level=2)

    out = capsys.readouterr().out
    special = [line for line in out.splitlines()
               if line.strip().startswith("special_codes")][0]

    # a container option is reported as present, not printed
    assert special.split()[1:] == ["yes", "*", "U"]
    assert "quantile" in out
