"""
Scenario-based stochastic optimal binning: edge cases and chaos inputs.

Companion to tests/test_binning_scenarios.py. Everything here targets
SBOptimalBinning paths that the happy-path suite never walks: degenerate
pre-binning, the weights argument, scenarios with different supports,
non-binary targets, and the whole public surface called with the wrong
thing.

Tests named ``test_defect_*`` are known failures kept on purpose; each one
states in its docstring what the near-identical sibling estimator does with
the same input.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from pytest import approx, raises

from optbinning import OptimalBinning
from optbinning.binning.binning_statistics import BinningTable
from optbinning.binning.preprocessing import split_data_scenarios
from optbinning.binning.uncertainty import SBOptimalBinning
from sklearn.exceptions import NotFittedError


def _make_scenario(n_samples, seed, loc=5.0):
    """Small logistic scenario: x ~ U(0, 10), P(y=1) rising through ``loc``."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n_samples)
    p = 1.0 / (1.0 + np.exp(-(x - loc)))
    y = (rng.rand(n_samples) < p).astype(int)

    return x, y


x1, y1 = _make_scenario(150, 1)
x2, y2 = _make_scenario(150, 2, loc=4.5)
x3, y3 = _make_scenario(150, 3, loc=5.5)

x_s = [x1, x2, x3]
y_s = [y1, y2, y3]


def _stepped_scenario():
    """A scenario whose first prebin is pure non-event, so pre-binning
    refinement always removes the split that closes it."""
    rng = np.random.RandomState(7)
    n_samples = 200
    x = np.arange(n_samples, dtype=float)
    y = (rng.rand(n_samples) < 0.15 + 0.7 * x / n_samples).astype(int)
    y[x < 25] = 0

    return x, y


x_step, y_step = _stepped_scenario()


# --------------------------------------------------------------------------
# Parameter validation
# --------------------------------------------------------------------------
def test_params_types_and_bounds():
    with raises(ValueError):
        SBOptimalBinning(max_n_prebins=2.5).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(max_n_prebins=1).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(min_prebin_size=0.).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(min_n_bins=2.5).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(max_n_bins=0).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(min_bin_size="0.1").fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(max_bin_size=1.5).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(min_event_rate_diff=-0.1).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(max_pvalue=0.).fit(x_s, y_s)

    with raises(TypeError):
        SBOptimalBinning(class_weight=1).fit(x_s, y_s)

    with raises(TypeError):
        SBOptimalBinning(user_splits=(1, 2)).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(split_digits=-1).fit(x_s, y_s)

    with raises(ValueError):
        SBOptimalBinning(time_limit="10").fit(x_s, y_s)

    with raises(TypeError):
        SBOptimalBinning(verbose="True").fit(x_s, y_s)


def test_params_boundaries_are_accepted():
    # The open/closed ends of the documented intervals must fit, not raise.
    sboptb = SBOptimalBinning(min_prebin_size=0.5, min_bin_size=0.5,
                              max_bin_size=1.0, max_pvalue=1.0,
                              min_event_rate_diff=1.0, split_digits=0,
                              max_n_prebins=2)
    sboptb.fit([x1], [y1])

    assert sboptb.status == "OPTIMAL"


# --------------------------------------------------------------------------
# X / Y / weights container checks
# --------------------------------------------------------------------------
def test_input_containers():
    # A tuple of scenarios is not a list, and a 2-D array is not a list.
    with raises(TypeError):
        SBOptimalBinning().fit((x1, x2), [y1, y2])

    with raises(TypeError):
        SBOptimalBinning().fit([x1, x2], (y1, y2))

    with raises(TypeError):
        SBOptimalBinning().fit(np.array([x1, x2]), np.array([y1, y2]))

    with raises(ValueError):
        SBOptimalBinning().fit(x_s, y_s, weights=[1, 1, 1, 1])


def test_input_mismatched_lengths_inside_a_scenario():
    # check_input=False: numpy blows up on the ragged masks.
    with raises(ValueError):
        SBOptimalBinning(max_n_prebins=5).fit([x1[:100]], [y1])

    # check_input=True: sklearn reports it properly.
    with raises(ValueError):
        SBOptimalBinning(max_n_prebins=5).fit([x1[:100]], [y1],
                                              check_input=True)


def test_empty_scenario_list():
    # Zero scenarios means zero samples, so min_samples_leaf becomes 0 and
    # the CART pre-binner rejects it. InvalidParameterError is a ValueError.
    with raises(ValueError):
        SBOptimalBinning().fit([], [])


def test_check_input_true_is_accepted():
    sboptb = SBOptimalBinning(max_n_prebins=5)
    sboptb.fit(x_s, y_s, check_input=True)

    assert sboptb.status == "OPTIMAL"


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def test_weights_equivalent_to_duplicated_scenarios():
    kwargs = dict(max_n_prebins=6, monotonic_trend="ascending")

    weighted = SBOptimalBinning(**kwargs).fit([x1, x2], [y1, y2],
                                              weights=[2, 1])
    repeated = SBOptimalBinning(**kwargs).fit([x1, x1, x2], [y1, y1, y2])

    assert weighted.status == "OPTIMAL"
    assert repeated.status == "OPTIMAL"
    assert weighted.splits == approx(repeated.splits)


def test_weights_accepts_floats_and_arrays():
    sboptb = SBOptimalBinning(max_n_prebins=5)
    sboptb.fit([x1, x2], [y1, y2], weights=[0.3, 0.7])
    assert sboptb.status == "OPTIMAL"

    sboptb = SBOptimalBinning(max_n_prebins=5)
    sboptb.fit([x1, x2], [y1, y2], weights=np.array([1, 3]))
    assert sboptb.status == "OPTIMAL"


def test_weights_unvalidated_values():
    # _check_X_Y_weights only checks the length; the values are handed
    # straight to the pre-binner and to OR-Tools, which reject them.
    with raises((TypeError, ValueError)):
        SBOptimalBinning(max_n_prebins=5).fit([x1, x2], [y1, y2],
                                              weights=[-1., 2.])

    with raises(ValueError):
        SBOptimalBinning(max_n_prebins=5).fit([x1, x2], [y1, y2],
                                              weights=[0, 0])

    with raises((TypeError, ValueError)):
        SBOptimalBinning(max_n_prebins=5).fit([x1, x2], [y1, y2],
                                              weights=["a", "b"])

    # A scalar is not sized, so the length check itself fails.
    with raises(TypeError):
        SBOptimalBinning(max_n_prebins=5).fit([x1, x2], [y1, y2], weights=2)


# --------------------------------------------------------------------------
# Single / duplicated / differently supported scenarios
# --------------------------------------------------------------------------
def test_single_scenario_matches_optimal_binning():
    # One equally weighted scenario is the deterministic problem, so the
    # extensive form must reproduce OptimalBinning exactly.
    kwargs = dict(max_n_prebins=6, monotonic_trend="ascending")

    optb = OptimalBinning(**kwargs)
    optb.fit(x1, y1)

    sboptb = SBOptimalBinning(**kwargs)
    sboptb.fit([x1], [y1])

    assert sboptb.status == optb.status == "OPTIMAL"
    assert sboptb.splits == approx(optb.splits)
    assert (sboptb.binning_table.build()["IV"].values[-1] ==
            approx(optb.binning_table.build()["IV"].values[-1]))


def test_duplicated_scenarios_match_a_single_one():
    kwargs = dict(max_n_prebins=6, monotonic_trend="ascending")

    single = SBOptimalBinning(**kwargs).fit([x1], [y1])
    triple = SBOptimalBinning(**kwargs).fit([x1, x1, x1], [y1, y1, y1])

    assert single.splits == approx(triple.splits)
    assert len(triple._binning_tables) == 3


def test_scenarios_with_different_supports():
    # Overlapping but not identical supports: prebins outside a scenario's
    # range are pure there, so refinement pushes every split into the
    # common region.
    rng = np.random.RandomState(11)
    n_samples = 200
    xa = rng.uniform(0, 8, n_samples)
    ya = (rng.rand(n_samples) < 1. / (1. + np.exp(-(xa - 4)))).astype(int)
    xb = rng.uniform(2, 10, n_samples)
    yb = (rng.rand(n_samples) < 1. / (1. + np.exp(-(xb - 5)))).astype(int)

    sboptb = SBOptimalBinning(max_n_prebins=8, monotonic_trend="ascending")
    sboptb.fit([xa, xb], [ya, yb])

    assert sboptb.status == "OPTIMAL"
    assert sboptb._n_refinements >= 1

    splits = sboptb.splits
    assert np.all(np.diff(splits) > 0)
    assert np.all(splits > max(xa.min(), xb.min()))
    assert np.all(splits < min(xa.max(), xb.max()))

    # Every scenario keeps its own support in its own table.
    assert sboptb.binning_table_scenario(0).max_x == approx(xa.max())
    assert sboptb.binning_table_scenario(1).min_x == approx(xb.min())


# --------------------------------------------------------------------------
# Degenerate pre-binning (see defects below for the crashes)
# --------------------------------------------------------------------------
def test_defect_constant_scenario_produces_no_prebins():
    """A constant x leaves the CART pre-binner with no split at all.

    OptimalBinning, ContinuousOptimalBinning and MulticlassOptimalBinning
    all fit the same data and return an empty ``splits`` with a one-bin
    table; SBOptimalBinning raises IndexError in post-processing because
    it indexes the empty ``n_nonevent`` as if it were 2-D.
    """
    x = np.ones(120)
    y = np.tile([0, 1], 60)

    sboptb = SBOptimalBinning(verbose=True)
    sboptb.fit([x, x.copy()], [y, y.copy()])

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits.size == 0
    assert sboptb.binning_table.build().shape[0] == 4


def test_defect_all_prebins_pure_are_removed():
    """A perfectly separable target makes every prebin pure, so pre-binning
    refinement recurses until no split is left. Same crash as above, reached
    through ``_compute_prebins`` instead of through the pre-binner.
    """
    x = np.arange(200, dtype=float)
    y = (x >= 100).astype(int)

    sboptb = SBOptimalBinning(max_n_prebins=5)
    sboptb.fit([x], [y])

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits.size == 0


def test_no_prebins_counts_every_sample_per_scenario():
    """The single-bin fallback must keep each scenario's own counts, not a
    pooled or a zeroed one. Two constant-x scenarios of different size and
    different event rate pin that.
    """
    xa = np.ones(120)
    ya = np.tile([0, 1], 60)
    xb = np.ones(80)
    yb = np.concatenate([np.zeros(60, int), np.ones(20, int)])

    sboptb = SBOptimalBinning()
    sboptb.fit([xa, xb], [ya, yb])

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits.size == 0

    ta = sboptb.binning_table_scenario(0).build(add_totals=False)
    assert ta["Count"].sum() == 120
    assert ta["Event"].sum() == 60

    tb = sboptb.binning_table_scenario(1).build(add_totals=False)
    assert tb["Count"].sum() == 80
    assert tb["Event"].sum() == 20

    t = sboptb.binning_table.build(add_totals=False)
    assert t["Count"].sum() == 200
    assert t["Event"].sum() == 80


def test_time_limit_zero_yields_a_single_bin():
    # The solver gets no time, returns a non-optimal status, and cp.solve
    # falls back to the all-in-one-bin solution. That path must still build
    # a table.
    sboptb = SBOptimalBinning(max_n_prebins=6, time_limit=0)
    sboptb.fit([x1], [y1])

    assert sboptb.status in ("OPTIMAL", "FEASIBLE", "UNKNOWN", "INFEASIBLE")

    table = sboptb.binning_table.build(add_totals=False)
    assert table["Count"].sum() == len(x1)


# --------------------------------------------------------------------------
# user_splits / user_splits_fixed / split_digits
# --------------------------------------------------------------------------
def test_split_digits():
    sboptb = SBOptimalBinning(max_n_prebins=6, split_digits=2)
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits == approx(np.round(sboptb.splits, 2))

    sboptb = SBOptimalBinning(max_n_prebins=6, split_digits=0)
    sboptb.fit(x_s, y_s)

    assert sboptb.splits == approx(np.floor(sboptb.splits))


def test_user_splits_unsorted_and_ndarray():
    unsorted_splits = np.array([7., 3., 5.])

    sboptb = SBOptimalBinning(user_splits=unsorted_splits)
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert np.all(np.diff(sboptb.splits) > 0)
    assert set(sboptb.splits).issubset(set(unsorted_splits))


def test_user_splits_rejected_values():
    with raises(ValueError, match="Input contains NaN"):
        SBOptimalBinning(user_splits=[3., np.nan]).fit(x_s, y_s)

    with raises(TypeError, match="Cannot cast array data"):
        SBOptimalBinning(user_splits=["a", "b"]).fit(x_s, y_s)


def test_empty_user_splits_fits_a_single_bin():
    """``user_splits=[]`` means "no split points", not an error.

    ``_fit`` had no empty-split branch, so the empty list reached
    ``check_array`` and came back as "Found array with 0 sample(s)
    (shape=(0,)) while a minimum of 1 is required". OptimalBinning,
    ContinuousOptimalBinning and MulticlassOptimalBinning all fit a single
    bin on the same input; ``_prebinning_refinement`` already early-returns
    the empty counts, so the fit delegates to it.
    """
    sboptb = SBOptimalBinning(user_splits=[])
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert len(sboptb.splits) == 0

    table = sboptb.binning_table.build()
    # One bin covering everything, plus the special, missing and totals rows.
    assert len(table) == 4
    assert table["Bin"].values[0] == "(-inf, inf)"
    assert table["Count"].values[0] == sum(len(x) for x in x_s)
    assert table["Non-event"].values[0] == sum(
        np.count_nonzero(yy == 0) for yy in y_s)
    assert table["Event"].values[0] == sum(np.count_nonzero(yy) for yy in y_s)

    # Every scenario gets its own single-bin table, not an empty one.
    for i, (xx, yy) in enumerate(zip(x_s, y_s)):
        scenario = sboptb.binning_table_scenario(i).build()
        assert scenario["Count"].values[0] == len(xx)
        assert scenario["Event"].values[0] == np.count_nonzero(yy)


def test_refit_does_not_mutate_user_splits_fixed():
    """``fit`` must leave its constructor parameters alone.

    ``_fit`` stored ``np.asarray(user_splits_fixed)[sorted_idx]`` back on the
    public attribute, so the second ``fit`` handed numpy ``bool_`` to
    ``_check_parameters`` and was rejected with "user_splits_fixed must be
    list of boolean".
    """
    user_splits = [7., 3., 5.]
    user_splits_fixed = [False, False, True]

    sboptb = SBOptimalBinning(user_splits=list(user_splits),
                              user_splits_fixed=user_splits_fixed)
    sboptb.fit(x_s, y_s)
    first_splits = sboptb.splits

    assert sboptb.user_splits_fixed is user_splits_fixed
    assert list(sboptb.user_splits) == user_splits
    # The private copies carry the sort and the refinement; 5. was flagged.
    assert list(sboptb._user_splits) == [5.]
    assert list(sboptb._user_splits_fixed) == [True]

    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits == approx(first_splits)
    assert 5. in sboptb.splits


def test_user_splits_fixed_pure_prebin_raises():
    # The bin below 25 is pure non-event, so refinement wants to drop the
    # split at 25 -- which is exactly the one pinned as fixed.
    sboptb = SBOptimalBinning(user_splits=[25., 60., 100., 140.],
                              user_splits_fixed=[True, False, False, False])

    with raises(ValueError, match="Fixed user_splits"):
        sboptb.fit([x_step, x_step], [y_step, y_step])


def test_defect_user_splits_fixed_not_reaching_the_solver():
    """``_compute_prebins`` shrinks ``self._user_splits_fixed`` after a prebin
    is removed, but ``_fit_optimizer`` hands ``self.user_splits_fixed`` -- the
    untouched, full-length array -- to BinningCP. The fixed flag therefore
    lands on the wrong split index.

    OptimalBinning, ContinuousOptimalBinning and MulticlassOptimalBinning all
    update the public attribute, which is the one the optimizer reads. Here
    the split at 25 is dropped, so the flag on 100 slides onto 140:
    OptimalBinning returns [100.], SBOptimalBinning returns [140.].
    """
    user_splits = [25., 60., 100., 140.]
    user_splits_fixed = [False, False, True, False]

    optb = OptimalBinning(user_splits=list(user_splits),
                          user_splits_fixed=list(user_splits_fixed),
                          max_n_bins=2)
    optb.fit(x_step, y_step)
    assert 100. in optb.splits

    sboptb = SBOptimalBinning(user_splits=list(user_splits),
                              user_splits_fixed=list(user_splits_fixed),
                              max_n_bins=2)
    sboptb.fit([x_step], [y_step])

    assert 100. in sboptb.splits


def test_defect_user_splits_fixed_unsorted_spurious_error():
    """Second face of the same defect: ``_compute_prebins`` reads the
    *unsorted* ``self._user_splits_fixed`` while ``mask_splits`` is in sorted
    order, so an unsorted ``user_splits`` misidentifies which split is fixed.

    Fixing 140. while 25. is the split that gets removed, OptimalBinning fits
    and returns [140.]; SBOptimalBinning raises "Fixed user_splits [140.] are
    removed because produce pure prebins".
    """
    user_splits = [140., 25., 100., 60.]
    user_splits_fixed = [True, False, False, False]

    optb = OptimalBinning(user_splits=list(user_splits),
                          user_splits_fixed=list(user_splits_fixed),
                          max_n_bins=2)
    optb.fit(x_step, y_step)
    assert 140. in optb.splits

    sboptb = SBOptimalBinning(user_splits=list(user_splits),
                              user_splits_fixed=list(user_splits_fixed),
                              max_n_bins=2)
    sboptb.fit([x_step], [y_step])

    assert 140. in sboptb.splits


def test_user_splits_fixed_on_the_last_split_is_honoured():
    """Third face of the same defect: a flag on the last split was silently
    dropped rather than misplaced. BinningCP.add_constraint_fixed_splits
    reads user_splits_fixed[i] for i in range(n - 1) with n taken from the
    *refined* prebins, so an over-long array never reaches its final entry
    and the constraint vanished with no error.
    """
    user_splits = [25., 60., 100., 140.]
    user_splits_fixed = [False, False, False, True]

    optb = OptimalBinning(user_splits=list(user_splits),
                          user_splits_fixed=list(user_splits_fixed),
                          max_n_bins=2)
    optb.fit(x_step, y_step)
    assert 140. in optb.splits

    sboptb = SBOptimalBinning(user_splits=list(user_splits),
                              user_splits_fixed=list(user_splits_fixed),
                              max_n_bins=2)
    sboptb.fit([x_step], [y_step])

    assert sboptb.status == "OPTIMAL"
    assert 140. in sboptb.splits


# --------------------------------------------------------------------------
# special codes
# --------------------------------------------------------------------------
def _scenarios_with_specials():
    rng = np.random.RandomState(13)
    xa = np.r_[x1, np.full(15, -1.), np.full(10, np.nan)]
    ya = np.r_[y1, rng.randint(0, 2, 15), rng.randint(0, 2, 10)]
    xb = np.r_[x2, np.full(12, -1.), np.full(8, np.nan)]
    yb = np.r_[y2, rng.randint(0, 2, 12), rng.randint(0, 2, 8)]

    return [xa, xb], [ya, yb]


def test_special_codes_list_and_ndarray():
    xx, yy = _scenarios_with_specials()

    sboptb = SBOptimalBinning(max_n_prebins=6, special_codes=[-1])
    sboptb.fit(xx, yy)
    table = sboptb.binning_table.build(add_totals=False)

    assert sboptb.status == "OPTIMAL"
    assert table["Count"].values[-2] == 27      # 15 + 12 special
    assert table["Count"].values[-1] == 18      # 10 + 8 missing

    # Per-scenario tables carry their own share.
    assert sboptb.binning_table_scenario(0).build(
        add_totals=False)["Count"].values[-2] == 15

    sboptb_arr = SBOptimalBinning(max_n_prebins=6,
                                  special_codes=np.array([-1]))
    sboptb_arr.fit(xx, yy)

    assert sboptb_arr.splits == approx(sboptb.splits)


def test_special_codes_dict_is_rejected():
    # SBOptimalBinning is the one estimator of the family whose
    # _check_parameters does not allow the dict (named buckets) form, and
    # its internals really cannot carry it: _prebinning_refinement collapses
    # the special samples with target_info (one scalar count), while the dict
    # form needs one count per named bucket. Pinned so the guard is not
    # relaxed without also fixing the counting.
    special_codes = {"a": -1, "b": [-2, -3]}
    xx, yy = _scenarios_with_specials()

    with raises(TypeError):
        SBOptimalBinning(special_codes=special_codes).fit(xx, yy)

    # The preprocessing step underneath does understand the dict form.
    out = split_data_scenarios(xx, yy, None, special_codes, False)
    assert len(out[4][0]) == 15

    # ... but a BinningTable built the way _fit builds it cannot: one scalar
    # special count against two named buckets.
    table = BinningTable("v", "numerical", special_codes, np.array([1., 2.]),
                         np.array([10, 20, 30, 5, 2]),
                         np.array([5, 10, 15, 3, 1]), 0., 3., None, None, None)
    with raises(ValueError):
        table.build()


def test_transform_special_and_missing_metrics():
    xx, yy = _scenarios_with_specials()

    sboptb = SBOptimalBinning(max_n_prebins=6, special_codes=[-1])
    sboptb.fit(xx, yy)

    default = sboptb.transform([-1., np.nan, 5.])
    assert default[:2] == approx([0., 0.])

    empirical = sboptb.transform([-1., np.nan, 5.], metric_special="empirical",
                                 metric_missing="empirical")
    assert empirical[0] != 0.
    assert empirical[1] != 0.
    assert empirical[2] == approx(default[2])


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------
def test_transform_metrics():
    sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend="ascending")
    sboptb.fit(x_s, y_s)

    n_bins = len(sboptb.splits) + 1
    probe = np.array([0.5, 5., 9.5])

    woe = sboptb.transform(probe)
    event_rate = sboptb.transform(probe, metric="event_rate")
    indices = sboptb.transform(probe, metric="indices")
    bins = sboptb.transform(probe, metric="bins")

    assert np.all(np.diff(event_rate) > 0)          # ascending trend
    assert np.all(np.diff(woe) < 0)                 # WoE mirrors it
    assert np.all(event_rate >= 0.) and np.all(event_rate <= 1.)
    assert indices.dtype.kind == "i"
    assert np.all(indices < n_bins)
    assert bins[0].startswith("(-inf")
    assert bins[-1].endswith("inf)")

    # show_digits reaches the formatting of the bin labels.
    assert "," in sboptb.transform([5.], metric="bins", show_digits=5)[0]

    with raises(ValueError):
        sboptb.transform(probe, metric="not_a_metric")

    # Empty input and 2-D input are both tolerated.
    assert sboptb.transform([]).size == 0
    assert sboptb.transform(np.column_stack([probe, probe])).shape == (3, 2)


def test_fit_transform_with_weights():
    sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend="ascending")
    x_transform = sboptb.fit_transform([1., 5., 9.], [x1, x2], [y1, y2],
                                       weights=[1, 3], metric="event_rate")

    assert sboptb.status == "OPTIMAL"
    assert np.all(np.diff(x_transform) > 0)


def test_unfitted_access():
    sboptb = SBOptimalBinning()

    with raises(NotFittedError):
        sboptb.transform([1., 2.])

    with raises(NotFittedError):
        sboptb.splits

    with raises(NotFittedError):
        sboptb.binning_table

    with raises(NotFittedError):
        sboptb.information()

    with raises(NotFittedError):
        sboptb.binning_table_scenario(0)


# --------------------------------------------------------------------------
# binning tables and information
# --------------------------------------------------------------------------
def test_binning_table_aggregates_the_scenarios():
    sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend="ascending")
    sboptb.fit(x_s, y_s)

    aggregated = sboptb.binning_table.build(add_totals=False)
    per_scenario = [sboptb.binning_table_scenario(s).build(add_totals=False)
                    for s in range(len(x_s))]

    total = sum(t["Count"].values for t in per_scenario)
    assert np.array_equal(aggregated["Count"].values, total)
    assert aggregated["Count"].sum() == sum(len(x) for x in x_s)

    assert sboptb.binning_table.min_x == approx(min(x.min() for x in x_s))
    assert sboptb.binning_table.max_x == approx(max(x.max() for x in x_s))

    assert all(isinstance(t, BinningTable)
               for t in sboptb._binning_tables)


def test_binning_table_scenario_bad_identifiers():
    sboptb = SBOptimalBinning(max_n_prebins=5)
    sboptb.fit(x_s, y_s)

    for bad in (-1, 3, 1.0, "0", None):
        with raises(ValueError):
            sboptb.binning_table_scenario(bad)


def test_information_print_levels(capsys):
    sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend="ascending")
    sboptb.fit(x_s, y_s)

    sboptb.information(print_level=0)
    assert "Status" in capsys.readouterr().out

    sboptb.information(print_level=1)
    assert "Solver statistics" in capsys.readouterr().out

    sboptb.information(print_level=2)
    out = capsys.readouterr().out
    assert "Begin options" in out
    assert "monotonic_trend" in out

    with raises(ValueError):
        sboptb.information(print_level=-1)

    with raises(ValueError):
        sboptb.information(print_level="2")


def test_binning_table_analysis_and_plot(tmp_path):
    sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend="ascending")
    sboptb.fit(x_s, y_s)

    table = sboptb.binning_table
    table.build()
    table.analysis(print_output=False)

    assert table.iv > 0
    assert table.js >= 0

    table.plot(savefig=str(tmp_path / "scenarios.png"))

    # A scenario table comes back unbuilt, so plot must refuse until built.
    scenario_table = sboptb.binning_table_scenario(0)
    with raises(NotFittedError):
        scenario_table.plot(savefig=str(tmp_path / "scenario_0.png"))

    scenario_table.build()
    scenario_table.plot(savefig=str(tmp_path / "scenario_0.png"))


# --------------------------------------------------------------------------
# Constraints and options
# --------------------------------------------------------------------------
def test_monotonic_trends_and_constraints():
    for trend in ("ascending", "descending", "concave", "convex", "peak",
                  "valley"):
        sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend=trend)
        sboptb.fit(x_s, y_s)

        assert sboptb.status == "OPTIMAL"
        assert np.all(np.diff(sboptb.splits) > 0)

    ascending = SBOptimalBinning(max_n_prebins=8, monotonic_trend="ascending")
    ascending.fit(x_s, y_s)
    event_rate = ascending.binning_table.build(
        add_totals=False)["Event rate"].values[:-2]
    assert np.all(np.diff(event_rate) > 0)


def test_prebinning_methods_and_class_weight():
    for method in ("cart", "quantile", "uniform"):
        sboptb = SBOptimalBinning(prebinning_method=method, max_n_prebins=6)
        sboptb.fit(x_s, y_s)

        assert sboptb.status == "OPTIMAL"
        assert np.all(np.diff(sboptb.splits) > 0)

    balanced = SBOptimalBinning(max_n_prebins=6, class_weight="balanced")
    balanced.fit(x_s, y_s)
    assert balanced.status == "OPTIMAL"

    weighted = SBOptimalBinning(max_n_prebins=6, class_weight={0: 1, 1: 3})
    weighted.fit(x_s, y_s)
    assert weighted.status == "OPTIMAL"


def test_min_max_n_bins_and_pvalue():
    sboptb = SBOptimalBinning(max_n_prebins=8, min_n_bins=2, max_n_bins=3)
    sboptb.fit(x_s, y_s)

    n_bins = len(sboptb.splits) + 1
    assert 2 <= n_bins <= 3

    for policy in ("consecutive", "all"):
        sboptb = SBOptimalBinning(max_n_prebins=8, max_pvalue=0.01,
                                  max_pvalue_policy=policy)
        sboptb.fit(x_s, y_s)
        assert sboptb.status == "OPTIMAL"

    sboptb = SBOptimalBinning(max_n_prebins=8, min_event_rate_diff=0.15,
                              monotonic_trend="ascending")
    sboptb.fit(x_s, y_s)
    event_rate = sboptb.binning_table.build(
        add_totals=False)["Event rate"].values[:-2]
    assert np.all(np.diff(event_rate) >= 0.15 - 1e-9)


# --------------------------------------------------------------------------
# Odd data
# --------------------------------------------------------------------------
def test_non_binary_target_is_treated_as_nonzero_equals_event():
    rng = np.random.RandomState(5)
    y_multi = rng.randint(0, 3, len(x1))

    sboptb = SBOptimalBinning(max_n_prebins=5)
    sboptb.fit([x1], [y_multi])

    table = sboptb.binning_table.build(add_totals=False)
    assert table["Event"].sum() == np.count_nonzero(y_multi != 0)
    assert table["Non-event"].sum() == np.count_nonzero(y_multi == 0)


def test_infinite_values_are_rejected():
    x_inf = x1.copy()
    x_inf[0] = np.inf
    x_inf[1] = -np.inf

    with raises(ValueError):
        SBOptimalBinning(max_n_prebins=5).fit([x_inf], [y1])


def test_extreme_magnitudes():
    scale = 1e12
    sboptb = SBOptimalBinning(max_n_prebins=6, monotonic_trend="ascending")
    sboptb.fit([x1 * scale, x2 * scale], [y1, y2])

    reference = SBOptimalBinning(max_n_prebins=6,
                                 monotonic_trend="ascending")
    reference.fit([x1, x2], [y1, y2])

    assert sboptb.status == "OPTIMAL"
    assert np.all(np.isfinite(sboptb.splits))
    assert sboptb.splits == approx(reference.splits * scale, rel=1e-6)


def test_duplicated_and_low_cardinality_values():
    # Three distinct values only: at most two splits can ever exist.
    x = np.repeat([1., 2., 3.], 60)
    rng = np.random.RandomState(17)
    y = (rng.rand(180) < np.repeat([0.2, 0.5, 0.8], 60)).astype(int)

    sboptb = SBOptimalBinning(max_n_prebins=10, monotonic_trend="ascending")
    sboptb.fit([x, x.copy()], [y, y.copy()])

    assert sboptb.status == "OPTIMAL"
    assert len(sboptb.splits) <= 2
    assert sboptb.binning_table.build(
        add_totals=False)["Count"].sum() == 2 * len(x)


def test_verbose_logs_the_user_split_count():
    # The "user splits supplied" line needs verbose=True together with
    # user_splits, a combination no other test uses.
    buf = StringIO()
    optb = SBOptimalBinning(user_splits=[0.0, 1.0], verbose=True)

    with redirect_stdout(buf):
        optb.fit(x_s, y_s)

    assert optb.status == "OPTIMAL"
