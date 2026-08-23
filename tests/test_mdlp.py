"""
MDLP testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import MDLP
from optbinning import OptimalBinning
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target


def test_params():
    with raises(ValueError):
        mdlp = MDLP(min_samples_split=-1)
        mdlp.fit(x, y)

    with raises(ValueError):
        mdlp = MDLP(min_samples_leaf=-1)
        mdlp.fit(x, y)

    with raises(ValueError):
        mdlp = MDLP(max_candidates=-1)
        mdlp.fit(x, y)


def test_numerical_default():
    # Properties of the split set, not its literal values: the values are
    # sensitive to numpy's percentile interpolation, which is what left the
    # original literal assertions commented out. See reports/DECISIONS.md.
    mdlp = MDLP()
    mdlp.fit(x, y)

    splits = mdlp.splits

    assert len(splits)
    assert np.all(np.diff(splits) > 0)
    assert x.min() < splits.min()
    assert splits.max() < x.max()

    # every split actually partitions the sample
    for split in splits:
        n_left = np.sum(x <= split)
        assert 0 < n_left < len(x)


def test_numerical_practical():
    min_samples_leaf = int(np.ceil(len(x) * 0.05))
    mdlp = MDLP(max_candidates=128, min_samples_leaf=min_samples_leaf)
    mdlp.fit(x, y)

    assert len(mdlp.splits)

    # no leaf smaller than min_samples_leaf on either side of a split
    for split in mdlp.splits:
        n_left = np.sum(x <= split)
        assert n_left >= min_samples_leaf
        assert len(x) - n_left >= min_samples_leaf


def test_deterministic():
    mdlp_1 = MDLP().fit(x, y)
    mdlp_2 = MDLP().fit(x, y)

    assert mdlp_1.splits == approx(mdlp_2.splits, rel=1e-12)


def test_refit_replaces_previous_splits():
    # fit must reset the estimator, not add to it: _recurse only appends to
    # _splits, and splits sorts them, so a second fit used to return each
    # split twice, interleaved rather than obviously doubled.
    # See reports/DECISIONS.md.
    mdlp = MDLP()

    first = mdlp.fit(x, y).splits
    second = mdlp.fit(x, y).splits

    assert second == approx(first, rel=1e-12)


def test_refit_on_different_data():
    # The second fit must not be contaminated by the first, even where the
    # two fits disagree about how many splits there are.
    x_sep = np.arange(200, dtype=float)
    y_sep = np.array([0] * 100 + [1] * 100)

    mdlp = MDLP()
    mdlp.fit(x, y)

    assert mdlp.fit(x_sep, y_sep).splits == approx([99.5], rel=1e-12)


def test_row_order_independent():
    # The result must depend on the (x, y) pairs, not on the order they arrive
    # in — _find_split reads candidate cuts off y[1:] != y[:-1], so the order
    # the sort gives tied x values changes the answer. "mean radius" has 24
    # tied values carrying both labels; before the fix the IV of
    # OptimalBinning(prebinning_method="mdlp") ranged over 3.85 to 4.82 across
    # orderings, which is how the macOS CI job disagreed with Linux.
    # See reports/DECISIONS.md.
    splits = MDLP().fit(x, y).splits

    rng = np.random.RandomState(0)
    for _ in range(5):
        idx = rng.permutation(len(x))

        assert MDLP().fit(x[idx], y[idx]).splits == approx(splits, rel=1e-12)


def test_separable_target():
    x_sep = np.arange(200, dtype=float)
    y_sep = np.array([0] * 100 + [1] * 100)

    mdlp = MDLP()
    mdlp.fit(x_sep, y_sep)

    assert mdlp.splits == approx([99.5], rel=1e-12)


def test_single_class_target():
    x_one = np.arange(200, dtype=float)
    y_one = np.zeros(200, dtype=int)

    mdlp = MDLP()
    mdlp.fit(x_one, y_one)

    assert not len(mdlp.splits)


def test_min_samples_leaf_larger_than_sample():
    # No candidate can leave min_samples_leaf observations on both sides.
    mdlp = MDLP(min_samples_leaf=len(x))
    mdlp.fit(x, y)

    assert not len(mdlp.splits)


def test_min_samples_leaf_reduces_splits():
    n_default = len(MDLP().fit(x, y).splits)
    n_restricted = len(MDLP(min_samples_leaf=50).fit(x, y).splits)

    assert n_restricted <= n_default


def test_prebinning_method():
    # MDLP as the prebinning stage of OptimalBinning; the only path that
    # reaches optbinning.binning.prebinning.PreBinning's "mdlp" branch.
    optb = OptimalBinning(prebinning_method="mdlp")
    optb.fit(x, y)

    optb.binning_table.build()

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.iv == approx(4.76862756, rel=1e-6)


def test_splits():
    mdlp = MDLP()

    with raises(NotFittedError):
        mdlp.splits


def test_no_split_without_information():
    # The MDLP criterion rejects every candidate on a target that carries no
    # information about x, so the discretisation is a single interval.
    rng = np.random.RandomState(0)
    x_noise = np.arange(200, dtype=float)
    y_noise = rng.randint(0, 2, 200)

    mdlp = MDLP()
    mdlp.fit(x_noise, y_noise)

    assert not len(mdlp.splits)


def test_min_samples_split():
    x_sep = np.arange(200, dtype=float)
    y_sep = np.array([0] * 100 + [1] * 100)

    # A node with fewer than min_samples_split distinct values is not split,
    # the root included.
    mdlp = MDLP(min_samples_split=len(x_sep) + 1)
    mdlp.fit(x_sep, y_sep)

    assert not len(mdlp.splits)


def test_float_target():
    # A binary target read from a pandas column is float, not int.
    mdlp_int = MDLP().fit(x, y)
    mdlp_float = MDLP().fit(x, y.astype(float))

    assert mdlp_float.splits == approx(mdlp_int.splits, rel=1e-12)


def test_target_labels():
    with raises(ValueError):
        MDLP().fit(x, y + 0.5)

    with raises(ValueError):
        MDLP().fit(x, y - 1)
