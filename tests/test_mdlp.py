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
    assert optb.binning_table.iv == approx(4.7913274, rel=1e-6)


def test_splits():
    mdlp = MDLP()

    with raises(NotFittedError):
        mdlp.splits
