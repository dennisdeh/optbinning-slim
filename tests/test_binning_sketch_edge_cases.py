"""
OptimalBinningSketch, BSketch/BCatSketch and GK edge-case testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import BinningProcessSketch
from optbinning import OptimalBinningSketch
from optbinning.binning.distributed import bsketch as bsketch_module
from optbinning.binning.distributed import binning_sketch as binning_sketch_mod
from optbinning.binning.distributed.bsketch import BCatSketch, BSketch
from optbinning.binning.distributed.gk import GK
from optbinning.exceptions import NotSolvedError

from sklearn.exceptions import NotFittedError


def _numerical_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return x, y


def _categorical_data(n=400, seed=42):
    rng = np.random.RandomState(seed)
    cats = np.array(["a", "b", "c", "d", "e", "f", "g", "h"], dtype=object)
    p = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])
    x = rng.choice(cats, size=n, p=p)
    rate = {"a": 0.1, "b": 0.2, "c": 0.35, "d": 0.5,
            "e": 0.6, "f": 0.7, "g": 0.8, "h": 0.9}
    y = np.array([rng.rand() < rate[v] for v in x]).astype(int)
    return x, y


# ---------------------------------------------------------------- GK sketch

def test_gk_add_compresses_on_threshold():
    # compress threshold is int(1 / eps) + 1
    gk = GK(eps=0.5)
    assert gk._compress_threshold == 3

    gk.add(1.0)
    gk.add(2.0)
    assert len(gk.incoming) == 2
    assert gk.entries == []

    gk.add(3.0)
    # the third add hits _count % threshold == 0 and flushes incoming
    assert gk.incoming == []
    assert len(gk.entries)
    assert gk.n == 3
    assert gk._min == 1.0
    assert gk._max == 3.0
    assert gk._sum == 6.0


def test_gk_merge_requires_same_eps():
    gk1 = GK(eps=0.01)
    gk2 = GK(eps=0.02)

    assert not gk1.mergeable(gk2)

    with raises(Exception, match="gk does not share signature."):
        gk1.merge(gk2)


def test_gk_merge_empty_other_is_a_noop():
    gk1 = GK(eps=0.1)
    for v in range(20):
        gk1.add(float(v))

    n_entries = len(gk1)
    gk1.merge(GK(eps=0.1))

    assert gk1.n == 20
    assert len(gk1) == n_entries


def test_gk_merge_into_empty_copies():
    gk1 = GK(eps=0.1)
    gk2 = GK(eps=0.1)
    for v in range(20):
        gk2.add(float(v))

    gk1.merge(gk2)

    assert gk1.n == gk2.n
    assert gk1._min == gk2._min
    assert gk1._max == gk2._max
    assert gk1._sum == gk2._sum
    # copy, not alias
    assert gk1.entries is not gk2.entries


def test_gk_merge_disjoint_ranges():
    rng = np.random.RandomState(0)
    gk1 = GK(eps=0.05)
    gk2 = GK(eps=0.05)

    for v in rng.rand(200):
        gk1.add(float(v))
    for v in rng.rand(200) + 10.0:
        gk2.add(float(v))

    gk1.merge(gk2)

    assert gk1.n == 400
    assert gk1._min < 1.0
    assert gk1._max > 10.0

    values = [e.value for e in gk1.entries]
    assert values == sorted(values)

    # the median of a 50/50 mixture of [0, 1) and [10, 11) sits in one of the
    # two clusters, and every quantile stays inside the observed range
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert gk1._min <= gk1.quantile(q) <= gk1._max


def test_gk_merge_compress_removal_paths():
    # eps=0.5 makes remove_threshold large enough that entries are dropped in
    # every branch of merge_compress
    rng = np.random.RandomState(7)
    gk1 = GK(eps=0.5)
    gk2 = GK(eps=0.5)

    for v in rng.rand(50):
        gk1.add(float(v))
    for v in rng.rand(50) + 0.5:
        gk2.add(float(v))

    gk1.merge(gk2)

    assert gk1.n == 100
    # aggressive compression keeps only a handful of entries
    assert len(gk1) < 10
    # g values accumulate as entries are removed
    assert sum(e.g for e in gk1.entries) <= gk1.n
    assert max(e.g for e in gk1.entries) > 1


def test_gk_quantile_guards():
    gk = GK(eps=0.1)

    with raises(ValueError, match="GK sketch does not contain values."):
        gk.quantile(0.5)

    with raises(ValueError, match=r"q must be a value in \[0, 1\]."):
        gk.quantile(-0.1)

    with raises(ValueError, match=r"q must be a value in \[0, 1\]."):
        gk.quantile(1.5)


def test_gk_quantile_flushes_incoming():
    gk = GK(eps=0.5)
    gk.add(3.0)
    gk.add(1.0)

    assert len(gk.incoming) == 2

    assert gk.quantile(1.0) == 3.0
    assert gk.incoming == []


def test_gk_quantile_returns_min_when_first_entry_dominates():
    rng = np.random.RandomState(0)
    gk = GK(eps=0.5)
    for v in rng.rand(20):
        gk.add(float(v))

    # the first surviving entry carries enough weight that rank + spread is
    # already exceeded, so quantile falls back to the minimum
    assert gk.quantile(0.0) == gk._min
    assert gk.quantile(0.0) == min(gk._min, gk.entries[0].value)


def test_gk_quantile_is_monotone_and_bounded():
    rng = np.random.RandomState(3)
    gk = GK(eps=0.01)
    for v in rng.normal(size=500):
        gk.add(float(v))

    qs = [gk.quantile(q) for q in np.linspace(0, 1, 11)]

    assert qs == sorted(qs)
    assert gk._min <= qs[0]
    assert qs[-1] <= gk._max
    assert gk.n == 500


def test_gk_handles_infinities_and_extreme_magnitudes():
    gk = GK(eps=0.1)
    for v in (-np.inf, -1e300, 0.0, 1e300, np.inf):
        gk.add(v)

    assert gk._min == -np.inf
    assert gk._max == np.inf
    assert gk.n == 5
    assert gk.quantile(1.0) == np.inf


def test_gk_single_value():
    gk = GK(eps=0.1)
    gk.add(7.0)

    assert gk.n == 1
    assert gk.quantile(0.0) == 7.0
    assert gk.quantile(0.5) == 7.0
    assert gk.quantile(1.0) == 7.0


# ------------------------------------------------------------------ BSketch

def test_bsketch_params():
    with raises(ValueError, match="Invalid value for sketch."):
        BSketch(sketch="new_sketch")

    with raises(ValueError, match="K must be a positive integer"):
        BSketch(K=0)

    with raises(ValueError, match="K must be a positive integer"):
        BSketch(K=2.5)

    with raises(TypeError, match="special_codes must be a list or"):
        BSketch(special_codes={1, 2, 3})

    # a dict of named special buckets is accepted by OptimalBinning but not
    # by the sketch estimators, whose docstrings say "array-like or None"
    with raises(TypeError, match="special_codes must be a list or"):
        BSketch(special_codes={"a": [1, 2]})


def test_bsketch_tdigest_unavailable(monkeypatch):
    monkeypatch.setattr(bsketch_module, "TDIGEST_AVAILABLE", False)

    with raises(ImportError, match="Cannot import tdigest."):
        BSketch(sketch="t-digest")


def test_defect_bsketch_accepts_out_of_range_eps():
    # OptimalBinningSketch rejects the same value, and so must BSketch: the
    # guard used to read "not isinstance(...) and not 0 <= eps <= 1", so the
    # range half never ran for a real number
    with raises(ValueError):
        OptimalBinningSketch(eps=-1.0)

    with raises(ValueError, match=r"eps must be a value in \[0, 1\]"):
        BSketch(eps=-1.0)

    with raises(ValueError, match=r"eps must be a value in \[0, 1\]"):
        BSketch(eps=2.0)


def test_bsketch_eps_guard_rejects_non_numbers():
    # a 0-d numpy array is not a numbers.Number, so the guard short-circuits
    # on the isinstance half before comparing
    with raises(ValueError, match=r"eps must be a value in \[0, 1\]"):
        BSketch(eps=np.array(2.0))

    with raises(ValueError, match=r"eps must be a value in \[0, 1\]"):
        BSketch(eps="not-a-number")


def test_bsketch_add_missing_and_special():
    x = np.array([1.0, 2.0, np.nan, 3.0, -1.0, np.nan, 4.0, -1.0])
    y = np.array([0, 1, 1, 1, 0, 0, 1, 1])

    sk = BSketch(sketch="gk", eps=0.1, special_codes=[-1.0])
    sk.add(x, y)

    assert sk._count_missing_e == 1
    assert sk._count_missing_ne == 1
    assert sk._count_special_e == 1
    assert sk._count_special_ne == 1

    assert sk.n_event == 5
    assert sk.n_nonevent == 3
    assert sk.n == 8


def test_bsketch_bins_counts_every_record():
    x, y = _numerical_data(n=200, seed=1)
    sk = BSketch(sketch="gk", eps=1e-3)
    sk.add(x, y)

    splits = np.array([-0.5, 0.0, 0.5])
    bins_e, bins_ne = sk.bins(splits)

    assert bins_e.shape == (4,)
    assert bins_ne.shape == (4,)
    assert bins_e.dtype == np.int64
    assert bins_e.sum() + bins_ne.sum() == 200


def test_bsketch_merge_requires_same_signature():
    sk1 = BSketch(sketch="gk", eps=0.01)
    sk2 = BSketch(sketch="gk", eps=0.02)

    assert not sk1._mergeable(sk2)

    with raises(Exception, match="bsketch does not share signature."):
        sk1.merge(sk2)


def test_bsketch_merge_empty_other_is_a_noop():
    x, y = _numerical_data(n=100, seed=2)
    sk1 = BSketch(sketch="gk", eps=1e-2)
    sk1.add(x, y)

    sk1.merge(BSketch(sketch="gk", eps=1e-2))

    assert sk1.n == 100


def test_bsketch_merge_into_empty_copies_counts():
    x = np.array([1.0, 2.0, np.nan, 3.0, -1.0])
    y = np.array([0, 1, 1, 1, 0])

    sk1 = BSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])
    sk2 = BSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])
    sk2.add(x, y)

    assert sk1.n == 0
    sk1.merge(sk2)

    assert sk1.n == sk2.n
    assert sk1._count_missing_e == sk2._count_missing_e
    assert sk1._count_missing_ne == sk2._count_missing_ne
    assert sk1._count_special_e == sk2._count_special_e
    assert sk1._count_special_ne == sk2._count_special_ne

    # the receiver copies the two sketches; aliasing them would make every
    # later add() on either instance grow the other one too
    assert sk1._sketch_e is not sk2._sketch_e
    assert sk1._sketch_ne is not sk2._sketch_ne

    sk1.add(np.array([10.0, 11.0]), np.array([1, 1]))

    assert sk1.n == 7
    assert sk2.n == 5


def test_bsketch_merge_keeps_the_other_sketch_special_counts():
    # a worker whose whole chunk was special codes or missing values has an
    # empty quantile sketch. merge() returned early on that, dropping every
    # count the worker carried.
    sk1 = BSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])
    sk2 = BSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])

    sk1.add(np.arange(10.0), np.array([0, 1] * 5))
    sk2.add(np.full(6, -1.0), np.array([0, 1] * 3))
    sk2.add(np.full(4, np.nan), np.array([0, 1] * 2))

    assert sk2._sketch_e.n == 0 and sk2._sketch_ne.n == 0

    sk1.merge(sk2)

    assert sk1._count_special_e == 3
    assert sk1._count_special_ne == 3
    assert sk1._count_missing_e == 2
    assert sk1._count_missing_ne == 2
    assert sk1.n == 20


def test_bsketch_mergeable_compares_special_codes_as_sets():
    sk1 = BSketch(sketch="gk", eps=1e-2, special_codes=[1, 2])
    sk2 = BSketch(sketch="gk", eps=1e-2, special_codes=[2, 1])
    sk3 = BSketch(sketch="gk", eps=1e-2, special_codes=[1, 3])
    sk4 = BSketch(sketch="gk", eps=1e-2)

    assert sk1._mergeable(sk2)
    assert not sk1._mergeable(sk3)
    # None means "no special codes", i.e. the empty set -- not a wildcard
    assert not sk1._mergeable(sk4)
    assert not sk4._mergeable(sk1)
    assert sk4._mergeable(BSketch(sketch="gk", eps=1e-2))


def test_bsketch_merge_refuses_an_uncoded_sketch():
    # sk2 has no special codes, so its -1.0 is a clean value. Merging it
    # into sk1, for which -1.0 is special, would put the same value in the
    # quantile sketch and in the special bucket of one merged stream.
    sk1 = BSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])
    sk2 = BSketch(sketch="gk", eps=1e-2)

    x = np.array([1.0, -1.0])
    y = np.array([0, 1])
    sk1.add(x, y)
    sk2.add(x, y)

    with raises(Exception, match="bsketch does not share signature."):
        sk1.merge(sk2)

    assert sk1._count_special_e == 1
    assert sk1._sketch_e.n == 0
    assert sk1.n == 2


def test_bsketch_merge_sketches_tdigest():
    x, y = _numerical_data(n=200, seed=4)
    sk = BSketch(sketch="t-digest", eps=1e-2)
    sk.add(x, y)

    merged = sk.merge_sketches()

    assert merged.n == 200
    assert merged.percentile(0) <= merged.percentile(50)
    assert merged.percentile(50) <= merged.percentile(100)


# --------------------------------------------------------------- BCatSketch

def test_bcatsketch_params():
    with raises(ValueError, match=r"cat_cutoff must be in \(0, 1.0\]"):
        BCatSketch(cat_cutoff=2.0)

    with raises(ValueError, match=r"cat_cutoff must be in \(0, 1.0\]"):
        BCatSketch(cat_cutoff="0.1")

    with raises(TypeError, match="special_codes must be a list or"):
        BCatSketch(special_codes={1, 2})


def test_bcatsketch_add_missing_and_special():
    x = np.array(["a", "b", None, "a", "SPECIAL", "b", None, "SPECIAL"],
                 dtype=object)
    y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

    sk = BCatSketch(special_codes=["SPECIAL"])
    sk.add(x, y)

    assert sk._count_missing_e == 1
    assert sk._count_missing_ne == 1
    assert sk._count_special_e == 1
    assert sk._count_special_ne == 1
    assert sk.n == 8


def test_bcatsketch_bins_without_cutoff_is_sorted_by_event_rate():
    x, y = _categorical_data(n=300, seed=5)
    sk = BCatSketch()
    sk.add(x, y)

    (splits, categories, bin_ne, bin_e, cat_others, bin_ne_others,
     bin_e_others) = sk.bins()

    assert cat_others == []
    assert bin_ne_others == []
    assert bin_e_others == []

    assert len(categories) == len(np.unique(x))
    assert len(splits) == len(categories) - 1
    assert bin_ne.sum() + bin_e.sum() == 300

    event_rate = bin_e / (bin_ne + bin_e)
    assert list(event_rate) == sorted(event_rate)


def test_bcatsketch_bins_with_cutoff_collects_others():
    x, y = _categorical_data(n=300, seed=6)
    sk = BCatSketch(cat_cutoff=0.1)
    sk.add(x, y)

    (splits, categories, bin_ne, bin_e, cat_others, bin_ne_others,
     bin_e_others) = sk.bins()

    assert len(cat_others)
    assert bin_ne_others + bin_e_others > 0
    assert set(categories).isdisjoint(set(cat_others))
    total = bin_ne.sum() + bin_e.sum() + bin_ne_others + bin_e_others
    assert total == 300


def test_bcatsketch_merge_accumulates_categories():
    x, y = _categorical_data(n=200, seed=8)

    sk1 = BCatSketch(special_codes=["SPECIAL"])
    sk2 = BCatSketch(special_codes=["SPECIAL"])

    sk1.add(x[:100], y[:100])
    sk2.add(x[100:], y[100:])

    sk_all = BCatSketch(special_codes=["SPECIAL"])
    sk_all.add(x, y)

    sk1.merge(sk2)

    assert sk1.n == 200
    assert sk1.n_event == sk_all.n_event
    assert sk1.n_nonevent == sk_all.n_nonevent
    assert set(sk1._d_categories) == set(sk_all._d_categories)
    for k, v in sk_all._d_categories.items():
        assert sk1._d_categories[k] == v


def test_bcatsketch_merge_of_disjoint_categories():
    sk1 = BCatSketch()
    sk2 = BCatSketch()

    sk1.add(np.array(["a", "a", "b"], dtype=object), np.array([0, 1, 1]))
    sk2.add(np.array(["c", "c"], dtype=object), np.array([0, 1]))

    sk1.merge(sk2)

    assert set(sk1._d_categories) == {"a", "b", "c"}
    assert sk1.n == 5


def test_bcatsketch_copy_does_not_alias_the_other_sketch():
    # BCatSketch.merge never calls _copy -- unlike BSketch.merge -- so it is
    # only reachable by a direct call, but it must still not hand the
    # receiver a reference to the other instance's category counts.
    sk1 = BCatSketch(special_codes=[1, 2])
    sk2 = BCatSketch(special_codes=[2, 1])

    sk2.add(np.array(["a", "b"], dtype=object), np.array([0, 1]))
    sk1._copy(sk2)

    assert sk1._d_categories == sk2._d_categories
    assert sk1._d_categories is not sk2._d_categories
    assert sk1._d_categories["a"] is not sk2._d_categories["a"]
    assert sk1.n == sk2.n

    sk1.add(np.array(["a"], dtype=object), np.array([0]))

    assert sk1._d_categories["a"] == [2, 0]
    assert sk2._d_categories["a"] == [1, 0]


def test_bcatsketch_mergeable_compares_special_codes_as_sets():
    sk1 = BCatSketch(special_codes=[1, 2])
    sk2 = BCatSketch(special_codes=[2, 1])
    sk3 = BCatSketch(special_codes=[3])
    sk4 = BCatSketch()

    assert sk1._mergeable(sk2)
    assert not sk1._mergeable(sk3)
    # None means "no special codes", i.e. the empty set -- not a wildcard
    assert not sk1._mergeable(sk4)
    assert not sk4._mergeable(sk1)
    assert sk4._mergeable(BCatSketch())


def test_bcatsketch_merge_refuses_an_uncoded_sketch():
    # sk2 has no special codes, so its "Z" is a plain category. Merging it
    # into sk1, for which "Z" is special, left "Z" both as a category and
    # in the special bucket -- the corruption the signature guard exists
    # for, reached through the None side the guard used to trust.
    sk1 = BCatSketch(special_codes=["Z"])
    sk2 = BCatSketch()

    x = np.array(["a", "Z"], dtype=object)
    sk1.add(x, np.array([0, 1]))
    sk2.add(x, np.array([0, 1]))

    with raises(Exception, match="bcatsketch does not share signature."):
        sk1.merge(sk2)

    assert set(sk1._d_categories) == {"a"}
    assert sk1._count_special_e == 1
    assert sk1.n == 2


def test_defect_bcatsketch_merge_ignores_the_special_codes_signature():
    # BSketch.merge refuses a sketch with a different signature; BCatSketch
    # .merge had no such guard, so a value that is a special code for one
    # instance and a plain category for the other landed in both the special
    # bucket and the categories of the merged sketch.
    sk1 = BCatSketch(special_codes=["Z"])
    sk2 = BCatSketch(special_codes=["Q"])

    x = np.array(["a", "b", "Z", "Q"], dtype=object)
    sk1.add(x, np.array([0, 1, 0, 1]))
    sk2.add(x, np.array([1, 0, 1, 0]))

    assert not sk1._mergeable(sk2)

    before = {k: list(v) for k, v in sk1._d_categories.items()}

    with raises(Exception, match="bcatsketch does not share signature."):
        sk1.merge(sk2)

    # the refused merge left the receiver untouched
    assert sk1._d_categories == before
    assert sk1._count_special_e == 0
    assert sk1._count_special_ne == 1
    assert sk1.n == 4


def test_bcatsketch_merge_accepts_an_equivalent_signature():
    sk1 = BCatSketch(special_codes=["Z", "Y"])
    sk2 = BCatSketch(special_codes=["Y", "Z"])

    sk1.add(np.array(["a", "Z"], dtype=object), np.array([0, 1]))
    sk2.add(np.array(["b", "Y"], dtype=object), np.array([1, 0]))

    sk1.merge(sk2)

    assert set(sk1._d_categories) == {"a", "b"}
    assert sk1.n == 4


def test_bcatsketch_merge_does_not_alias_the_other_sketch_counts():
    # a category only the other instance has used to be inserted by
    # reference, so a later add() on either sketch mutated both
    sk1 = BCatSketch()
    sk2 = BCatSketch()

    sk1.add(np.array(["a"], dtype=object), np.array([0]))
    sk2.add(np.array(["c", "c"], dtype=object), np.array([0, 1]))

    sk1.merge(sk2)

    assert sk1._d_categories["c"] is not sk2._d_categories["c"]

    sk1.add(np.array(["c"] * 5, dtype=object), np.array([1] * 5))

    assert sk1._d_categories["c"] == [1, 6]
    assert sk2._d_categories["c"] == [1, 1]
    assert sk2.n == 2


# ------------------------------------------------- OptimalBinningSketch

def test_sketch_params_extra():
    with raises(TypeError, match="cat_unknown must be a number or string"):
        OptimalBinningSketch(cat_unknown=[1, 2])

    with raises(TypeError, match="special_codes must be a list or"):
        OptimalBinningSketch(special_codes={"a": [1, 2]})

    with raises(ValueError, match=r"eps must be a value in \[0, 1\]"):
        OptimalBinningSketch(eps=2.0)


def test_sketch_pympler_unavailable(monkeypatch):
    monkeypatch.setattr(binning_sketch_mod, "PYMPLER_AVAILABLE", False)

    with raises(ImportError, match="Cannot import pympler."):
        OptimalBinningSketch()


def test_solve_before_add():
    optb = OptimalBinningSketch()

    with raises(NotFittedError, match="No data was added."):
        optb.solve()


def test_unsolved_accessors():
    optb = OptimalBinningSketch()

    with raises(NotSolvedError):
        optb.binning_table

    with raises(NotSolvedError):
        optb.splits

    with raises(NotSolvedError):
        optb.status

    with raises(NotSolvedError):
        optb.transform(np.array([1.0]))

    with raises(NotSolvedError):
        optb.plot_progress()


def test_merge_requires_same_params():
    x, y = _numerical_data(n=100, seed=9)

    optb1 = OptimalBinningSketch(eps=1e-2)
    optb2 = OptimalBinningSketch(eps=1e-3)

    optb1.add(x, y)
    optb2.add(x, y)

    assert not optb1.mergeable(optb2)

    with raises(Exception, match="optbsketch does not share signature."):
        optb1.merge(optb2)


def test_merge_verbose():
    x, y = _numerical_data(n=200, seed=10)

    optb1 = OptimalBinningSketch(eps=1e-2, max_n_prebins=8, verbose=True)
    optb2 = OptimalBinningSketch(eps=1e-2, max_n_prebins=8, verbose=True)

    optb1.add(x[:100], y[:100])
    optb2.add(x[100:], y[100:])
    optb1.merge(optb2)

    optb1.solve()

    assert optb1.status == "OPTIMAL"
    assert optb1._bsketch.n == 200


def test_constant_x_yields_a_single_bin():
    x = np.ones(200)
    y = np.array([0, 1] * 100)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=5,
                                verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table.build()
    assert table["Count"].values[0] == 200
    assert optb.binning_table.iv == approx(0.0, abs=1e-12)


def test_defect_information_after_single_bin_solve(capsys):
    # the solver is not run when pre-binning leaves a single bin, so
    # self._optimizer is never assigned; like every sibling estimator,
    # __init__ must leave it None so information() omits the solver section
    x = np.ones(100)
    y = np.array([0, 1] * 50)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=5)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert optb._optimizer is None

    optb.information(print_level=1)

    out = capsys.readouterr().out
    assert "Status  : OPTIMAL" in out
    assert "Pre-binning statistics" in out
    # print_solver_statistics is skipped when the solver was not run
    assert "Solver statistics" not in out


def test_two_rows_solve():
    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=4)
    optb.add(np.array([1.0, 2.0]), np.array([0, 1]))
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0


def test_defect_solve_crashes_on_degenerate_numerical_input():
    # every prebin of a single-class stream is dropped, so the refinement
    # recurses down to one bin and used to index mask_remove[-2] on a
    # length-1 array, making solve() raise IndexError. Pre-binning must
    # return that bin instead, exactly as OptimalBinning._compute_prebins
    # does: a single-class target is degenerate but legal.
    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=4)
    optb.add(np.arange(50.0), np.zeros(50, dtype=int))

    splits, n_nonevent, n_event = optb._prebinning_data()

    assert len(splits) == 0
    assert n_nonevent.tolist() == [50]
    assert n_event.tolist() == [0]

    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table.build()
    assert table["Count"].values.tolist() == [50, 0, 0, 50]
    assert table["Non-event"].values.tolist() == [50, 0, 0, 50]
    assert table["Event"].values.tolist() == [0, 0, 0, 0]
    assert optb.binning_table.iv == approx(0.0, abs=1e-12)
    assert optb.binning_table.gini == approx(0.0, abs=1e-12)


def test_defect_solve_crashes_on_single_row():
    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=4)
    optb.add(np.array([1.0]), np.array([1]))

    splits, n_nonevent, n_event = optb._prebinning_data()

    assert len(splits) == 0
    assert n_nonevent.tolist() == [0]
    assert n_event.tolist() == [1]

    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table.build()
    assert table["Count"].values.tolist() == [1, 0, 0, 1]
    assert table["Non-event"].values.tolist() == [0, 0, 0, 0]
    assert table["Event"].values.tolist() == [1, 0, 0, 1]


def test_defect_solve_crashes_on_degenerate_categorical_input():
    # the same missing base case in the sibling refinement
    # _compute_cat_prebins: every category is pure, so the merge recurses
    # down to one bin holding all of them
    x = np.array(["a"] * 10 + ["b"] * 10, dtype=object)
    y = np.zeros(20, dtype=int)

    optb = OptimalBinningSketch(dtype="categorical", max_n_prebins=4)
    optb.add(x, y)

    splits, n_nonevent, n_event = optb._prebinning_data()

    assert len(splits) == 0
    assert n_nonevent.tolist() == [20]
    assert n_event.tolist() == [0]

    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 1
    assert sorted(optb.splits[0]) == ["a", "b"]

    table = optb.binning_table.build()
    assert table["Count"].values.tolist() == [20, 0, 0, 20]
    assert table["Non-event"].values.tolist() == [20, 0, 0, 20]
    assert table["Event"].values.tolist() == [0, 0, 0, 0]


def test_single_prebin_with_special_codes_is_not_double_counted():
    # all the events sit in the special bucket, so the clean prebins are pure
    # and the refinement collapses them into one bin -- the path that used to
    # raise IndexError. The clean bin must hold the clean records only: the
    # whole-stream totals count the special and missing buckets in, and
    # bin_info appends those as rows of their own.
    x = np.concatenate([np.arange(20.0), np.full(10, -1.0),
                        np.full(4, np.nan)])
    y = np.array([0] * 20 + [1] * 10 + [0, 1, 0, 1])

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, special_codes=[-1.0],
                                max_n_prebins=4)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table.build()
    assert table["Count"].values[0] == 20
    assert table["Count"].values[-1] == len(x)


def test_single_cat_prebin_with_special_codes_is_not_double_counted():
    x = np.array(["a"] * 10 + ["b"] * 10 + ["Z"] * 10, dtype=object)
    y = np.array([0] * 20 + [1] * 10)

    optb = OptimalBinningSketch(dtype="categorical", special_codes=["Z"],
                                max_n_prebins=4)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    assert table["Count"].values[0] == 20
    assert table["Count"].values[-1] == len(x)


def test_single_prebin_totals_exclude_special_and_missing_buckets():
    # constant clean x collapses to one prebin without ever hitting the
    # refinement base case, so this isolates the post-processing bug: the
    # single bin used to be given the whole-stream totals, which already
    # count the special and missing buckets bin_info appends separately
    x = np.concatenate([np.ones(20), np.full(10, -1.0), np.full(4, np.nan)])
    y = np.array([0, 1] * 10 + [0] * 5 + [1] * 5 + [0, 1, 0, 1])

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, special_codes=[-1.0],
                                max_n_prebins=5)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table.build()
    assert table["Count"].values[0] == 20      # clean records only
    assert table["Count"].values[1] == 10      # special
    assert table["Count"].values[2] == 4       # missing
    assert table["Count"].values[-1] == len(x)


def test_single_prebin_totals_exclude_the_cat_others_bucket():
    # the third bucket bin_info appends as a row of its own. The whole-stream
    # totals self._bsketch.n_* count it in, so handing them to the single
    # clean bin counted every "others" record twice as well.
    x = np.array(["a"] * 10 + ["b"] * 10 + ["r"] * 2 + ["Z"] * 6 +
                 [None] * 4, dtype=object)
    y = np.array([0] * 20 + [0, 1] + [0, 0, 0, 1, 1, 1] + [0, 0, 1, 1])

    optb = OptimalBinningSketch(dtype="categorical", special_codes=["Z"],
                                cat_cutoff=0.1, max_n_prebins=4)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    # the whole-stream totals count every bucket in ...
    assert optb._bsketch.n_nonevent == 26
    assert optb._bsketch.n_event == 6

    table = optb.binning_table.build()

    # ... so they belong to the Totals row, never to the single clean bin
    assert table["Count"].values.tolist() == [20, 2, 6, 4, 32]
    assert table["Non-event"].values.tolist() == [20, 1, 3, 2, 26]
    assert table["Event"].values.tolist() == [0, 1, 3, 2, 6]
    assert table["Count"].values[-1] == len(x)


def test_solve_on_a_stream_whose_events_have_not_arrived_yet():
    # a partial stream is the canonical streaming case for the missing base
    # case: solve() over the non-event chunk alone used to raise IndexError
    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=4)
    optb.add(np.arange(30.0), np.zeros(30, dtype=int))

    splits, n_nonevent, n_event = optb._prebinning_data()
    assert len(splits) == 0

    optb.add(np.arange(30.0, 60.0), np.ones(30, dtype=int))
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert optb.binning_table.build()["Count"].values[-1] == 60


def test_solve_on_an_all_special_numerical_stream():
    # every record is a special code, so the quantile sketch is empty and
    # pre-binning has nothing to split. A degenerate stream is legal input:
    # solve() succeeds with a single, empty clean bin and the table still
    # accounts for every record.
    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])
    optb.add(np.full(10, -1.0), np.array([0, 1] * 5))
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0

    table = optb.binning_table.build()

    assert list(table["Count"].values) == [0, 10, 0, 10]
    assert list(table["Event"].values) == [0, 5, 0, 5]
    assert table.loc["Totals", "Count"] == 10

    optb.binning_table.analysis(print_output=False)


def test_solve_on_an_all_special_categorical_stream():
    # the categorical sibling: BCatSketch.bins() returns no category at
    # all, so bin_info received empty count arrays and raised IndexError
    optb = OptimalBinningSketch(dtype="categorical", special_codes=["Z"])
    optb.add(np.array(["Z"] * 10, dtype=object), np.array([0, 1] * 5))
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert [list(b) for b in optb.splits] == [[]]

    table = optb.binning_table.build()

    assert list(table["Count"].values) == [0, 10, 0, 10]
    assert list(table["Event"].values) == [0, 5, 0, 5]
    assert table.loc["Totals", "Count"] == 10

    optb.binning_table.analysis(print_output=False)


def test_solve_on_an_all_missing_categorical_stream():
    optb = OptimalBinningSketch(dtype="categorical")
    optb.add(np.array([np.nan] * 10, dtype=object), np.array([0, 1] * 5))
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert [list(b) for b in optb.splits] == [[]]

    table = optb.binning_table.build()

    assert list(table["Count"].values) == [0, 0, 10, 10]
    assert table.loc["Totals", "Count"] == 10


def test_merge_into_an_empty_aggregator_does_not_alias_the_worker():
    # the distributed pattern: an aggregator that has so far seen only
    # special codes merges a worker's sketch. BSketch._copy handed over the
    # worker's GK objects by reference, so every later add() on the
    # aggregator also grew the worker's own counts.
    agg = OptimalBinningSketch(sketch="gk", eps=1e-2, special_codes=[-1.0])
    worker = OptimalBinningSketch(sketch="gk", eps=1e-2,
                                  special_codes=[-1.0])

    agg.add(np.full(4, -1.0), np.array([0, 1, 0, 1]))
    worker.add(np.arange(20.0), np.array([0, 1] * 10))

    agg.merge(worker)

    assert agg._bsketch._sketch_e is not worker._bsketch._sketch_e
    assert agg._bsketch._sketch_ne is not worker._bsketch._sketch_ne
    assert agg._bsketch.n == 24

    agg.add(np.arange(1000., 1020.), np.ones(20))

    assert agg._bsketch.n == 44
    assert worker._bsketch.n == 20

    worker.solve()

    assert worker.binning_table.build()["Count"].values[-1] == 20


def test_bin_size_bounds():
    x, y = _numerical_data(n=300, seed=11)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=10,
                                min_bin_size=0.1, max_bin_size=0.9)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    counts = table["Count"].values[:-3]
    assert counts.min() >= np.ceil(0.1 * 300)
    assert counts.max() <= np.ceil(0.9 * 300)


def test_defect_split_digits_is_not_applied():
    # split_digits was validated and documented but never applied: the
    # sketch estimator has its own pre-binning and never rounded.
    x, y = _numerical_data(n=2000, seed=0)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-4, max_n_prebins=20,
                                split_digits=2)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) > 1
    assert optb.splits.tolist() == np.round(optb.splits, 2).tolist()


def test_split_digits_none_leaves_every_digit():
    x, y = _numerical_data(n=2000, seed=0)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-4, max_n_prebins=20)
    optb.add(x, y)
    optb.solve()

    assert optb.splits.tolist() != np.round(optb.splits, 2).tolist()


def test_split_digits_collision_leaves_no_duplicate_split():
    # rounding can collapse two quantiles onto one value. _compute_prebins
    # drops the empty prebin that leaves only on the iv/js branch; under
    # "hellinger"/"triangular" it merely raises _flag_min_n_event_nonevent,
    # so the duplicate splits used to reach the optimizer -- MODEL_INVALID
    # for "hellinger", a TypeError out of OR-Tools for "triangular".
    x, y = _numerical_data(n=2000, seed=0)

    for divergence in ("iv", "js", "hellinger", "triangular"):
        optb = OptimalBinningSketch(sketch="gk", eps=1e-4, max_n_prebins=20,
                                    split_digits=0, divergence=divergence)
        optb.add(x, y)
        optb.solve()

        assert optb.status == "OPTIMAL"

        prebins = optb._splits_prebinning
        assert len(prebins) == len(np.unique(prebins))

        splits = optb.splits
        assert len(splits) == len(np.unique(splits))
        assert splits.tolist() == np.round(splits, 0).tolist()

        table = optb.binning_table.build()
        assert table["Count"].values[-1] == 2000
        # every bin holds records: no empty prebin survived the collision.
        # The table carries n_bins + 3 rows (Special, Missing, Totals).
        assert (table["Count"].values[:-3] > 0).all()


def test_split_digits_does_not_round_categorical_positions():
    # a categorical split is the ordinal position of a category boundary,
    # not a user-facing number, so rounding it would move categories
    # between bins. Only the numerical branch rounds.
    x, y = _categorical_data(n=400, seed=42)

    optb_none = OptimalBinningSketch(dtype="categorical", max_n_prebins=10)
    optb_none.add(x, y)
    optb_none.solve()

    optb_zero = OptimalBinningSketch(dtype="categorical", max_n_prebins=10,
                                     split_digits=0)
    optb_zero.add(x, y)
    optb_zero.solve()

    assert len(optb_zero.splits) == len(optb_none.splits)
    for bin_zero, bin_none in zip(optb_zero.splits, optb_none.splits):
        assert sorted(bin_zero) == sorted(bin_none)


def test_split_digits_reaches_the_binning_process_sketch():
    rng = np.random.RandomState(3)
    x = rng.normal(size=1000)
    df = pd.DataFrame({"v": x})
    target = (x + rng.normal(scale=0.5, size=1000) > 0).astype(int)

    bpsketch = BinningProcessSketch(["v"], max_n_prebins=20, split_digits=2)
    bpsketch.add(df, target)
    bpsketch.solve()

    splits = bpsketch.get_binned_variable("v").splits

    assert len(splits) > 1
    assert splits.tolist() == np.round(splits, 2).tolist()


def test_divergence_hellinger_flags_min_n_event_nonevent():
    x, y = _numerical_data(n=300, seed=0)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=20,
                                divergence="hellinger")
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert optb._flag_min_n_event_nonevent

    optb.binning_table.build()
    assert optb.binning_table.hellinger > 0


def test_divergence_triangular_with_explicit_min_counts():
    x, y = _numerical_data(n=300, seed=0)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=20,
                                divergence="triangular",
                                min_bin_n_nonevent=5, min_bin_n_event=5)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    assert optb._flag_min_n_event_nonevent

    table = optb.binning_table.build()
    assert optb.binning_table.triangular > 0

    n_nonevent = table["Non-event"].values[:-3]
    n_event = table["Event"].values[:-3]
    assert n_nonevent.min() >= 5
    assert n_event.min() >= 5


def test_divergence_js():
    x, y = _numerical_data(n=300, seed=0)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=10,
                                divergence="js")
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    optb.binning_table.build()
    # _solve_stats is keyed by _n_solve, which is incremented before the
    # stats are recorded, so the first solve is key 1
    assert optb._solve_stats[1]["divergence"] == approx(
        optb.binning_table.js)


def test_monotonic_trend_ascending():
    x, y = _numerical_data(n=300, seed=12)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=10,
                                monotonic_trend="ascending")
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    event_rate = table["Event rate"].values[:-3]
    assert list(event_rate) == sorted(event_rate)


def test_monotonic_trend_none_verbose():
    x, y = _numerical_data(n=200, seed=13)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=8,
                                monotonic_trend=None, verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"


def _peak_data(n=400, seed=14):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-3, 3, size=n)
    rate = np.exp(-x ** 2)
    y = (rng.rand(n) < rate).astype(int)
    return x, y


def test_monotonic_trend_peak_heuristic_verbose():
    x, y = _peak_data()

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=10,
                                monotonic_trend="peak_heuristic",
                                verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"


def test_monotonic_trend_auto_heuristic_peak():
    x, y = _peak_data()

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=10,
                                monotonic_trend="auto_heuristic",
                                verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    event_rate = table["Event rate"].values[:-3]
    # a peak: the maximum is interior, and the series rises then falls
    assert len(event_rate) >= 3
    imax = int(np.argmax(event_rate))
    assert 0 < imax < len(event_rate) - 1


def test_solver_mip_and_transform_metrics():
    x, y = _numerical_data(n=200, seed=15)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=8,
                                solver="mip", mip_solver="cbc")
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    x_new = np.array([-10.0, 0.0, 10.0, np.nan])
    for metric in ("woe", "event_rate", "indices", "bins"):
        out = optb.transform(x_new, metric=metric)
        assert out.shape == (4,)


def test_special_codes_end_to_end():
    x, y = _numerical_data(n=300, seed=16)
    x = x.copy()
    x[:20] = -99.0
    x[20:30] = np.nan

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=8,
                                special_codes=[-99.0])
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    assert table["Count"].values[-2] == 10   # missing
    assert table["Count"].values[-3] == 20   # special
    assert table["Count"].values[-1] == 300

    out = optb.transform(np.array([-99.0, np.nan]), metric="event_rate",
                         metric_special="empirical",
                         metric_missing="empirical")
    assert out.shape == (2,)


def test_categorical_splits_and_heuristic():
    x, y = _categorical_data(n=400, seed=42)

    optb = OptimalBinningSketch(dtype="categorical", max_n_prebins=4,
                                cat_heuristic=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    # bin_categorical returns a list of per-bin category arrays, exactly as
    # OptimalBinning.splits does for dtype="categorical"
    splits = optb.splits
    assert isinstance(splits, list)
    seen = set()
    for group in splits:
        seen |= set(group)
    assert seen == set(np.unique(x))


def test_categorical_hellinger_with_heuristic():
    x, y = _categorical_data(n=400, seed=42)

    optb = OptimalBinningSketch(dtype="categorical", max_n_prebins=4,
                                cat_heuristic=True, divergence="hellinger")
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
    optb.binning_table.build()
    assert optb.binning_table.hellinger > 0


def test_categorical_monotonic_trend_is_forced_ascending():
    x, y = _categorical_data(n=400, seed=42)

    optb = OptimalBinningSketch(dtype="categorical", max_n_prebins=10,
                                monotonic_trend="descending")
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    event_rate = table["Event rate"].values[:-3]
    assert list(event_rate) == sorted(event_rate)


def test_monotonic_trend_auto_heuristic_valley():
    rng = np.random.RandomState(14)
    x = rng.uniform(-3, 3, size=400)
    y = (rng.rand(400) < 1 - np.exp(-x ** 2)).astype(int)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=10,
                                monotonic_trend="auto_heuristic",
                                verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    event_rate = table["Event rate"].values[:-3]
    assert len(event_rate) >= 3
    imin = int(np.argmin(event_rate))
    assert 0 < imin < len(event_rate) - 1


def test_tdigest_end_to_end_matches_gk():
    x, y = _numerical_data(n=300, seed=17)

    optb_gk = OptimalBinningSketch(sketch="gk", eps=1e-4, max_n_prebins=10)
    optb_gk.add(x, y)
    optb_gk.solve()

    optb_td = OptimalBinningSketch(sketch="t-digest", eps=1e-4,
                                   max_n_prebins=10)
    optb_td.add(x, y)
    optb_td.solve()

    assert optb_gk.status == "OPTIMAL"
    assert optb_td.status == "OPTIMAL"

    optb_gk.binning_table.build()
    optb_td.binning_table.build()

    # the two sketches approximate the same quantiles, so the IV agrees to
    # within the sketch error rather than exactly
    assert optb_td.binning_table.iv == approx(optb_gk.binning_table.iv,
                                              rel=0.15)


def test_bsketch_tdigest_bins_and_merge():
    x, y = _numerical_data(n=200, seed=18)

    sk1 = BSketch(sketch="t-digest", eps=1e-2)
    sk2 = BSketch(sketch="t-digest", eps=1e-2)
    sk1.add(x[:100], y[:100])
    sk2.add(x[100:], y[100:])

    sk1.merge(sk2)

    assert sk1.n == 200

    bins_e, bins_ne = sk1.bins(np.array([-0.5, 0.0, 0.5]))
    assert bins_e.shape == (4,)
    assert bins_e.sum() + bins_ne.sum() == 200


def test_information_reports_memory_usage(capsys):
    x, y = _numerical_data(n=200, seed=19)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=8)
    optb.add(x, y)
    optb.solve()

    with raises(ValueError, match="print_level must be an integer >= 0"):
        optb.information(print_level=-1)

    with raises(ValueError, match="print_level must be an integer >= 0"):
        optb.information(print_level=1.5)

    optb.information(print_level=1)
    out = capsys.readouterr().out

    assert "Sketch memory usage" in out
    assert "Processed records" in out
    assert "Solver statistics" in out


def test_plot_progress_across_two_solves(tmp_path):
    x, y = _numerical_data(n=300, seed=20)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=8,
                                name="chaos")
    optb.add(x[:150], y[:150])
    optb.solve()
    optb.add(x[150:], y[150:])
    optb.solve()

    assert set(optb._solve_stats) == {1, 2}
    assert optb._solve_stats[2]["n_add"] == 2
    assert optb._solve_stats[2]["n_records"] == 300

    path = tmp_path / "progress.png"
    optb.plot_progress(savefig=str(path))

    assert path.stat().st_size


def test_add_mismatched_lengths():
    sk = BSketch(sketch="gk", eps=0.1)

    with raises(ValueError, match="inconsistent numbers of samples"):
        sk.add(np.array([1.0, 2.0, 3.0]), np.array([0, 1]), check_input=True)

    # without check_input the mismatch surfaces from numpy instead
    with raises(ValueError, match="could not be broadcast together"):
        sk.add(np.array([1.0, 2.0, 3.0]), np.array([0, 1]))


def test_add_accepts_lists_and_series():
    import pandas as pd

    sk = BSketch(sketch="gk", eps=0.1)
    sk.add([1.0, 2.0, 3.0], [0, 1, 0])
    assert sk.n == 3

    sk.add(pd.Series([4.0, 5.0]), pd.Series([1, 0]))
    assert sk.n == 5


def test_all_nan_x_is_a_single_empty_bin():
    # this used to raise -- ValueError("GK sketch does not contain values.")
    # for "gk", ValueError("Tree is empty") for "t-digest" -- because the
    # quantile sketch was asked for percentiles it had no value for. A
    # stream of nothing but missing values is degenerate but legal.
    for sketch in ("gk", "t-digest"):
        optb = OptimalBinningSketch(sketch=sketch, eps=1e-2,
                                    max_n_prebins=4)
        optb.add(np.full(50, np.nan), np.array([0, 1] * 25))
        optb.solve()

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) == 0

        table = optb.binning_table.build()

        assert list(table["Count"].values) == [0, 0, 50, 50]
        assert list(table["Event"].values) == [0, 0, 25, 25]


def test_transform_invalid_metric():
    x, y = _numerical_data(n=100, seed=21)

    optb = OptimalBinningSketch(sketch="gk", eps=1e-2, max_n_prebins=6)
    optb.add(x, y)
    optb.solve()

    with raises(ValueError, match="Invalid value for metric."):
        optb.transform(x, metric="bogus")


def test_extreme_magnitudes_and_duplicates():
    rng = np.random.RandomState(22)
    x = np.concatenate([np.full(100, 1e-12), np.full(100, 1e12)])
    y = np.concatenate([rng.binomial(1, 0.1, 100),
                        rng.binomial(1, 0.9, 100)])

    optb = OptimalBinningSketch(sketch="gk", eps=1e-3, max_n_prebins=6)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    assert table["Count"].values[-1] == 200
    assert optb.binning_table.iv > 0


def test_special_codes_as_ndarray_and_unusual_categories():
    x = np.array(["  spaced  ", "ÜNICODE", "", "a,b", "  spaced  ", "ÜNICODE",
                  "", "a,b", "SPECIAL", "SPECIAL"], dtype=object)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])

    sk = BCatSketch(special_codes=np.array(["SPECIAL"], dtype=object))
    sk.add(x, y)

    assert sk._count_special_e == 1
    assert sk._count_special_ne == 1
    assert set(sk._d_categories) == {"  spaced  ", "ÜNICODE", "", "a,b"}
    assert sk.n == 10


def test_categorical_cat_unknown_transform():
    x, y = _categorical_data(n=300, seed=23)

    optb = OptimalBinningSketch(dtype="categorical", max_n_prebins=6,
                                cat_unknown=-1.0)
    optb.add(x, y)
    optb.solve()

    out = optb.transform(np.array(["a", "unseen"], dtype=object),
                         metric="woe")

    assert out.shape == (2,)
    assert out[1] == -1.0


def test_tied_values_leave_no_empty_prebin():
    # the sibling of test_split_digits_collision_leaves_no_duplicate_split
    # with no split_digits at all: a zero-inflated discrete column makes the
    # sketch report the same quantile several times, and the prebin between
    # two equal splits holds no record. model_data then computes
    # s_event / (s_nonevent + s_event) = 0/0 for it and casts the nan with
    # astype(np.int64) -- MODEL_INVALID under "hellinger", a TypeError out
    # of OR-Tools under "triangular", because those two keep every prebin.
    rng = np.random.RandomState(0)
    x = rng.choice([0., 0., 0., 0., 1., 2., 3.], size=2000)
    y = (rng.uniform(size=2000) < 0.2 + 0.15 * x).astype(int)

    for divergence in ("iv", "js", "hellinger", "triangular"):
        optb = OptimalBinningSketch(sketch="gk", eps=1e-4, max_n_prebins=20,
                                    divergence=divergence)
        optb.add(x, y)
        optb.solve()

        assert optb.status == "OPTIMAL"
        assert len(optb.splits) >= 1

        prebins = optb._splits_prebinning
        assert len(prebins) == len(np.unique(prebins))

        table = optb.binning_table.build()
        assert table["Count"].values[-1] == 2000
        # every clean bin holds records: Special, Missing and Totals follow
        assert (table["Count"].values[:-3] > 0).all()
