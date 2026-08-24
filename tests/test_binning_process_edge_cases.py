"""
Binning process edge-case and chaos testing.

Covers the guards, the parallel and on-disk paths, the selection-criteria
branches and the piecewise/updated-variable paths of
``optbinning/binning/binning_process.py`` and
``optbinning/binning/distributed/binning_process_sketch.py`` that the main
test modules never reach, plus degenerate and hostile inputs.

Tests named ``test_defect_*`` assert the behaviour the code documents or that
its siblings implement. Each was written red against the source of the day and
is kept as the regression pin for the defect its docstring describes.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import os
import tempfile
import warnings

from decimal import Decimal

import numpy as np
import pandas as pd

from pytest import approx, raises, warns

from optbinning import BinningProcess
from optbinning import BinningProcessSketch
from optbinning import ContinuousOptimalBinning
from optbinning import ContinuousOptimalBinning2D
from optbinning import ContinuousOptimalPWBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning import OptimalBinning2D
from optbinning import OptimalPWBinning
from optbinning.binning.binning_process import _check_selection_criteria
from optbinning.binning.binning_process import _check_variable_dtype
from optbinning.binning.binning_process import BaseBinningProcess
from sklearn.exceptions import NotFittedError


# Small synthetic problem: binning fits are solver runs, so everything here
# stays at a few hundred rows and a handful of prebins.
_RNG = np.random.RandomState(0)
N = 240

variable_names = ["v0", "v1", "v2", "v3"]

X_df = pd.DataFrame({
    "v0": _RNG.normal(size=N),
    "v1": _RNG.uniform(size=N),
    "v2": _RNG.normal(size=N),
    "v3": _RNG.normal(size=N),
})

X_arr = X_df.values

y_binary = (X_df["v0"].values + _RNG.normal(scale=0.8, size=N) > 0).astype(int)
y_continuous = 2.0 * X_df["v0"].values + _RNG.normal(scale=0.5, size=N)
y_multiclass = np.digitize(
    X_df["v0"].values + _RNG.normal(scale=0.5, size=N), [-0.5, 0.5])

# Scratch csv files go to a temporary directory, not to tests/results.
# That directory is for artifacts the suite is meant to keep; these are
# throwaway fit_disk inputs and transform_disk outputs, and leaving them
# behind makes `git status` dirty after every run.
_SCRATCH = tempfile.mkdtemp(prefix="optbinning_process_edge_")


def _process(names=None, **kwargs):
    """A BinningProcess sized so that a fit is a fraction of a second."""
    kwargs.setdefault("max_n_prebins", 5)
    kwargs.setdefault("min_prebin_size", 0.1)
    return BinningProcess(variable_names if names is None else names, **kwargs)


def _fitted(names=None, **kwargs):
    process = _process(names, **kwargs)
    process.fit(X_df if names is None else X_df[names], y_binary)
    return process


def _disk_input():
    """Write the synthetic frame and its target to csv; return the path."""
    path = os.path.join(_SCRATCH, "binning_process_edge_input.csv")
    df = X_df.copy()
    df["target"] = y_binary
    df["label"] = np.where(y_binary == 1, "yes", "no")
    df.to_csv(path, index=False)
    return path


def _fresh_output(name):
    path = os.path.join(_SCRATCH, name)
    if os.path.exists(path):
        os.remove(path)
    return path


# ---------------------------------------------------------------------------
# _check_parameters guards
# ---------------------------------------------------------------------------

def test_fixed_variables_must_be_list_or_array():
    with raises(TypeError, match="fixed_variables must be a list"):
        _process(fixed_variables="v0").fit(X_df, y_binary)

    with raises(TypeError, match="fixed_variables must be a list"):
        _process(fixed_variables=("v0",)).fit(X_df, y_binary)

    # A numpy array of names is accepted.
    process = _process(fixed_variables=np.array(["v0"]),
                       selection_criteria={"iv": {"min": 1e9}})
    process.fit(X_df, y_binary)
    assert list(process.get_support(names=True)) == ["v0"]


def test_special_codes_empty_dict_rejected():
    with raises(ValueError, match="special_codes empty"):
        _process(special_codes={}).fit(X_df, y_binary)


def test_special_codes_list_and_dict():
    x = X_df["v0"].values.copy()
    x[:30] = -999.0
    df = pd.DataFrame({"v0": x})

    as_list = BinningProcess(["v0"], max_n_prebins=5, min_prebin_size=0.1,
                             special_codes=[-999.0])
    as_list.fit(df, y_binary)
    bins_list = as_list.get_binned_variable("v0").binning_table.build()["Bin"]
    assert "Special" in list(bins_list)

    as_dict = BinningProcess(["v0"], max_n_prebins=5, min_prebin_size=0.1,
                             special_codes={"sentinel": -999.0})
    as_dict.fit(df, y_binary)
    bins_dict = as_dict.get_binned_variable("v0").binning_table.build()["Bin"]
    assert "sentinel" in list(bins_dict)
    assert "Special" not in list(bins_dict)


def test_n_jobs_must_be_integral_or_none():
    with raises(ValueError, match="n_jobs must be an integer"):
        _process(n_jobs=1.5).fit(X_df, y_binary)

    process = _process(n_jobs=None)
    process.fit(X_df, y_binary)
    assert process._is_fitted


# ---------------------------------------------------------------------------
# _check_variable_dtype
# ---------------------------------------------------------------------------

def test_check_variable_dtype_of_every_column_kind():
    assert _check_variable_dtype(pd.Series([1, 2, 3])) == "numerical"
    assert _check_variable_dtype(pd.Series([1.0, np.nan])) == "numerical"
    assert _check_variable_dtype(pd.Series([True, False])) == "numerical"
    assert _check_variable_dtype(pd.Series(["a", "b"])) == "categorical"
    assert _check_variable_dtype(
        pd.Series(pd.Categorical(["a", "b"]))) == "categorical"
    assert _check_variable_dtype(
        pd.Series(pd.date_range("2020-01-01", periods=2))) == "categorical"


def test_mixed_column_dtypes_are_classified_and_binned():
    df = pd.DataFrame({
        "num": X_df["v0"].values,
        "flag": X_df["v0"].values > 0,
        "catdt": pd.Categorical(np.tile(["a", "b", "c", "d"], N // 4)),
        "dt": pd.date_range("2020-01-01", periods=N),
    })

    process = BinningProcess(list(df.columns), max_n_prebins=5,
                             min_prebin_size=0.1)
    process.fit(df, y_binary)

    dtypes = process.summary().set_index("name")["dtype"]
    assert dtypes["num"] == "numerical"
    assert dtypes["flag"] == "numerical"
    assert dtypes["catdt"] == "categorical"
    # datetime64 is not numeric, so it is binned as categorical. See
    # reports/DECISIONS.md.
    assert dtypes["dt"] == "categorical"
    assert set(process.summary()["status"]) == {"OPTIMAL"}


def test_categorical_variables_forces_a_numeric_column_categorical():
    process = _process(names=["v0", "v1"], categorical_variables=["v1"])
    process.fit(X_df[["v0", "v1"]], y_binary)

    dtypes = process.summary().set_index("name")["dtype"]
    assert dtypes["v0"] == "numerical"
    assert dtypes["v1"] == "categorical"

    # A categorical binning has one bin per split, a numerical one has one
    # more than its number of splits.
    stats = process.summary().set_index("name")
    optb = process.get_binned_variable("v1")
    assert stats.loc["v1", "n_bins"] == len(optb.splits)


def test_multiclass_target_rejects_categorical_variable():
    df = pd.DataFrame({
        "v0": X_df["v0"].values,
        "cat": np.tile(["a", "b", "c", "d"], N // 4),
    })

    process = BinningProcess(["v0", "cat"], max_n_prebins=5,
                             min_prebin_size=0.1)

    with raises(ValueError, match="does not support categorical"):
        process.fit(df, y_multiclass)


# ---------------------------------------------------------------------------
# _fit / _fit_block: the n_jobs > 1 path
# ---------------------------------------------------------------------------

def test_parallel_fit_matches_serial_fit_dataframe():
    serial = _fitted()
    parallel = _process(n_jobs=2)
    parallel.fit(X_df, y_binary)

    assert list(parallel.summary()["name"]) == variable_names
    assert set(parallel.summary()["status"]) == {"OPTIMAL"}
    assert parallel.summary()["iv"].values == approx(
        serial.summary()["iv"].values, rel=1e-9)
    assert parallel._variable_dtypes == serial._variable_dtypes


def test_parallel_fit_matches_serial_fit_ndarray():
    serial = _process()
    serial.fit(X_arr, y_binary)

    parallel = _process(n_jobs=3)
    parallel.fit(X_arr, y_binary)

    assert list(parallel.summary()["name"]) == variable_names
    assert parallel.summary()["iv"].values == approx(
        serial.summary()["iv"].values, rel=1e-9)


def test_sample_weight_only_supported_for_binary_target():
    weight = np.ones(N)

    with raises(ValueError, match="continuous does not support sample weight"):
        _process(names=["v0"]).fit(X_df[["v0"]], y_continuous,
                                   sample_weight=weight)

    with raises(ValueError, match="multiclass does not support sample weight"):
        _process(names=["v0"]).fit(X_df[["v0"]], y_multiclass,
                                   sample_weight=weight)

    # Binary accepts it, and unit weights reproduce the unweighted fit.
    weighted = _process(names=["v0"])
    weighted.fit(X_df[["v0"]], y_binary, sample_weight=weight)
    assert weighted.summary()["iv"].values == approx(
        _fitted(names=["v0"]).summary()["iv"].values, rel=1e-9)


def test_number_of_columns_must_match_variable_names():
    with raises(ValueError, match="number of columns"):
        _process().fit(X_df[["v0", "v1"]], y_binary)

    with raises(ValueError, match="number of columns"):
        _process(names=["v0"]).fit(X_arr, y_binary)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trip_and_path_guards():
    process = _fitted()

    with raises(TypeError, match="path must be a string"):
        process.save(1)

    with raises(TypeError, match="path must be a string"):
        BinningProcess.load(1)

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "process_pkl")
    process.save(path)

    loaded = BinningProcess.load(path)
    assert isinstance(loaded, BinningProcess)
    assert loaded._is_fitted
    assert list(loaded.summary()["name"]) == variable_names
    assert loaded.summary()["iv"].values == approx(
        process.summary()["iv"].values, rel=1e-12)
    assert loaded.transform(X_df).values == approx(
        process.transform(X_df).values, rel=1e-12)


# ---------------------------------------------------------------------------
# _support_selection_criteria branches
# ---------------------------------------------------------------------------

def test_selection_criteria_top_as_fraction_and_lowest_strategy():
    process = _process(
        selection_criteria={"iv": {"strategy": "lowest", "top": 0.5}})
    process.fit(X_df, y_binary)

    summary = process.summary()
    # top=0.5 of four valid variables selects ceil(4 * 0.5) = 2.
    assert np.count_nonzero(process.get_support()) == 2

    # ``summary()`` is built with DataFrame.from_dict(...).T, so its metric
    # columns arrive as object dtype and have to be cast before comparison.
    iv = summary["iv"].astype(float)
    selected = set(summary.loc[summary["selected"], "name"])
    lowest_two = set(summary.loc[iv.nsmallest(2).index, "name"])
    assert selected == lowest_two


def test_selection_criteria_top_larger_than_the_valid_set():
    process = _process(
        selection_criteria={"iv": {"strategy": "highest", "top": 99}})
    process.fit(X_df, y_binary)

    assert np.count_nonzero(process.get_support()) == len(variable_names)


def test_fixed_variables_survive_an_unsatisfiable_criterion():
    unsatisfiable = {"iv": {"min": 1e9}}

    without = _process(selection_criteria=unsatisfiable)
    without.fit(X_df, y_binary)
    assert not without.get_support().any()

    with_fixed = _process(selection_criteria=unsatisfiable,
                          fixed_variables=["v1", "v3"])
    with_fixed.fit(X_df, y_binary)
    assert list(with_fixed.get_support(names=True)) == ["v1", "v3"]
    assert list(with_fixed.get_support(indices=True)) == [1, 3]


# ---------------------------------------------------------------------------
# get_binned_variable / update_binned_variable
# ---------------------------------------------------------------------------

def test_update_binned_variable_replaces_and_restates_the_summary():
    process = _fitted()
    before = process.summary().set_index("name")

    pw = OptimalPWBinning(name="v0")
    pw.fit(X_df["v0"].values, y_binary)

    process.update_binned_variable("v0", pw)
    assert process._is_updated
    assert process.get_binned_variable("v0") is pw

    after = process.summary().set_index("name")
    # summary() rebuilds the statistics and clears the flag.
    assert not process._is_updated
    # A piecewise object has no ``dtype``; the process reports it numerical.
    assert after.loc["v0", "dtype"] == "numerical"
    assert after.loc["v0", "n_bins"] == len(pw.splits) + 1
    assert after.loc["v0", "status"] == "OPTIMAL"
    # The untouched variables keep their statistics.
    assert after.loc["v1", "iv"] == approx(before.loc["v1", "iv"], rel=1e-12)


def test_update_binned_variable_target_dtype_guards():
    x = X_df["v0"].values

    continuous = _process(names=["v0"])
    continuous.fit(X_df[["v0"]], y_continuous)
    optb = OptimalBinning(name="v0")
    optb.fit(x, y_binary)
    with raises(TypeError, match="target is continuous"):
        continuous.update_binned_variable("v0", optb)

    coptb = ContinuousOptimalBinning(name="v0")
    coptb.fit(x, y_continuous)
    continuous.update_binned_variable("v0", coptb)
    assert continuous.get_binned_variable("v0") is coptb

    multiclass = _process(names=["v0"])
    multiclass.fit(X_df[["v0"]], y_multiclass)
    with raises(TypeError, match="target is multiclass"):
        multiclass.update_binned_variable("v0", optb)

    moptb = MulticlassOptimalBinning(name="v0")
    moptb.fit(x, y_multiclass)
    multiclass.update_binned_variable("v0", moptb)
    assert multiclass.get_binned_variable("v0") is moptb

    binary = _fitted(names=["v0"])
    cpw = ContinuousOptimalPWBinning(name="v0")
    cpw.fit(x, y_continuous)
    with raises(TypeError, match="target is binary"):
        binary.update_binned_variable("v0", cpw)


def test_update_binned_variable_when_the_old_object_is_unnamed():
    # Objects fitted outside the process may carry no name, which is the only
    # way to reach the "name and object name must coincide" guard.
    dict_optb = {}
    for name in ["v0", "v1"]:
        optb = OptimalBinning(max_n_prebins=5, min_prebin_size=0.1)
        optb.fit(X_df[name], y_binary)
        dict_optb[name] = optb

    process = BinningProcess(["v0", "v1"])
    process.fit_from_dict(dict_optb)
    assert not process.get_binned_variable("v0").name

    named = OptimalBinning(name="v1", max_n_prebins=5, min_prebin_size=0.1)
    named.fit(X_df["v1"], y_binary)

    with raises(ValueError, match="name and object name must coincide"):
        process.update_binned_variable("v0", named)

    process.update_binned_variable("v1", named)
    assert process.get_binned_variable("v1") is named


def test_accessors_before_fit():
    process = _process()

    with raises(NotFittedError):
        process.get_binned_variable("v0")

    with raises(NotFittedError):
        process.update_binned_variable("v0", OptimalBinning())

    with raises(NotFittedError):
        process.information()

    with raises(NotFittedError):
        process.summary()

    with raises(NotFittedError):
        process.get_support()

    with raises(NotFittedError):
        process.transform_disk("a.csv", "b.csv", chunksize=10)


# ---------------------------------------------------------------------------
# fit_from_dict
# ---------------------------------------------------------------------------

def test_fit_from_dict_continuous_and_multiclass_targets():
    continuous = {}
    for name in ["v0", "v1"]:
        optb = ContinuousOptimalBinning(name=name, max_n_prebins=5,
                                        min_prebin_size=0.1)
        optb.fit(X_df[name], y_continuous)
        continuous[name] = optb

    process = BinningProcess(["v0", "v1"],
                             selection_criteria={"woe": {"min": 0.0}})
    process.fit_from_dict(continuous)
    assert process._target_dtype == "continuous"
    assert list(process.summary().columns[-2:]) == ["woe", "quality_score"]
    assert process._n_samples == 0

    multiclass = {}
    for name in ["v0", "v1"]:
        optb = MulticlassOptimalBinning(name=name, max_n_prebins=5,
                                        min_prebin_size=0.1)
        optb.fit(X_df[name], y_multiclass)
        multiclass[name] = optb

    process = BinningProcess(["v0", "v1"],
                             selection_criteria={"js": {"min": 0.0}})
    process.fit_from_dict(multiclass)
    assert process._target_dtype == "multiclass"
    assert list(process.summary().columns[-2:]) == ["js", "quality_score"]


def test_fit_from_dict_object_guards():
    fitted = OptimalBinning(name="v0", max_n_prebins=5, min_prebin_size=0.1)
    fitted.fit(X_df["v0"], y_binary)

    unnamed = OptimalBinning(max_n_prebins=5, min_prebin_size=0.1)
    unnamed.fit(X_df["v1"], y_binary)

    with raises(TypeError, match="Object key must be a string"):
        BinningProcess([1]).fit_from_dict({1: unnamed})

    with raises(TypeError, match="must be of type"):
        BinningProcess(["v0"]).fit_from_dict({"v0": OptimalPWBinning()})

    coptb = ContinuousOptimalBinning(name="v1", max_n_prebins=5,
                                     min_prebin_size=0.1)
    coptb.fit(X_df["v1"], y_continuous)
    with raises(TypeError, match="same class"):
        BinningProcess(["v0", "v1"]).fit_from_dict({"v0": fitted,
                                                    "v1": coptb})

    with raises(NotFittedError, match="is not fitted yet"):
        BinningProcess(["v0"]).fit_from_dict({"v0": OptimalBinning()})

    with raises(ValueError, match="those must coincide"):
        BinningProcess(["v1"]).fit_from_dict({"v1": fitted})


# ---------------------------------------------------------------------------
# fit_disk
# ---------------------------------------------------------------------------

def test_fit_disk_unsupported_target_type():
    path = _disk_input()

    # ``**kwargs`` reach pandas.read_csv, so a converter can hand the process
    # a target sklearn cannot classify.
    with raises(ValueError, match="Target type unknown is not supported"):
        _process().fit_disk(path, "target",
                            converters={"target": lambda s: Decimal(s)})


def test_fit_disk_selection_criteria_and_fixed_variables():
    path = _disk_input()

    with raises(ValueError, match="metric for binary target must be in"):
        _process(selection_criteria={"woe": {"min": 0}}).fit_disk(
            path, "target")

    with raises(ValueError, match="Variable zzz to be fixed"):
        _process(fixed_variables=["zzz"]).fit_disk(path, "target")

    process = _process(selection_criteria={"iv": {"min": 1e9}},
                       fixed_variables=["v2"])
    process.fit_disk(path, "target")
    assert list(process.get_support(names=True)) == ["v2"]
    assert set(process.summary()["status"]) == {"OPTIMAL"}


def test_fit_disk_reads_only_the_named_columns():
    path = _disk_input()

    process = _process(names=["v1", "v3"])
    process.fit_disk(path, "target")

    assert process._n_samples == N
    assert process._n_variables == 2
    assert list(process.summary()["name"]) == ["v1", "v3"]
    assert set(process.summary()["status"]) == {"OPTIMAL"}


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def test_transform_requires_every_selected_column():
    process = _fitted()

    with raises(ValueError, match="must be a column in the input dataframe"):
        process.transform(X_df.drop(columns=["v2"]))


def test_transform_index_and_extra_columns_are_tolerated():
    process = _fitted()

    extra = X_df.copy()
    extra["unused"] = 1.0
    extra.index = pd.Index([3 * i for i in range(N)], name="row")

    transformed = process.transform(extra)
    assert list(transformed.columns) == variable_names
    pd.testing.assert_index_equal(transformed.index, extra.index)


def test_transform_indices_and_bins_rejected_for_piecewise():
    process = _fitted()
    pw = OptimalPWBinning(name="v0")
    pw.fit(X_df["v0"].values, y_binary)
    process.update_binned_variable("v0", pw)

    with raises(TypeError, match="metric indices not supported"):
        process.transform(X_df, metric="indices")

    with raises(TypeError, match="metric bins not supported"):
        process.transform(X_df, metric="bins")


def test_transform_drops_show_digits_for_piecewise_objects():
    process = _fitted()
    pw = OptimalPWBinning(name="v0")
    pw.fit(X_df["v0"].values, y_binary)
    process.update_binned_variable("v0", pw)

    # show_digits is not a parameter of the piecewise transform; passing it
    # must not raise.
    transformed = process.transform(X_df, show_digits=4)
    assert transformed.shape == (N, len(variable_names))
    assert transformed["v0"].values == approx(
        pw.transform(X_df["v0"].values), nan_ok=True)


def test_transform_params_bins_metric_for_every_variable():
    params = {name: {"metric": "bins"} for name in variable_names}
    process = _process(binning_transform_params=params)
    process.fit(X_df, y_binary)

    transformed = process.transform(X_df, metric="bins")
    assert transformed.shape == (N, len(variable_names))
    assert all(isinstance(v, str) for v in transformed.iloc[0])


def test_transform_params_mixing_bins_with_numeric_metrics():
    params = {"v0": {"metric": "bins"}, "v1": {"metric": "woe"}}
    process = _process(binning_transform_params=params)
    process.fit(X_df, y_binary)

    with raises(ValueError, match="cannot be mixed with numeric metrics"):
        process.transform(X_df)


def test_transform_params_override_special_and_missing():
    x = X_df["v0"].values.copy()
    x[:20] = -999.0
    x[20:40] = np.nan
    df = pd.DataFrame({"v0": x, "v1": X_df["v1"].values})

    params = {"v0": {"metric_special": 1.5, "metric_missing": -2.5}}
    process = BinningProcess(["v0", "v1"], max_n_prebins=5,
                             min_prebin_size=0.1, special_codes=[-999.0],
                             binning_transform_params=params)
    process.fit(df, y_binary)

    transformed = process.transform(df, metric="woe")
    assert transformed["v0"].values[:20] == approx(1.5)
    assert transformed["v0"].values[20:40] == approx(-2.5)


# ---------------------------------------------------------------------------
# transform_disk
# ---------------------------------------------------------------------------

def test_transform_disk_with_bins_metric_for_every_variable():
    path = _disk_input()
    output = _fresh_output("binning_process_edge_bins.csv")

    params = {name: {"metric": "bins"} for name in variable_names}
    process = _process(binning_transform_params=params)
    process.fit(X_df, y_binary)
    process.transform_disk(path, output, chunksize=100)

    written = pd.read_csv(output)
    assert list(written.columns) == variable_names
    assert len(written) == N
    assert written.dtypes.map(lambda d: d == object or d == "str").all()


def test_transform_disk_with_indices_metric():
    path = _disk_input()
    output = _fresh_output("binning_process_edge_indices.csv")

    params = {name: {"metric": "indices"} for name in variable_names}
    process = _process(binning_transform_params=params)
    process.fit(X_df, y_binary)
    # The explicit metric and the per-variable metrics agree, so the process
    # keeps the integer output array.
    process.transform_disk(path, output, chunksize=100, metric="indices")

    written = pd.read_csv(output)
    assert len(written) == N
    assert all(pd.api.types.is_integer_dtype(written[c]) for c in written)
    assert (written.values >= 0).all()


def test_transform_disk_mixing_bins_with_numeric_metrics():
    path = _disk_input()
    output = _fresh_output("binning_process_edge_mixed.csv")

    params = {"v0": {"metric": "bins"}, "v1": {"metric": "woe"}}
    process = _process(binning_transform_params=params)
    process.fit(X_df, y_binary)

    with raises(ValueError, match="cannot be mixed with numeric metrics"):
        process.transform_disk(path, output, chunksize=100)


def test_transform_disk_with_a_piecewise_variable():
    path = _disk_input()
    output = _fresh_output("binning_process_edge_pw.csv")

    process = _fitted()
    pw = OptimalPWBinning(name="v0")
    pw.fit(X_df["v0"].values, y_binary)
    process.update_binned_variable("v0", pw)

    with raises(TypeError, match="metric bins not supported"):
        process.transform_disk(path, output, chunksize=100, metric="bins")

    with raises(TypeError, match="metric indices not supported"):
        process.transform_disk(path, output, chunksize=100, metric="indices")

    process.transform_disk(path, output, chunksize=100)
    written = pd.read_csv(output)
    assert list(written.columns) == variable_names
    assert len(written) == N


def test_transform_disk_extension_and_chunksize_guards():
    path = _disk_input()
    output = _fresh_output("binning_process_edge_guard.csv")
    process = _fitted()

    with raises(ValueError, match="must be csv files"):
        process.transform_disk(path, output.replace(".csv", ".parquet"),
                               chunksize=100)

    with raises(ValueError, match="chunksize must be a positive integer"):
        process.transform_disk(path, output, chunksize=-1)

    with raises(ValueError, match="chunksize must be a positive integer"):
        process.transform_disk(path, output, chunksize=1.5)


# ---------------------------------------------------------------------------
# Chaos: degenerate and hostile inputs
# ---------------------------------------------------------------------------

def test_constant_column_yields_a_single_bin():
    df = pd.DataFrame({"const": np.ones(N), "v1": X_df["v1"].values})
    process = BinningProcess(["const", "v1"], max_n_prebins=5,
                             min_prebin_size=0.1)
    process.fit(df, y_binary)

    optb = process.get_binned_variable("const")
    assert optb.status == "OPTIMAL"
    assert len(optb.splits) == 0
    assert process.summary().set_index("name").loc["const", "n_bins"] == 1
    assert process.summary().set_index("name").loc["const", "iv"] == approx(
        0.0, abs=1e-12)

    transformed = process.transform(df, metric="woe")
    assert transformed["const"].values == approx(0.0, abs=1e-10)


def test_duplicated_values_column():
    df = pd.DataFrame({"dup": np.repeat([1.0, 2.0, 3.0], N // 3)})
    process = BinningProcess(["dup"], max_n_prebins=5, min_prebin_size=0.1)
    process.fit(df, y_binary)

    optb = process.get_binned_variable("dup")
    assert optb.status == "OPTIMAL"
    assert np.all(np.diff(optb.splits) > 0)
    assert np.all(optb.splits > 1.0) and np.all(optb.splits < 3.0)


def test_all_nan_column_raises():
    df = pd.DataFrame({"nans": np.full(N, np.nan)})
    with raises(ValueError, match="minimum of 1 is required"):
        BinningProcess(["nans"], max_n_prebins=5).fit(df, y_binary)


def test_single_class_target_fits_one_degenerate_bin():
    """A binary target carrying a single class is degenerate but legal.

    ``fit`` succeeds with no split points, the solver reports "OPTIMAL" and
    the binning table reports honest zeros for every divergence metric. A
    single row is the same input viewed through a smaller sample and travels
    the identical path, so it is covered here rather than separately.

    The process transform must report the same honest numbers: the event
    rate is a property of the bin on its own, so an all-event bin is 1.0 and
    an all-non-event bin 0.0 -- never 0.0 for both.
    """
    cases = ((X_df[["v0"]], np.zeros(N, dtype=int), 0.0),
             (X_df[["v0"]], np.ones(N, dtype=int), 1.0),
             (X_df[["v0"]].iloc[:1], y_binary[:1], float(y_binary[0])))

    for X, y, event_rate in cases:
        process = _process(names=["v0"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process.fit(X, y)

        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]

        assert list(process.get_support()) == [True]
        assert list(process.get_support(names=True)) == ["v0"]

        optb = process.get_binned_variable("v0")
        assert optb.status == "OPTIMAL"
        assert len(optb.splits) == 0

        table = optb.binning_table
        table.build()
        assert table.iv == 0
        assert table.js == 0
        assert table.gini == 0

        # WoE compares the bin against the rest of the sample and stays
        # gated on the bin holding both classes, so it is 0; the event rate
        # is gated on records instead. There being one bin, every index is 0.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rates = process.transform(X, metric="event_rate")
            woes = process.transform(X, metric="woe")
            indices = process.transform(X, metric="indices")

        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        assert rates["v0"].values == approx(event_rate)
        assert woes["v0"].values == approx(0.0)
        assert (indices["v0"].values == 0).all()


def test_infinite_and_extreme_values():
    x = X_df["v0"].values.copy()
    x[0], x[1] = np.inf, -np.inf
    df = pd.DataFrame({"v0": x})

    with raises(ValueError, match="[Ii]nfinity"):
        _process(names=["v0"]).fit(df, y_binary)

    with raises(ValueError, match="[Ii]nfinity"):
        _process(names=["v0"]).fit(df, y_binary, check_input=True)

    # Finite but extreme magnitudes are binned; float32 prebinning caps them.
    big = X_df["v0"].values.copy()
    big[0], big[1] = 1e30, -1e30
    process = _process(names=["v0"])
    process.fit(pd.DataFrame({"v0": big}), y_binary)
    optb = process.get_binned_variable("v0")
    assert optb.status == "OPTIMAL"
    assert np.all(np.isfinite(optb.splits))


def test_mismatched_lengths():
    with raises(ValueError, match="inconsistent numbers of samples"):
        _process(names=["v0"]).fit(X_df[["v0"]], y_binary[:10],
                                   check_input=True)


def test_unusual_category_labels():
    labels = np.tile(["", "  ", "ü/ñ", "1,000"], N // 4)
    df = pd.DataFrame({"cat": labels})

    process = BinningProcess(["cat"], max_n_prebins=5, min_prebin_size=0.1)
    process.fit(df, y_binary)

    optb = process.get_binned_variable("cat")
    assert optb.status == "OPTIMAL"
    assert process.summary().set_index("name").loc["cat", "dtype"] == \
        "categorical"

    transformed = process.transform(df, metric="woe")
    assert transformed.shape == (N, 1)
    assert np.all(np.isfinite(transformed["cat"].values))


def test_user_splits_and_user_splits_fixed():
    user_splits = np.array([-1.0, 0.0, 1.0])

    free = _process(names=["v0"],
                    binning_fit_params={"v0": {"user_splits": user_splits}})
    free.fit(X_df[["v0"]], y_binary)
    splits = free.get_binned_variable("v0").splits
    assert free.get_binned_variable("v0").status == "OPTIMAL"
    # The optimiser may merge user splits, never invent one.
    assert set(splits) <= set(user_splits)
    assert np.all(np.diff(splits) > 0)

    fixed_splits = np.array([-0.5, 0.5])
    fixed = _process(names=["v0"], binning_fit_params={
        "v0": {"user_splits": fixed_splits,
               "user_splits_fixed": [True, True]}})
    fixed.fit(X_df[["v0"]], y_binary)
    assert fixed.get_binned_variable("v0").splits == approx(fixed_splits)

    # A fixed split that would leave a pure prebin is refused by name.
    pure = _process(names=["v0"], binning_fit_params={
        "v0": {"user_splits": user_splits,
               "user_splits_fixed": [True, True, True]}})
    with raises(ValueError, match="Fixed user_splits"):
        pure.fit(X_df[["v0"]], y_binary)


def test_no_variables_selected_warns_and_returns_empty():
    process = _process(selection_criteria={"iv": {"min": 1e9}})
    process.fit(X_df, y_binary)

    with warns(UserWarning, match="No variables were selected"):
        transformed = process.transform(X_df)

    assert transformed.shape == (N, 0)


# ---------------------------------------------------------------------------
# BinningProcessSketch
# ---------------------------------------------------------------------------

_SKETCH_NAMES = ["v0", "v1", "cat"]


def _sketch_frame():
    return pd.DataFrame({
        "v0": X_df["v0"].values,
        "v1": X_df["v1"].values,
        "cat": np.tile(["a", "b", "c", "d"], N // 4),
    })


def _solved_sketch(**kwargs):
    kwargs.setdefault("max_n_prebins", 5)
    kwargs.setdefault("categorical_variables", ["cat"])
    sketch = BinningProcessSketch(_SKETCH_NAMES, **kwargs)
    sketch.add(_sketch_frame(), y_binary)
    sketch.solve()
    return sketch


def test_sketch_categorical_variables_and_summary():
    sketch = _solved_sketch()

    assert sketch._n_categorical == 1
    assert sketch._n_numerical == 2

    summary = sketch.summary()
    assert list(summary.columns) == [
        "name", "dtype", "status", "selected", "n_bins",
        "iv", "js", "gini", "quality_score"]
    assert list(summary["name"]) == _SKETCH_NAMES
    assert set(summary["status"]) == {"OPTIMAL"}
    assert summary.set_index("name").loc["cat", "dtype"] == "categorical"
    assert summary.set_index("name").loc["v0", "dtype"] == "numerical"
    assert summary["selected"].all()
    assert (summary["iv"].values >= 0).all()


def test_sketch_verbose_add_solve_and_merge():
    frame = _sketch_frame()

    first = BinningProcessSketch(_SKETCH_NAMES, max_n_prebins=5,
                                 categorical_variables=["cat"], verbose=True)
    second = BinningProcessSketch(_SKETCH_NAMES, max_n_prebins=5,
                                  categorical_variables=["cat"], verbose=True)

    first.add(frame.iloc[:120], y_binary[:120])
    second.add(frame.iloc[120:], y_binary[120:])

    assert first.mergeable(second)
    first.merge(second)
    first.solve()

    assert first._n_add == 1
    assert first._n_solve == 1
    assert set(first.summary()["status"]) == {"OPTIMAL"}
    first.information(print_level=0)


def test_sketch_merge_signature_mismatch():
    first = BinningProcessSketch(_SKETCH_NAMES, max_n_prebins=5)
    second = BinningProcessSketch(_SKETCH_NAMES, max_n_prebins=6)

    assert not first.mergeable(second)
    with raises(Exception, match="does not share signature"):
        first.merge(second)


def test_sketch_add_ignores_columns_that_are_not_variables():
    frame = _sketch_frame()
    frame["extra"] = 1.0

    sketch = BinningProcessSketch(["v0", "v1"], max_n_prebins=5)
    sketch.add(frame, y_binary)
    sketch.solve()

    assert list(sketch.summary()["name"]) == ["v0", "v1"]
    assert sketch._n_samples == N


def test_sketch_selection_criteria_validated_on_add():
    sketch = BinningProcessSketch(
        ["v0"], max_n_prebins=5,
        selection_criteria={"woe": {"min": 0.0}})

    with raises(ValueError, match="metric for binary target must be in"):
        sketch.add(_sketch_frame()[["v0"]], y_binary)


def test_sketch_get_binned_variable_guards():
    sketch = _solved_sketch()

    with raises(TypeError, match="name must be a string"):
        sketch.get_binned_variable(1)

    with raises(ValueError, match="does not match a binned variable"):
        sketch.get_binned_variable("zzz")

    assert sketch.get_binned_variable("v0").status == "OPTIMAL"


def test_sketch_get_support_indices_and_names():
    sketch = _solved_sketch()

    with raises(ValueError, match="Only indices or names can be True"):
        sketch.get_support(indices=True, names=True)

    assert list(sketch.get_support(indices=True)) == [0, 1, 2]
    assert list(sketch.get_support(names=True)) == _SKETCH_NAMES
    assert sketch.get_support().dtype == bool


def test_sketch_transform_requires_every_selected_column():
    sketch = _solved_sketch()

    with raises(ValueError, match="must be a column in the input dataframe"):
        sketch.transform(_sketch_frame().drop(columns=["v1"]))


def test_sketch_transform_binning_transform_params():
    frame = _sketch_frame()
    x = frame["v0"].values.copy()
    x[:20] = -999.0
    frame["v0"] = x

    params = {"v0": {"metric_special": 0.25}}
    sketch = BinningProcessSketch(
        ["v0", "v1"], max_n_prebins=5, special_codes=[-999.0],
        binning_transform_params=params)
    sketch.add(frame[["v0", "v1"]], y_binary)
    sketch.solve()

    transformed = sketch.transform(frame, metric="woe")
    assert list(transformed.columns) == ["v0", "v1"]
    assert transformed["v0"].values[:20] == approx(0.25)


def test_sketch_no_variables_selected_warns_and_returns_empty():
    sketch = BinningProcessSketch(["v0", "v1"], max_n_prebins=5,
                                  selection_criteria={"iv": {"min": 1e9}})

    sketch.add(_sketch_frame()[["v0", "v1"]], y_binary)
    sketch.solve()

    assert not sketch.get_support().any()

    with warns(UserWarning, match="No variables were selected"):
        transformed = sketch.transform(_sketch_frame())

    assert transformed.shape == (N, 0)


# ---------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------

def test_defect_sketch_selection_criteria_reads_fixed_variables():
    """BinningProcessSketch.solve() raises AttributeError whenever
    ``selection_criteria`` is set.

    ``BaseBinningProcess._support_selection_criteria`` returns early when
    ``selection_criteria is None`` and otherwise reads
    ``self.fixed_variables``. ``BinningProcess`` defines that attribute;
    ``BinningProcessSketch`` does not, and has no ``fixed_variables``
    parameter -- yet it documents ``selection_criteria`` and validates it in
    ``add``.
    """
    sketch = BinningProcessSketch(["v0", "v1"], max_n_prebins=5,
                                  selection_criteria={"iv": {"min": 0.0}})
    sketch.add(_sketch_frame()[["v0", "v1"]], y_binary)
    sketch.solve()

    assert sketch.get_support().all()


def test_defect_sketch_transform_rejects_documented_metrics():
    """BinningProcessSketch.transform allows only "woe" and "event_rate".

    Its own docstring lists "indices" and "bins", its own body has the
    ``metric == "indices"`` / ``metric == "bins"`` branches that build the
    output array, and both ``OptimalBinningSketch.transform`` and the sibling
    ``BinningProcess.transform`` accept them. The guard makes those branches
    dead code.
    """
    sketch = _solved_sketch()

    indices = sketch.transform(_sketch_frame(), metric="indices")
    assert indices.values.dtype == int

    bins = sketch.transform(_sketch_frame(), metric="bins")
    assert all(isinstance(v, str) for v in bins["v0"])


def test_defect_selection_criteria_strategy_message_arguments_swapped():
    """The "strategy" error interpolates its two arguments the wrong way round.

    ``_check_selection_criteria`` raises
    ``.format(value, metric)`` into a template whose first slot is the metric
    and whose second is the offending value. Every other raise in the same
    function formats ``(metric, value, ...)``.
    """
    with raises(ValueError) as exc:
        _check_selection_criteria({"iv": {"strategy": "sideways"}}, "binary")

    message = str(exc.value)
    assert "strategy value for metric iv" in message
    assert "got sideways" in message


def test_defect_fit_does_not_validate_fixed_variables():
    """``fit`` accepts a fixed variable that is not a variable name.

    ``_fit_disk`` checks each entry of ``fixed_variables`` against
    ``variable_names`` and raises "Variable {} to be fixed is not a valid
    variable name."; ``_fit`` and ``_fit_from_dict`` do not. With
    ``selection_criteria=None`` the bad name is silently ignored, and with a
    criterion set it surfaces as a bare ``ValueError: 'zzz' is not in list``
    from ``list.index`` inside ``_support_selection_criteria``.
    """
    with raises(ValueError, match="Variable zzz to be fixed"):
        _process(fixed_variables=["zzz"]).fit(X_df, y_binary)

    with raises(ValueError, match="Variable zzz to be fixed"):
        _process(fixed_variables=["zzz"],
                 selection_criteria={"iv": {"min": 0.0}}).fit(X_df, y_binary)

    with raises(ValueError, match="Variable zzz to be fixed"):
        _process(fixed_variables=np.array(["zzz"])).fit_transform(
            X_df, y_binary)

    dict_optb = {}
    for name in ["v0", "v1"]:
        optb = OptimalBinning(name=name, max_n_prebins=5, min_prebin_size=0.1)
        optb.fit(X_df[name], y_binary)
        dict_optb[name] = optb

    with raises(ValueError, match="Variable zzz to be fixed"):
        _process(names=["v0", "v1"],
                 fixed_variables=["zzz"]).fit_from_dict(dict_optb)


def test_defect_update_binned_variable_binary_guard_accepts_continuous():
    """The "target is binary" guard lets a continuous binning through.

    ``update_binned_variable`` checks ``isinstance(optb, (OptimalBinning,
    OptimalPWBinning))`` for a binary target, but ``ContinuousOptimalBinning``
    and ``MulticlassOptimalBinning`` both *subclass* ``OptimalBinning``, so
    they satisfy it. The continuous and multiclass branches name leaf classes
    and are not fooled. The process is left broken: ``summary()`` then dies
    with ``AttributeError: 'ContinuousBinningTable' object has no attribute
    'gini'``.
    """
    process = _fitted(names=["v0"])

    coptb = ContinuousOptimalBinning(name="v0", max_n_prebins=5,
                                     min_prebin_size=0.1)
    coptb.fit(X_df["v0"], y_continuous)
    with raises(TypeError, match="target is binary"):
        process.update_binned_variable("v0", coptb)

    moptb = MulticlassOptimalBinning(name="v0", max_n_prebins=5,
                                     min_prebin_size=0.1)
    moptb.fit(X_df["v0"], y_multiclass)
    with raises(TypeError, match="target is binary"):
        process.update_binned_variable("v0", moptb)


def test_defect_transform_metric_indices_mixed_is_silently_truncated():
    """A per-variable metric too wide for the output array is coerced.

    ``_transform`` allocates one output array from the reconciled base
    metric and then transforms each variable with
    ``binning_transform_params[name]["metric"]``. The reconciliation raised
    on a "bins" mix but not on an "indices" one, so the integer array a base
    metric of "indices" allocates received WoE and truncated it: -2.407 was
    stored as -2.

    Only this direction loses data. The reverse -- a numeric base metric
    with a per-variable "indices" -- is exact and must keep working; it is
    pinned by
    ``test_transform_params_indices_override_under_a_numeric_base_metric``.
    """
    params = {"v0": {"metric": "woe"}}
    process = _process(names=["v0", "v1"], binning_transform_params=params)
    process.fit(X_df[["v0", "v1"]], y_binary)

    with raises(ValueError, match="'indices' cannot be mixed"):
        process.transform(X_df[["v0", "v1"]], metric="indices")


def test_transform_params_indices_metric_for_every_variable():
    """Agreeing metrics are not rejected: the guard is about mixing."""
    params = {name: {"metric": "indices"} for name in ["v0", "v1"]}
    process = _process(names=["v0", "v1"], binning_transform_params=params)
    process.fit(X_df[["v0", "v1"]], y_binary)

    transformed = process.transform(X_df[["v0", "v1"]], metric="indices")
    assert transformed.values.dtype == int
    assert (transformed.values >= 0).all()


def test_transform_params_indices_override_under_a_numeric_base_metric():
    """A per-variable "indices" override is lossless under a float array.

    Only one direction of the dtype clash loses data. An "indices" base
    metric allocates an integer array, so a numeric override written into it
    is truncated. The reverse -- a numeric or default base metric with a
    per-variable "indices" -- allocates a float array, and bin indices are
    small integers that float64 holds exactly. Both the explicit metric and
    the documented ``metric=None`` default keep working.
    """
    frame = X_df[["v0", "v1"]]
    params = {"v0": {"metric": "indices"}}
    process = _process(names=["v0", "v1"], binning_transform_params=params)
    process.fit(frame, y_binary)

    optb0 = process.get_binned_variable("v0")
    optb1 = process.get_binned_variable("v1")
    expected = optb0.transform(frame["v0"].values, metric="indices")

    transformed = process.transform(frame, metric="woe")
    assert np.array_equal(transformed["v0"].values, expected)
    assert transformed["v1"].values == approx(
        optb1.transform(frame["v1"].values, metric="woe"))

    default = process.transform(frame)
    assert np.array_equal(default["v0"].values, expected)
    assert default["v1"].values == approx(
        optb1.transform(frame["v1"].values))


def test_transform_disk_indices_override_under_a_numeric_base_metric():
    """``_transform_disk`` reconciles metrics through the same helper."""
    path = _disk_input()
    output = _fresh_output("binning_process_edge_indices_override.csv")

    params = {"v0": {"metric": "indices"}}
    process = _process(names=["v0", "v1"], binning_transform_params=params)
    process.fit(X_df[["v0", "v1"]], y_binary)
    process.transform_disk(path, output, chunksize=100, metric="woe")

    written = pd.read_csv(output)
    optb0 = process.get_binned_variable("v0")
    optb1 = process.get_binned_variable("v1")

    assert np.array_equal(
        written["v0"].values,
        optb0.transform(X_df["v0"].values, metric="indices"))
    assert written["v1"].values == approx(
        optb1.transform(X_df["v1"].values, metric="woe"))


def test_sketch_transform_indices_override_under_a_numeric_base_metric():
    """The sketch shares the helper, and the same lossless direction."""
    frame = _sketch_frame()[["v0", "v1"]]

    sketch = BinningProcessSketch(
        ["v0", "v1"], max_n_prebins=5,
        binning_transform_params={"v0": {"metric": "indices"}})
    sketch.add(frame, y_binary)
    sketch.solve()

    transformed = sketch.transform(frame, metric="woe")
    optb0 = sketch.get_binned_variable("v0")
    optb1 = sketch.get_binned_variable("v1")

    assert np.array_equal(
        transformed["v0"].values,
        optb0.transform(frame["v0"].values, metric="indices"))
    assert transformed["v1"].values == approx(
        optb1.transform(frame["v1"].values, metric="woe"))


def test_defect_transform_disk_metric_indices_mixed_is_silently_truncated():
    """``_transform_disk`` carried the same gap, and the same reconciliation.

    It allocates its chunk array from the same reconciled base metric, so
    the truncation described in
    ``test_defect_transform_metric_indices_mixed_is_silently_truncated``
    reached the written csv too. The lossless direction is pinned by
    ``test_transform_disk_indices_override_under_a_numeric_base_metric``.
    """
    path = _disk_input()
    output = _fresh_output("binning_process_edge_mixed_indices.csv")

    params = {"v0": {"metric": "woe"}}
    process = _process(names=["v0", "v1"], binning_transform_params=params)
    process.fit(X_df[["v0", "v1"]], y_binary)

    with raises(ValueError, match="'indices' cannot be mixed"):
        process.transform_disk(path, output, chunksize=100, metric="indices")


def test_defect_sketch_transform_ignores_per_variable_metric_dtype():
    """``BinningProcessSketch.transform`` picked its output dtype from the
    top-level ``metric`` alone.

    Each variable is transformed with ``params.get("metric", metric)``, so a
    per-variable override of a different dtype used to be written into the
    wrong array: "bins" strings into a float array raised numpy's opaque
    ``could not convert string to float``, and a numeric override under an
    "indices" base metric was silently truncated to zero. Round 1 widened
    the metric guard to accept "indices" and "bins" without porting the
    ``base_metric`` reconciliation ``BinningProcess._transform`` already had;
    the sketch now shares that helper and raises instead. The lossless
    direction is pinned by
    ``test_sketch_transform_indices_override_under_a_numeric_base_metric``.
    """
    frame = _sketch_frame()[["v0", "v1"]]

    sketch = BinningProcessSketch(
        ["v0", "v1"], max_n_prebins=5,
        binning_transform_params={"v0": {"metric": "bins"}})
    sketch.add(frame, y_binary)
    sketch.solve()

    with raises(ValueError, match="'bins' cannot be mixed"):
        sketch.transform(frame, metric="woe")

    sketch = BinningProcessSketch(
        ["v0", "v1"], max_n_prebins=5,
        binning_transform_params={"v0": {"metric": "event_rate"}})
    sketch.add(frame, y_binary)
    sketch.solve()

    with raises(ValueError, match="'indices' cannot be mixed"):
        sketch.transform(frame, metric="indices")


def test_sketch_transform_params_metric_for_every_variable():
    """Agreeing metrics keep working, for both non-numeric dtypes."""
    frame = _sketch_frame()[["v0", "v1"]]
    params = {name: {"metric": "bins"} for name in ["v0", "v1"]}

    sketch = BinningProcessSketch(["v0", "v1"], max_n_prebins=5,
                                  binning_transform_params=params)
    sketch.add(frame, y_binary)
    sketch.solve()

    bins = sketch.transform(frame, metric="bins")
    assert all(isinstance(v, str) for v in bins["v0"])

    params = {name: {"metric": "indices"} for name in ["v0", "v1"]}
    sketch = BinningProcessSketch(["v0", "v1"], max_n_prebins=5,
                                  binning_transform_params=params)
    sketch.add(frame, y_binary)
    sketch.solve()

    indices = sketch.transform(frame, metric="indices")
    assert indices.values.dtype == int


def test_defect_base_binning_process_does_not_declare_fixed_variables():
    """``_support_selection_criteria`` reads an attribute the base never
    declares.

    ``BinningProcess`` sets ``self.fixed_variables`` in ``__init__``;
    ``BinningProcessSketch`` has no such parameter, and the shared base
    method reads it. Declaring the default on ``BaseBinningProcess`` states
    the contract once, instead of every reader guessing with ``getattr``.
    """
    assert BaseBinningProcess.fixed_variables is None

    sketch = BinningProcessSketch(["v0", "v1"], max_n_prebins=5)
    assert sketch.fixed_variables is None
    assert "fixed_variables" not in sketch.get_params()


def test_defect_two_dimensional_binning_accepted_as_a_process_variable():
    """The binning process accepts a 2D binning and is left unusable.

    ``OptimalBinning2D`` subclasses ``OptimalBinning`` and
    ``ContinuousOptimalBinning2D`` subclasses ``OptimalBinning2D``, so both
    satisfy the ``_OPTB_TYPES`` isinstance test and the "target is binary"
    guard -- which names the 1D continuous and multiclass leaves, not these.
    A binning process reads ``optb.dtype``, which no 2D estimator sets, so
    the next ``summary()`` dies with ``AttributeError``; ``fit_from_dict``
    dies the same way while still inside ``fit_from_dict``.
    """
    x, y2 = X_df["v0"].values, X_df["v1"].values

    optb2d = OptimalBinning2D(name_x="v0", name_y="v1", max_n_prebins_x=3,
                              max_n_prebins_y=3)
    optb2d.fit(x, y2, y_binary)

    coptb2d = ContinuousOptimalBinning2D(name_x="v0", name_y="v1",
                                         max_n_prebins_x=3,
                                         max_n_prebins_y=3)
    coptb2d.fit(x, y2, y_continuous)

    # The 2D name is "name_x-name_y", so the name check only bites when the
    # process variable is not called that.
    assert optb2d.name == "v0-v1"
    assert not hasattr(optb2d, "dtype")
    assert not hasattr(coptb2d, "dtype")

    frame = pd.DataFrame({"v0-v1": x, "v1": y2})
    process = BinningProcess(["v0-v1", "v1"], max_n_prebins=5,
                             min_prebin_size=0.1)
    process.fit(frame, y_binary)

    # The rejection gets its own sentence: interpolating the accepted 1D
    # types would read "OptimalBinning2D must be of type (OptimalBinning,
    # ...)", which contradicts itself.
    for optb in (optb2d, coptb2d):
        with raises(TypeError, match="two-dimensional estimators bin a pair"):
            process.update_binned_variable("v0-v1", optb)

        with raises(TypeError, match="two-dimensional estimators bin a pair"):
            BinningProcess(["v0-v1"]).fit_from_dict({"v0-v1": optb})

    # The process is still the one that was fitted.
    assert process.summary()["status"].tolist() == ["OPTIMAL", "OPTIMAL"]


def test_defect_transform_params_override_leaks_to_later_variables():
    """A per-variable transform option becomes the default for the rest.

    ``_transform`` rebound its own ``metric`` / ``metric_special`` /
    ``metric_missing`` arguments from ``binning_transform_params[name]``
    inside the loop over the selected variables, so the first override
    silently became the default every later variable inherited. Only "v0" is
    configured here; "v1" must still be transformed with the call's metric.
    """
    frame = X_df[["v0", "v1"]]
    params = {"v0": {"metric": "event_rate"}}
    process = _process(names=["v0", "v1"], binning_transform_params=params)
    process.fit(frame, y_binary)

    transformed = process.transform(frame, metric="woe")

    optb0 = process.get_binned_variable("v0")
    optb1 = process.get_binned_variable("v1")

    assert transformed["v0"].values == approx(
        optb0.transform(frame["v0"].values, metric="event_rate"))
    assert transformed["v1"].values == approx(
        optb1.transform(frame["v1"].values, metric="woe"))


def test_defect_transform_disk_params_override_leaks_to_later_variables():
    """``_transform_disk`` leaked the same way, and across chunks too."""
    path = _disk_input()
    output = _fresh_output("binning_process_edge_leak.csv")

    params = {"v0": {"metric": "event_rate"}}
    process = _process(binning_transform_params=params)
    process.fit(X_df, y_binary)
    process.transform_disk(path, output, chunksize=100, metric="woe")

    written = pd.read_csv(output)
    for name in variable_names:
        optb = process.get_binned_variable(name)
        expected = "event_rate" if name == "v0" else "woe"
        assert written[name].values == approx(
            optb.transform(X_df[name].values, metric=expected))


def test_defect_sketch_transform_params_override_leaks_to_later_variables():
    """``BinningProcessSketch.transform`` leaked metric_special and
    metric_missing the same way."""
    frame = _sketch_frame()[["v0", "v1"]]
    for name in ("v0", "v1"):
        column = frame[name].values.copy()
        column[:20] = -999.0
        frame[name] = column

    sketch = BinningProcessSketch(
        ["v0", "v1"], max_n_prebins=5, special_codes=[-999.0],
        binning_transform_params={"v0": {"metric_special": 0.25}})
    sketch.add(frame, y_binary)
    sketch.solve()

    transformed = sketch.transform(frame, metric="woe")

    # Only v0 asked for 0.25; v1 keeps the call's metric_special of 0.
    assert transformed["v0"].values[:20] == approx(0.25)
    assert transformed["v1"].values[:20] == approx(0.0)


def test_sketch_per_variable_metric_none_falls_back_to_the_default():
    # A per-variable {"metric": None} drops the key entirely so the estimator
    # applies its own default. The top-level `metric` guard rejects None, so
    # the per-variable override is the only route into that branch.
    sketch = BinningProcessSketch(
        variable_names, binning_transform_params={"v0": {"metric": None}})
    sketch.add(X_df, y_binary)
    sketch.solve()

    out = sketch.transform(X_df, metric="woe")
    default = sketch.get_binned_variable("v0").transform(X_df["v0"].values)

    assert np.allclose(out["v0"].values, default)
