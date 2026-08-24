"""
Agreement between what the library documents and what it accepts.

Every value named in a docstring or in a validation error message must be a
value the corresponding parameter actually accepts. These are mechanical
checks: they exist because a docstring advertised ``outlier_detector="zcore"``
and three error messages named ``"helliger"``, ``"pd_odds"`` and
``"split_digist"`` -- none of which the code accepts -- and nothing caught it.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import ast
import inspect
import pathlib
import re

import numpy as np

from pytest import raises

from optbinning import ContinuousOptimalBinning
from optbinning import ContinuousOptimalPWBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning import OptimalPWBinning
from optbinning.binning.outlier import ModifiedZScoreDetector


PACKAGE = (pathlib.Path(inspect.getfile(OptimalBinning)).parents[2]
           / "optbinning")

_RNG = np.random.RandomState(0)
x = _RNG.normal(size=200)
y = (_RNG.rand(200) > 0.5).astype(int)
y_cont = 2.0 * x + _RNG.normal(scale=0.5, size=200)
y_multi = _RNG.randint(0, 3, 200)


def _sources():
    return sorted(PACKAGE.rglob("*.py"))


def test_no_docstring_advertises_zcore():
    # "zcore" is not a value any outlier_detector accepts; the detector is
    # "zscore". Following the docstring used to raise ValueError.
    offenders = [str(p) for p in _sources()
                 if "zcore" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_documented_outlier_detectors_are_accepted():
    # Every value the docstring names must survive _check_parameters.
    for cls, target in ((OptimalBinning, y),
                        (ContinuousOptimalBinning, y_cont),
                        (MulticlassOptimalBinning, y_multi)):
        doc = inspect.getdoc(cls)
        section = re.search(
            r"^outlier_detector : (.*?)(?=^\w[\w ]*\s:\s)", doc,
            re.S | re.M)
        assert section is not None, cls.__name__
        named = set(re.findall(r'"([a-z_]+)"', section.group(1)))
        assert named, cls.__name__
        for value in sorted(named):
            cls(outlier_detector=value).fit(x, target)


def test_error_messages_only_name_accepted_values():
    # For every `if <name> not in (...): raise ValueError(...)` guard, the
    # quoted values in the message must be a subset of the accepted tuple.
    bad = []
    for path in _sources():
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.NotIn)
                    and isinstance(test.left, ast.Name)
                    and isinstance(test.comparators[0],
                                   (ast.Tuple, ast.List, ast.Set))):
                continue
            allowed = {e.value for e in test.comparators[0].elts
                       if isinstance(e, ast.Constant)
                       and isinstance(e.value, str)}
            if not allowed:
                continue
            param = test.left.id
            for raise_node in [n for n in ast.walk(node)
                               if isinstance(n, ast.Raise)]:
                msg = ast.get_source_segment(src, raise_node) or ""
                if param.lower() not in msg.lower():
                    continue
                named = set(re.findall(r'"([a-z_][a-z_0-9]+)"', msg))
                unknown = named - allowed - {"got"}
                if unknown:
                    bad.append((path.name, param, sorted(unknown)))
    assert bad == []


def test_scorecard_scaling_method_message_names_the_accepted_value():
    # The message used to name "pd_odds", which Scorecard rejects; the
    # value it accepts is "pdo_odds".
    from sklearn.linear_model import LogisticRegression

    from optbinning import BinningProcess
    from optbinning.scorecard.scorecard import _check_parameters

    with raises(ValueError, match="pdo_odds"):
        _check_parameters(binning_process=BinningProcess(["v"]),
                          estimator=LogisticRegression(),
                          scaling_method="nope",
                          scaling_method_params={"a": 1},
                          intercept_based=False, reverse_scorecard=False,
                          rounding=False, verbose=False)


def test_divergence_message_names_hellinger():
    with raises(ValueError, match="hellinger"):
        OptimalBinning(divergence="nope").fit(x, y)


def test_special_codes_message_says_dict_not_dit():
    for cls, target in ((OptimalBinning, y),
                        (ContinuousOptimalBinning, y_cont),
                        (MulticlassOptimalBinning, y_multi)):
        with raises(TypeError, match="must be a dict, list"):
            cls(special_codes=3.14).fit(x, target)


def test_split_digits_message_spells_the_parameter():
    for cls, target in ((OptimalBinning, y),
                        (ContinuousOptimalBinning, y_cont),
                        (MulticlassOptimalBinning, y_multi),
                        (OptimalPWBinning, y),
                        (ContinuousOptimalPWBinning, y_cont)):
        with raises(ValueError, match="split_digits must be"):
            cls(split_digits=99).fit(x, target)


def test_max_n_prebins_message_spells_the_parameter():
    # The parameter is max_n_prebins; the message used to say "max_prebins",
    # which is not a keyword any estimator accepts.
    for cls, target in ((OptimalBinning, y),
                        (ContinuousOptimalBinning, y_cont),
                        (MulticlassOptimalBinning, y_multi)):
        with raises(ValueError, match="max_n_prebins must be"):
            cls(max_n_prebins=1).fit(x, target)


def test_modified_zscore_detector_docstring_matches_its_signature():
    documented = set(re.findall(r"^([a-z_]+) : ",
                                inspect.getdoc(ModifiedZScoreDetector),
                                re.M))
    signature = set(inspect.signature(
        ModifiedZScoreDetector.__init__).parameters) - {"self"}
    assert documented == signature


def test_time_limit_is_documented_as_int_or_float():
    # Every _check_parameters validates time_limit as numbers.Number, and
    # both solver backends honour a fractional value, so `int` alone was
    # never the accurate type. Mechanical because the entry is duplicated
    # across eight files and drifted for the whole life of the project.
    entries = []
    for path in _sources():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("time_limit : "):
                entries.append((path.name, stripped))

    assert entries, "no time_limit docstring entries found"
    wrong = [(name, text) for name, text in entries
             if not text.startswith("time_limit : int or float ")]
    assert wrong == []
