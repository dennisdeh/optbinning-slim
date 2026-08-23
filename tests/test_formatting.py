"""
Printing utilities testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd

from pytest import raises

from optbinning.formatting import dataframe_to_string


df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})


def test_params():
    with raises(TypeError):
        dataframe_to_string("not a dataframe")

    with raises(ValueError):
        dataframe_to_string(df, tab=-1)

    with raises(ValueError):
        dataframe_to_string(df, tab=1.5)


def test_no_tab():
    string = dataframe_to_string(df)

    assert "a" in string
    assert not string.startswith(" ")


def test_tab():
    tab = 4
    string = dataframe_to_string(df, tab=tab)

    for line in string.splitlines():
        assert line.startswith(" " * tab)
