"""
Test suite configuration.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import matplotlib


# The plotting tests only ever save figures, so pin the non-interactive
# backend rather than letting matplotlib pick one. It picks a GUI backend
# wherever it believes there is a display -- always, on Windows -- and the
# hosted Python on the Windows CI runners ships a Tcl installation that
# tkinter cannot initialise, which failed test_multiclass_binning.py the
# moment fail-fast stopped hiding that job. See reports/DECISIONS.md.
matplotlib.use("Agg", force=True)
