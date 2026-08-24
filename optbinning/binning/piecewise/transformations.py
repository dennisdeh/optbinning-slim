"""
Piecewise binning transformations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from sklearn.utils import check_array

from ...binning.transformations import transform_event_rate_to_woe
from ...binning.transformations import _check_metric_special_dict
from ...binning.transformations import _check_metric_special_missing
from ...binning.transformations import _mask_special_missing


def _apply_transform(x, c, lb, ub, special_codes, metric_special,
                     metric_missing, clean_mask, special_mask, missing_mask,
                     indices, x_clean, n_bins, n_special, event_rate_special,
                     event_rate_missing):

    x_transform = np.zeros(x.shape)
    x_clean_transform = np.zeros(x_clean.shape)

    for i in range(n_bins):
        mask = (indices == i)
        x_clean_transform[mask] = np.polyval(c[i, :][::-1], x_clean[mask])

    # Clip values using LB/UB
    bounded = (lb is not None or ub is not None)
    if bounded:
        x_clean_transform = np.clip(x_clean_transform, lb, ub)

    x_transform[clean_mask] = x_clean_transform

    # bool() on an ndarray of more than one code raises; the checker
    # and the docstrings accept a numpy.ndarray here.
    _check_metric_special_dict(metric_special, special_codes)

    if special_codes is not None and len(special_codes):
        if isinstance(special_codes, dict):
            xt = pd.Series(x)
            for i, (k, s) in enumerate(special_codes.items()):
                sl = s if isinstance(s, (list, np.ndarray)) else [s]
                mask = xt.isin(sl).values
                # A dict metric_special carries one value per named bucket;
                # anything else applies to all of them.
                ms = (metric_special[k] if isinstance(metric_special, dict)
                      else metric_special)
                if ms == "empirical":
                    x_transform[mask] = event_rate_special[i]
                else:
                    x_transform[mask] = ms
        else:
            if metric_special == "empirical":
                x_transform[special_mask] = event_rate_special
            else:
                x_transform[special_mask] = metric_special

    if metric_missing == "empirical":
        x_transform[missing_mask] = event_rate_missing
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def transform_binary_target(splits, x, c, lb, ub, n_nonevent, n_event,
                            n_nonevent_special, n_event_special,
                            n_nonevent_missing, n_event_missing,
                            special_codes, metric, metric_special,
                            metric_missing, check_input=False):

    if metric not in ("event_rate", "woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "event_rate" and "woe".')

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)

    special_mask, missing_mask, clean_mask, n_special = _mask_special_missing(
        x, special_codes)

    x_clean = x[clean_mask]

    if len(splits):
        indices = np.digitize(x_clean, splits, right=False)
    else:
        indices = np.zeros(x_clean.shape)

    n_bins = len(splits) + 1

    # Compute event rate for special and missing bin
    event_rate_special = metric_special
    event_rate_missing = metric_missing

    if metric_special == "empirical":
        # The non-dict form reports one pooled bucket as scalars; make both
        # forms a length-n_special array so a single kernel serves them.
        n_event_special = np.atleast_1d(
            np.asarray(n_event_special, dtype=float))
        n_nonevent_special = np.atleast_1d(
            np.asarray(n_nonevent_special, dtype=float))

        event_rate_special = np.zeros(n_special)
        n_records_special = n_event_special + n_nonevent_special

        # The event rate is a property of the bucket on its own, so it is
        # gated on records: an all-event bucket reports 1 and an
        # all-non-event one 0. An empty bucket keeps 0.
        mask_records = n_records_special > 0
        event_rate_special[mask_records] = (
            n_event_special[mask_records] /
            n_records_special[mask_records])

        if metric == "woe":
            # WoE compares the bucket against the whole sample, so it is 0
            # wherever the bucket lacks either class -- the gate the binning
            # table uses, and the one that keeps log(0) out of the result.
            mask = (n_event_special > 0) & (n_nonevent_special > 0)
            woe_special = np.zeros(n_special)

            if mask.any():
                woe_special[mask] = transform_event_rate_to_woe(
                    event_rate_special[mask], n_nonevent, n_event)

            event_rate_special = woe_special

    if metric_missing == "empirical":
        n_records_missing = n_event_missing + n_nonevent_missing

        if n_records_missing > 0:
            event_rate_missing = n_event_missing / n_records_missing
        else:
            event_rate_missing = 0

        if metric == "woe":
            if n_event_missing > 0 and n_nonevent_missing > 0:
                event_rate_missing = transform_event_rate_to_woe(
                    event_rate_missing, n_nonevent, n_event)
            else:
                event_rate_missing = 0

    x_transform = _apply_transform(
        x, c, lb, ub, special_codes, metric_special, metric_missing,
        clean_mask, special_mask, missing_mask, indices, x_clean, n_bins,
        n_special, event_rate_special, event_rate_missing)

    if metric == "woe":
        # The piecewise fit is a regression, so its prediction is not
        # confined to [0, 1] unless the caller passed lb/ub -- and
        # log(1 / p - 1) is not finite outside it. Bound it the way
        # piecewise/metrics.py::binary_metrics and PWBinningTable.plot
        # already do, so metric="woe" cannot return NaN on ordinary data.
        # The bound is far below the resolution of any reported WoE, and
        # metric="event_rate" is left unbounded: there the raw prediction
        # is the answer the caller asked for.
        min_pred = 1e-8
        clean = np.clip(x_transform[clean_mask], min_pred, 1 - min_pred)
        x_transform[clean_mask] = transform_event_rate_to_woe(
            clean, n_nonevent, n_event)

    return x_transform


def transform_continuous_target(splits, x, c, lb, ub, n_records_special,
                                sum_special, n_records_missing, sum_missing,
                                special_codes, metric_special, metric_missing,
                                check_input=False):

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)

    special_mask, missing_mask, clean_mask, n_special = _mask_special_missing(
        x, special_codes)

    x_clean = x[clean_mask]

    if len(splits):
        indices = np.digitize(x_clean, splits, right=False)
    else:
        indices = np.zeros(x_clean.shape)

    n_bins = len(splits) + 1

    # Compute event rate for special and missing bin
    mean_special = metric_special
    mean_missing = metric_missing

    if metric_special == "empirical":
        sum_special = np.asarray(sum_special)
        n_records_special = np.asarray(n_records_special)

        mean_special = np.zeros(n_special)

        mask = (n_records_special > 0)

        if n_special > 1:
            mean_special[mask] = sum_special[mask] / n_records_special[mask]
        elif mask:
            mean_special = sum_special / n_records_special

    if metric_missing == "empirical":
        if n_records_missing > 0:
            mean_missing = sum_missing / n_records_missing
        else:
            mean_missing = 0

    x_transform = _apply_transform(
        x, c, lb, ub, special_codes, metric_special, metric_missing,
        clean_mask, special_mask, missing_mask, indices, x_clean, n_bins,
        n_special, mean_special, mean_missing)

    return x_transform
