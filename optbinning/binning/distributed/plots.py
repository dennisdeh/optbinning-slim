"""
Binning sketch plots.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import matplotlib.pyplot as plt
import numpy as np


def plot_progress_divergence(df, divergence, savefig=None, save_kwargs=None):
    n = len(df)
    n_add = df.n_add
    n_records = df.n_records
    div = df.divergence

    mv_div_mean = div.rolling(n, min_periods=1).mean()
    mv_div_std = div.rolling(n, min_periods=1).std()
    mv_div_std /= np.sqrt(np.arange(1, n+1))

    div_low = np.maximum(0, div - mv_div_std * 1.959963984540054)
    div_high = div + mv_div_std * 1.959963984540054

    div_label = "divergence ({:.5f})".format(div.values[-1])
    mv_div_label = "moving mean ({:.5f})".format(mv_div_mean.values[-1])
    mv_std_label = "standard error ({:.5f})".format(mv_div_std.values[-1])

    plt.plot(n_records, div, label=div_label)
    plt.plot(n_records, mv_div_mean, linestyle="-.", color="green",
             label=mv_div_label)
    plt.fill_between(n_records, div_low, div_high, alpha=0.2, color="green",
                     label=mv_std_label)

    plt.title("Progress after {:} add and {} processed records".
              format(int(n_add.values[-1]), int(n_records.values[-1])),
              fontsize=14)
    plt.xlabel("Processed records", fontsize=12)
    plt.ylabel("Divergence: {}".format(divergence), fontsize=12)
    plt.legend(fontsize=12)

    if savefig is None:
        plt.show()
    else:
        if not isinstance(savefig, str):
            raise TypeError("savefig must be a string path; got {}."
                            .format(savefig))
        if save_kwargs is None:
            save_kwargs = {}
        else:
            if not isinstance(save_kwargs, dict):
                raise TypeError("save_kwargs must be a dictionary; got {}."
                                .format(save_kwargs))

        plt.savefig(savefig, **save_kwargs)
        plt.close()
