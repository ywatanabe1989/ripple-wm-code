#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 19:25:41 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts import utils
from scripts.ripple.NT.distance.from_O_lineplot import calc_dist_by_condi

"""Functions & Classes"""


def plot_box(df):
    df = df[df.period.isin(["pre", "mid", "post"])]

    conditions = list(
        (
            df[["SWR_type", "period", "match", "phase"]]
            .drop_duplicates()
            .T.to_dict()
        ).values()
    )

    fig, axes = mngs.plt.subplots(
        nrows=1,
        ncols=len(conditions) // 2 // 3,
    )

    PHASES = list(df.phase.unique())

    for i_match, match in enumerate([1, 2]):
        for i_phase, phase in enumerate(PHASES):
            i_ax = i_match * len(PHASES) + i_phase
            ax = axes[i_ax]
            data = []
            labels = []
            for swr_type in ["SWR-", "SWR+"]:
                for period in ["pre", "mid", "post"]:
                    condi = {
                        "match": match,
                        "phase": phase,
                        "SWR_type": swr_type,
                        "period": period,
                    }
                    dd = mngs.pd.slice(df, condi).distances.iloc[0]
                    dd = np.nanmean(dd, axis=-1)
                    dd = dd[~np.isnan(dd)]
                    print(dd)
                    data.append(dd)
                    label = mngs.gen.dict2str(condi)
                    labels.append(label)

            ax.boxplot(
                data,
                showfliers=False,
                positions=np.arange(len(data)),
                labels=labels,
                # positions=labels,
            )
            ax.legend(loc="upper left")
            print(labels)
            ax.set_xyt(
                None,
                None,
                {1: "Match IN", 2: "Mismatch OUT"}[condi["match"]]
                + "\n"
                + phase,
            )
    fig.supxyt("SWR period", "Distance from O [a.u.]", None)
    return fig


def main():
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    data = calc_dist_by_condi(swr_p_all, swr_m_all)

    # Box Plot
    fig = plot_box(data)
    mngs.io.save(fig, "./SWR-triggered_distance_from_O.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        fig_scale=2,
        font_size_legend=3,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
