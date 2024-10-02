#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-01 08:47:03 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts import utils

"""Config"""
CONFIG = mngs.io.load_configs()

"""Functions & Classes"""


def calc_dist_by_condi(swr_p_all, swr_m_all):
    def _NT2dist(xx_all):
        nt_xx = np.stack(xx_all.NT, axis=0)
        dd_xx = np.sqrt((nt_xx**2).sum(axis=1))
        return dd_xx

    PHASES = ["Encoding", "Retrieval"]
    condi_all = {"match": CONFIG.MATCHES, "phase": PHASES}
    PERIODS = ["pre", "mid", "post", "all"]

    data_list = []
    for xx_all, swr_type in zip([swr_p_all, swr_m_all], ["SWR+", "SWR-"]):
        for condi in mngs.gen.yield_grids(condi_all):
            df = mngs.pd.slice(xx_all, condi)
            dd_xx = _NT2dist(df)

            for period in PERIODS:
                if period == "all":
                    dd_period = dd_xx
                else:
                    start, end = CONFIG.RIPPLE.BINS[period]
                    dd_period = dd_xx[
                        :,
                        dd_xx.shape[1] // 2
                        + start : dd_xx.shape[1] // 2
                        + end,
                    ]

                data_list.append(
                    {
                        "SWR_type": swr_type,
                        "period": period,
                        "phase": condi["phase"],
                        "match": condi["match"],
                        "distances": dd_period,
                    }
                )

    result_df = pd.DataFrame(data_list)
    return result_df


def plot_line(df):
    df = df[df.period == "all"]

    conditions = list(
        (
            df[["SWR_type", "match", "phase"]].drop_duplicates().T.to_dict()
        ).values()
    )

    fig, axes = mngs.plt.subplots(
        nrows=1,
        ncols=len(conditions) // 2,
    )

    PHASES = list(df.phase.unique())

    for condi in conditions:
        i_ax = (condi["match"] - 1) * len(PHASES) + mngs.gen.search(
            condi["phase"], PHASES
        )[0][0]
        ax = axes[i_ax]

        dd = mngs.pd.slice(df, condi).distances.iloc[0]

        described = mngs.gen.describe(dd, "mean_ci", axis=0)

        tt = np.arange(len(described["mean"])) * CONFIG.GPFA.BIN_SIZE_MS
        tt -= int(tt.mean())

        ax.plot_(
            xx=tt,
            **described,
            label=condi["SWR_type"],
            id=mngs.gen.dict2str(condi),
            alpha=0.1,
            color=CC["purple" if condi["SWR_type"] == "SWR+" else "yellow"],
        )

        ax.set_xyt(
            None, None, {1: "Match IN", 2: "Mismatch OUT"}[condi["match"]]
        )

    fig.supxyt("Time from SWR peak [ms]", "Distance from O [a.u.]", None)
    return fig


def main():
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    df = calc_dist_by_condi(swr_p_all, swr_m_all)

    # Line Plot
    fig = plot_line(df)
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
