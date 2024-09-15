#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 15:10:11 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""
This version creates a 3x3 grid of plots:
- Columns: All data, Match IN, Mismatch OUT
- Rows:
1. eSWR+*rSWR+, eSWR+*vER, rSWR+*vER
2. eSWR-*rSWR-, eSWR-*vER, rSWR-*vER
3. Differences
"""

"""Imports"""
import os
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scripts import utils
from scripts.ripple.NT.distance.from_O_lineplot import calc_dist_by_condi

"""Functions & Classes"""
XLIM = (0, np.pi)
# XLIM = (None, None)


def process_data(swr_all, match, SWR_direction_def):
    radian = {}
    for ca1 in CONFIG.ROI.CA1:
        swr_all["sub"] = swr_all["subject"]
        df = mngs.pd.slice(swr_all, ca1)
        if match is not None:
            df = df[df.match == match]

        for swr_type in CONFIG.RIPPLE.TYPES:
            df_E = df[(df["swr_type"] == swr_type) & (df.phase == "Encoding")]
            df_R = df[(df["swr_type"] == swr_type) & (df.phase == "Retrieval")]

            v1_SWR = np.vstack(df_E[f"vSWR_def{SWR_direction_def}"])
            v2_SWR = np.vstack(df_R[f"vSWR_def{SWR_direction_def}"])
            v_ER = np.vstack(df_R[f"vER"])

            radian[f"{ca1.values()}_{swr_type}_eSWR_vs_rSWR"] = np.arccos(
                1 - cdist(v1_SWR, v2_SWR, metric="cosine").reshape(-1)
            )
            radian[f"{ca1.values()}_{swr_type}_eSWR_vs_vER"] = np.arccos(
                1 - cdist(v1_SWR, v_ER, metric="cosine")[:, 0]
            )
            radian[f"{ca1.values()}_{swr_type}_rSWR_vs_vER"] = np.arccos(
                1 - cdist(v2_SWR, v_ER, metric="cosine")[:, 0]
            )

    df = mngs.pd.force_df(radian)
    return df


def main(SWR_direction_def=1):
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    df_all = process_data(swr_all, None, SWR_direction_def)
    df_in = process_data(swr_all, 1, SWR_direction_def)
    df_out = process_data(swr_all, 2, SWR_direction_def)

    fig, axes = mngs.plt.subplots(
        ncols=3, nrows=3, figsize=(15, 15), sharex=True, sharey=True
    )

    for col, (df, title) in enumerate(
        zip([df_all, df_in, df_out], ["All", "Match IN", "Mismatch OUT"])
    ):
        for i_swr_type, swr_type in enumerate(CONFIG.RIPPLE.TYPES):
            ax = axes[i_swr_type, col]
            ax.set_title(f"{title} - {swr_type}")
            ax.set_xlim(*XLIM)
            # ax.set_ylim(0 * 1e-4, 20 * 1e-4)

            for comparison in ["eSWR_vs_rSWR", "eSWR_vs_vER", "rSWR_vs_vER"]:
                data = df[
                    [
                        col
                        for col in df.columns
                        if f"{swr_type}_{comparison}" in col
                    ]
                ]
                data = data.values.flatten()
                data = data[~np.isnan(data)]
                ax.kde(
                    data,
                    label=f"{comparison}",
                    id=f"{title}-{comparison}-{swr_type}",
                    color=CC[CONFIG.COLORS[comparison]],
                    xlim=XLIM,
                )
            ax.legend()

    # Difference plot
    plotted = axes.to_sigma()

    for col, (df, title) in enumerate(
        zip([df_all, df_in, df_out], ["All", "Match IN", "Mismatch OUT"])
    ):
        ax = axes[2, col]
        ax.set_title(f"{title} - Diff (SWR+ - SWR-)")

        for comparison in ["eSWR_vs_rSWR", "eSWR_vs_vER", "rSWR_vs_vER"]:
            x_p = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-SWR\+_kde_x", plotted.columns
                )[1]
            ]
            kde_p = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-SWR\+_kde_kde", plotted.columns
                )[1]
            ]
            x_m = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-SWR\-_kde_x", plotted.columns
                )[1]
            ]
            kde_m = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-SWR\-_kde_kde", plotted.columns
                )[1]
            ]
            kde_diff = np.array(kde_p) - np.array(kde_m)
            ax.plot(
                x_p,
                kde_diff,
                label=f"{comparison}",
                id=f"{title}-{comparison}-diff (SWR+ - SWR-)",
                color=CC[CONFIG.COLORS[f"{comparison}"]],
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        # ax.set_ylim(-10e-4, 10e-4)
        ax.set_xlim(*XLIM)
        ax.legend()

    fig.supxyt(f"Radian (vSWR def. {SWR_direction_def})", "KDE density")
    mngs.io.save(fig, f"kde_vSWR_def{SWR_direction_def}.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        os=os,
        random=random,
        np=np,
        torch=None,
        fig_scale=2,
        font_size_base=8,
        font_size_title=8,
        font_size_axis_label=7,
        font_size_tick_label=7,
        font_size_legend=8,
        line_width=3,
    )
    main(SWR_direction_def=1)
    main(SWR_direction_def=2)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
