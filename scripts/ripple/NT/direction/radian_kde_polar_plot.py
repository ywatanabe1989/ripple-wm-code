#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 19:36:26 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""
This script creates a 3x3 grid of plots:
- Columns: All data, Match IN, Mismatch OUT
- Rows:
1. eSWR+*rSWR+, eSWR+*vER, rSWR+*vER
2. eSWR-*rSWR-, eSWR-*vER, rSWR-*vER
3. Differences
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scripts import utils
from scripts.ripple.NT.distance.from_O_lineplot import calc_dist_by_condi

XLIM = (0, np.pi)
COMPARISONS = [
    "eSWR_vs_eSWR",
    "rSWR_vs_rSWR",
    "eSWR_vs_rSWR",
    "eSWR_vs_vER",
    "rSWR_vs_vER",
]


def process_data(swr_all, match, set_size, swr_direction_def):
    radian = {}
    for ca1 in CONFIG.ROI.CA1:
        swr_all["sub"] = swr_all["subject"]
        df = mngs.pd.slice(swr_all, ca1)

        if match != "all":
            df = df[df.match == match]

        if set_size != "all":
            df = df[df.set_size == set_size]

        for swr_type in CONFIG.RIPPLE.TYPES:
            df_encoding = df[
                (df["swr_type"] == swr_type) & (df.phase == "Encoding")
            ]
            df_retrieval = df[
                (df["swr_type"] == swr_type) & (df.phase == "Retrieval")
            ]

            if df_encoding.empty or df_retrieval.empty:
                radian[f"{ca1.values()}_{swr_type}_eSWR_vs_rSWR"] = [np.nan]
                radian[f"{ca1.values()}_{swr_type}_eSWR_vs_vER"] = [np.nan]
                radian[f"{ca1.values()}_{swr_type}_rSWR_vs_vER"] = [np.nan]
            else:
                v1_swr = np.vstack(df_encoding[f"vSWR_def{swr_direction_def}"])
                v2_swr = np.vstack(
                    df_retrieval[f"vSWR_def{swr_direction_def}"]
                )
                v_er = np.vstack(df_retrieval["vER"])

                radian[f"{ca1.values()}_{swr_type}_eSWR_vs_rSWR"] = np.arccos(
                    1 - cdist(v1_swr, v2_swr, metric="cosine").reshape(-1)
                )
                radian[f"{ca1.values()}_{swr_type}_eSWR_vs_vER"] = np.arccos(
                    1 - cdist(v1_swr, v_er, metric="cosine")[:, 0]
                )
                radian[f"{ca1.values()}_{swr_type}_rSWR_vs_vER"] = np.arccos(
                    1 - cdist(v2_swr, v_er, metric="cosine")[:, 0]
                )
                radian[f"{ca1.values()}_{swr_type}_eSWR_vs_eSWR"] = np.arccos(
                    1 - cdist(v1_swr, v1_swr, metric="cosine").reshape(-1)
                )
                radian[f"{ca1.values()}_{swr_type}_rSWR_vs_rSWR"] = np.arccos(
                    1 - cdist(v2_swr, v2_swr, metric="cosine").reshape(-1)
                )

    return mngs.pd.force_df(radian)


def main(swr_direction_def=1, set_size=4):
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    df_all = process_data(swr_all, "all", set_size, swr_direction_def)
    df_in = process_data(swr_all, 1, set_size, swr_direction_def)
    df_out = process_data(swr_all, 2, set_size, swr_direction_def)

    _fig, _axes = mngs.plt.subplots(
        ncols=3,
        nrows=3,
    )

    fig, axes = mngs.plt.subplots(
        ncols=3,
        nrows=3,
        figsize=(15, 15),
        subplot_kw=dict(projection="polar"),
        sharex=False,
        sharey=False,
    )

    for col, (df, title) in enumerate(
        zip([df_all, df_in, df_out], ["All", "Match IN", "Mismatch OUT"])
    ):
        for i_swr_type, swr_type in enumerate(CONFIG.RIPPLE.TYPES):
            ax = axes[i_swr_type, col]
            _ax = _axes[i_swr_type, col]
            ax.set_title(f"{title} - {swr_type}")

            for comparison in COMPARISONS:
                data = df[
                    [
                        col
                        for col in df.columns
                        if f"{swr_type}_{comparison}" in col
                    ]
                ]
                data = data.values.flatten()
                data = data[~np.isnan(data)]

                # Plot to _ax with kde() to store data
                id = f"{title}-{comparison}-{set_size}-{swr_type}"
                _ax.kde(
                    data,
                    id=id,
                )
                id = id.replace("+", "\+").replace("-", "\-")
                dd = _ax.to_sigma()
                x = np.array(
                    dd[mngs.gen.search(id + "_kde_x", dd.columns)[1]]
                ).reshape(-1)
                kde = np.array(
                    dd[mngs.gen.search(id + "_kde_kde", dd.columns)[1]]
                ).reshape(-1)

                # Plot to ax for polar representation
                ax.plot(
                    x,
                    kde,
                    label=f"{comparison}",
                    id=id,
                    color=CC[CONFIG.COLORS[comparison]],
                )
            ax.legend()

    _plotted = _axes.to_sigma()

    for col, (df, title) in enumerate(
        zip([df_all, df_in, df_out], ["All", "Match IN", "Mismatch OUT"])
    ):
        ax = axes[2, col]
        ax.set_title(f"{title} - Diff (SWR+ - SWR-)")

        for comparison in COMPARISONS:
            x_p = _plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\+_kde_x",
                    _plotted.columns,
                )[1]
            ]
            kde_p = _plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\+_kde_kde",
                    _plotted.columns,
                )[1]
            ]
            x_m = _plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\-_kde_x",
                    _plotted.columns,
                )[1]
            ]
            kde_m = _plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\-_kde_kde",
                    _plotted.columns,
                )[1]
            ]
            kde_diff = np.array(kde_p) - np.array(kde_m)

            ax.plot(
                x_p,
                kde_diff,
                label=f"{comparison}",
                color=CC[CONFIG.COLORS[f"{comparison}"]],
            )

        ax.legend()

    # mngs.plt.ax.sharey(*axes.flat[:-3])
    # mngs.plt.ax.sharey(*axes.flat[-3:])

    for ax in axes.flat:
        ax.set_ylim(0, ax.get_ylim()[1])

    for ax in axes.flat[-2:]:
        ax.set_ylim(-10e-4, 10e-4)

    # # Set y-limit for difference plots
    # for ax in axes.flat[-2:]:
    #     ax.set_ylim(-1, 1)

    spath = f"./kde_vSWR_def{swr_direction_def}/set_size_{set_size}.jpg"
    fig.supxyt(
        f"Radian (vSWR def. {swr_direction_def}) (similar <---> dissimilar)",
        "KDE density",
        spath,
    )
    mngs.io.save(fig, spath)


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
        line_width=3,
    )
    for set_size in ["all"] + CONFIG.SET_SIZES:
        main(swr_direction_def=1, set_size=set_size)
        main(swr_direction_def=2, set_size=set_size)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
