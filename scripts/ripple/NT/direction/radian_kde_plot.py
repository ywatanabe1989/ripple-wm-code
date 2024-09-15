#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 21:02:31 (ywatanabe)"
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
YLIM = (-3e-4, 20e-4)
COMPARISONS = [
    "eSWR_vs_eSWR",
    "rSWR_vs_rSWR",
    "eSWR_vs_rSWR",
    "eSWR_vs_vER",
    "rSWR_vs_vER",
]


def calculate_radian(v1, v2, reshape=True):
    result = 1 - cdist(v1, v2, metric="cosine")
    return np.arccos(result.reshape(-1) if reshape else result[:, 0])


def process_data(swr_all, match, set_size, SWR_direction_def, control):
    radian = {}
    for ca1 in CONFIG.ROI.CA1:
        df = mngs.pd.slice(swr_all.assign(sub=swr_all.subject), ca1)
        df = df[df.match == match] if match != "all" else df
        df = df[df.set_size == set_size] if set_size != "all" else df

        for swr_type in CONFIG.RIPPLE.TYPES:
            df_E = df[(df["swr_type"] == swr_type) & (df.phase == "Encoding")]
            df_R = df[(df["swr_type"] == swr_type) & (df.phase == "Retrieval")]

            if df_E.empty or df_R.empty:
                for comparison in [
                    "eSWR_vs_rSWR",
                    "eSWR_vs_vER",
                    "rSWR_vs_vER",
                ]:
                    radian[f"{ca1.values()}_{swr_type}_{comparison}"] = [
                        np.nan
                    ]
            else:
                v_eSWR = np.vstack(df_E[f"vSWR_def{SWR_direction_def}"])
                v_rSWR = np.vstack(df_R[f"vSWR_def{SWR_direction_def}"])
                v_ER = np.vstack(df_R["vER"])

                def control_vector(v):
                    return np.where(
                        np.isnan(v), np.nan, np.random.randn(*v.shape)
                    )

                # def control_vector(v):
                #     mask = ~np.isnan(v)
                #     resampled = np.random.choice(
                #         v[mask].flatten(), size=v.shape, replace=True
                #     )
                #     return np.where(mask, resampled, np.nan)

                v_eSWR_c, v_rSWR_c, v_ER_c = map(
                    control_vector, [v_eSWR, v_rSWR, v_ER]
                )

                comparisons = [
                    ("eSWR_vs_rSWR", v_eSWR, v_rSWR, v_eSWR_c, v_rSWR_c, True),
                    ("eSWR_vs_vER", v_eSWR, v_ER, v_eSWR_c, v_ER_c, False),
                    ("rSWR_vs_vER", v_rSWR, v_ER, v_rSWR_c, v_ER_c, False),
                    ("eSWR_vs_eSWR", v_eSWR, v_eSWR, v_eSWR_c, v_eSWR_c, True),
                    ("rSWR_vs_rSWR", v_rSWR, v_rSWR, v_rSWR_c, v_rSWR_c, True),
                ]

                for name, v1, v2, v1_c, v2_c, reshape in comparisons:
                    key = f"{ca1.values()}_{swr_type}_{name}"
                    radian[key] = calculate_radian(
                        v1_c if control else v1,
                        v2_c if control else v2,
                        reshape,
                    )

    return mngs.pd.force_df(radian)


# def process_data(swr_all, match, set_size, SWR_direction_def, control):
#     radian = {}
#     for ca1 in CONFIG.ROI.CA1:
#         swr_all["sub"] = swr_all["subject"]
#         df = mngs.pd.slice(swr_all, ca1)

#         if match != "all":
#             df = df[df.match == match]

#         if set_size != "all":
#             df = df[df.set_size == set_size]

#         for swr_type in CONFIG.RIPPLE.TYPES:
#             df_E = df[(df["swr_type"] == swr_type) & (df.phase == "Encoding")]
#             df_R = df[(df["swr_type"] == swr_type) & (df.phase == "Retrieval")]

#             if df_E.empty or df_R.empty:
#                 radian[f"{ca1.values()}_{swr_type}_eSWR_vs_rSWR"] = [np.nan]
#                 radian[f"{ca1.values()}_{swr_type}_eSWR_vs_vER"] = [np.nan]
#                 radian[f"{ca1.values()}_{swr_type}_rSWR_vs_vER"] = [np.nan]

#             else:
#                 v_eSWR = np.vstack(df_E[f"vSWR_def{SWR_direction_def}"])
#                 v_rSWR = np.vstack(df_R[f"vSWR_def{SWR_direction_def}"])
#                 v_ER = np.vstack(df_R["vER"])

#                 v_eSWR_control = np.where(
#                     np.isnan(v_eSWR), np.nan, np.random.randn(*v_eSWR.shape)
#                 )
#                 v_rSWR_control = np.where(
#                     np.isnan(v_rSWR), np.nan, np.random.randn(*v_rSWR.shape)
#                 )
#                 v_ER_control = np.where(
#                     np.isnan(v_ER), np.nan, np.random.randn(*v_ER.shape)
#                 )

#                 if not control:
#                     radian[f"{ca1.values()}_{swr_type}_eSWR_vs_rSWR"] = (
#                         np.arccos(
#                             1
#                             - cdist(v_eSWR, v_rSWR, metric="cosine").reshape(
#                                 -1
#                             )
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_eSWR_vs_vER"] = (
#                         np.arccos(
#                             1 - cdist(v_eSWR, v_ER, metric="cosine")[:, 0]
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_rSWR_vs_vER"] = (
#                         np.arccos(
#                             1 - cdist(v_rSWR, v_ER, metric="cosine")[:, 0]
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_eSWR_vs_eSWR"] = (
#                         np.arccos(
#                             1
#                             - cdist(v_eSWR, v_eSWR, metric="cosine").reshape(
#                                 -1
#                             )
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_rSWR_vs_rSWR"] = (
#                         np.arccos(
#                             1
#                             - cdist(v_rSWR, v_rSWR, metric="cosine").reshape(
#                                 -1
#                             )
#                         )
#                     )
#                 else:
#                     radian[f"{ca1.values()}_{swr_type}_eSWR_vs_rSWR"] = (
#                         np.arccos(
#                             1
#                             - cdist(
#                                 v_eSWR, v_rSWR_control, metric="cosine"
#                             ).reshape(-1)
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_eSWR_vs_vER"] = (
#                         np.arccos(
#                             1
#                             - cdist(v_eSWR, v_ER_control, metric="cosine")[
#                                 :, 0
#                             ]
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_rSWR_vs_vER"] = (
#                         np.arccos(
#                             1
#                             - cdist(v_rSWR, v_ER_control, metric="cosine")[
#                                 :, 0
#                             ]
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_eSWR_vs_eSWR"] = (
#                         np.arccos(
#                             1
#                             - cdist(
#                                 v_eSWR, v_eSWR_control, metric="cosine"
#                             ).reshape(-1)
#                         )
#                     )
#                     radian[f"{ca1.values()}_{swr_type}_rSWR_vs_rSWR"] = (
#                         np.arccos(
#                             1
#                             - cdist(
#                                 v_rSWR, v_rSWR_control, metric="cosine"
#                             ).reshape(-1)
#                         )
#                     )

#     df = mngs.pd.force_df(radian)
#     return df


def main(SWR_direction_def=1, set_size=4, control=False):
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    df_all = process_data(swr_all, "all", set_size, SWR_direction_def, control)
    df_in = process_data(swr_all, 1, set_size, SWR_direction_def, control)
    df_out = process_data(swr_all, 2, set_size, SWR_direction_def, control)

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
                ax.kde(
                    data,
                    label=f"{comparison}",
                    id=f"{title}-{comparison}-{set_size}-{swr_type}",
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

        for comparison in COMPARISONS:
            x_p = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\+_kde_x",
                    plotted.columns,
                )[1]
            ]
            kde_p = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\+_kde_kde",
                    plotted.columns,
                )[1]
            ]
            x_m = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\-_kde_x",
                    plotted.columns,
                )[1]
            ]
            kde_m = plotted[
                mngs.gen.search(
                    rf"{title}-{comparison}-{set_size}-SWR\-_kde_kde",
                    plotted.columns,
                )[1]
            ]
            kde_diff = np.array(kde_p) - np.array(kde_m)
            ax.plot(
                x_p,
                kde_diff,
                label=f"{comparison}",
                id=f"{title}-{comparison}-{set_size}-diff (SWR+ - SWR-)",
                color=CC[CONFIG.COLORS[f"{comparison}"]],
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.legend()

    spath = f"./kde_vSWR_def{SWR_direction_def}/set_size_{set_size}.jpg"
    if control:
        spath = spath.replace(".jpg", "_control.jpg")
    fig.supxyt(
        f"Radian (vSWR def. {SWR_direction_def}) (similar <---> dissimilar)",
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
        # font_size_base=10,
        # font_size_title=10,
        # font_size_axis_label=9,
        # font_size_tick_label=9,
        # font_size_legend=8,
        line_width=3,
    )
    for set_size in ["all"] + CONFIG.SET_SIZES:
        main(SWR_direction_def=1, set_size=set_size, control=False)
        main(SWR_direction_def=2, set_size=set_size, control=False)
        main(SWR_direction_def=1, set_size=set_size, control=True)
        main(SWR_direction_def=2, set_size=set_size, control=True)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
