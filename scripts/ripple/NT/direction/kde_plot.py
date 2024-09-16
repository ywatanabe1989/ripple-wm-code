#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-16 10:41:37 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""
This script calculate and visualize distributions of radian/cosine of vectors. All plotted data are saved not only as jpg and but as csv (data values), due to the mngs package.

- Vectors
  - eSWR+ (SWR during Encoding)
  - eSWR- (Control SWR during Encoding)
  - rSWR+ (SWR during Retrieval phase)
  - rSWR- (Control SWR during Retrieval phase)
  - ER (vector of geometric medians of Encoding to Retrieval)

- Sternburg Task
  - set size
    - The number of alphabetical letters presented
  - Match IN task
    - When the probe letter is included in the letters to be encoded
  - Mismatch OUT task
    - When the probe letter is not included in the letters to be encoded
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
XLIM = {
    "radian": (0, np.pi),
    "cosine": (-1, 1),
}
YLIM = {
    "radian": (-3e-4, 20e-4),
    "cosine": (-4e-4, 20e-4),
}
COMPARISONS = [
    "eSWR_vs_eSWR",
    "rSWR_vs_rSWR",
    "eSWR_vs_rSWR",
    "eSWR_vs_vER",
    "rSWR_vs_vER",
]


def calc_radian(v1, v2, reshape=True):
    result = 1 - cdist(v1, v2, metric="cosine")
    return np.arccos(result.reshape(-1) if reshape else result[:, 0])


def calc_cosine(v1, v2, reshape=True):
    result = 1 - cdist(v1, v2, metric="cosine")
    return result.reshape(-1) if reshape else result[:, 0]


def process_data(
    swr_all, match, set_size, SWR_direction_def, control, calc_fn
):
    cosine_or_radian = {}
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
                    cosine_or_radian[
                        f"{ca1.values()}_{swr_type}_{comparison}"
                    ] = [np.nan]
            else:
                v_eSWR = np.vstack(df_E[f"vSWR_def{SWR_direction_def}"])
                v_rSWR = np.vstack(df_R[f"vSWR_def{SWR_direction_def}"])
                v_ER = np.vstack(df_R["vER"])

                def control_vector(v):
                    return np.where(
                        np.isnan(v), np.nan, np.random.randn(*v.shape)
                    )

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
                    cosine_or_radian[key] = calc_fn(
                        v1_c if control else v1,
                        v2_c if control else v2,
                        reshape,
                    )

    return mngs.pd.force_df(cosine_or_radian)


def main(
    SWR_direction_def=1, set_size=4, control=False, cosine_or_radian="cosine"
):
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    calc_fn = {
        "cosine": calc_cosine,
        "radian": calc_radian,
    }[cosine_or_radian]

    df_all = process_data(
        swr_all, "all", set_size, SWR_direction_def, control, calc_fn
    )
    df_in = process_data(
        swr_all, 1, set_size, SWR_direction_def, control, calc_fn
    )
    df_out = process_data(
        swr_all, 2, set_size, SWR_direction_def, control, calc_fn
    )

    fig, axes = mngs.plt.subplots(
        ncols=3, nrows=3, figsize=(15, 15), sharex=True, sharey=True
    )

    for col, (df, title) in enumerate(
        zip([df_all, df_in, df_out], ["All", "Match IN", "Mismatch OUT"])
    ):
        for i_swr_type, swr_type in enumerate(CONFIG.RIPPLE.TYPES):
            ax = axes[i_swr_type, col]
            ax.set_title(f"{title} - {swr_type}")
            ax.set_xlim(*XLIM[cosine_or_radian])

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
                    xlim=XLIM[cosine_or_radian],
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
                np.hstack(np.array(x_p)),
                np.hstack(kde_diff),
                label=f"{comparison}",
                id=f"{title}-{comparison}-{set_size}-diff (SWR+ - SWR-)",
                color=CC[CONFIG.COLORS[f"{comparison}"]],
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(*XLIM[cosine_or_radian])
        ax.set_ylim(*YLIM[cosine_or_radian])
        ax.legend()

    spath = f"./kde_vSWR_def{SWR_direction_def}/{cosine_or_radian}/set_size_{set_size}.jpg"
    if control:
        spath = spath.replace(".jpg", "_control.jpg")

    if cosine_or_radian == "cosine":
        xlabel = (
            f"{cosine_or_radian} (vSWR def. {SWR_direction_def}) (dissimilar <---> similar)",
        )
    else:
        xlabel = (
            f"{cosine_or_radian} (vSWR def. {SWR_direction_def}) (similar <---> dissimilar)",
        )
    fig.supxyt(
        xlabel,
        "KDE density",
        spath,
    )
    mngs.io.save(fig, spath, from_cwd=True)


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
    for cosine_or_radian in ["cosine", "radian"]:
        for set_size in ["all"] + CONFIG.SET_SIZES:
            for control in [True, False]:
                for SWR_direction_def in [1, 2]:
                    main(
                        SWR_direction_def=SWR_direction_def,
                        set_size=set_size,
                        control=control,
                        cosine_or_radian=cosine_or_radian,
                    )

    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
