#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-16 19:36:24 (ywatanabe)"
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


def plot_first_two_rows(dfs, fig, axes, _fig, _axes, MATCHES, SWR_TYPES):
    for col, match in enumerate(MATCHES):
        df = dfs[match]

        match_str = CONFIG.MATCHES_STR[str(match)]

        for i_swr_type, swr_type in enumerate(SWR_TYPES[:-1]):
            ax = axes[i_swr_type, col]
            _ax = _axes[i_swr_type, col]
            ax.set_title(f"{match_str} - {swr_type}")
            ax.set_xlim(*XLIM[cosine_or_radian])

            for i_comparison, comparison in enumerate(COMPARISONS):
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
                    id=f"{match_str}-{comparison}-{set_size}-{swr_type}",
                    color=CC[CONFIG.COLORS[comparison]],
                    xlim=XLIM[cosine_or_radian],
                )

                _ax.boxplot(
                    data,
                    label=f"{comparison}",
                    id=f"{match_str}-{comparison}-{set_size}-{swr_type}",
                    positions=[i_comparison],
                    # c=CC[CONFIG.COLORS[comparison]],
                )
            ax.legend()
            _ax.set_xyt(COMPARISONS, None, None)
    return fig, axes, _fig, _axes


def plot_last_row(dfs, fig, axes, _fig, _axes, MATCHES, SWR_TYPES):
    # Difference plot
    plotted = axes.to_sigma()

    for col, match in enumerate(MATCHES):
        df = dfs[match]
        match_str = CONFIG.MATCHES_STR[str(match)]

        ax = axes[2, col]
        _ax = _axes[2, col]

        swr_type = SWR_TYPES[-1]
        ax.set_title(f"{match_str} - {swr_type}")

        for i_comparison, comparison in enumerate(COMPARISONS):
            x_p = plotted[
                mngs.gen.search(
                    rf"{match_str}-{comparison}-{set_size}-{SWR_TYPES[0]}_kde_x",
                    plotted.columns,
                )[1]
            ]
            kde_p = plotted[
                mngs.gen.search(
                    rf"{match_str}-{comparison}-{set_size}-{SWR_TYPES[0]}_kde_kde",
                    plotted.columns,
                )[1]
            ]
            x_m = plotted[
                mngs.gen.search(
                    rf"{match_str}-{comparison}-{set_size}-{SWR_TYPES[1]}_kde_x",
                    plotted.columns,
                )[1]
            ]
            kde_m = plotted[
                mngs.gen.search(
                    rf"{match_str}-{comparison}-{set_size}-{SWR_TYPES[1]}_kde_kde",
                    plotted.columns,
                )[1]
            ]
            kde_diff = np.array(kde_p) - np.array(kde_m)

            ax.plot(
                np.hstack(np.array(x_p)),
                np.hstack(kde_diff),
                label=f"{comparison}",
                id=f"{match_str}-{comparison}-{set_size}-{SWR_TYPES[2]}",
                color=CC[CONFIG.COLORS[f"{comparison}"]],
            )

            _ax.boxplot(
                np.hstack(kde_diff),
                label=f"{comparison}",
                id=f"{match_str}-{comparison}-{set_size}-{SWR_TYPES[2]}",
                positions=[i_comparison],
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(*XLIM[cosine_or_radian])
        ax.set_ylim(*YLIM[cosine_or_radian])
        ax.legend()
        _ax.set_xyt(COMPARISONS, None, None)

    return fig, axes, _fig, _axes


def main(
    SWR_direction_def=1, set_size=4, control=False, cosine_or_radian="cosine"
):
    MATCHES = ["all"] + CONFIG.MATCHES
    SWR_TYPES = CONFIG.RIPPLE.TYPES + ["Diff (SWR+ - SWR-)"]

    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    calc_fn = {
        "cosine": calc_cosine,
        "radian": calc_radian,
    }[cosine_or_radian]

    dfs = {
        match: process_data(
            swr_all, match, set_size, SWR_direction_def, control, calc_fn
        )
        for match in MATCHES
    }

    # Plotting
    fig, axes = mngs.plt.subplots(
        ncols=len(MATCHES),
        nrows=len(SWR_TYPES),
        figsize=(15, 15),
        sharex=True,
        sharey=True,
    )

    # For referencing raw data afterwards
    _fig, _axes = mngs.plt.subplots(
        ncols=len(MATCHES),
        nrows=len(SWR_TYPES),
        sharex=True,
        sharey=True,
    )

    # Storing KDE data as well
    fig, axes, _fig, _axes = plot_first_two_rows(
        dfs, fig, axes, _fig, _axes, MATCHES, SWR_TYPES
    )
    # Takes difference between the KDE data
    fig, axes, _fig, _axes = plot_first_two_rows(
        dfs, fig, axes, _fig, _axes, MATCHES, SWR_TYPES
    )

    # Saving
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

    # For referencing raw data afterwards
    spath = spath.replace(".jpg", "_box.jpg")
    _fig.supxyt(None, xlabel, spath)
    mngs.io.save(_fig, spath, from_cwd=True)

    # Cleanup
    plt.close("all")


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
