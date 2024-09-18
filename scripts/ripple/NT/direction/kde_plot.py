#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-19 09:10:39 (ywatanabe)"
# /ssh:ywatanabe@crest:/mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/kde_plot.py

"""
This script calculate and visualize distributions of radian/cosine of vectors. All plotted data are saved not only as jpg and but as csv (data values), due to the mngs package. Note that statistical tests are not performed in this script but in the dedicated script named stats.py, using the data collected from this script.

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
import logging

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
    # "eSWR_vs_eSWR",
    # "rSWR_vs_rSWR",
    "eSWR_vs_rSWR",
    "eSWR_vs_vER",
    "rSWR_vs_vER",
]


def calc_measure(v1, v2, measure, all_combinations):
    def calc_radian(v1, v2, all_combinations):
        result = 1 - cdist(v1, v2, metric="cosine")
        return np.arccos(result.reshape(-1) if all_combinations else result[:, 0])


    def calc_cosine(v1, v2, reshape):
        result = 1 - cdist(v1, v2, metric="cosine")
        return result.reshape(-1) if all_combinations else result[:, 0]

    if measure == "cosine":
        return calc_cosine(v1, v2, all_combinations)
    elif measure == "radian":
        return calc_radian(v1, v2, all_combinations)

def process_comparisons(
    swr_all, match, set_size, SWR_direction_def, measure
):
    data = {}
    for ca1 in CONFIG.ROI.CA1:
        df = mngs.pd.slice(swr_all.assign(sub=swr_all.subject), ca1)
        df = df[df.match == match] if match != "all" else df
        df = df[df.set_size == set_size] if set_size != "all" else df

        for swr_type in CONFIG.RIPPLE.TYPES:
            # vER, eSWR, rSWR
            vER = np.vstack(df["vER"])
            assert np.unique(vER, axis=0).shape[0] == 1
            vER = vER[0]
            df_eSWR = df[(df["swr_type"] == swr_type) & (df.phase == "Encoding")]
            df_rSWR = df[(df["swr_type"] == swr_type) & (df.phase == "Retrieval")]

            if df_eSWR.empty or df_rSWR.empty:
                for comparison in COMPARISONS:
                    key = f"{swr_type}_{comparison}"
                    data[key] = [np.nan]
            else:
                vs = {
                    "eSWR": np.vstack(df_eSWR[f"vSWR_def{SWR_direction_def}"]),
                    "rSWR": np.vstack(df_rSWR[f"vSWR_def{SWR_direction_def}"]),
                    "vER": vER[np.newaxis, :],
                }

                for comparison in COMPARISONS:
                    v1_str = comparison.split("_")[0]
                    v2_str = comparison.split("_")[-1]

                    v1 = vs[v1_str]
                    v2 = vs[v2_str]

                    all_combinations = False if v1_str == "vER" else True

                    key = f"{comparison}".replace("SWR", swr_type)
                    data[key] = calc_measure(v1, v2, measure, all_combinations)

    return mngs.pd.force_df(data)


def plot_first_two_rows(dfs, fig, axes, _fig, _axes, MATCHES, SWR_TYPES):
    for col, match in enumerate(MATCHES):
        df = dfs[match]

        match_str = CONFIG.MATCHES_STR[str(match)]

        for i_swr_type, swr_type in enumerate(SWR_TYPES[:-1]):
            i_ax = i_swr_type, col
            ax, _ax = axes[i_ax], _axes[i_ax]
            ax.set_title(f"{match_str} - {swr_type}")
            ax.set_xlim(*XLIM[measure])

            for i_comparison, comparison in enumerate(COMPARISONS):
                key = f"{comparison}".replace("SWR", swr_type)
                data = df[key]
                data = data.values.flatten()
                data = data[~np.isnan(data)]

                # print(len(data)) # Some kde is plotted in small samples

                try:
                    ax.kde(
                        data,
                        label=key,
                        id=f"{match_str}-{set_size}-{key}",
                        color=CC[CONFIG.COLORS[comparison]],
                        xlim=XLIM[measure],
                    )
                    _ax.boxplot_(
                        data,
                        label=key,
                        id=f"{match_str}-{set_size}-{key}",
                        positions=[i_comparison],
                        # c=CC[CONFIG.COLORS[comparison]],
                    )
                except Exception as e:
                    pass
                    # logging.warn(e)

            ax.legend()
            _ax.set_xyt(COMPARISONS, None, None)
    return fig, axes, _fig, _axes


def plot_last_row(dfs, fig, axes, _fig, _axes, MATCHES, SWR_TYPES):
    # Difference plot
    plotted = axes.to_sigma()

    for i_match, match in enumerate(MATCHES):
        df = dfs[match]
        match_str = CONFIG.MATCHES_STR[str(match)]

        # for _, swr_type in enumerate(SWR_TYPES[-1:]):
        i_swr_type, swr_type = len(SWR_TYPES)-1, SWR_TYPES[-1]
        i_ax = i_swr_type, i_match
        ax, _ax = axes[i_ax], _axes[i_ax]
        ax.set_title(f"{match_str} - {swr_type}")
        ax.set_xlim(*XLIM[measure])
        ax.set_title(f"{match_str} - {swr_type}")

        for i_comparison, comparison in enumerate(COMPARISONS):

            __import__("ipdb").set_trace()

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
        ax.set_xlim(*XLIM[measure])
        ax.set_ylim(*YLIM[measure])
        ax.legend()
        _ax.set_xyt(COMPARISONS, None, None)

    return fig, axes, _fig, _axes


def main(
    SWR_direction_def=1, set_size=4, measure="cosine"
):
    MATCHES = ["all"] + CONFIG.MATCHES
    SWR_TYPES = CONFIG.RIPPLE.TYPES + ["Diff (SWR+ - SWR-)"]

    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    dfs = {
        match: process_comparisons(
            swr_all, match, set_size, SWR_direction_def, measure
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
    spath = f"./kde_vSWR_def{SWR_direction_def}/{measure}/set_size_{set_size}.jpg"

    if measure == "cosine":
        xlabel = (
            f"{measure} (vSWR def. {SWR_direction_def}) (dissimilar <---> similar)",
        )
    else:
        xlabel = (
            f"{measure} (vSWR def. {SWR_direction_def}) (similar <---> dissimilar)",
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
    for measure in ["cosine", "radian"]:
        for set_size in ["all"] + CONFIG.SET_SIZES:
            for SWR_direction_def in [1, 2]:
                main(
                    SWR_direction_def=SWR_direction_def,
                    set_size=set_size,
                    measure=measure,
                )

    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
