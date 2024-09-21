#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-21 10:32:30 (ywatanabe)"
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


def calc_measure(v1, v2, measure):
    def calc_cosine(v1, v2):
        result = 1 - cdist(v1, v2, metric="cosine")
        return result.reshape(-1)

    def calc_radian(v1, v2):
        result = 1 - cdist(v1, v2, metric="cosine")
        return np.arccos(result.reshape(-1))

    if measure == "cosine":
        return calc_cosine(v1, v2)
    elif measure == "radian":
        return calc_radian(v1, v2)

def get_eSWR_rSWR_vER(swr, swr_type, vSWR_def):
    swr = swr[swr.swr_type == swr_type].copy()

    # vER
    vER = np.vstack(swr["vER"])
    assert np.unique(vER, axis=0).shape[0] == 1
    vER = vER[0]

    # eSWR / rSWR
    eSWR = swr[swr.phase == "Encoding"]
    rSWR = swr[swr.phase == "Retrieval"]

    if eSWR.empty:
        return None

    if rSWR.empty:
        return None

    vectors = {
        "eSWR": np.vstack(eSWR[vSWR_def]),
        "rSWR": np.vstack(rSWR[vSWR_def]),
        "vER": vER[np.newaxis, :],
    }
    return vectors

def process_comparisons(
    swr_all, match, set_size, vSWR_def, measure
):
    swr = swr_all[swr_all.match == match] if match != "all" else swr_all
    swr = swr[swr.set_size == set_size] if set_size != "all" else swr

    data = mngs.gen.listed_dict()
    for ca1 in CONFIG.ROI.CA1:
        _ca1 = ca1.copy()
        _ca1["subject"] = _ca1["sub"]
        del _ca1["sub"]
        swr_ca1 = mngs.pd.slice(swr, _ca1)
        for swr_type in CONFIG.RIPPLE.TYPES:
            vectors = get_eSWR_rSWR_vER(swr_ca1, swr_type, vSWR_def)
            if vectors is not None:
                for comparison in COMPARISONS:
                    try:
                        v1_str, v2_str = comparison.split("_vs_")
                        v1, v2 = vectors[v1_str], vectors[v2_str]
                        key = comparison.replace("SWR", swr_type)
                        data[key].append(calc_measure(v1, v2, measure))
                    except:
                        __import__("ipdb").set_trace()
            else:
                data[key].append([np.nan])

    for k,v in data.items():
        data[k] = np.hstack(v)

    df = mngs.pd.force_df(data)
    return df


def plot_first_two_rows(dfs, fig, axes, _fig, _axes, MATCHES, set_size, SWR_TYPES):

    for col, match in enumerate(MATCHES):
        df = dfs[match]
        match_str = CONFIG.MATCHES_STR[str(match)]

        for i_swr_type, swr_type in enumerate(SWR_TYPES[:-1]):
            i_ax = i_swr_type, col
            ax, _ax = axes[i_ax], _axes[i_ax]
            ax.set_title(f"{match_str} - {swr_type}")
            ax.set_xlim(*XLIM[measure])

            for i_comparison, comparison in enumerate(COMPARISONS):
                key = comparison.replace("SWR", swr_type)
                data = np.array(df[key])
                data = data[~np.isnan(data)]

                try:
                    ax.kde(
                        data,
                        label=key,
                        id=f"{match_str}-{set_size}-{key}",
                        color=CC[CONFIG.COLORS[comparison]],
                        xlim=XLIM[measure],
                    )
                    _ax.boxplot(
                        data,
                        label=key,
                        id=f"{match_str}-{set_size}-{key}",
                        positions=[i_comparison],
                    )
                except Exception as e:
                    print(e)


            ax.legend()
            _ax.set_xyt(COMPARISONS, None, None)
    return fig, axes, _fig, _axes


def calc_kde_diff(plotted, match_str, comparison, set_size, SWR_TYPES):
    x_p = plotted[
        mngs.gen.search(
            rf"{match_str}-{set_size}-{comparison.replace('SWR', 'SWR\+')}_kde_x",
            plotted.columns,
        )[1]
    ]
    kde_p = plotted[
        mngs.gen.search(
            rf"{match_str}-{set_size}-{comparison.replace('SWR', 'SWR\+')}_kde_kde",
            plotted.columns,
        )[1]
    ]
    x_m = plotted[
        mngs.gen.search(
            rf"{match_str}-{set_size}-{comparison.replace('SWR', 'SWR\-')}_kde_x",
            plotted.columns,
        )[1]
    ]
    kde_m = plotted[
        mngs.gen.search(
            rf"{match_str}-{set_size}-{comparison.replace('SWR', 'SWR\-')}_kde_kde",
            plotted.columns,
        )[1]
    ]
    kde_diff = np.array(kde_p) - np.array(kde_m)
    assert (np.array(x_p) == np.array(x_m)).all()
    return x_p, kde_diff

def plot_kde_diff(dfs, fig, axes, _fig, _axes, MATCHES, set_size, SWR_TYPES):
    # Difference plot
    plotted = axes.to_sigma()

    for i_match, match in enumerate(MATCHES):
        df = dfs[match]
        match_str = CONFIG.MATCHES_STR[str(match)]

        i_swr_type, swr_type = len(SWR_TYPES)-1, SWR_TYPES[-1]
        i_ax = i_swr_type, i_match
        ax, _ax = axes[i_ax], _axes[i_ax]
        ax.set_title(f"{match_str} - {swr_type}")
        ax.set_xlim(*XLIM[measure])
        ax.set_title(f"{match_str} - {swr_type}")

        for i_comparison, comparison in enumerate(COMPARISONS):
            xx, kde_diff = calc_kde_diff(plotted, match_str, comparison, set_size, SWR_TYPES)

            if kde_diff.any():
                n = (~np.isnan(xx)).sum()
                ax.plot_(
                    np.hstack(np.array(xx)),
                    np.hstack(kde_diff),
                    label=f"{comparison} (n={n})",
                    id=f"{match_str}-{comparison}-{set_size}-{SWR_TYPES[2]}",
                    color=CC[CONFIG.COLORS[f"{comparison}"]],
                    n=n,
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
    vSWR_def="vSWR_NT", set_size=4, measure="cosine"
):
    MATCHES = ["all"] + CONFIG.MATCHES
    SWR_TYPES = CONFIG.RIPPLE.TYPES + ["Diff (SWR+ - SWR-)"]

    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    dfs = {
        match: process_comparisons(
            swr_all, match, set_size, vSWR_def, measure
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
        dfs, fig, axes, _fig, _axes, MATCHES, set_size, SWR_TYPES
    )
    # Takes difference between the KDE data
    fig, axes, _fig, _axes = plot_kde_diff(
        dfs, fig, axes, _fig, _axes, MATCHES, set_size, SWR_TYPES
    )

    # Saving
    spath = f"./kde_{vSWR_def}/{measure}/set_size_{set_size}.jpg"

    left, right = ("dissimilar", "similar") if measure == "cosine" else ("similar", "dissimilar")
    xlabel = f"{measure} ({vSWR_def}) ({left} <---> {right})"
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
            for vSWR_def in CONFIG.RIPPLE.DIRECTIONS:
                main(
                    vSWR_def=vSWR_def,
                    set_size=set_size,
                    measure=measure,
                )

    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
