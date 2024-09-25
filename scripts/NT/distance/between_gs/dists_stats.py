#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-26 08:06:49 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_stats.py


"""This script does XYZ."""


"""Imports"""
import sys
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from itertools import combinations, product

import scipy.stats as stats
from scipy.stats import rankdata

mngs.pd.ignore_SettingWithCopyWarning()

"""Functions & Classes"""


def perform_pairwise_statistical_test(df):

    results = []
    for col1, col2 in product(df.group.unique(), df.group.unique()):

        x1 = df.loc[df.group == col1, "dist"]
        x2 = df.loc[df.group == col2, "dist"]

        if np.all(np.array(x1) == np.array(x2)):
            statistic, p_value = np.nan, 1.0
        else:
            statistic, p_value = stats.wilcoxon(x1, x2)

        result = {
            "col1": col1,
            "col2": col2,
            "n1": len(x1),
            "n2": len(x2),
            "statistic": statistic,
            "p_val_unc": p_value,
        }

        results.append(pd.Series(result))

    results = pd.DataFrame(results)
    results["p_val"] = (results["p_val_unc"] * len(results)).clip(upper=1.0)
    results["statistic"] = results["statistic"]
    results["p_val_unc"] = results["p_val_unc"].round(3)
    results["p_val"] = results["p_val"].round(3)

    return results


def plot_kde(df):
    fig, ax = mngs.plt.subplots()
    ax.sns_kdeplot(
        data=df,

        x="dist",
        hue="group",
        xlim=(df.dist.min(), df.dist.max()),
        id="_".join(phases_to_plot),
    )
    ax.legend()
    return fig


def plot_heatmap(stats, z):
    vmin = 0 if z == "p_val" else np.nanmin(stats[z])
    vmax = 1 if z == "p_val" else np.nanmax(stats[z])
    cmap = "viridis_r" if z == "p_val" else "viridis"
    hm = mngs.pd.from_xyz(stats, x="col1", y="col2", z=z)
    fig, ax = mngs.plt.subplots()
    ax.imshow2d(hm, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.rotate_labels()
    ax.set_xyt(None, None, f"{z}")
    ax.set_ticks(
        xvals="auto",
        xticks=hm.columns,
        yvals="auto",
        yticks=hm.index,
    )
    fig.tight_layout()
    return fig


def main(phases_to_plot):
    """
    phases_to_plot = ["Encoding", "Retrieval"]
    """

    # Loading
    lpath = mngs.io.glob(
        f"./scripts/NT/distance/between_gs/to_rank_dists/{'_'.join(phases_to_plot)}/dist_ca1.csv",
        ensure_one=True,
    )[0]
    df = mngs.io.load(lpath)

    # Verify balanced data
    df["n"] = 1
    df.groupby("group").agg({"n": "sum", "dist": ["mean", "min", "max"]})

    # KDE plot
    fig = plot_kde(df)

    # Statistical test
    stats = perform_pairwise_statistical_test(df)
    stats_in = perform_pairwise_statistical_test(df[df.match == 1])
    stats_out = perform_pairwise_statistical_test(df[df.match == 2])

    # Saving
    sdir = f"./{'_'.join(phases_to_plot)}/"
    mngs.io.save(fig, sdir + "kde.jpg")
    mngs.io.save(stats, sdir + "stats.csv")
    mngs.io.save(stats_in, sdir + "stats_in.csv")
    mngs.io.save(stats_out, sdir + "stats_out.csv")
    for metric in ["statistic", "p_val"]:
        for obj, spath in [
            (plot_heatmap(stats, metric), sdir + f"{metric}_heatmap.jpg"),
            (
                plot_heatmap(stats_in, metric),
                sdir + f"{metric}_heatmap_in.jpg",
            ),
            (
                plot_heatmap(stats_out, metric),
                sdir + f"{metric}_heatmap_out.jpg",
            ),
        ]:
            mngs.io.save(obj, spath)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, line_width=1.0, np=np, agg=True
    )
    for phases_to_plot in [
        ["Encoding", "Retrieval"],
        ["Fixation", "Encoding", "Maintenance", "Retrieval"],
    ]:
        main(phases_to_plot)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
