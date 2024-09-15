#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 10:16:38 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import os
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts import utils
from scripts.ripple.NT.distance.from_O_lineplot import calc_dist_by_condi

"""Functions & Classes"""


def main():
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    # Plotting
    fig, axes = mngs.plt.subplots(ncols=1, nrows=3, sharex=True, sharey=False)

    for i_swr_type, swr_type in enumerate(CONFIG.RIPPLE.TYPES):
        i_ax = i_swr_type
        ax = axes.flat[i_ax]
        ax.set_xyt(None, None, f"{swr_type}")
        ax.set_ylim(5 * 1e-4, 15 * 1e-4)

        for phase in CONFIG.PHASES.keys():
            if not phase in ["Encoding", "Retrieval"]:
                continue
            condi = {
                "swr_type": swr_type,
                "phase": phase,
            }
            _df = mngs.pd.slice(swr_all, condi)["cosine_with_vER"]
            _df = _df[~_df.isna()]
            ax.kde(
                _df,
                id=f"{phase}-{swr_type}",
                color=CONFIG.COLORS[phase],
                xlim=(-1, 1),
            )
            ax.legend()

    df = axes.to_sigma()

    for phase in CONFIG.PHASES.keys():
        if not phase in ["Encoding", "Retrieval"]:
            continue
        i_ax = -1
        ax = axes.flat[i_ax]
        ax.set_xyt(None, None, f"Diff (SWR+ - SWR-)")
        x_p = df[mngs.gen.search(rf"{phase}-SWR\+_kde_x", df.columns)[1]]
        kde_p = df[mngs.gen.search(rf"{phase}-SWR\+_kde_kde", df.columns)[1]]
        x_m = df[mngs.gen.search(rf"{phase}-SWR\-_kde_x", df.columns)[1]]
        kde_m = df[mngs.gen.search(rf"{phase}-SWR\-_kde_kde", df.columns)[1]]

        assert np.array((np.array(x_p) == np.array(x_m))).all()

        kde_diff = np.array(kde_p) - np.array(kde_m)

        ax.plot(
            x_p,
            kde_diff,
            # label=f"{match_str}-{phase}-diff (SWR+ - SWR-)",
            # label=f"Diff (SWR+ - SWR-)",
            id=f"{phase}-diff (SWR+ - SWR-)",
            color=CONFIG.COLORS[phase],
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-5 * 1e-4, 5 * 1e-4)

    for ax in axes.flat:
        ax.legend()
    fig.supxyt("Cosine similarity", "KDE density")
    mngs.io.save(fig, "kde.jpg")


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
        font_size_legend=6,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
