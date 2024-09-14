#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 08:11:21 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import sys
import os
import random
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
    fig, axes = mngs.plt.subplots(
        ncols=len(CONFIG.MATCHES), nrows=3, sharex=True, sharey=True
    )

    for i_match, match in enumerate(CONFIG.MATCHES):
        match_str = {1: "Match IN", 2: "Mismatch OUT"}[match]
        for i_swr_type, swr_type in enumerate(CONFIG.RIPPLE.TYPES):
            i_ax = i_match * len(CONFIG.RIPPLE.TYPES) + i_swr_type
            ax = axes.flat[i_ax]
            ax.set_xyt(None, None, match_str)

            for phase in CONFIG.PHASES.keys():
                condi = {
                    "match": match,
                    "swr_type": swr_type,
                    "phase": phase,
                }
                _df = mngs.pd.slice(swr_all, condi)["cosine_with_vER"]
                _df = _df[~_df.isna()]
                ax.kde(
                    _df,
                    label=phase,
                    color=CONFIG.COLORS[phase],
                )

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
        font_size_legend="auto",
        font_size_base="auto",
        font_size_title="auto",
        font_size_axis_label="auto",
        font_size_tick_label="auto",
        font_size_legend="auto",
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
