#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 20:43:40 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/from_O_of_MTL_regions.py

"""This script does XYZ."""


"""Imports"""
import itertools
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import norm
import utils
from copy import deepcopy

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def roi2mtl(roi):
    for mtl, subregions in CONFIG.ROI.MTL.items():
        if roi in subregions:
            return mtl


def load_dists_match_set_size():
    LPATHS_DIST_BETWEEN_GS_MATCH_SET_SIZE = mngs.gen.glob(
        CONFIG.PATH.NT_DIST_BETWEEN_GS_MATCH_SET_SIZE
    )
    dists = []
    for lpath_dists in LPATHS_DIST_BETWEEN_GS_MATCH_SET_SIZE:
        parsed = utils.parse_lpath(lpath_dists)
        if parsed["session"] not in CONFIG.SESSION.FIRST_TWO:
            continue
        _dists = mngs.io.load(lpath_dists)
        _dists["MTL"] = roi2mtl(parsed["roi"])
        for k, v in parsed.items():
            _dists[k] = v
        dists.append(_dists)
    dists = pd.concat(dists)
    return dists


def main():
    # Loading
    df = load_dists_match_set_size()
    df.match = df.match.replace(CONFIG.MATCHES_STR)

    # Plotting
    fig, axes = mngs.plt.subplots(
        ncols=len(CONFIG.MATCHES_STR), nrows=len(CONFIG.ROI.MTL.keys())
    )
    sorted_CONFIG_STR = list(CONFIG.MATCHES_STR.values())[-1:] + list(CONFIG.MATCHES_STR.values())[:-1]
    for i_mtl, mtl in enumerate(CONFIG.ROI.MTL.keys()):
        for i_match, match_str in enumerate(sorted_CONFIG_STR):
            ax = axes[i_match, i_mtl]

            indi_MTL = df.MTL == mtl
            indi_match = df.match == match_str

            ax.sns_boxplot(
                df[indi_MTL * indi_match],
                x="phase_combination",
                y="distance",
                hue="set_size",
                showfliers=False,
                id=mtl,
            )

            ax.set_xyt("Phase combination", "Distance [a.u.]", f"{mtl}-{match_str}")
            ax.get_legend().remove()

    fig.tight_layout()
    mngs.io.save(fig, "box.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        fig_scale=4,
        font_size_base=6,
        font_size_title=6,
        font_size_axis_label=6,
        font_size_tick_label=6,
        font_size_legend=6,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
