#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 09:50:24 (ywatanabe)"
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


"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""
def roi2mtl(roi):
    for mtl, subregions in CONFIG.ROI.MTL.items():
        if roi in subregions:
            return mtl

def main():
    LPATHS_GS_TRIAL = mngs.gen.glob(CONFIG.PATH.NT_DIST_BETWEEN_GS_TRIAL)

    gs = []
    for lpath_gs in LPATHS_GS_TRIAL:
        parsed = utils.parse_lpath(lpath_gs)
        if parsed["session"] not in CONFIG.SESSION.FIRST_TWO:
            continue
        _gs = mngs.io.load(lpath_gs)
        _gs["MTL"] = roi2mtl(parsed["roi"])
        gs.append(_gs)
    gs = pd.concat(gs)

    df = mngs.pd.melt_cols(gs, [f"{p1[0]}{p2[0]}" for p1,p2 in combinations(CONFIG.PHASES.keys(), 2)])
    df = df.rename(columns={"variable": "phase_combination", "value": "distance"})

    fig, axes = mngs.plt.subplots(ncols=len(CONFIG.MATCHES_STR), nrows=len(CONFIG.ROI.MTL.keys()))
    for i_mtl, mtl in enumerate(CONFIG.ROI.MTL.keys()):
        for i_match, match in enumerate(CONFIG.MATCHES_STR.keys()):
            ax = axes[i_match, i_mtl]

            indi_MTL = df.MTL == mtl
            indi_match = np.full(len(df), True) if match == "all" else df.match == int(match)

            match_str = CONFIG.MATCHES_STR[match]

            ax.sns_boxplot(
                df[indi_MTL * indi_match],
                x="phase_combination",
                y="distance",
                hue="set_size",
                showfliers=False,
                id=mtl,
                )

            ax.set_xyt(None, None, f"{mtl}-{match_str}")

    mngs.io.save(fig, "box.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
