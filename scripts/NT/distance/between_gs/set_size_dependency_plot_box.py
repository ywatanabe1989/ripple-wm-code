#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 21:24:11 (ywatanabe)"
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

def load_dists():
    LPATHS_DIST_BETWEEN_GS_MATCH_SET_SIZE = mngs.gen.glob(CONFIG.PATH.NT_DIST_BETWEEN_GS_MATCH_SET_SIZE)
    dists = []
    for lpath_dists in LPATHS_DIST_BETWEEN_GS_MATCH_SET_SIZE:
        parsed = utils.parse_lpath(lpath_dists)
        if parsed["session"] not in CONFIG.SESSION.FIRST_TWO:
            continue
        _dists = mngs.io.load(lpath_dists)
        _dists["MTL"] = roi2mtl(parsed["roi"])
        for k,v in parsed.items():
            _dists[k] = v
        dists.append(_dists)
    dists = pd.concat(dists)
    return dists

def add_match_all(df):
    df_match_all = deepcopy(df)
    df_match_all["match"] = -1
    df = pd.concat([df, df_match_all])
    match_mapper = deepcopy(CONFIG.MATCHES_STR)
    match_mapper["-1"] = match_mapper.pop("all")
    df["match"] = df["match"].astype(str).replace(match_mapper)
    return df

def main():
    # Loading
    df = load_dists()
    df = add_match_all(df)

    # Plotting
    fig, axes = mngs.plt.subplots(ncols=len(CONFIG.MATCHES_STR), nrows=len(CONFIG.ROI.MTL.keys()))
    for i_mtl, mtl in enumerate(CONFIG.ROI.MTL.keys()):
        for i_match, match_str in enumerate(CONFIG.MATCHES_STR.values()):
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
