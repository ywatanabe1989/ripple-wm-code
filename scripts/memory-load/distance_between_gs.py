#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-09 09:00:39 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/memory-load/distance_between_gs.py

"""This script does XYZ."""

"""Imports"""
import importlib
import logging
import os
import re
import sys
import warnings
from glob import glob
from itertools import combinations
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from scipy.spatial.distance import norm
from tqdm import tqdm

sys.path = ["."] + sys.path
try:
    from scripts import load, utils
except Exception as e:
    pass

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def load(mtl):
    LPATHS_GS = mngs.gen.glob(CONFIG.PATH.NT_GS_TRIAL)
    LPATHS_GS = mngs.gen.search(CONFIG.SESSION.FIRST_TWO, LPATHS_GS)[1]
    LPATHS_GS = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS_GS)[1]

    GS, TI = [], []
    for lpath_gs in LPATHS_GS:
        lpath_ti = mngs.gen.replace(
            CONFIG.PATH.TRIALS_INFO, utils.parse_lpath(lpath_gs)
        )
        GS.append(mngs.io.load(lpath_gs))
        TI.append(mngs.io.load(lpath_ti))

    GS = np.vstack(GS)
    TI = pd.concat(TI)
    mask = ~np.isnan(GS).any(axis=(1, 2))
    return GS[mask], TI[mask]


def calc_dist_between_gs(GS):
    dists = {}
    for i_p1, i_p2 in combinations(np.arange(len(CONFIG.PHASES.keys())), 2):
        p1 = list(CONFIG.PHASES.keys())[i_p1]
        p2 = list(CONFIG.PHASES.keys())[i_p2]
        v1 = GS[..., i_p1]
        v2 = GS[..., i_p2]
        mask = ~(np.isnan(v1).any(axis=-1) + np.isnan(v2).any(axis=-1))
        dists[f"{p1[0]}{p2[0]}"] = pd.Series(
            norm(v1[mask] - v2[mask], axis=-1)
        )
    return pd.concat(dists, axis=1)


def main_mtl(mtl):
    GS, TI = load(mtl)
    DS = calc_dist_between_gs(GS)

    # DS_reset = np.log10(DS.reset_index(drop=True))
    DS_reset = DS.reset_index(drop=True)
    TI_reset = TI.reset_index(drop=True)
    merged_df = pd.concat(
        [DS_reset, TI_reset[["correct", "response_time", "set_size"]]], axis=1
    )

    # Calculate correlations
    correlations = merged_df[
        ["FE", "FM", "FR", "EM", "ER", "MR", "correct", "response_time"]
    ].corrwith(merged_df["set_size"])


def main():
    main_mtl("HIP")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF


#    for lpath in LPATHS:
#         parsed = utils.parse_lpath(lpath)

#         if int(parsed["sub"]) > CONFIG.SESSION.THRES:
#             continue

#         df = mngs.io.load(lpath)
#         for k, v in parsed.items():
#             df[k] = v
#         dfs.append(df)
#     df = pd.concat(dfs)
#     df = (
#         df.groupby(["sub", "session", "set_size"])[correct_or_response_time]
#         .mean()
#         .groupby("set_size")
#         .agg(["mean", "std"])
#     ).reset_index()
#     return df


# def main():
#     df = _load_data("correct")
#     fig, ax = mngs.plt.subplots()
#     ax.bar(
#         df["set_size"],
#         df["mean"],
#         yerr=df["std"],
#     )
#     ax.set_xyt("Set size", "Correct rate", None)

#     mngs.io.save(fig, "./data/memory_load/correct_rate.jpg", from_cwd=True)
#     mngs.io.save(
#         fig.to_sigma(), "./data/memory_load/correct_rate.csv", from_cwd=True
#     )
#     return fig


# if __name__ == "__main__":
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys,
#         plt,
#         verbose=False,
#         agg=True,
#         font_size_axis_label=6,
#         font_size_title=6,
#         alpha=0.5,
#         dpi_display=100,
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)

# # EOF
