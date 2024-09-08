#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-08 17:21:42 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/figures/01.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys
import warnings
from glob import glob
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
from scipy.stats import gaussian_kde
# sys.path = ["."] + sys.path
from scripts import utils  # , load
from tqdm import tqdm

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""

# def aggregate_trials_info():
#     trials_info_paths = mngs.gen.glob(CONFIG.PATH.TRIALS_INFO)
#     trials_dataframes = []
#     for trials_info_path in trials_info_paths:
#         parsed_info = utils.parse_lpath(trials_info_path)
#         trials_df = mngs.io.load(trials_info_path)
#         for key, value in parsed_info.items():
#             trials_df[key] = value
#         trials_dataframes.append(trials_df)
#     combined_trials_df = pd.concat(trials_dataframes)

#     subject_set_size_accuracy = combined_trials_df.groupby(["sub", "set_size"]).agg({
#         "correct": [np.mean]
#     })

#     set_size_accuracy_stats = (combined_trials_df.groupby(['sub', 'set_size'])['correct']
#                                .mean()
#                                .groupby('set_size')
#                                .agg(["mean", "std"]))

#     return subject_set_size_accuracy, set_size_accuracy_stats


def _load_data(correct_or_response_time):
    LPATHS = mngs.gen.glob(CONFIG.PATH.TRIALS_INFO)
    dfs = []
    for lpath in LPATHS:
        parsed = utils.parse_lpath(lpath)

        if int(parsed["sub"]) > CONFIG.SESSION.THRES:
            continue

        df = mngs.io.load(lpath)
        for k, v in parsed.items():
            df[k] = v
        dfs.append(df)
    df = pd.concat(dfs)
    df = (
        df.groupby(["sub", "set_size"])[correct_or_response_time]
        .mean()
        .groupby("set_size")
        .agg(["mean", "std"])
    ).reset_index()
    return df


def A():
    df = _load_data("correct")
    fig, ax = mngs.plt.subplots()
    ax.bar(
        df["set_size"],
        df["mean"],
        yerr=df["std"],
    )
    ax.set_xyt("Set size", "Correct rate", None)


def B():
    df = _load_data("response_time")

    fig, ax = mngs.plt.subplots()
    ax.bar(
        df["set_size"],
        df["mean"],
        yerr=df["std"],
    )
    ax.set_xyt("Set size", "Response time [s]", None)


def C():
    LPATHS = mngs.gen.glob(CONFIG.PATH.NT_GS)
    LPATHS = mngs.gen.search(CONFIG.ROI.MTL.HIP, LPATHS)[1]
    LPATHS = mngs.gen.search(["Session_01", "Session_02"], LPATHS)[1]


def main():
    fig_A = A()
    fig_B = B()
    plt.show()


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        # agg=True,
        font_size_axis_label=6,
        font_size_title=6,
        alpha=0.5,
        dpi_display=100,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
