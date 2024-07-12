#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-12 00:52:52 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/utils/_load_NTdf.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import mngs

importlib.reload(mngs)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from glob import glob
from pprint import pprint
import warnings
import logging
from tqdm import tqdm
import xarray as xr
import utils

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
CONFIG = mngs.gen.load_configs()
CC = mngs.plt.PARAMS["RGBA_NORM"]


"""
Functions & Classes
"""


def load_NTdf(lpath_NT, sample_type, znorm=False, symlog=False, unbias=False):
    # Loading
    NT = mngs.io.load(lpath_NT)

    if znorm:
        NT = mngs.gen.to_z(NT, axis=(0, 2))

    if unbias:
        NT -= NT.min(axis=1, keepdims=True) + 1e-5

    if symlog:
        NT = mngs.gen.symlog(NT, 1e-5)

    parsed = utils.parse_lpath(lpath_NT)

    trials_info = mngs.io.load(
        mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
    ).set_index("trial_number")
    trials_info.index = trials_info.index.astype(int)

    df = _add_meta_from_trials_info(NT, trials_info)

    if sample_type == "SWR+":
        ripple = mngs.io.load(mngs.gen.replace(CONFIG.PATH.RIPPLE, parsed))
        df = _add_ripple_tag(df, ripple)
    elif sample_type == "SWR-":
        ripple = mngs.io.load(
            mngs.gen.replace(CONFIG.PATH.RIPPLE_MINUS, parsed)
        )
        df = _add_ripple_tag(df, ripple)

    df = mngs.pd.merge_columns(df, *["phase", "match", "set_size"])

    # Color
    df = _add_color(df)

    return df


def _add_meta_from_trials_info(NT, trials_info):
    dfs = []

    # Conditonal data
    conditions = list(
        mngs.gen.yield_grids({"match": [1, 2], "set_size": [4, 6, 8]})
    )

    for i_cc, cc in enumerate(conditions):
        indi = mngs.pd.find_indi(trials_info, cc)
        indi = np.array(indi[indi].index) - 1

        NTc = NT[indi]
        indi_bin = np.arange(NTc.shape[-1])[
            np.newaxis, np.newaxis
        ] * np.ones_like(NTc).astype(int)

        for phase_str in CONFIG.PHASES:
            NTcp = NTc[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]
            indi_trial_cp = (
                indi.reshape(-1, 1, 1) * np.ones_like(NTcp)
            ).astype(int)

            indi_bin_cp = indi_bin[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]

            _df = pd.DataFrame(
                {
                    f"factor_{kk+1}": NTcp[:, kk, :].reshape(-1)
                    for kk in range(NTcp.shape[1])
                }
            )
            _df["i_trial"] = indi_trial_cp[:, 0, :].reshape(-1)
            _df["i_bin"] = indi_bin_cp[:, 0, :].reshape(-1)

            cc.update({"phase": phase_str})
            for k, v in cc.items():
                _df[k] = v
            dfs.append(_df)

    dfs = pd.concat(dfs)

    return dfs


def _add_ripple_tag(df, ripple):
    ripple["i_trial"] = ripple.index - 1
    ripple.set_index("i_trial", inplace=True)

    # bin
    ripple["start_bin"] = np.floor(
        ripple.start_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)
    ripple["end_bin"] = np.ceil(
        ripple.end_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)
    ripple["peak_bin"] = np.array(
        ripple.peak_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)

    # labeling
    df = df.sort_values(["i_trial", "i_bin"])
    df["within_ripple"] = False
    for i_rip, rip in ripple.iterrows():
        indi_trial = df.i_trial == rip.name
        # indi_bin = df.i_bin == rip.peak_bin
        indi_bin = (rip.peak_bin - 2 <= df.i_bin) * (
            df.i_bin <= rip.peak_bin + 2
        )
        indi = indi_trial * indi_bin

        assert 0 < indi.sum()

        df.loc[indi, "within_ripple"] = True

    return df


def define_color(phase, set_size):
    n_set_sizes = len(CONFIG.SET_SIZES)

    # Base Color
    base_color = np.array(CC[CONFIG.PHASES[phase].color])
    # Gradient color
    n_split = n_set_sizes * 4
    i_color = {4: n_split - 1, 6: n_split // 2, 8: 0}[set_size]
    color = mngs.plt.gradiate_color(base_color, n=n_split)[i_color]
    return color


def _add_color(df):
    def __add_color(row):
        return define_color(row.phase, row.set_size)

    df["color"] = df.apply(__add_color, axis=1)
    return df


def main():
    lpath_NT = mngs.gen.natglob(CONFIG.PATH.NT)[0]
    df = load_NTdf(lpath_NT, sample_type="all")
    print(df)


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
