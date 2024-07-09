#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-10 00:22:07 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/define_SWR-.py


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
import mngs
import seaborn as sns

mngs.gen.reload(mngs)
import random
import warnings
from functools import partial
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from scripts.ripple.detect_SWR_p import add_firing_patterns, transfer_metadata
from scripts.utils import parse_lpath
from tqdm import tqdm

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


"""
Functions & Classes
"""


def add_rel_peak_pos(row, xxr, fs_r):
    trial_number = row.name
    i_trial = trial_number - 1
    _xxr = xxr[i_trial][int(row.start_s * fs_r) : int(row.end_s * fs_r)]
    return _xxr.argmax() / len(_xxr)


def add_peak_amp_sd(row, xxr, fs_r):
    trial_number = row.name
    i_trial = trial_number - 1
    _xxr = xxr[i_trial][int(row.start_s * fs_r) : int(row.end_s * fs_r)]
    return _xxr.max()


def main():
    LPATHS_RIPPLE = mngs.gen.natglob(CONFIG.PATH.RIPPLE)
    LPATHS_iEEG_RIPPLE_BAND = mngs.gen.natglob(CONFIG.PATH.iEEG_RIPPLE_BAND)

    for lpath_ripple, lpath_iEEG in zip(
        LPATHS_RIPPLE, LPATHS_iEEG_RIPPLE_BAND
    ):
        main_lpath(lpath_ripple, lpath_iEEG)


def main_lpath(lpath_ripple, lpath_iEEG):
    # Loading
    # SWR+
    df_p = mngs.io.load(lpath_ripple)
    (iEEG_ripple_band, fs_r) = (xxr, fs_r) = mngs.io.load(lpath_iEEG)

    # Parsing lpath
    parsed = parse_lpath(lpath_ripple)
    sub = parsed["sub"]
    session = parsed["session"]
    roi = parsed["roi"]

    # Trials info
    trials_info = mngs.io.load(
        mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
    )
    trials_info.set_index("trial_number", inplace=True)

    # Starts defining SWR- using SWR+, iEEG signal, and trials_info
    df_m = df_p[["start_s", "end_s", "duration_s"]].copy()

    # Shuffle ripple period (row) within a session as controls
    df_m = df_m.iloc[np.random.permutation(np.arange(len(df_m)))]

    # Override trial_number
    new_trial_numbers = pd.Series(
        np.random.choice(trials_info.index, len(df_m)).astype(int),
        name="trial_number",
    )

    assert trials_info.index.min() <= new_trial_numbers.min()
    assert new_trial_numbers.max() <= trials_info.index.max()

    df_m.index = new_trial_numbers
    df_m = df_m.sort_index()

    # Adds metadata for the control data
    df_m = transfer_metadata(df_m, trials_info)

    # rel_peak_pos
    df_m["rel_peak_pos"] = df_m.apply(
        partial(add_rel_peak_pos, xxr=xxr, fs_r=fs_r), axis=1
    )

    # peak_amp_sd
    df_m["peak_amp_sd"] = df_m.apply(
        partial(add_peak_amp_sd, xxr=xxr, fs_r=fs_r), axis=1
    )

    # subject
    df_m.loc[:, "subject"] = sub

    # session
    df_m.loc[:, "session"] = session

    # session
    df_m.loc[:, "roi"] = roi

    # Firing patterns
    df_m = add_firing_patterns(df_m)

    assert len(df_p) == len(df_m)

    # Saving
    spath = lpath_ripple.replace("SWR_p", "SWR_m")
    mngs.io.save(df_m, spath, from_cwd=True)


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, random=random, np=np
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
