#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-29 18:05:04 (ywatanabe)"
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
import warnings
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


def main():
    ################################################################################
    ## Delete us
    ################################################################################
    LPATHS_RIPPLE = mngs.gen.natglob(CONFIG["PATH_RIPPLE"])
    LPATHS_iEEG_RIPPLE_BAND = mngs.gen.natglob(CONFIG["PATH_iEEG_RIPPLE_BAND"])

    lpath_ripple = LPATHS_RIPPLE[0]
    lpath_iEEG = LPATHS_iEEG_RIPPLE_BAND[0]
    ################################################################################

    # Loading
    # SWR+
    df_p = mngs.io.load(lpath_ripple)
    (iEEG_ripple_band, fs) = (xxr, fs) = mngs.io.load(lpath_iEEG)

    # Starts defining SWR-
    df_m = df_p.copy()[["start_s", "end_s", "duration_s"]]

    # Shuffle ripple period (row) within a session as controls
    df_m = df_m.iloc[np.random.permutation(np.arange(len(df_m)))]
    trial_numbers = [
        random.randint(0, df_p.index.max() - 1) for _ in range(len(df_m))
    ]
    df_m.index = trial_numbers
    df_m = df_m.sort_index()

    # Adds metadata for the control data
    # df_p.columns
    # ['rel_peak_pos', 'peak_amp_sd', 'incidence_hz', 'set_size', 'match', 'correct',
    #  'response_time', 'subject', 'session']
    row = df_m.iloc[0]  # fixme

    _xxr = xxr[row.name][int(row.start_s * fs) : int(row.end_s * fs)]
    print(_xxr.shape)
    __import__("ipdb").set_trace()

    # __import__("ipdb").set_trace()

    # print(LPATHS_RIPPLE)

    # for lpath in LPATHS_RIPPLE:
    #     df_plus = mngs.io.load(lpath)

    #     df_plus[["subject", "session"]].drop_duplicates()

    #     df_minus = df_plus.copy()

    #     df_minus["peak_s"] = np.nan
    #     df_minus["rel_peak_pos"] = np.nan

    #     print()

    # __import__("ipdb").set_trace()
    # # rips_df = mngs.io.load(f"./data/rips_df/{iEEG_roi_connected}.pkl")

    # pass


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
