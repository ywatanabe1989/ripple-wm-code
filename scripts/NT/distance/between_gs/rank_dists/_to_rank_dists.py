#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 19:29:04 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/to_rank_dists.py

"""This script does XYZ."""

"""Imports"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import mngs

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

try:
    from scripts import utils
except:
    pass
from scipy.stats import rankdata
"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Parameters"""

"""Functions & Classes"""

def under_sample_by_session(df):
    # Drops rows with nan dist
    df = df[~df["dist"].isna()]

    # Calculates the n_min to balance the number of samples per session
    n_min = np.inf
    df["n"] = 1
    nn = df.groupby("session_id").agg("sum")
    n_min = nn.n.min()

    for sid in df.session_id.unique():
        indi_session = df.session_id == sid
        non_nan_mask = (~df[indi_session].isna()).index
        indi_balanced = np.random.permutation(non_nan_mask)[:n_min]

        # Temporalily extract data to use
        stashed = df.loc[indi_balanced, "dist"].copy()

        # Remove dist data of the session
        df.loc[indi_session, "dist"] = np.nan
        df.loc[indi_balanced, "dist"] = stashed

    # Drops rows with nan dist
    df = df[~df["dist"].isna()]

    nn = df.groupby("session_id").agg("sum")
    assert (nn.n == n_min).all()

    return df

def main():
    PHASES_TO_PLOT = [
        ["Fixation", "Encoding", "Maintenance", "Retrieval"],
        ["Encoding", "Retrieval"],
    ]

    for phases_to_plot in PHASES_TO_PLOT:
        df = mngs.io.load(f"./scripts/NT/distance/between_gs/calc_dist/{'_'.join(phases_to_plot)}/dist_ca1.csv")
        df = mngs.pd.merge_cols(df, "sub", "session", "roi", name="session_id")
        for sid in df.session_id.unique():
            indi_session = df.session_id == sid
            df.loc[indi_session, "dist"] = rankdata(df.loc[indi_session, "dist"])

        mngs.io.save(df, f"./{'_'.join(phases_to_plot)}/dist_ca1.csv")

if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
