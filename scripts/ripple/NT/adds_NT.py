#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 10:02:15 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/adds_NT.py

"""This script adds neural trajectory during SWR."""

"""Imports"""
import importlib
import logging
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
from tqdm import tqdm

try:
    from scripts import utils
except:
    pass

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def find_peak_i(xx):
    NT_TIME = eval(CONFIG.NT.TIME_AXIS)
    xx["peak_i"] = xx["peak_s"].apply(
        lambda x: mngs.gen.find_closest(NT_TIME, x)[1]
    )
    return xx


def add_NT(xx):
    """Add Neural Trajectory (NT) data to the DataFrame."""

    def _load_nt(row):
        return mngs.io.load(
            mngs.gen.replace(
                CONFIG.PATH.NT_Z,
                dict(sub=row.subject, session=row.session, roi=row.roi),
            )
        )

    def _slice_and_pad(nt, row, i_trial):
        # Slicing
        lim = (0, eval(CONFIG.NT.N_BINS))
        start = row.peak_i + CONFIG.RIPPLE.BINS.pre[0]
        end = row.peak_i + CONFIG.RIPPLE.BINS.post[1]
        width_ideal = end - start
        start_clipped, end_clipped = np.clip([start, end], *lim).astype(int)
        nt_slice = nt[i_trial, :, start_clipped:end_clipped]

        # Padding
        if nt_slice.shape[1] == width_ideal:
            return nt_slice

        padded = np.full((nt_slice.shape[0], width_ideal), np.nan)
        n_left_pad = abs(start - start_clipped)
        padded[:, n_left_pad : n_left_pad + nt_slice.shape[1]] = nt_slice
        return padded

    def _add_NT_single(row):
        i_trial = row.name - 1
        nt = _load_nt(row)
        nt_padded = _slice_and_pad(nt, row, i_trial)
        return nt_padded

    xx["NT"] = xx.apply(_add_NT_single, axis=1)
    np.vstack(xx.NT)
    return xx


def main():
    for ripple_type in ["RIPPLE", "RIPPLE_MINUS"]:
        for ca1 in CONFIG.ROI.CA1:
            # PATHs
            lpath = mngs.gen.replace(CONFIG.PATH[ripple_type], ca1)
            spath = mngs.gen.replace(
                CONFIG.PATH[f"{ripple_type}_WITH_NT"], ca1
            )

            # Main
            swr = mngs.io.load(lpath)
            swr = find_peak_i(swr)
            swr = add_NT(swr)
            # swr = swr.drop(columns=["peak_i"])

            # Saving
            mngs.io.save(
                swr,
                spath,
                dry_run=False,
                from_cwd=True,
                verbose=True,
            )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
