#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 09:25:27 (ywatanabe)"
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
    def _add_NT(row):
        i_trial = row.name - 1
        nt = mngs.io.load(
            mngs.gen.replace(
                CONFIG.PATH.NT_Z,
                dict(
                    sub=row.subject,
                    session=row.session,
                    roi=row.roi,
                ),
            )
        )
        lim = (0, eval(CONFIG.NT.N_BINS))
        start = row.peak_i + CONFIG.RIPPLE.BINS.pre[0]
        end = row.peak_i + CONFIG.RIPPLE.BINS.post[1]
        start = np.clip(start, *lim).astype(int)
        end = np.clip(end, *lim).astype(int)
        nt = nt[
            i_trial,
            :,
            start:end,
        ]
        return nt

    xx["NT"] = xx.apply(_add_NT, axis=1)
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
