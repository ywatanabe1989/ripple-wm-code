#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 10:14:04 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import importlib
import logging
import os
import re
import sys
import warnings
from bisect import bisect_left
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
from scripts import utils
from tqdm import tqdm

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def add_phase(xx_all):
    xx_all["phase"] = str(np.nan)
    for phase, phase_data in CONFIG.PHASES.items():
        indi_phase = (phase_data.start <= xx_all.peak_i) * (
            xx_all.peak_i < phase_data.end
        )
        xx_all.loc[indi_phase, "phase"] = phase
    return xx_all


def main():
    xxp_all, xxm_all = utils.load_ripples(with_NT=True)
    xxp_all, xxm_all = add_phase(xxp_all), add_phase(xxm_all)

    nt_xxp = np.stack(xxp_all.NT, axis=0)
    nt_xxm = np.stack(xxm_all.NT, axis=0)

    dd_xxp = np.sqrt((nt_xxp**2).sum(axis=1))
    dd_xxm = np.sqrt((nt_xxm**2).sum(axis=1))

    nnp = np.nansum(~np.isnan(dd_xxp), axis=0)
    mmp = np.nanmean(dd_xxp, axis=0)
    ssp = np.nanstd(dd_xxp, axis=0)
    cip = 1.96 * ssp / np.sqrt(nnp)

    # Plotting
    fig, ax = mngs.plt.subplots()
    tt = np.arange(dd_xxp.shape[1]) * CONFIG.GPFA.BIN_SIZE_MS
    tt -= int(tt.mean())
    ax.plot_with_ci(tt, mmp, cip)
    mngs.io.save(fig, "./tmp.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
