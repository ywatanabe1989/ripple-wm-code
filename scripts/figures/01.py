#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-06 18:20:08 (ywatanabe)"
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
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def A():
    # Loading
    xx = mngs.io.load(
        eval(
            mngs.gen.replace(
                CONFIG["PATH_iEEG"], eval(CONFIG["REPRESENTATIVE"])
            )
        )
    )
    time = mngs.dsp.time(0, CONFIG["TRIAL_SEC"], CONFIG["FS_iEEG"]) - 6
    trial_number = int(eval(CONFIG["REPRESENTATIVE"])["trial"])
    i_trial = trial_number - 1

    # Plotting
    fig, ax = mngs.plt.subplots()
    ax.plot(time, xx[i_trial].T)

    # Saving
    mngs.io.save(fig, "./data/figures/01/A.jpg", from_cwd=True)
    mngs.io.save(ax.to_sigma(), "./data/figures/01/A.csv", from_cwd=True)
    return fig


def B():
    # Loading
    xx = mngs.io.load(
        eval(
            mngs.gen.replace(
                CONFIG["PATH_iEEG"], eval(CONFIG["REPRESENTATIVE"])
            )
        )
    )
    time = mngs.dsp.time(0, CONFIG["TRIAL_SEC"], CONFIG["FS_iEEG"]) - 6
    trial_number = int(eval(CONFIG["REPRESENTATIVE"])["trial"])
    i_trial = trial_number - 1

    # Loading
    xxr = mngs.dsp.filt.bandpass(
        xx, CONFIG["FS_iEEG"], CONFIG["RIPPLE_BANDS"]
    ).squeeze()

    # Plotting
    fig, ax = mngs.plt.subplots()
    ax.plot(time, xxr[i_trial].T)

    # Saving
    mngs.io.save(fig, "./data/figures/01/B.jpg", from_cwd=True)
    mngs.io.save(ax.to_sigma(), "./data/figures/01/B.csv", from_cwd=True)
    return fig


def C():
    # Loading
    xx = mngs.io.load(
        eval(
            mngs.gen.replace(
                CONFIG["PATH_SPIKE_TIMES"], eval(CONFIG["REPRESENTATIVE"])
            )
        )
    )
    time = mngs.dsp.time(0, CONFIG["TRIAL_SEC"], CONFIG["FS_iEEG"]) - 6
    trial_number = int(eval(CONFIG["REPRESENTATIVE"])["trial"])
    i_trial = trial_number - 1

    xi = xx[i_trial]

    spikes = [
        xi[col].replace("", np.nan).dropna().tolist() for col in xi.columns
    ]

    # Plotting
    fig, ax = mngs.plt.subplots()
    ax.raster(
        spikes,
        time,
    )

    # Saving
    mngs.io.save(fig, "./data/figures/01/C.jpg", from_cwd=True)
    mngs.io.save(ax.to_sigma(), "./data/figures/01/C.csv", from_cwd=True)

    return fig


def D():
    # Loading
    xx = mngs.io.load(
        eval(
            mngs.gen.replace(
                CONFIG["PATH_NT_Z"], eval(CONFIG["REPRESENTATIVE"])
            )
        )
    )
    time = np.linspace(0, CONFIG["TRIAL_SEC"], xx.shape[-1]) - 6
    trial_number = int(eval(CONFIG["REPRESENTATIVE"])["trial"])
    i_trial = trial_number - 1

    xi = xx[i_trial]
    xi = mngs.gen.symlog(xi)

    # Plotting
    fig, ax = mngs.plt.subplots()
    ax.plot(time, xi.T)

    # Saving
    mngs.io.save(fig, "./data/figures/01/D.jpg", from_cwd=True)
    mngs.io.save(ax.to_sigma(), "./data/figures/01/D.csv", from_cwd=True)

    return fig


def E():
    # Loading
    xx = mngs.io.load(
        eval(
            mngs.gen.replace(
                CONFIG["PATH_NT_Z"], eval(CONFIG["REPRESENTATIVE"])
            )
        )
    )
    time = np.linspace(0, CONFIG["TRIAL_SEC"], xx.shape[-1]) - 6
    trial_number = int(eval(CONFIG["REPRESENTATIVE"])["trial"])
    i_trial = trial_number - 1

    xi = xx[i_trial]
    # xi.shape # (3, 160)
    dist = []
    for ii in range(xi.shape[-1]):
        dist.append(mngs.linalg.euclidean_distance(xi[:, ii], 0 * xi[:, ii]))
    dist = np.array(dist)

    dist = mngs.gen.symlog(dist)

    # Plotting
    fig, ax = mngs.plt.subplots()
    ax.plot(time, dist)

    # Saving
    mngs.io.save(fig, "./data/figures/01/E.jpg", from_cwd=True)
    mngs.io.save(ax.to_sigma(), "./data/figures/01/E.csv", from_cwd=True)

    return fig


def main():
    fig_A = A()
    fig_B = B()
    fig_C = C()
    fig_D = D()
    fig_E = E()


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
