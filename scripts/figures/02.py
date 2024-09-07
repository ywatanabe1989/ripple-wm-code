#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-07 21:11:20 (ywatanabe)"
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
from scripts import utils  # , load
from scipy.stats import gaussian_kde

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


def A():
    # Loading
    NT = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.NT, CONFIG.REPRESENTATIVE))
    )
    # mngs.gen.replace(CONFIG.PATH.NT_Z, CONFIG.REPRESENTATIVE)
    trials_info = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, CONFIG.REPRESENTATIVE))
    )

    # Takes the first two factors
    NT = NT[:, :2, :]

    # conditonal data
    params_grid = {"match": [1, 2], "set_size": [4, 6, 8]}
    conditions = list(mngs.gen.yield_grids(params_grid))

    # Phases
    CONFIG.PHASES.update(
        {"Merged": {"color": "black", "dur_sec": 8, "start": 0, "end": 160}}
    )

    # Plotting
    fig, axes = mngs.plt.subplots(
        nrows=len(CONFIG.PHASES),
        ncols=len(conditions),
        sharex=True,
        sharey=True,
    )
    fig.supxyt("Factor 1", "Factor 2", None)

    xmin = NT[:, 0, :].min()
    xmax = NT[:, 0, :].max()
    ymin = NT[:, 1, :].min()
    ymax = NT[:, 1, :].max()

    # Conditions
    for i_cc, cc in enumerate(conditions):
        NTc = NT[mngs.pd.find_indi(trials_info, cc)]
        n_trials = len(NTc)
        # Phases
        for i_phase, phase_str in enumerate(CONFIG.PHASES):
            phase = CONFIG.PHASES[phase_str]
            ax = axes[i_phase, i_cc]

            ################################################################################
            ## KDE heatmap
            ################################################################################
            # Sample data
            _x = NTc[:, 0, phase.start : phase.end]  # (10, 20)
            _y = NTc[:, 1, phase.start : phase.end]  # (10, 20)

            # Flatten the arrays
            _x = _x.flatten()
            _y = _y.flatten()

            # Creating the grid
            _xx, _yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([_xx.ravel(), _yy.ravel()])

            # Fit the KDE
            values = np.vstack([_x, _y])
            kde = gaussian_kde(values)
            f = np.reshape(kde(positions).T, _xx.shape)

            # Apply the symlog transformation if necessary
            f = mngs.gen.symlog(f, 1e-3)

            ax.imshow2d(f, cbar=None)

        axes[0, i_cc].set_xyt(
            None,
            None,
            str(cc).replace(", ", "\n") + f"\nn = {n_trials} trials",
        )

    # Saving
    mngs.io.save(fig, "./data/figures/02/A.jpg", from_cwd=True)
    mngs.io.save(ax.to_sigma(), "./data/figures/02/A.csv", from_cwd=True)
    return fig


def main():
    fig_A = A()


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
        agg=True,
        font_size_axis_label=6,
        font_size_title=6,
        alpha=0.5,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
