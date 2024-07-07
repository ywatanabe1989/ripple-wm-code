#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-07 19:54:14 (ywatanabe)"
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
    xx = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.NT, CONFIG.REPRESENTATIVE))
    )
    # mngs.gen.replace(CONFIG.PATH.NT_Z, CONFIG.REPRESENTATIVE)
    trials_info = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, CONFIG.REPRESENTATIVE))
    )

    # Takes the first two factors
    xx = xx[:, :2, :]
    xx = mngs.gen.symlog(xx, 1e-3)

    time = np.linspace(0, CONFIG.TRIAL.DUR_SEC, xx.shape[-1]) - 6

    # conditonal data
    params_grid = {"match": [1, 2], "set_size": [4, 6, 8]}
    conditions = list(mngs.gen.yield_grids(params_grid))

    # Phases
    CONFIG.PHASES.update(
        {"Merged": {"color": "black", "dur_sec": 8, "start": 0, "end": 160}}
    )

    # Plotting
    colors = utils.define_transitional_colors()
    fig, axes = mngs.plt.subplots(
        nrows=len(CONFIG.PHASES),
        ncols=len(conditions),
        sharex=True,
        sharey=True,
    )
    fig.supxyt("Factor 1", "Factor 2", None)

    n_max_trials = 5

    xmin = xx[:, 0, :].min()
    xmax = xx[:, 0, :].max()
    ymin = xx[:, 1, :].min()
    ymax = xx[:, 1, :].max()

    # Conditions
    for i_cc, cc in enumerate(conditions):
        xc = xx[mngs.pd.find_indi(trials_info, cc)]
        # # Random
        # for ii, i_rand_trial in enumerate(
        #     np.random.permutation(range(len(xc)))
        # ):

        # # In order
        # for ii, i_trial in enumerate(
        #     range(len(xc))
        # ):

        # Phases
        for i_phase, phase_str in enumerate(CONFIG.PHASES):
            phase = CONFIG.PHASES[phase_str]
            ax = axes[i_phase, i_cc]

            # ii += 1
            # if ii > n_max_trials:
            #     continue

            # ################################################################################
            # ## Scatter
            # ################################################################################
            # ax.scatter(
            #     xc[i_rand_trial][0, phase.start : phase.end],
            #     xc[i_rand_trial][1, phase.start : phase.end],
            #     color=CC[phase.color],
            #     s=10,
            # )

            # # Transitional Line plot
            # for tt in range(xc.shape[-1] - 1):
            #     if (phase.start <= tt) and (tt < phase.end):
            #         ax.plot(
            #             xc[i_rand_trial][0][tt : tt + 2],
            #             xc[i_rand_trial][1][tt : tt + 2],
            #             color=colors[tt + 1],
            #         )

            ################################################################################
            ## KDE heatmap
            ################################################################################
            from scipy.stats import gaussian_kde

            # Sample data
            _x = xc[:, 0, phase.start : phase.end]  # (10, 20)
            _y = xc[:, 1, phase.start : phase.end]  # (10, 20)

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

        axes[0, i_cc].set_xyt(None, None, str(cc).replace(", ", "\n"))

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
        font_size_title=6,
        alpha=0.5,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
