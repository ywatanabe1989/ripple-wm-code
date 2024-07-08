#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-08 21:16:25 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/kde.py


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
import mngs

mngs.gen.reload(mngs)
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
from tqdm import tqdm
import xarray as xr
from scipy.stats import gaussian_kde
import logging

# sys.path = ["."] + sys.path
from scripts import utils

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


def calc_kde(NTc, start, end, xmin, xmax, ymin, ymax, take_diff=False):
    # Sample data
    _x = NTc[:, 0, start:end]  # (10, 20)
    _y = NTc[:, 1, start:end]  # (10, 20)

    if take_diff:
        _start, _end = (
            CONFIG.PHASES.Fixation.start,
            CONFIG.PHASES.Fixation.end,
        )
        _x_f = NTc[:, 0, _start:_end]
        _y_f = NTc[:, 1, _start:_end]

        _x -= _x_f
        _y -= _y_f

    # Flatten the arrays
    _x = _x.flatten()
    _y = _y.flatten()

    # Creating the grid
    _xx, _yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([_xx.ravel(), _yy.ravel()])

    # Fit the KDE
    values = np.vstack([_x, _y])

    try:
        kde = gaussian_kde(values)
        f = np.reshape(kde(positions).T, _xx.shape)
        return f

    except Exception as e:
        logging.warn(e)


def kde_plot(lpath_NT):
    # Loading
    NT = mngs.io.load(lpath_NT)
    parsed = utils.parse_lpath(lpath_NT)
    trials_info = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed))
    )

    if not parsed in CONFIG.ROI.CA1:
        return

    NT = mngs.gen.symlog(NT)

    # min and max
    xmin = NT[:, 0, :].min()
    xmax = NT[:, 0, :].max()
    ymin = NT[:, 1, :].min()
    ymax = NT[:, 1, :].max()

    # Takes the first two factors
    NT = NT[:, :2, :]

    # Conditonal data
    params_grid = {"match": [1, 2], "set_size": [4, 6, 8]}
    conditions = list(mngs.gen.yield_grids(params_grid))

    # Plotting
    fig, axes = mngs.plt.subplots(
        nrows=len(CONFIG.PHASES),
        ncols=len(conditions),
        sharex=True,
        sharey=True,
    )
    fig.supxyt("Factor 1", "Factor 2", None)

    # Conditions
    for i_cc, cc in enumerate(conditions):
        NTc = NT[mngs.pd.find_indi(trials_info, cc)]
        n_trials_c = len(NTc)

        # Phases
        for i_phase, phase_str in enumerate(CONFIG.PHASES):

            phase = CONFIG.PHASES[phase_str]
            ax = axes[i_phase, i_cc]

            f = calc_kde(
                NTc,
                phase.start,
                phase.end,
                xmin,
                xmax,
                ymin,
                ymax,
                take_diff=False,
            )

            if f is not None:
                # Heatmap
                ax.imshow2d(f, cbar=None)
                ax.set_ticks(
                    xvals=np.linspace(xmin, xmax, f.shape[0]),
                    yvals=np.linspace(ymin, ymax, f.shape[1]),
                )

            axes[0, i_cc].set_xyt(
                None,
                None,
                str(cc).replace(", ", "\n") + f"\nn = {n_trials_c} trials",
            )
            axes[i_phase, 0].set_xyt(
                None,
                phase_str,
                None,
            )

    # Saving
    spath_fig = lpath_NT.replace("/NT/", "/NT/kde/").replace(".npy", ".jpg")
    mngs.io.save(fig, spath_fig, from_cwd=True, dry_run=False)
    spath_csv = spath_fig.replace(".jpg", ".csv")
    mngs.io.save(ax.to_sigma(), spath_csv, from_cwd=True, dry_run=False)

    spath_fig = (
        "./CA1/" + "_".join("-".join(item) for item in parsed.items()) + ".jpg"
    )
    mngs.io.save(fig, spath_fig, from_cwd=False, dry_run=False)
    return fig


def main():
    LPATHS_NT = mngs.gen.natglob(CONFIG.PATH.NT_Z)
    for lpath_NT in LPATHS_NT:
        kde_plot(lpath_NT)
        plt.close()


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
