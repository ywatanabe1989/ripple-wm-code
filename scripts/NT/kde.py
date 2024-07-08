#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-09 04:07:50 (ywatanabe)"
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
from IPython.lib import deepreload

with mngs.gen.suppress_output():
    deepreload.reload(mngs)

# mngs.gen.reload(mngs)
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


# def _calc_kde(NTc, start, end, xmin, xmax, ymin, ymax, take_diff=False):
#     # Sample data
#     _x = NTc[:, 0, start:end]  # (10, 20)
#     _y = NTc[:, 1, start:end]  # (10, 20)

#     if take_diff:
#         _start, _end = (
#             CONFIG.PHASES.Fixation.start,
#             CONFIG.PHASES.Fixation.end,
#         )
#         _x_f = NTc[:, 0, _start:_end]
#         _y_f = NTc[:, 1, _start:_end]

#         _x -= _x_f
#         _y -= _y_f

#     # Flatten the arrays
#     _x = _x.flatten()
#     _y = _y.flatten()

#     # Creating the grid
#     _xx, _yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#     positions = np.vstack([_xx.ravel(), _yy.ravel()])

#     # Fit the KDE
#     values = np.vstack([_x, _y])

#     try:
#         kde = gaussian_kde(values)
#         f = np.reshape(kde(positions).T, _xx.shape)
#         return f

#     except Exception as e:
#         logging.warn(e)


def NT2df(NT, trials_info):
    dfs = []
    # Conditonal data
    conditions = list(
        mngs.gen.yield_grids({"match": [1, 2], "set_size": [4, 6, 8]})
    )
    for i_cc, cc in enumerate(conditions):
        NTc = NT[mngs.pd.find_indi(trials_info, cc)]
        for phase_str in CONFIG.PHASES:
            NT_c_p = NTc[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]
            _df = pd.DataFrame(
                {
                    "factor_1": NT_c_p[:, 0, :].reshape(-1),
                    "factor_2": NT_c_p[:, 1, :].reshape(-1),
                }
            )

            cc.update({"phase": phase_str})
            for k, v in cc.items():
                _df[k] = v
            dfs.append(_df)

    dfs = pd.concat(dfs)
    return dfs


def kde_plot(lpath_NT, znorm=False, symlog=False, unbias=False):
    # Loading
    NT = mngs.io.load(lpath_NT)

    if unbias:
        NT[:, 0, :] -= NT[:, 0, :].min() + 1e-5
        NT[:, 1, :] -= NT[:, 1, :].min() + 1e-5

    if symlog:
        NT = mngs.gen.symlog(NT, 1e-5)

    parsed = utils.parse_lpath(lpath_NT)
    trials_info = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed))
    )

    if not parsed in CONFIG.ROI.CA1:
        return

    # Takes the first two factors
    NT = NT[:, :2, :]

    df = NT2df(NT, trials_info)
    df = mngs.pd.merge_columns(df, *["phase", "match", "set_size"])

    # Plotting
    n_phases = len(CONFIG.PHASES)
    n_matches = len(CONFIG.MATCHES)
    fig, axes = plt.subplots(
        nrows=n_matches, ncols=n_phases, sharex=True, sharey=True
    )

    for i_phase in range(n_phases):
        for i_match in range(n_matches):

            ax = axes[i_match, i_phase]

            phase = list(CONFIG.PHASES.keys())[i_phase]
            match = list(CONFIG.MATCHES)[i_match]

            queries = [
                f"{phase}_{match}_4",
                # f"{phase}_{match}_6",
                f"{phase}_{match}_8",
            ]

            base_color = np.array(CC[CONFIG.PHASES[phase].color])

            colors = []
            for ic in range(len(queries)):
                factor = 0.5**ic

                _c = factor * base_color
                _c[-1] = base_color[-1]
                _c = list(_c)
                colors.append(_c)

            # Main scatter plot
            sns.scatterplot(
                data=df[df["phase_match_set_size"].isin(queries)],
                x="factor_1",
                y="factor_2",
                hue="phase_match_set_size",
                palette=colors,
                ax=ax,
                legend=False,
                s=1,
            )

            # Adding marginal KDEs
            [
                sns.kdeplot(
                    data=df[df["phase_match_set_size"] == qq],
                    x="factor_1",
                    ax=ax,
                    legend=False,
                    common_norm=True,
                    color=colors[i_qq],
                )
                for i_qq, qq in enumerate(queries)
            ]
            [
                sns.kdeplot(
                    data=df[df["phase_match_set_size"] == qq],
                    x="factor_2",
                    ax=ax,
                    legend=False,
                    common_norm=True,
                    color=colors[i_qq],
                    vertical=True,
                )
                for i_qq, qq in enumerate(queries)
            ]

            ax._legend = None

    fig.tight_layout()

    # Saving
    # spath_fig = lpath_NT.replace("/NT/", "/NT/kde/").replace(".npy", ".jpg")
    # mngs.io.save(fig, spath_fig, from_cwd=True, dry_run=False)
    # spath_csv = spath_fig.replace(".jpg", ".csv")
    # mngs.io.save(ax.to_sigma(), spath_csv, from_cwd=True, dry_run=False)

    scale = "linear" if not symlog else "symlog"
    znorm_str = "NT" if not znorm else "NT_z"
    unbias_str = "unbiased" if unbias else "orig"
    spath_fig = (
        f"./CA1/{znorm_str}/{scale}/{unbias_str}/"
        + "_".join("-".join(item) for item in parsed.items())
        + ".jpg"
    )
    mngs.io.save(fig, spath_fig, from_cwd=False, dry_run=False)
    plt.close()
    return fig


def main():
    from itertools import product

    for znorm, symlog, unbias in product(
        [False, True], [False, True], [False, True]
    ):
        # for znorm in [False, True]:
        if znorm:
            LPATHS_NT = mngs.gen.natglob(CONFIG.PATH.NT_Z)
        else:
            LPATHS_NT = mngs.gen.natglob(CONFIG.PATH.NT)

        for lpath_NT in LPATHS_NT:
            kde_plot(lpath_NT, znorm=znorm, symlog=symlog, unbias=unbias)

    # for lpath_NT in LPATHS_NT:
    # for symlog in [False, True]:
    # kde_plot(lpath_NT, znorm=znorm, symlog=symlog)
    # plt.close()


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
        alpha=0.9,
        fig_scale=2,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
