#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-09 23:47:22 (ywatanabe)"
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
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

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


def gradiate_colors(base_color, n_colors):
    colors = []
    factor = 1.0
    for ic in range(n_colors):
        _c = factor * base_color
        _c[-1] = base_color[-1]
        _c = list(_c)
        colors.append(_c)
        factor *= 0.8
    colors = np.clip(colors, 0, 1)
    return colors[::-1]


def calc_max_density(data):
    # Calculate global KDE maxima for consistent scale in density plots
    max_density_x = 0
    max_density_y = 0
    for _data in data:
        for df in _data:
            kde_x = gaussian_kde(df["factor_1"])
            kde_y = gaussian_kde(df["factor_2"])
            x_values = np.linspace(
                df["factor_1"].min(), df["factor_1"].max(), 1000
            )
            y_values = np.linspace(
                df["factor_2"].min(), df["factor_2"].max(), 1000
            )
            max_density_x = max(max_density_x, max(kde_x(x_values)))
            max_density_y = max(max_density_y, max(kde_y(y_values)))

    # Global max density and override them for compatibility
    max_density = max(max_density_x, max_density_y)
    max_density_x = max_density_y = max_density

    return max_density_x, max_density_y


def prepare_marginal_axes(ax):
    divider = make_axes_locatable(ax)

    ax_marg_x = divider.append_axes("top", size="20%", pad=0.1)
    ax_marg_x.set_box_aspect(0.2)

    ax_marg_y = divider.append_axes("right", size="20%", pad=0.1)
    ax_marg_y.set_box_aspect(0.2 ** (-1))

    return ax_marg_x, ax_marg_y


def cleanup_axes(ax, ax_marg_x, ax_marg_y, max_density_x, max_density_y):
    # Set the same density limits for all marginal plots
    ax_marg_x.set_xlim(ax.get_xlim())
    ax_marg_y.set_ylim(ax.get_ylim())
    ax_marg_x.set_ylim(0, max_density_x * 1.25)
    ax_marg_y.set_xlim(0, max_density_y * 1.25)

    # Hide spines
    mngs.plt.ax.hide_spines(ax_marg_x, bottom=False)
    mngs.plt.ax.hide_spines(ax_marg_y, left=False)

    # Hide ticks
    for ax_marg in [ax_marg_x, ax_marg_y]:
        ax_marg.set_xticks([])
        ax_marg.set_yticks([])
        ax_marg.set_xlabel(None)
        ax_marg.set_ylabel(None)


def custom_joint_plot(data, nrows, ncols, sample_type, figsize=(15, 10)):
    # Data is expected to be listed list.
    assert mngs.gen.is_listed_X(data, list)

    # Main
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
    )

    max_density_x, max_density_y = calc_max_density(data)

    for i, ax in enumerate(axes.flat):
        if i >= len(data):
            ax.axis("off")
            continue

        ax.set_box_aspect(1)

        ax_marg_x, ax_marg_y = prepare_marginal_axes(ax)

        for i_dd, dd in enumerate(data[i]):

            # Color
            base_color = np.array(CC[CONFIG.PHASES[dd.phase.iloc[0]].color])
            n_queries = len(data[i])
            color = gradiate_colors(base_color, n_queries)[i_dd]

            # Label and Title
            match_str = {1: "Match IN", 2: "Mismatch OUT"}[dd["match"].iloc[0]]
            setsize_str = {4: "Set Size 4", 6: "Set Size 6", 8: "Set Size 8"}[
                dd["set_size"].iloc[0]
            ]
            label = f"{match_str}, {setsize_str}, {sample_type}"
            ax.set_title(f"{dd['phase'].iloc[0]}")

            # Main
            # only mask scatter as kde needs NT values
            if sample_type == "all":
                indi = np.ones(len(dd)).astype(bool)
            if "SWR" in sample_type:
                indi = dd["within_ripple"]

            nn = indi.sum()
            label += f" (n = {nn:,})"

            sns.scatterplot(
                data=dd[indi],
                x="factor_1",
                y="factor_2",
                ax=ax,
                s=10,
                color=color,
                alpha=0.6,
                label=label,
            )
            ax = mngs.plt.ax.set_n_ticks(ax)

            sns.kdeplot(
                data=dd[indi],
                x="factor_1",
                fill=True,
                ax=ax_marg_x,
                color=color,
                common_norm=True,
            )
            sns.kdeplot(
                data=dd[indi],
                x="factor_2",
                fill=True,
                ax=ax_marg_y,
                color=color,
                vertical=True,
                common_norm=True,
            )

        cleanup_axes(ax, ax_marg_x, ax_marg_y, max_density_x, max_density_y)

    plt.tight_layout()

    return fig


def NT2df(NT, trials_info):
    dfs = []
    # Conditonal data
    conditions = list(
        mngs.gen.yield_grids({"match": [1, 2], "set_size": [4, 6, 8]})
    )

    for i_cc, cc in enumerate(conditions):
        indi = mngs.pd.find_indi(trials_info, cc)
        NTc = NT[indi]

        indi_bin = np.arange(NTc.shape[-1])[
            np.newaxis, np.newaxis
        ] * np.ones_like(NTc).astype(int)

        for phase_str in CONFIG.PHASES:
            NT_c_p = NTc[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]
            indi_trial = (
                np.array(indi[indi].index).reshape(-1, 1, 1)
                * np.ones_like(NT_c_p)
            ).astype(int)

            indi_bin_c_p = indi_bin[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]

            _df = pd.DataFrame(
                {
                    "factor_1": NT_c_p[:, 0, :].reshape(-1),
                    "factor_2": NT_c_p[:, 1, :].reshape(-1),
                    "i_trial": indi_trial[:, 0, :].reshape(-1),
                    "i_bin": indi_bin_c_p[:, 0, :].reshape(-1),
                }
            )
            cc.update({"phase": phase_str})
            for k, v in cc.items():
                _df[k] = v
            dfs.append(_df)

    dfs = pd.concat(dfs)
    return dfs


def add_ripple_tag(df, ripple):
    ripple["i_trial"] = ripple.index - 1
    ripple.set_index("i_trial", inplace=True)

    df["within_ripple"] = False
    ripple["start_bin"] = np.floor(
        ripple.start_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)
    ripple["end_bin"] = np.ceil(
        ripple.end_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)
    for i_rip, rip in ripple.iterrows():
        indi = ((rip.start_bin <= df.i_bin) * (df.i_bin <= rip.end_bin)) * (
            df.i_trial == rip.name
        )
        if not 0 < indi.sum():
            __import__("ipdb").set_trace()
        # assert 0 < indi.sum()
        df.loc[indi, "within_ripple"] = True
    return df


def kde_plot(lpath_NT, sample_type, znorm=False, symlog=False, unbias=False):
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

    if sample_type == "SWR+":
        ripple = mngs.io.load(mngs.gen.replace(CONFIG.PATH.RIPPLE, parsed))
        df = add_ripple_tag(df, ripple)
    elif sample_type == "SWR-":
        ripple = mngs.io.load(
            mngs.gen.replace(CONFIG.PATH.RIPPLE_MINUS, parsed)
        )
        df = add_ripple_tag(df, ripple)

    df = mngs.pd.merge_columns(df, *["phase", "match", "set_size"])
    df["color"] = ""

    # Plotting
    data_list = []
    n_phases = len(CONFIG.PHASES)
    n_matches = len(CONFIG.MATCHES)
    for i_match in range(n_matches):
        for i_phase in range(n_phases):

            phase = list(CONFIG.PHASES.keys())[i_phase]
            match = list(CONFIG.MATCHES)[i_match]

            queries = [
                f"{phase}_{match}_4",
                f"{phase}_{match}_6",
                f"{phase}_{match}_8",
            ]

            data = [df[df["phase_match_set_size"] == qq] for qq in queries]
            data_list.append(data)

    nrows, ncols = n_matches, n_phases
    fig = custom_joint_plot(
        data_list,
        nrows,
        ncols,
        sample_type,
        figsize=(15, 10),
    )

    fig.suptitle(str(parsed))

    scale = "linear" if not symlog else "symlog"
    znorm_str = "NT" if not znorm else "NT_z"
    unbias_str = "unbiased" if unbias else "orig"
    spath_fig = (
        f"./CA1/{znorm_str}/{scale}/{unbias_str}/{sample_type}/"
        + "_".join("-".join(item) for item in parsed.items())
        + ".jpg"
    )
    mngs.io.save(fig, spath_fig, from_cwd=False, dry_run=False)
    plt.close()
    return fig


def main():
    from itertools import product

    for znorm, symlog, unbias, sample_type in product(
        # [False, True], [False, True], [False, True]
        [False],
        [False],
        [False],
        ["all", "SWR+", "SWR-"],
    ):
        if znorm:
            LPATHS_NT = mngs.gen.natglob(CONFIG.PATH.NT_Z)
        else:
            LPATHS_NT = mngs.gen.natglob(CONFIG.PATH.NT)

        for lpath_NT in LPATHS_NT:
            kde_plot(
                lpath_NT,
                znorm=znorm,
                symlog=symlog,
                unbias=unbias,
                sample_type=sample_type,
            )


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
        # font_size_axis_label=6,
        # font_size_title=6,
        alpha=0.3,
        fig_scale=2,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
