#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-10 14:42:07 (ywatanabe)"
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

from itertools import product

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


# def gradiate_colors(base_color, n_colors):
#     colors = []
#     factor = 1.0
#     for ic in range(n_colors):
#         _c = factor * base_color
#         _c[-1] = base_color[-1]
#         _c = list(_c)
#         colors.append(_c)
#         factor *= 0.8
#     colors = np.clip(colors, 0, 1)
#     return colors[::-1]


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


def add_marginal_axes(ax):
    divider = make_axes_locatable(ax)

    ax_marg_x = divider.append_axes("top", size="20%", pad=0.1)
    ax_marg_x.set_box_aspect(0.2)

    ax_marg_y = divider.append_axes("right", size="20%", pad=0.1)
    ax_marg_y.set_box_aspect(0.2 ** (-1))

    return ax_marg_x, ax_marg_y


def cleanup_axes(ax, ax_marg_x, ax_marg_y, max_density_x, max_density_y):
    # # Set the same density limits for all marginal plots
    # ax_marg_x.set_xlim(ax.get_xlim())
    # ax_marg_y.set_ylim(ax.get_ylim())
    # ax_marg_x.set_ylim(0, max_density_x * 1.25)
    # ax_marg_y.set_xlim(0, max_density_y * 1.25)
    ax = mngs.plt.ax.set_n_ticks(ax)
    ax_marg_x = mngs.plt.ax.set_n_ticks(ax_marg_x)
    ax_marg_y = mngs.plt.ax.set_n_ticks(ax_marg_y)

    # Hide spines
    mngs.plt.ax.hide_spines(ax_marg_x, bottom=False)
    mngs.plt.ax.hide_spines(ax_marg_y, left=False)

    # Hide ticks
    for ax_marg in [ax_marg_x, ax_marg_y]:
        ax_marg.set_xticks([])
        ax_marg.set_yticks([])
        ax_marg.set_xlabel(None)
        ax_marg.set_ylabel(None)


def get_global_xlim(*multiple_axes):
    xmin, xmax = np.inf, -np.inf
    for axes in multiple_axes:
        for ax in axes.flat:
            _xmin, _xmax = ax.get_xlim()
            xmin = min(xmin, _xmin)
            xmax = max(xmax, _xmax)
    return (xmin, xmax)


def get_global_ylim(*multiple_axes):
    ymin, ymax = np.inf, -np.inf
    for axes in multiple_axes:
        for ax in axes.flat:
            _ymin, _ymax = ax.get_ylim()
            ymin = min(ymin, _ymin)
            ymax = max(ymax, _ymax)
    return (ymin, ymax)


def sharex(*multiple_axes, xlim=None):
    if xlim is None:
        raise ValueError

    for axes in multiple_axes:
        for ax in axes.flat:
            ax.set_xlim(xlim)

    if len(multiple_axes) == 1:
        return multiple_axes[0], xlim
    else:
        return multiple_axes, xlim


def sharey(*multiple_axes, ylim=None):
    if ylim is None:
        raise ValueError

    for axes in multiple_axes:
        for ax in axes.flat:
            ax.set_ylim(ylim)

    if len(multiple_axes) == 1:
        return multiple_axes[0], ylim
    else:
        return multiple_axes, ylim


def prepare_fig(nrows, ncols, figsize):
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize  # , sharex=True, sharey=True
    )
    axes_marg_x = np.full(axes.shape, np.nan).astype(object)
    axes_marg_y = np.full(axes.shape, np.nan).astype(object)

    for ii in range(axes.shape[0]):
        for jj in range(axes.shape[1]):
            ax = axes[ii, jj]
            ax.set_box_aspect(1)
            ax_marg_x, ax_marg_y = add_marginal_axes(ax)
            axes_marg_x[ii, jj] = ax_marg_x
            axes_marg_y[ii, jj] = ax_marg_y

    return fig, axes, axes_marg_x, axes_marg_y


def custom_joint_plot(data, nrows, ncols, sample_type, figsize=(15, 10)):
    # Data is expected to be listed list.
    assert mngs.gen.is_listed_X(data, list)

    # Calc lims of marginal axes
    max_density_x, max_density_y = calc_max_density(data)

    # Main
    fig, axes, axes_marg_x, axes_marg_y = prepare_fig(nrows, ncols, figsize)
    for i, (ax, ax_marg_x, ax_marg_y) in enumerate(
        zip(axes.flat, axes_marg_x.flat, axes_marg_y.flat)
    ):
        if i >= len(data):
            ax.axis("off")
            continue

        for i_dd, dd in enumerate(data[i]):

            # Base Color
            base_color = np.array(CC[CONFIG.PHASES[dd.phase.iloc[0]].color])
            # Gradient color
            n_queries = len(data[i])
            ss = dd["set_size"].iloc[0]
            n_split = n_queries * 4
            i_color = {4: n_split - 1, 6: n_split // 2, 8: 0}[ss]
            color = mngs.plt.gradiate_color(base_color, n=n_split)[i_color]

            # Label and Title
            match_str = {1: "IN", 2: "OUT"}[dd["match"].iloc[0]]
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

            for _xy, _ax_marg, _vertical in zip(
                ["factor_1", "factor_2"], [ax_marg_x, ax_marg_y], [False, True]
            ):
                sns.kdeplot(
                    data=dd[indi],
                    x=_xy,
                    fill=False,
                    ax=_ax_marg,
                    color=color,
                    common_norm=True,
                    vertical=_vertical,
                    linewidth=0.5,
                )
            # sns.kdeplot(
            #     data=dd[indi],
            #     x="factor_1",
            #     fill=fill,
            #     ax=ax_marg_x,
            #     color=color,
            #     common_norm=True,
            # )
            # sns.kdeplot(
            #     data=dd[indi],
            #     x="factor_2",
            #     fill=fill,
            #     ax=ax_marg_y,
            #     color=color,
            #     vertical=True,
            #     common_norm=True,
            # )

        cleanup_axes(ax, ax_marg_x, ax_marg_y, max_density_x, max_density_y)

    plt.tight_layout()

    return fig, axes, axes_marg_x, axes_marg_y


def NT2df(NT, trials_info):
    dfs = []
    # Conditonal data
    conditions = list(
        mngs.gen.yield_grids({"match": [1, 2], "set_size": [4, 6, 8]})
    )

    indi_all = []
    for i_cc, cc in enumerate(conditions):
        indi = mngs.pd.find_indi(trials_info, cc)
        indi = np.array(indi[indi].index)

        NTc = NT[indi]
        indi_bin = np.arange(NTc.shape[-1])[
            np.newaxis, np.newaxis
        ] * np.ones_like(NTc).astype(int)

        for phase_str in CONFIG.PHASES:
            NTcp = NTc[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]
            indi_trial_cp = (
                indi.reshape(-1, 1, 1) * np.ones_like(NTcp)
            ).astype(int)

            indi_bin_cp = indi_bin[
                ...,
                CONFIG.PHASES[phase_str].start : CONFIG.PHASES[phase_str].end,
            ]

            _df = pd.DataFrame(
                {
                    "factor_1": NTcp[:, 0, :].reshape(-1),
                    "factor_2": NTcp[:, 1, :].reshape(-1),
                    "i_trial": indi_trial_cp[:, 0, :].reshape(-1),
                    "i_bin": indi_bin_cp[:, 0, :].reshape(-1),
                }
            )
            cc.update({"phase": phase_str})
            for k, v in cc.items():
                _df[k] = v
            dfs.append(_df)

    dfs = pd.concat(dfs)
    return dfs


def add_ripple_tag(df, ripple):
    # ripple
    ripple["i_trial"] = ripple.index - 1
    ripple.set_index("i_trial", inplace=True)

    ripple["start_bin"] = np.floor(
        ripple.start_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)
    ripple["end_bin"] = np.ceil(
        ripple.end_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)
    ripple["peak_bin"] = np.array(
        ripple.peak_s / CONFIG.GPFA.BIN_SIZE_MS * 1e3
    ).astype(int)

    # labeling
    df = df.sort_values(["i_trial", "i_bin"])
    df["within_ripple"] = False
    for i_rip, rip in ripple.iterrows():
        indi_trial = df.i_trial == rip.name
        # indi_bin = df.i_bin == rip.peak_bin
        indi_bin = (rip.peak_bin - 2 <= df.i_bin) * (
            df.i_bin <= rip.peak_bin + 2
        )
        indi = indi_trial * indi_bin

        # indi = ((rip.start_bin <= df.i_bin) * (df.i_bin <= rip.end_bin)) * (
        #     df.i_trial == rip.name
        # )
        # n_bins = rip.end_bin - rip.start_bin

        assert 0 < indi.sum()  # <= n_bins + 1
        # assert indi.sum() == 1
        # print(indi.sum())

        df.loc[indi, "within_ripple"] = True
    return df


def kde_plot(lpath_NT, sample_type, znorm=False, symlog=False, unbias=False):
    # Loading
    NT = mngs.io.load(lpath_NT)
    # Takes the first two factors
    NT = NT[:, :2, :]

    if unbias:
        NT[:, 0, :] -= NT[:, 0, :].min() + 1e-5
        NT[:, 1, :] -= NT[:, 1, :].min() + 1e-5

    if symlog:
        NT = mngs.gen.symlog(NT, 1e-5)

    parsed = utils.parse_lpath(lpath_NT)

    trials_info = mngs.io.load(
        eval(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed))
    )

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

    fig, axes, axes_marg_x, axes_marg_y = custom_joint_plot(
        data_list,
        n_matches,
        n_phases,
        sample_type,
        figsize=(15, 10),
    )

    fig.suptitle(str(parsed))

    # Saving
    scale = "linear" if not symlog else "symlog"
    znorm_str = "NT" if not znorm else "NT_z"
    unbias_str = "unbiased" if unbias else "orig"
    spath_fig = (
        f"./CA1/{znorm_str}/{scale}/{unbias_str}/{sample_type}/"
        + "_".join("-".join(item) for item in parsed.items())
        + ".jpg"
    )

    return fig, spath_fig, axes, axes_marg_x, axes_marg_y


def main():
    for znorm, symlog, unbias in product(
        # [False, True], [False, True], [False, True]
        [False],
        [False],
        [False],
    ):

        if znorm:
            LPATH_EXP = CONFIG.PATH.NT_Z
        else:
            LPATH_EXP = CONFIG.PATH.NT

        for ca1_region in CONFIG.ROI.CA1:
            lpath_NT = mngs.gen.replace(LPATH_EXP, ca1_region)
            parsed = utils.parse_lpath(lpath_NT)
            if not parsed in CONFIG.ROI.CA1:
                continue

            cache = mngs.gen.listed_dict()
            for sample_type in ["SWR+", "SWR-", "all"]:

                fig, spath_fig, axes, axes_marg_x, axes_marg_y = kde_plot(
                    lpath_NT,
                    znorm=znorm,
                    symlog=symlog,
                    unbias=unbias,
                    sample_type=sample_type,
                )

                cache["fig"].append(fig)
                cache["spath_fig"].append(spath_fig)
                cache["axes"].append(axes)
                cache["axes_marg_x"].append(axes_marg_x)
                cache["axes_marg_y"].append(axes_marg_y)

            # xlim and ylim
            # main axes
            xlim_main = get_global_xlim(*cache["axes"])
            ylim_main = get_global_ylim(*cache["axes"])
            sharex(*cache["axes"], xlim=xlim_main)
            sharey(*cache["axes"], ylim=ylim_main)

            # marginal axes
            sharex(*cache["axes_marg_x"], xlim=xlim_main)
            sharey(*cache["axes_marg_y"], ylim=ylim_main)

            # # marginal axes, height
            # ylim_marg_x = get_global_ylim(*cache["axes_marg_x"])
            # xlim_marg_y = get_global_xlim(*cache["axes_marg_y"])
            # height = max(max(ylim_marg_x), max(xlim_marg_y))
            # sharey(*cache["axes_marg_x"], ylim=(0, height))
            # sharex(*cache["axes_marg_y"], xlim=(0, height))

            for fig, spath_fig in zip(cache["fig"], cache["spath_fig"]):
                mngs.io.save(fig, spath_fig, from_cwd=False, dry_run=False)

            plt.close()
    # # xlim and ylim
    # # main axes
    # xlim_main = get_global_xlim(axes)
    # ylim_main = get_global_ylim(axes)
    # sharex(axes, xlim=xlim_main)
    # sharey(axes, ylim=ylim_main)

    # # marginal axes
    # sharex(axes_marg_x, xlim=xlim_main)
    # sharey(axes_marg_y, ylim=ylim_main)

    # # marginal axes, height
    # ylim_marg_x = get_global_ylim(axes_marg_x)
    # xlim_marg_y = get_global_xlim(axes_marg_y)
    # height = max(max(ylim_marg_x), max(xlim_marg_y))
    # sharey(axes_marg_x, ylim=(0, height))
    # sharex(axes_marg_y, xlim=(0, height))
    # mngs.io.save(fig, spath_fig, from_cwd=False, dry_run=False)


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
