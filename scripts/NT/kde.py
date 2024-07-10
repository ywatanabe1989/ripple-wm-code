#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-11 07:55:05 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/kde.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import mngs
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from mngs.plt.ax import (
    get_global_xlim,
    get_global_ylim,
    sharex,
    sharey,
    add_marginal_ax,
)
import utils
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


def calc_max_density(data):
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


def cleanup_axes(ax, ax_marg_x, ax_marg_y, max_density_x, max_density_y):
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


def prepare_fig(nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes_marg_x = np.full(axes.shape, np.nan).astype(object)
    axes_marg_y = np.full(axes.shape, np.nan).astype(object)

    for ii in range(axes.shape[0]):
        for jj in range(axes.shape[1]):
            ax = axes[ii, jj]
            ax.set_box_aspect(1)
            ax_marg_x = add_marginal_ax(ax, place="top", size=0.2, pad=0.1)
            ax_marg_y = add_marginal_ax(ax, place="right", size=0.2, pad=0.1)
            axes_marg_x[ii, jj] = ax_marg_x
            axes_marg_y[ii, jj] = ax_marg_y

    return fig, axes, axes_marg_x, axes_marg_y


def NT2df(NT, trials_info):
    dfs = []

    # Conditonal data
    conditions = list(
        mngs.gen.yield_grids({"match": [1, 2], "set_size": [4, 6, 8]})
    )

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
    ripple["i_trial"] = ripple.index - 1
    ripple.set_index("i_trial", inplace=True)

    # bin
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

        assert 0 < indi.sum()

        df.loc[indi, "within_ripple"] = True

    return df


def custom_joint_plot(data, nrows, ncols, sample_type, figsize=(15, 10)):
    # Data is expected to be listed list.
    assert mngs.gen.is_listed_X(data, list)

    # Calc lims of marginal axes
    max_density_x, max_density_y = calc_max_density(data)

    # Main
    fig, axes, axes_marg_x, axes_marg_y = prepare_fig(nrows, ncols, figsize)
    fig_mngs, axes_mngs = mngs.plt.subplots(nrows, ncols)
    for i, (ax, ax_marg_x, ax_marg_y, ax_mngs) in enumerate(
        zip(axes.flat, axes_marg_x.flat, axes_marg_y.flat, axes_mngs.flat)
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
            match = dd["match"].iloc[0]
            set_size = dd["set_size"].iloc[0]

            match_str = {1: "IN", 2: "OUT"}[match]
            setsize_str = {4: "Set Size 4", 6: "Set Size 6", 8: "Set Size 8"}[
                set_size
            ]
            label = f"{match_str}, {setsize_str}, {sample_type}"
            ax.set_title(f"{dd['phase'].iloc[0]}")

            # Main
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
            ax_mngs.scatter(
                dd[indi]["factor_1"],
                dd[indi]["factor_2"],
                s=10,
                color=color,
                alpha=0.6,
                label=label,
                id=f"match: {match}, set_size: {set_size}, sample_type: {sample_type}",
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

                # ax_mngs.kde(
                #     dd[indi][_xy],
                #     color=color,
                #     linewidth=0.5,
                #     id=label,
                # )

        cleanup_axes(ax, ax_marg_x, ax_marg_y, max_density_x, max_density_y)

    plt.tight_layout()

    return fig, axes, axes_marg_x, axes_marg_y, axes_mngs


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

    fig, axes, axes_marg_x, axes_marg_y, axes_mngs = custom_joint_plot(
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
        f"./data/CA1/{znorm_str}/{scale}/{unbias_str}/{sample_type}/"
        + "_".join("-".join(item) for item in parsed.items())
        + ".jpg"
    )

    mngs.io.save(
        axes_mngs.to_sigma(), spath_fig.replace(".jpg", ".csv"), from_cwd=True
    )

    return fig, spath_fig, axes, axes_marg_x, axes_marg_y


def main():
    for znorm, symlog, unbias in product(
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

            if parsed not in CONFIG.ROI.CA1:
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

            # Share xlim and ylim
            # main axes
            xlim_main = get_global_xlim(*cache["axes"])
            ylim_main = get_global_ylim(*cache["axes"])
            sharex(*cache["axes"], xlim=xlim_main)
            sharey(*cache["axes"], ylim=ylim_main)

            # marginal axes
            sharex(*cache["axes_marg_x"], xlim=xlim_main)
            sharey(*cache["axes_marg_y"], ylim=ylim_main)

            for fig, spath_fig in zip(cache["fig"], cache["spath_fig"]):
                mngs.io.save(fig, spath_fig, from_cwd=True, dry_run=False)

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
        alpha=0.3,
        fig_scale=2,
    )

    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
