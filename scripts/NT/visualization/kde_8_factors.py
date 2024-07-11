#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-11 22:38:37 (ywatanabe)"
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

import utils
from itertools import product
import logging

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


def gen_query(phase, match, set_size):
    return f"{phase}_{match}_{set_size}"

    # # Plotting
    # data_list = []
    # n_phases = len(CONFIG.PHASES)
    # n_matches = len(CONFIG.MATCHES)
    # for i_match in range(n_matches):
    #     for i_phase in range(n_phases):

    #         phase = list(CONFIG.PHASES.keys())[i_phase]
    #         match = list(CONFIG.MATCHES)[i_match]

    #         queries = [
    #             f"{phase}_{match}_4",
    #             f"{phase}_{match}_6",
    #             f"{phase}_{match}_8",
    #         ]

    #         data = [df[df["phase_match_set_size"] == qq] for qq in queries]
    #         data_list.append(data)


def NT2df(NT, trials_info):
    dfs = []

    # Conditonal data
    conditions = list(
        mngs.gen.yield_grids({"match": [1, 2], "set_size": [4, 6, 8]})
    )

    for i_cc, cc in enumerate(conditions):
        indi = mngs.pd.find_indi(trials_info, cc)
        indi = np.array(indi[indi].index) - 1

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
                    f"factor_{kk+1}": NTcp[:, kk, :].reshape(-1)
                    for kk in range(NTcp.shape[1])
                }
            )
            _df["i_trial"] = indi_trial_cp[:, 0, :].reshape(-1)
            _df["i_bin"] = indi_bin_cp[:, 0, :].reshape(-1)

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


def define_color(phase, set_size):
    n_set_sizes = len(CONFIG.SET_SIZES)

    # Base Color
    base_color = np.array(CC[CONFIG.PHASES[phase].color])
    # Gradient color
    n_split = n_set_sizes * 4
    i_color = {4: n_split - 1, 6: n_split // 2, 8: 0}[set_size]
    color = mngs.plt.gradiate_color(base_color, n=n_split)[i_color]
    return color


def kde_plot_8_factors(data, sample_type):
    # n_phases = len(CONFIG.PHASES)
    n_matches = len(CONFIG.MATCHES)
    # n_set_sizes = len(CONFIG.SET_SIZES)
    n_factors = len(mngs.gen.search("factor_", data.columns)[1])

    # # Calc lims of marginal axes
    # max_density_x, max_density_y = calc_max_density(data)

    # Main
    fig, axes = mngs.plt.subplots(ncols=n_matches, nrows=n_factors)

    for i_match, match in enumerate(CONFIG.MATCHES):
        for i_factor in range(n_factors):
            ax = axes[i_factor, i_match]

            for i_set_size, set_size in enumerate([8]):
                for i_phase, phase in enumerate(CONFIG.PHASES.keys()):

                    dd = mngs.pd.slice(
                        data,
                        {"match": match, "set_size": set_size, "phase": phase},
                    )
                    color = define_color(phase, set_size)

                    # label
                    match_str = ["IN", "OUT"][match - 1]
                    set_size_str = f"Set Size {set_size}"
                    label = f"{match_str}, {phase}, {set_size_str}, {sample_type}, {i_factor}"

                    if i_factor == 0:
                        ax.set_title(match_str)

                    # Main
                    if sample_type == "all":
                        indi = np.ones(len(dd)).astype(bool)
                    if "SWR" in sample_type:
                        indi = dd["within_ripple"]

                    nn = indi.sum()
                    label += f" (n = {nn:,})"

                    xlim_width = 0.6
                    try:
                        ax.kde(
                            dd[indi][f"factor_{i_factor+1}"],
                            color=color,
                            linewidth=0.5,
                            id=label,
                            xlim=(-xlim_width / 2, xlim_width / 2),
                            label=f"{phase} (n = {nn} {CONFIG.GPFA.BIN_SIZE_MS}-ms bins)",
                        )
                        ax.legend(loc="upper right")

                    except Exception as e:
                        logging.warn(e)

    for i_factor in range(len(axes)):
        ax_left = axes[i_factor][0]
        ax_right = axes[i_factor][1]
        mngs.plt.ax.sharey(ax_left, ax_right)
        mngs.plt.ax.set_n_ticks(ax_left)
        mngs.plt.ax.set_n_ticks(ax_right)
        mngs.plt.ax.sci_note(ax_left, y=True)
        ax_right.set_yticklabels([])

    return fig, axes


def kde_plot(lpath_NT, sample_type, znorm=False, symlog=False, unbias=False):
    # Loading
    NT = mngs.io.load(lpath_NT)

    if unbias:
        NT -= NT.min(axis=1, keepdims=True) + 1e-5

    if symlog:
        NT = mngs.gen.symlog(NT, 1e-5)

    parsed = utils.parse_lpath(lpath_NT)

    trials_info = mngs.io.load(
        mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
    ).set_index("trial_number")
    trials_info.index = trials_info.index.astype(int)

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
    fig, axes = kde_plot_8_factors(df, sample_type)
    return fig, axes


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

                fig, axes = kde_plot(
                    lpath_NT,
                    znorm=znorm,
                    symlog=symlog,
                    unbias=unbias,
                    sample_type=sample_type,
                )

                # Saving
                scale = "linear" if not symlog else "symlog"
                znorm_str = "NT" if not znorm else "NT_z"
                unbias_str = "unbiased" if unbias else "orig"

                spath_fig = (
                    f"./data/CA1/8_factors/{znorm_str}/{scale}/{unbias_str}/{sample_type}/"
                    + "_".join("-".join(item) for item in parsed.items())
                    + ".jpg"
                )
                fig.supxyt("Factor value", "KDE density", spath_fig)

                mngs.io.save(
                    axes.to_sigma(),
                    spath_fig.replace(".jpg", ".csv"),
                    from_cwd=True,
                )
                mngs.io.save(fig, spath_fig, from_cwd=True)


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
        alpha=0.75,
        fig_scale=2,
        font_size_legend=2,
    )

    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
