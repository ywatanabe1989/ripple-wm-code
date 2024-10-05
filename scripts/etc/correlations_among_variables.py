#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-04 21:13:54 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/etc/correlations_among_variables.py

"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""

"""Imports"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import mngs

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
import logging
from tqdm import tqdm
import xarray as xr

try:
    from scripts import utils
except:
    pass

"""Aliases"""
pt = print

"""Configs"""
# CONFIG = mngs.gen.load_configs()

"""Parameters"""

"""Functions & Classes"""


def add_previous_trial_parameters(rip):
    rip = rip.reset_index()
    for ca1 in CONFIG.ROI.CA1:
        TI = mngs.io.load(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1))
        for k, v in ca1.items():
            TI[k] = v
        TI = TI.rename(columns={"sub": "subject"})
        rip = _add_previous_trial_parameters(rip, TI)

    return rip


def _add_previous_trial_parameters(rips_df, trials_df):
    rips_df["previous_set_size"] = np.nan
    rips_df["previous_correct"] = np.nan
    rips_df["previous_responset_time"] = np.nan
    for i_rip, (_, rip) in enumerate(rips_df.iterrows()):
        if rip.trial_number == 1:
            continue

        indi_subject = trials_df.subject == rip.subject
        indi_session = trials_df.session == rip.session
        indi_trial_prev = trials_df.trial_number == rip.trial_number - 1
        indi = indi_subject * indi_session * indi_trial_prev

        if indi.sum():
            rips_df.loc[i_rip, "previous_set_size"] = float(
                trials_df[indi].set_size.iloc[0]
            )
            rips_df.loc[i_rip, "previous_correct"] = float(
                trials_df[indi].correct.iloc[0]
            )
            rips_df.loc[i_rip, "previous_response_time"] = float(
                trials_df[indi].response_time.iloc[0]
            )
    return rips_df


def if_mean(firing_pattern):
    fp = np.array(firing_pattern)
    if isinstance(fp, float):
        return fp
    else:
        return fp.any().mean()


def transform_variables(rip):
    try:
        rip["mean_n_spikes_per_unit"] = rip["firing_pattern"].apply(if_mean)
        rip["log10(duration_ms)"] = np.log(rip["duration_s"] * 1e3 + 1e-5)
        rip["log10(peak_amp_sd)"] = np.log(rip["peak_amp_sd"] + 1e-5)
        rip["|vER|"] = rip["vER"].apply(mngs.linalg.nannorm)
        rip["|vOR|"] = rip["vOR"].apply(mngs.linalg.nannorm)
        rip["|vSWR_NT|"] = rip["vSWR_NT"].apply(mngs.linalg.nannorm)
        rip["|vSWR_JUMP|"] = rip["vSWR_JUMP"].apply(mngs.linalg.nannorm)
    except Exception as e:
        print(e)
        __import__("ipdb").set_trace()
    return rip


def pairwise_correlation(df):
    columns = df.columns
    n_cols = len(columns)
    corr_matrix = pd.DataFrame(index=columns, columns=columns)

    for i in range(n_cols):
        for j in range(i, n_cols):
            col1, col2 = columns[i], columns[j]
            mask = ~(df[col1].isna() | df[col2].isna())
            if mask.sum() > 1:  # Ensure at least 2 non-NaN values
                corr = df.loc[mask, col1].corr(df.loc[mask, col2])
                corr_matrix.loc[col1, col2] = corr
                corr_matrix.loc[col2, col1] = corr
            else:
                corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = (
                    np.nan
                )

    return corr_matrix


# def pairwise_correlation(df):
#     columns = df.columns
#     n_cols = len(columns)
#     corr_matrix = pd.DataFrame(index=columns, columns=columns)

#     for i in range(n_cols):
#         for j in range(i, n_cols):
#             col1, col2 = columns[i], columns[j]
#             mask = ~(df[col1].isna() | df[col2].isna())
#             corr = df.loc[mask, col1].corr(df.loc[mask, col2])
#             corr_matrix.loc[col1, col2] = corr
#             corr_matrix.loc[col2, col1] = corr

#     return corr_matrix

mapper = {
    "previous_set_size": "Prev. set size",
    "previous_correct": "Prev. correct",
    "previous_response_time": "Prev. response time [s]",
    "set_size": "Set size",
    "correct": "Correct",
    "response_time": "Response time [s]",
    "match": "Match",
    "log10(duration_ms)": "Log_{10}(Duration [ms])",
    "log10(peak_amp_sd)": "Log_{10}(Peak Amp. [SD])",
    "rel_peak_pos": "Rel. peak pos",
    "mean_n_spikes_per_unit": "Unit participation ratio",
    "|vER|": "| " + mngs.tex.to_vec("g_{E}g_{R}") + " |",
    "|vOR|": "| " + mngs.tex.to_vec("g_{R}") + " |",
    "|vSWR_NT|": "| " + mngs.tex.to_vec("SWR_{time}") + " |",
    "|vSWR_JUMP|": "| " + mngs.tex.to_vec("SWR_{jump}") + " |",
}

def main():

    import utils

    rip_p, rip_m = utils.load_ripples(with_NT=True)
    rip_p, rip_m = map(add_previous_trial_parameters, [rip_p, rip_m])
    rip_p, rip_m = map(transform_variables, [rip_p, rip_m])

    # Correlation among variables
    corr_df = rip_p[
        [
            "previous_set_size",
            "previous_correct",
            "previous_response_time",
            "set_size",
            "correct",
            "response_time",
            "match",
            "log10(duration_ms)",
            "log10(peak_amp_sd)",
            "rel_peak_pos",
            "mean_n_spikes_per_unit",
            "|vER|",
            "|vOR|",
            "|vSWR_NT|",
            "|vSWR_JUMP|",
            # "radian_NT",
            # "radian_peak",
        ]
    ]
    corr_df = corr_df.rename(columns=mapper, index=mapper)

    # corr_matrix = pairwise_correlation(corr_df)
    corr_matrix = corr_df.corr()

    # Plotting
    fig, ax = mngs.plt.subplots()
    ax.sns_heatmap(corr_matrix, annot=True, cmap="vlag", vmin=-1, vmax=1, xyz=True)
    # fig.to_sigma()
    # fig.axes
    # hasattr(fig.axes, "to_sigma")
    # ax.to_sigma()
    mngs.io.save(fig, "heatmap.jpg")


    # # heatmap
    # fig, ax = plt.subplots(figsize=(6.4*2, 4.8*2))
    # sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="vlag", vmin=-1, vmax=1)
    # mngs.io.save(fig, "./tmp/figs/heatmap/correlations.tif")
    # # plt.show()

    # # # variables pair
    # # g = sns.pairplot(corr_matrix, height=2.5)
    # # mngs.io.save(g, "./tmp/figs/pair/variables.png")
    # # # plt.show()

    # # # effect on duration
    # # fig, ax = plt.subplots()
    # # sns.boxplot(
    # #     data=rips_df[(rips_df.previous_set_size == 6)],  # + (rips_df.set_size == 8)],
    # #     x="phase",
    # #     y="log10(duration_ms)",
    # #     hue="set_size",
    # #     order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
    # #     ax=ax,
    # # )
    # # # ax.set_ylim(0, 200)
    # # plt.show()

    # pass


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        fig_scale=4,
        font_size_base=6,
        font_size_title=6,
        font_size_axis_label=6,
        font_size_tick_label=6,
        font_size_legend=6,
    )

    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
