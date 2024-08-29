#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-26 11:46:57 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_stats.py


"""
This script does XYZ.
"""


"""
Imports
"""
import importlib
import logging
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
from scipy.stats import rankdata
from tqdm import tqdm

# import joypy
mngs.pd.ignore_SettingWithCopyWarning()
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


from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def under_sample(df, cols_NT):
    # Balance the number of samples
    n_min = np.inf
    for col in cols_NT:
        nn = (~df[col].isna()).sum()
        n_min = min(n_min, nn)

    for col in cols_NT:
        non_nan_mask = ~df[col].isna()
        indi = non_nan_mask[non_nan_mask].index
        indi_balanced = np.random.permutation(indi)[:n_min]
        tmp = df[col][indi_balanced]
        df[col] = np.nan
        df.loc[indi_balanced, col] = tmp

    for col in cols_NT:
        assert (~df[col].isna()).sum() == n_min

    return df


def NT_to_rank(df, cols_NT):
    df_info = df[list(set(df.columns) - set(cols_NT))].copy()  # (9860, 4)
    df_NT = df[cols_NT].copy()  # (9860, 8)
    df_rank = np.nan * df_NT.copy()  # (9860, 8)

    for col_session in df["sub_session_roi"].unique():
        indi_session = df["sub_session_roi"] == col_session

        # Slice the session data
        NT_session = np.array(df_NT)[indi_session]  # (1000, 8)
        rank_session = np.full(NT_session.shape, np.nan)  # (1000, 8)

        # NT to rank
        non_nan_mask = ~np.isnan(NT_session)
        rank_session[non_nan_mask] = rankdata(
            NT_session[non_nan_mask]
        )  # , method="average"

        # Buffering
        df_rank[indi_session] = rank_session

    val_rank_max = np.array(df_NT)[np.where(df_rank == np.nanmax(df_rank))]
    val_rank_min = np.array(df_NT)[np.where(df_rank == np.nanmin(df_rank))]
    # print(val_rank_max)
    # print(val_rank_min)

    return pd.concat([df_info, df_rank], axis=1)


def perform_pairwise_statistical_test(df, cols_NT):

    results = []
    for col1, col2 in combinations(cols_NT, 2):

        x1 = df[col1]
        x2 = df[col2]

        x1 = x1[~np.isnan(x1)]
        x2 = x2[~np.isnan(x2)]

        statistic, p_value = stats.wilcoxon(x1, x2)

        result = {
            "col1": col1,
            "col2": col2,
            "statistic": statistic,
            "p_val_unc": p_value,
        }

        results.append(pd.Series(result))

    results = pd.DataFrame(results)
    results["p_val"] = (results["p_val_unc"] * len(results)).clip(upper=1.0)

    results["statistic"] = results["statistic"].astype(int)
    results["p_val_unc"] = results["p_val_unc"].round(3)
    results["p_val"] = results["p_val"].round(3)

    print(results.sort_values(["p_val"]))


# def plot_box(df, cols_NT):
#     fig, ax = mngs.plt.subplots()
#     df_plot = df[cols_NT]
#     df_plot = df_plot.melt()
#     ax.sns_boxplot(
#         data=df_plot[~df_plot.value.isna()],
#         x="variable",
#         y="value",
#     )
#     return fig


# # Working
# def plot_violin(df, cols_NT):
#     fig, ax = mngs.plt.subplots(figsize=(10, 6))
#     df_plot = df[cols_NT].melt()
#     sns.violinplot(
#         data=df_plot[~df_plot.value.isna()],
#         x="variable",
#         y="value",
#         ax=ax,
#         inner="quartile",
#     )
#     ax.set_title("Distribution of Ranked Data")
#     ax.set_xlabel("Variables")
#     ax.set_ylabel("Rank")
#     return fig


def plot_violin(df, cols_NT):
    fig, ax = mngs.plt.subplots()
    df_plot = df[cols_NT].melt()

    ax.sns_violinplot(
        data=df_plot[~df_plot.value.isna()],
        x="variable",
        y="value",
        inner="quartile",
    )
    return fig


def plot_joy(df, cols_NT):
    df_plot = df[cols_NT]
    fig, ax = mngs.plt.subplots()
    ax.joyplot(df[cols_NT])
    # fig, axes = joypy.joyplot(
    #     data=df_plot,
    #     colormap=plt.cm.viridis,
    #     title="Distribution of Ranked Data",
    #     labels=cols_NT,
    #     overlap=0.5,
    #     orientation="vertical",
    # )
    # plt.xlabel("Variables")
    # plt.ylabel("Rank")
    return fig


# # working
# def plot_kde(df, cols_NT):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for col in cols_NT:

#         color = define_color(col)
#         linestyle = "-" if "1.0" in col else "--"

#         sns.kdeplot(
#             data=df[col],
#             ax=ax,
#             vertical=False,
#             label=col,
#             color=CC[color],
#             linestyle=linestyle,
#         )
#     ax.set_title("Distribution of Ranked Data")
#     ax.set_ylabel("KDE")
#     ax.set_xlabel("Ranked distance")
#     ax.legend()
#     return fig


def define_color(col):
    if "g_E-NT_E" in col:
        return "blue"
    elif "g_E-NT_R" in col:
        return "light_blue"
    elif "g_R-NT_E" in col:
        return "pink"
    elif "g_R-NT_R" in col:
        return "red"


def plot_kde(df, cols_NT):
    # Data Preparation
    df_melt = df[cols_NT].melt()
    df_melt = df_melt.rename(
        columns={"variable": "Group", "value": "Ranked Distance"}
    )
    df_melt = df_melt.dropna()

    df_melt["Group"] = df_melt["Group"].replace(
        {"1.0-": "Match IN: ", "2.0-": "Mismatch OUT: "},  # , "-NT": "--NT"
        regex=True,
    )

    # hue_colors = {
    #     "Match IN: $g_E-NT_E$": CC["blue"],
    #     "Match IN: $g_E-NT_R$": CC["light_blue"],
    #     "Match IN: $g_R-NT_E$": CC["pink"],
    #     "Match IN: $g_R-NT_R$": CC["red"],
    #     "Mismatch OUT: $g_E-NT_E$": CC["blue"],
    #     "Mismatch OUT: $g_E-NT_R$": CC["light_blue"],
    #     "Mismatch OUT: $g_R-NT_E$": CC["pink"],
    #     "Mismatch OUT: $g_R-NT_R$": CC["red"],
    # }

    # how can I replace CC["*"], to "*" on emacs?

    hue_colors = {
        "Match IN: $g_E-NT_E$": CC["blue"],
        "Match IN: $g_E-NT_R$": CC["light_blue"],
        "Match IN: $g_R-NT_E$": CC["pink"],
        "Match IN: $g_R-NT_R$": CC["red"],
        "Mismatch OUT: $g_E-NT_E$": CC["blue"],
        "Mismatch OUT: $g_E-NT_R$": CC["light_blue"],
        "Mismatch OUT: $g_R-NT_E$": CC["pink"],
        "Mismatch OUT: $g_R-NT_R$": CC["red"],
    }
    hue_order = list(hue_colors.keys())
    hue_line_styles = {
        "Match IN: $g_E-NT_E$": "-",
        "Match IN: $g_E-NT_R$": "-",
        "Match IN: $g_R-NT_E$": "-",
        "Match IN: $g_R-NT_R$": "-",
        "Mismatch OUT: $g_E-NT_E$": "--",
        "Mismatch OUT: $g_E-NT_R$": "--",
        "Mismatch OUT: $g_R-NT_E$": "--",
        "Mismatch OUT: $g_R-NT_R$": "--",
    }

    # Main
    fig, ax = mngs.plt.subplots()
    ax.sns_kdeplot(
        data=df_melt,
        x="Ranked Distance",
        hue="Group",
        hue_order=hue_order,
        hue_colors=hue_colors,
    )
    # fig, ax = plt.subplots()
    # sns.kdeplot(
    #     data=df_melt,
    #     x="Ranked Distance",
    #     hue="Group",
    #     hue_order=hue_order,
    #     palette=hue_colors,
    #     ax=ax,
    # )

    # Apply line styles
    for line, group in zip(ax.lines, hue_order):
        line.set_linestyle(hue_line_styles[group])

    return fig


def main():
    # Loading
    LPATHS = mngs.io.glob("./scripts/NT/distance/plot_dists/*.csv")
    df = pd.concat([mngs.io.load(lpath) for lpath in LPATHS]).reset_index()
    df = df.drop(columns=["index"])

    mngs.pd.merge_cols(df, "sub", "session", "roi")
    cols_NT = mngs.gen.search("NT", df.columns)[1]

    df = under_sample(df, cols_NT)

    # Smaller rank represents smaller distance
    df = NT_to_rank(df, cols_NT)

    # Plotting
    # fig = plot_box(df, cols_NT)
    # fig = plot_joy(df, cols_NT)
    fig = plot_kde(df, cols_NT)
    plt.show()
    __import__("ipdb").set_trace()

    perform_pairwise_statistical_test(df, cols_NT)

    __import__("ipdb").set_trace()


# def perform_kruskal_wallis(df, cols_NT):
#     results = []
#     for col in cols_NT:
#         __import__("ipdb").set_trace()
#         groups = [group[~np.isnan(group)] for _, group in df.groupby('session')[col] if not group.empty]
#         print(groups)

#         h_statistic, p_value = stats.kruskal(*groups)
#         results.append({
#             'column': col,
#             'H-statistic': h_statistic,
#             'p-value': p_value
#         })
#     return pd.DataFrame(results)


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
        line_width=1.0,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
