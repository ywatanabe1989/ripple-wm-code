#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-25 21:08:56 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_stats.py


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
from scipy.stats import rankdata
import joypy

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


import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


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


def to_rank(df, cols_NT):
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
    print(val_rank_max)
    print(val_rank_min)

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


def plot_violin(df, cols_NT):
    fig, ax = mngs.plt.subplots(figsize=(10, 6))
    df_plot = df[cols_NT].melt()
    sns.violinplot(
        data=df_plot[~df_plot.value.isna()],
        x="variable",
        y="value",
        ax=ax,
        inner="quartile",
    )
    ax.set_title("Distribution of Ranked Data")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Rank")
    return fig


def plot_joy(df, cols_NT):
    df_plot = df[cols_NT]
    fig, axes = joypy.joyplot(
        data=df_plot,
        colormap=plt.cm.viridis,
        title="Distribution of Ranked Data",
        labels=cols_NT,
        overlap=0.5,
        orientation="vertical",
    )
    plt.xlabel("Variables")
    plt.ylabel("Rank")
    return fig


def define_color(col):
    if "g_E-NT_E" in col:
        return "blue"
    elif "g_E-NT_R" in col:
        return "light_blue"
    elif "g_R-NT_E" in col:
        return "pink"
    elif "g_R-NT_R" in col:
        return "red"


def plot_density(df, cols_NT):
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in cols_NT:

        color = define_color(col)
        linestyle = "-" if "1.0" in col else "--"

        sns.kdeplot(
            data=df[col],
            ax=ax,
            vertical=False,
            label=col,
            color=CC[color],
            linestyle=linestyle,
        )
    ax.set_title("Distribution of Ranked Data")
    ax.set_ylabel("KDE")
    ax.set_xlabel("Ranked distance")
    ax.legend()
    return fig


def main():
    # Loading
    LPATHS = mngs.io.glob(
        "/mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_in_spheres_bp/*.csv"
    )
    df = pd.concat([mngs.io.load(lpath) for lpath in LPATHS]).reset_index()
    df = df.drop(columns=["index"])

    mngs.pd.merge_cols(df, "sub", "session", "roi")
    cols_NT = mngs.gen.search("NT", df.columns)[1]

    df = under_sample(df, cols_NT)

    # Smaller rank represents smaller distance
    df = to_rank(df, cols_NT)

    # Plotting
    # fig = plot_box(df, cols_NT)
    # fig = plot_ranked_data(df, cols_NT)
    # fig = plot_joy(df, cols_NT)
    fig = plot_density(df, cols_NT)
    plt.show()
    __import__("ipdb").set_trace()

    perform_pairwise_statistical_test(df, cols_NT)

    __import__("ipdb").set_trace()


# def get_task(col):
#     return '1.0 Match IN' if col.startswith('1.0') else '2.0 Mismatch OUT'

# def get_measure(col):
#     return 'NT_E' if 'NT_E' in col else 'NT_R'

# results['task1'] = results['col1'].apply(get_task)
# results['task2'] = results['col2'].apply(get_task)
# results['measure1'] = results['col1'].apply(get_measure)
# results['measure2'] = results['col2'].apply(get_measure)

# # Organize results
# organized_results = []
# for task_combo in ['Within Match IN', 'Within Mismatch OUT', 'Between Tasks']:
#     for measure_combo in ['NT_E vs NT_E', 'NT_R vs NT_R', 'NT_E vs NT_R']:
#         subset = results[
#             ((results['task1'] == results['task2']) if 'Within' in task_combo else (results['task1'] != results['task2'])) &
#             ((results['measure1'] == results['measure2'] == measure_combo[:4]) if 'vs' not in measure_combo else
#              (results['measure1'] != results['measure2']))
#         ].sort_values('p_value_corrected')
#         organized_results.append(subset)

# final_results = pd.concat(organized_results)
# print(final_results[['col1', 'col2', 'statistic', 'p_value_corrected']])

# fig, ax = mngs.plt.subplots()
# ax.sns_boxplot(
#     data=df,
#     x=
# )


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
