#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 23:29:25 (ywatanabe)"
# _set_size_dependency_stats_helper.py

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

from typing import List, Tuple, Dict, Any, Union, Sequence, Literal
from collections.abc import Iterable
from scipy import stats
import itertools

ArrayLike = Union[
    List,
    Tuple,
    np.ndarray,
    pd.Series,
    pd.DataFrame,
    xr.DataArray,
    torch.Tensor,
]
from functools import partial


try:
    from scripts import utils
except:
    pass

"""Parameters"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""

# def calculate_effect_size(groups: List[ArrayLike]) -> float:
#     """
#     Calculate effect size for Kruskal-Wallis test.

#     Parameters:
#     -----------
#     groups : List[ArrayLike]
#         List of groups to compare.

#     Returns:
#     --------
#     float
#         Effect size.
#     """
#     group_means = [np.mean(group) for group in groups]
#     group_vars = [np.var(group, ddof=1) for group in groups]
#     pooled_sd = np.sqrt(np.mean(group_vars))
#     return max(group_means) / pooled_sd if pooled_sd != 0 else np.nan


def run_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_effect_size_kruskal_wallis(groups: List[ArrayLike]) -> float:
        """
        Calculate epsilon-squared (ε²) effect size for Kruskal-Wallis test.

        Parameters:
        -----------
        groups : List[ArrayLike]
            List of groups to compare.

        Returns:
        --------
        float
            Epsilon-squared effect size.
        """
        n = sum(len(group) for group in groups)
        k = len(groups)
        H = stats.kruskal(*groups)[0]
        return (H - k + 1) / (n - k)

    results = []
    for mtl in df["MTL"].unique():
        for phase_combi in df["phase_combination"].unique():
            for match in df["match"].unique():
                subset = df[
                    (df["MTL"] == mtl)
                    & (df["phase_combination"] == phase_combi)
                    & (df["match"] == match)
                ].dropna(subset=["distance"])

                groups = [
                    subset[subset["set_size"] == str(size)]["distance"]
                    for size in [4, 6, 8]
                ]

                if all(len(group) > 0 for group in groups):
                    statistic, p_value = stats.kruskal(*groups)
                    p_value = float(p_value)
                    effect_size = calculate_effect_size_kruskal_wallis(groups)
                    sample_sizes = [len(group) for group in groups]
                    results.append(
                        {
                            "MTL": mtl,
                            "phase_combination": phase_combi,
                            "match": match,
                            "statistic": statistic,
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "sample_sizes": sample_sizes,
                            "dof": len(groups) - 1,
                            "test_name": "Kruskal-Wallis H-test",
                        }
                    )
    return pd.DataFrame(results), None


def run_brunner_munzel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Brunner-Munzel test on the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the data.

    Returns:
    --------
    pd.DataFrame
        Results of the Brunner-Munzel test.
    """
    # def calculate_cles(x, y):
    #     # For the Brunner-Munzel test, you can use the Common Language Effect Size (CLES) instead of the current calculation:
    #     return np.mean([1 if xi > yi else 0.5 if xi == yi else 0 for xi in x for yi in y])

    results = []
    for mtl in df["MTL"].unique():
        for phase_combi in df["phase_combination"].unique():
            for match in df["match"].unique():
                subset = df[
                    (df["MTL"] == mtl)
                    & (df["phase_combination"] == phase_combi)
                    & (df["match"] == match)
                ].dropna(subset=["distance"])
                for size1, size2 in itertools.combinations([4, 6, 8], 2):
                    xx = subset[subset["set_size"] == str(size1)]["distance"]
                    yy = subset[subset["set_size"] == str(size2)]["distance"]
                    if len(xx) > 0 and len(yy) > 0:
                        bm_out = mngs.stats.brunner_munzel_test(xx, yy)
                        results.append(
                            {
                                "MTL": mtl,
                                "phase_combination": phase_combi,
                                "match": match,
                                "comparison": f"{size1}vs{size2}",
                                **bm_out,
                            }
                        )
    return pd.DataFrame(results), None


def run_corr_test(
    df: pd.DataFrame, test: Literal["spearman", "pearson"]
) -> pd.DataFrame:
    agg_stats = []
    agg_surrogate = []
    for mtl in df["MTL"].unique():
        for phase_combi in df["phase_combination"].unique():
            for match in df["match"].unique():
                subset = (
                    df[
                        (df["MTL"] == mtl)
                        & (df["phase_combination"] == phase_combi)
                        & (df["match"] == match)
                    ]
                    .dropna(subset=["distance", "set_size"])
                    .copy()
                )
                if len(subset) > 1:
                    stats = mngs.stats.corr_test(
                        subset["set_size"],
                        subset["distance"],
                        n_perm=10_000,
                        test=test,
                    )
                    stats["MTL"] = mtl
                    stats["phase_combination"] = phase_combi
                    stats["match"] = match
                    surrogate = stats.pop("surrogate")
                    agg_stats.append(pd.Series(stats).T)
                    agg_surrogate.append(surrogate)

    stats = pd.concat(agg_stats, axis=1).T
    surrogate = np.vstack(agg_surrogate)
    return stats, surrogate


def sort_columns(stats):
    if stats.empty:
        return stats

    stats = stats.reset_index().drop(columns="index")

    stats = stats.set_index(
        ["MTL", "match", "phase_combination"]
    ).reset_index()

    mtl_order = ["HIP", "EC", "AMY"]
    match_order = ["Match ALL", "Match IN", "Mismatch OUT"]
    phase_order = ["FE", "FM", "FR", "EM", "ER", "MR"]

    if "phase_combination" in stats.columns:
        indi = np.hstack(
            [
                mngs.gen.search(key, stats["phase_combination"])[0]
                for key in phase_order
            ]
        )
        stats = stats.iloc[indi]

    indi = np.hstack(
        [mngs.gen.search(key, stats["match"])[0] for key in match_order]
    )
    stats = stats.iloc[indi]

    # stats = stats.sort_values(["MTL", "match", "phase_combination"])
    indi = np.hstack(
        [mngs.gen.search(key, stats["MTL"])[0] for key in mtl_order]
    )
    stats = stats.iloc[indi]

    stats = stats.reset_index().drop(columns="index")

    return stats
