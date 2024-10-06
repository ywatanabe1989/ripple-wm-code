#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 18:27:52 (ywatanabe)"
# set_size_dependency_stats.py

"""
Functionality:
    - Analyzes the relationship between set size and distance in MTL regions
Input:
    - Neural trajectory distance data between ground states
Output:
    - Statistical test results for set size dependency
Prerequisites:
    - mngs package, scipy, pandas, numpy
"""

"""Imports"""
import os
import re
import sys
import itertools
from typing import List, Tuple, Dict, Any, Union, Sequence, Optional, Literal
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import xarray as xr
from itertools import combinations
from copy import deepcopy
import mngs
from functools import partial
try:
    from scripts import utils
except ImportError:
    pass

ArrayLike = Union[
    List,
    Tuple,
    np.ndarray,
    pd.Series,
    pd.DataFrame,
    xr.DataArray,
    torch.Tensor,
]

"""Parameters"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def load_NT_dist_between_gs_trial_all() -> pd.DataFrame:
    """
    Load neural trajectory distance data between ground states for all trials and MTL regions.

    Returns:
    --------
    pd.DataFrame
        Combined dataframe of neural trajectory distances for all MTL regions.
    """

    def roi2mtl(roi: str) -> str:
        for mtl, subregions in CONFIG.ROI.MTL.items():
            if roi in subregions:
                return mtl
        return None

    LPATHS = mngs.gen.glob(CONFIG.PATH.NT_DIST_BETWEEN_GS_TRIAL)

    phase_combinations = [f"{p1[0]}{p2[0]}" for p1, p2 in combinations(CONFIG.PHASES.keys(), 2)]

    dfs = mngs.gen.listed_dict()
    for lpath in LPATHS:
        df = mngs.io.load(lpath)
        # Z norm
        df[phase_combinations] = mngs.gen.to_z(df[phase_combinations], axis=(0,1))
        parsed = utils.parse_lpath(lpath)
        if parsed["session"] not in CONFIG.SESSION.FIRST_TWO:
            continue
        mtl = roi2mtl(parsed["roi"])
        if mtl:
            dfs[mtl].append(df)

    for mtl, df_list in dfs.items():
        dfs[mtl] = pd.concat(df_list)
        dfs[mtl]["MTL"] = mtl

    df = pd.concat(dfs.values())

    return df


def calculate_effect_size(groups: List[ArrayLike]) -> float:
    """
    Calculate effect size for Kruskal-Wallis test.

    Parameters:
    -----------
    groups : List[ArrayLike]
        List of groups to compare.

    Returns:
    --------
    float
        Effect size.
    """
    group_means = [np.mean(group) for group in groups]
    group_vars = [np.var(group, ddof=1) for group in groups]
    pooled_sd = np.sqrt(np.mean(group_vars))
    return max(group_means) / pooled_sd if pooled_sd != 0 else np.nan


def run_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
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
                    subset[subset["set_size"] == size]["distance"]
                    for size in [4, 6, 8]
                ]

                if all(len(group) > 0 for group in groups):
                    statistic, p_value = stats.kruskal(*groups)
                    p_value = float(p_value)
                    effect_size = calculate_effect_size(groups)
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
                    xx = subset[subset["set_size"] == size1]["distance"]
                    yy = subset[subset["set_size"] == size2]["distance"]
                    if len(xx) > 0 and len(yy) > 0:
                        statistic, p_value = stats.brunnermunzel(xx, yy)
                        effect_size = abs(np.mean(xx) - np.mean(yy)) / np.sqrt(
                            (np.var(xx) + np.var(yy)) / 2
                        )
                        results.append(
                            {
                                "MTL": mtl,
                                "phase_combination": phase_combi,
                                "match": match,
                                "comparison": f"{size1}vs{size2}",
                                "statistic": statistic,
                                "p_value": p_value,
                                "effect_size": effect_size,
                                "sample_sizes": [len(xx), len(yy)],
                                "dof": len(xx) + len(yy) - 2,
                                "test_name": "Brunner-Munzel test",
                            }
                        )
    return pd.DataFrame(results), None


def run_corr_test(df: pd.DataFrame, test: Literal["spearman", "pearson"]) -> pd.DataFrame:
    agg_stats = []
    agg_surrogate = []
    for mtl in df["MTL"].unique():
        for phase_combi in df["phase_combination"].unique():
            for match in df["match"].unique():
                subset = df[
                    (df["MTL"] == mtl)
                    & (df["phase_combination"] == phase_combi)
                    & (df["match"] == match)
                ].dropna(subset=["distance", "set_size"]).copy()
                if len(subset) > 1:
                    stats = mngs.stats.corr_test(
                        subset["set_size"], subset["distance"], test=test
                    )
                    stats["MTL"] = mtl
                    stats["phase_combination"] = phase_combi
                    stats["match"] = match
                    surrogate = stats.pop("surrogate")
                    agg_stats.append(pd.Series(stats).T)
                    agg_surrogate.append(surrogate)

    stats = pd.concat(agg_stats, axis=1).T
    surrogate = np.vstack(agg_surrogate)
    __import__("ipdb").set_trace()
    mngs.io.save(surrogate, f"surrogate_{test}.npy")
    return stats, surrogate



def run_stats(df: pd.DataFrame, scale: Optional[str] = "linear") -> Dict[str, pd.DataFrame]:
    """
    Run all statistical tests on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    scale : Optional[str], default "linear"
        Scale of the distance data. Can be "linear" or "log10".

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing results of all statistical tests.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'distance': np.random.rand(100),
    ...     'set_size': np.random.choice([4, 6, 8], 100),
    ...     'match': np.random.choice(['Match IN', 'Mismatch OUT'], 100),
    ...     'MTL': np.random.choice(['HIP', 'EC', 'AMY'], 100),
    ...     'phase_combination': np.random.choice(['FE', 'FM', 'FR', 'EM', 'ER', 'MR'], 100)
    ... })
    >>> results = run_stats(df)
    >>> print(list(results.keys()))
    ['kw', 'bm', 'corr']
    """
    if scale == "log10":
        df = df.copy()
        df["distance"] = np.log10(df["distance"] + 1e-5)

    stats_agg = {}
    for test, run_fn in [
        ("kw", run_kruskal_wallis),
        ("bm", run_brunner_munzel),
        ("corr-spearman", partial(run_corr_test, test="spearman")),
        ("corr-pearson", partial(run_corr_test, test="pearson")),
    ]:
        # Stats
        _df = deepcopy(df)
        stats, _surrogate = run_fn(_df)
        stats = stats.dropna(subset=["p_value"])
        stats = mngs.stats.fdr_correction(stats)
        stats = mngs.stats.p2stars(stats)
        stats = mngs.pd.round(stats)

        # Decoding match
        match_mapper = deepcopy(CONFIG.MATCHES_STR)
        match_mapper["-1"] = match_mapper.pop("all")
        stats["match"] = stats["match"].astype(float).astype(int).astype(str).replace(match_mapper)

        # Sorting
        stats = sort_columns(stats)
        stats_agg[test] = stats

    return stats_agg



def sort_columns(stats):
    if stats.empty:
        return stats

    stats = stats.reset_index().drop(columns="index")

    stats = stats.set_index(["MTL", "match", "phase_combination"]).reset_index()

    mtl_order = ["HIP", "EC", "AMY"]
    match_order = ["Match ALL", "Match IN", "Mismatch OUT"]
    phase_order = ['FE', 'FM', 'FR', 'EM', 'ER', 'MR']

    if "phase_combination" in stats.columns:
        indi = np.hstack([mngs.gen.search(key, stats["phase_combination"])[0] for key in phase_order])
        stats = stats.iloc[indi]

    indi = np.hstack([mngs.gen.search(key, stats["match"])[0] for key in match_order])
    stats = stats.iloc[indi]

    # stats = stats.sort_values(["MTL", "match", "phase_combination"])
    indi = np.hstack([mngs.gen.search(key, stats["MTL"])[0] for key in mtl_order])
    stats = stats.iloc[indi]



    return stats

def main():
    df = load_NT_dist_between_gs_trial_all()
    df = mngs.pd.melt_cols(df, cols=["FE", "FM", "FR", "EM", "ER", "MR"])
    df = df.rename(
        columns={"variable": "phase_combination", "value": "distance"}
    )

    # Adds Match ALL
    df_match_all = deepcopy(df)
    df_match_all["match"] = -1 # "Match ALL"
    df = pd.concat([df, df_match_all])

    # Run statistical tests
    linear_stats = run_stats(df, scale="linear")

    # Log-transform the distance and run tests again
    log10_stats = run_stats(df, scale="log10")

    # Save results
    for key, value in linear_stats.items():
        mngs.io.save(value, f"stats_{key}.csv")

    for key, value in log10_stats.items():
        mngs.io.save(value, f"stats_{key}_log10.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True, np=np
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
