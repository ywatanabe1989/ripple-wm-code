#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 01:10:32 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/set_size_dependency_stats.py

"""Analyzes the relationship between set size and distance in MTL regions."""

"""Imports"""
import itertools
import re
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import utils
import xarray as xr
from scipy import stats
from scipy.linalg import norm

# from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import fdrcorrection
from typing import List, Dict, Any, Tuple

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def parse_variable(variable: str) -> pd.Series:
    """
    Parse variable string to extract MTL, phase combination, and set size.

    Example
    -------
    >>> parse_variable("ax_1_AMY_sns_boxplot_encoding-set_size_4")
    pd.Series({'mtl': 'AMY', 'phase_combi': 'encoding', 'set_size': 4})

    Parameters
    ----------
    variable : str
        Input string to parse

    Returns
    -------
    pd.Series
        Parsed information
    """
    pattern = r"ax_(\d+)_(\w+)_sns_boxplot_(\w+)-set_size_(\d+)"
    match = re.match(pattern, variable)
    if match:
        _, mtl, phase, set_size = match.groups()
        return pd.Series(
            {"mtl": mtl, "phase_combi": phase, "set_size": int(set_size)}
        )
    return pd.Series({"mtl": None, "phase_combi": None, "set_size": None})


def apply_fdr_correction(results: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FDR correction to p-values in the results DataFrame.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing statistical test results

    Returns
    -------
    pd.DataFrame
        DataFrame with added FDR-corrected p-values and stars
    """
    if "p_value" not in results.columns:
        return results

    _, fdr_corrected_pvals = fdrcorrection(results["p_value"])
    results["fdr_p_value"] = fdr_corrected_pvals
    results["fdr_stars"] = results["fdr_p_value"].apply(mngs.stats.p2stars)
    return results


def calculate_effect_size(groups: List[np.ndarray]) -> float:
    """
    Calculate effect size for given groups.

    Parameters
    ----------
    groups : List[np.ndarray]
        List of arrays containing data for each group

    Returns
    -------
    float
        Calculated effect size
    """
    group_means = [np.mean(group) for group in groups]
    group_vars = [np.var(group, ddof=1) for group in groups]
    pooled_sd = np.sqrt(np.mean(group_vars))
    return max(group_means) / pooled_sd if pooled_sd != 0 else np.nan


def run_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Kruskal-Wallis H-test on the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Results of Kruskal-Wallis H-test
    """
    results = {}
    for mtl in df["mtl"].unique():
        for phase_combi in df["phase_combi"].unique():
            subset = df[
                (df["mtl"] == mtl) & (df["phase_combi"] == phase_combi)
            ].dropna(subset=["distance"])
            groups = [
                subset[subset["set_size"] == size]["distance"]
                for size in [4, 6, 8]
            ]
            if all(len(group) > 0 for group in groups):
                statistic, p_value = stats.kruskal(*groups)
                effect_size = calculate_effect_size(groups)
                sample_sizes = [len(group) for group in groups]
                results[(mtl, phase_combi)] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "sample_sizes": sample_sizes,
                    "dof": len(groups) - 1,
                    "test_name": "Kruskal-Wallis H-test",
                }
    return pd.DataFrame(results).T


def run_brunner_munzel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Brunner-Munzel test on the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Results of Brunner-Munzel test
    """
    results = {}
    for mtl in df["mtl"].unique():
        for phase_combi in df["phase_combi"].unique():
            subset = df[
                (df["mtl"] == mtl) & (df["phase_combi"] == phase_combi)
            ].dropna(subset=["distance"])
            for size1, size2 in itertools.combinations([4, 6, 8], 2):
                x = subset[subset["set_size"] == size1]["distance"]
                y = subset[subset["set_size"] == size2]["distance"]
                if len(x) > 0 and len(y) > 0:
                    statistic, p_value = stats.brunnermunzel(x, y)
                    effect_size = abs(np.mean(x) - np.mean(y)) / np.sqrt(
                        (np.var(x) + np.var(y)) / 2
                    )
                    results[(mtl, phase_combi, f"{size1}vs{size2}")] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "sample_sizes": [len(x), len(y)],
                        "dof": len(x) + len(y) - 2,
                        "test_name": "Brunner-Munzel test",
                    }
    return pd.DataFrame(results).T


def run_spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Spearman correlation test on the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Results of Spearman correlation test
    """
    results = {}
    for mtl in df["mtl"].unique():
        for phase_combi in df["phase_combi"].unique():
            subset = df[
                (df["mtl"] == mtl) & (df["phase_combi"] == phase_combi)
            ].dropna(subset=["distance", "set_size"])
            if len(subset) > 1:
                corr, p_value = stats.spearmanr(
                    subset["set_size"], subset["distance"]
                )
                results[(mtl, phase_combi)] = {
                    "correlation": corr,
                    "p_value": p_value,
                    "sample_size": len(subset),
                    "dof": len(subset) - 2,
                    "test_name": "Spearman correlation",
                }
    return pd.DataFrame(results).T


def run_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Run all statistical tests on the data and apply FDR correction.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing results of all statistical tests with FDR correction
    """
    stats_agg = {}
    for test, run_fn in [
        ("kw", run_kruskal_wallis),
        ("bm", run_brunner_munzel),
        ("corr", run_spearman_correlation),
    ]:
        stats = run_fn(df)
        stats = apply_fdr_correction(stats)
        stats["stars"] = stats["p_value"].apply(mngs.stats.p2stars)
        stats = mngs.pd.round(stats)
        stats_agg[test] = stats
    return stats_agg


def process_data(
    file_path: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Process the data and run statistical tests.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Processed dataframe, linear stats results, and log10 stats results
    """
    df = mngs.io.load(file_path).melt()
    parsed = df["variable"].apply(parse_variable)
    df = pd.concat([df, parsed], axis=1)
    df = df.drop(columns="variable")
    df = df.rename(columns={"value": "distance"})

    linear_stats = run_stats(df)
    df["distance"] = np.log10(df["distance"] + 1e-5)
    log10_stats = run_stats(df)

    return df, linear_stats, log10_stats

def get_mtl(roi):
    for mtl in CONFIG.ROI.MTL.keys():
        if any([roi == subregion for subregion in CONFIG.ROI.MTL[mtl]]):
            return mtl

def load_NT_dist_between_gs_trial_all():
    LPATHS = mngs.gen.glob(CONFIG.PATH.NT_DIST_BETWEEN_GS_TRIAL)

    dfs = mngs.gen.listed_dict()
    for lpath in LPATHS:
        # Loading
        df = mngs.io.load(lpath)
        # # to rank if you lik

        # MTL
        parsed = utils.parse_lpath(lpath)
        mtl = get_mtl(parsed["roi"])

        # Aggregation
        dfs[mtl].append(df)

    # Formatting
    for k,v in dfs.items():
        dfs[k] = pd.concat(v)
        dfs[k]["MTL"] = k
    df = pd.concat(dfs.values())
    return df


def main():
    df = load_NT_dist_between_gs_trial_all()

    # Formatting
    df = mngs.pd.melt_cols(df, cols=["FE", "FM", "FR", "EM", "ER", "MR"])
    df = df.rename(columns={"variable": "phase_combination", "value": "distance"})

    # Plotting
    MTL_REGIONS = CONFIG.ROI.MTL.keys()

    fig, axes = mngs.plt.subplots(ncols=len(MTL_REGIONS), nrows=len(CONFIG.MATCHES_STR))
    for i_mtl, mtl in enumerate(MTL_REGIONS):
        for i_match, match_str in enumerate(CONFIG.MATCHES_STR.keys()):
            ax = axes[i_mtl, i_match]
            indi_MTL = df.MTL == mtl
            indi_match = np.full(len(df), True) if match_str == "all" else df.match == int(match_str)

            ax.sns_boxplot(
                data=df[indi_MTL * indi_match],
                x="phase_combination",
                y="distance",
            )
    mngs.io.save(fig, "fig.jpg")


    # condi = {
    #     "MTL": df.MTL.unique(),
    #     "set_size": CONFIG.SET_SIZES,
    #     "match": CONFIG.MATCHES_STR.keys(),
    #     "phase_combination": df.phase_combination.unique(),
    # }

    # # ll = pd.DataFrame([list(ll) for ll in mngs.gen.list_module_contents(mngs)])
    # # mngs.gen.search("grid", ll[1])
    # for cc in mngs.gen.yield_grids(condi):
    #     if cc["MTL"]
    #     print(cc)
    # __import__("ipdb").set_trace()



    # # Plotting
    # fig, ax = mngs.plt.subplots()
    # # ax.box(
    # #     df,
    # #     x="MTL",
    # #     y=
    # # )

    # # """Main function to process data and save results."""
    # # df, linear_stats, log10_stats = process_data(
    # #     "./scripts/NT/distance/between_gs/set_size_dependency_plot_box/box.csv"
    # # )

    # # for key, value in linear_stats.items():
    # #     mngs.io.save(value, f"stats_{key}.csv")

    # # for key, value in log10_stats.items():
    # #     mngs.io.save(value, f"stats_{key}_log10.csv")


if __name__ == "__main__":
    np.random.seed(42)
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
