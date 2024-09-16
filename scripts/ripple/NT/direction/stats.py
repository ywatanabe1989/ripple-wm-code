#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-16 20:36:12 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/stats.py

"""
This script performs statistical tests.

Null hypotheses are as follows:
H0-1. The direction (radians/cosines) of eSWR+ and eSWR- are the same
H0-2. The direction (radians/cosines) of rSWR+ and rSWR- are the same
H0-3. The direction (radians/cosines) of eSWR+ and rSWR+ are the same
H0-4. The direction (radians/cosines) of eSWR+ and ER are the same
H0-5. The direction (radians/cosines) of rSWR+ and ER are the same
H0-6. The direction (radians/cosines) of the above combinations are not dependent on set size

Data are stored in the following csv files:
find ./scripts/ripple/NT/direction -type f -name "*.csv" | grep -v .old
(CREST) code $ find ./scripts/ripple/NT/direction -type f -name "*.csv" | grep -v .old
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_4_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_all_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_6.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_8.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_6_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_4.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_all.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/radian/set_size_8_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_4_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_all_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_6.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_8.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_6_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_4.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_all.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def1/cosine/set_size_8_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_4_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_all_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_6.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_8.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_6_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_4.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_all.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/radian/set_size_8_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_4_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_all_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_6.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_8.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_6_control.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_4.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_all.csv
./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def2/cosine/set_size_8_control.csv
"""

"""Imports"""
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
from tqdm import tqdm

try:
    from scripts import utils
except:
    pass
from scipy import stats

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


# def run_ks_test(data1, data2):
#     """
#     Perform a two-sample Kolmogorov-Smirnov test.

#     Parameters
#     ----------
#     data1 : array-like
#         First sample data
#     data2 : array-like
#         Second sample data

#     Returns
#     -------
#     statistic : float
#         KS statistic
#     p_value : float
#         p-value of the test
#     """
#     statistic, p_value = stats.ks_2samp(data1, data2)
#     return statistic, p_value


def determine_col(df, comparison, swr_type, match, set_size):
    match_str = CONFIG.MATCHES_STR[str(match)]
    exp = rf"{match_str}-{comparison}-{set_size}-{swr_type}_boxplot".replace(
        "SWR+_", "SWR\+_"
    ).replace("SWR-_", "SWR\-_")
    cols = mngs.gen.search(
        exp,
        df.columns,
    )[1]
    if len(cols) != 1:
        pprint(exp)
        pprint(cols)
        __import__("ipdb").set_trace()
    return cols[0]


def perform_tests(df, hypotheses, match, set_size, control):
    """
    Perform statistical tests based on given hypotheses.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    hypotheses : list of tuples
        List of hypotheses to test

    Returns
    -------
    dict
        Dictionary containing test results for each hypothesis
    """

    results = {}
    for hypothesis, comparison, swr_type1, swr_type2 in hypotheses:

        col1 = determine_col(df, comparison, swr_type1, match, set_size)
        col2 = determine_col(df, comparison, swr_type2, match, set_size)

        data1 = np.array(df[col1])
        data2 = np.array(df[col2])

        results[hypothesis] = mngs.stats.brunner_munzel_test(data1, data2)

    results = pd.DataFrame(results).T
    results["stars"] = results.p_value.apply(mngs.stats.p2stars)
    # results["match"] = match
    # results["set_size"] = set_size
    # results["control"] = control
    return results


def main():
    base_path = "./scripts/ripple/NT/direction/kde_plot"
    hypotheses = [
        ("H0-1", "eSWR_vs_eSWR", "SWR+", "SWR-"),
        ("H0-2", "rSWR_vs_rSWR", "SWR+", "SWR-"),
        ("H0-3", "eSWR_vs_rSWR", "SWR+", "SWR+"),
        ("H0-4", "eSWR_vs_vER", "SWR+", "SWR+"),
        ("H0-5", "rSWR_vs_vER", "SWR+", "SWR+"),
    ]

    results = []

    for def_num in [1, 2]:
        for measure in ["radian", "cosine"]:
            for match in ["all"] + CONFIG.MATCHES:
                for set_size in ["all"] + CONFIG.SET_SIZES:
                    for control in [True, False]:
                        exp = (
                            f"{base_path}/kde_vSWR_def{def_num}/{measure}/"
                            f"set_size_{set_size}{'_control' if control else ''}_box.csv"
                        )
                        lpaths = mngs.gen.glob(exp)
                        if len(lpaths) != 1:
                            print(exp)
                            __import__("ipdb").set_trace()
                        df = mngs.io.load(lpaths[0])

                        # implement calculation of results here
                        _results = perform_tests(
                            df, hypotheses, match, set_size, control
                        )
                        _results["SWR_direction_definition"] = def_num
                        _results["measure"] = measure
                        _results["match"] = match
                        _results["set_size"] = set_size
                        _results["control"] = control
                        print(
                            f"\nResults for vSWR_def{def_num}, {measure}, "
                            f"set_size_{set_size}, {'control' if control else 'non-control'}:"
                        )
                        for hypothesis, result in _results.items():
                            results.append(_results)
                            # p_stars = mngs.stats.p2stars(result["p_value"])
                            print(_results)
                            # print(
                            #     f"{hypothesis}: Statistic = {result['statistic']:.3f}, "
                            #     f"p-value = {result['p_value']:.3f}{p_stars} "
                            #     f"(n={result['sample_size']:,}, eff={result['effect_size']:.3f})"
                            # )
    results = pd.concat(results)
    print(results)
    results = results.set_index(
        ["match", "measure", "set_size", "control", "SWR_direction_definition"]
    )
    results = results[["p_value", "stars", "effsize", "dof", "w_statistic"]]
    mngs.io.save(results, "stats.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
