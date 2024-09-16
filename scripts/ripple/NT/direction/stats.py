#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-16 21:04:11 (ywatanabe)"
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


def determine_col(df, comparison, swr_type, match, set_size):
    match_str = CONFIG.MATCHES_STR[str(match)]
    with mngs.gen.suppress_output():
        exp = (
            rf"{match_str}-{comparison}-{set_size}-{swr_type}_boxplot".replace(
                "SWR+_", "SWR\+_"
            ).replace("SWR-_", "SWR\-_")
        )
    cols = mngs.gen.search(
        exp,
        df.columns,
    )[1]
    if len(cols) != 1:
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

                        for hypothesis, result in _results.items():
                            results.append(_results)

    results = pd.concat(results)
    results = results.reset_index()
    results = results.rename(columns={"index": "H0"})
    results = results.set_index(
        [
            "H0",
            "match",
            "measure",
            "set_size",
            "control",
            "SWR_direction_definition",
        ]
    )
    results = results[["p_value", "stars", "effsize", "dof", "w_statistic"]]

    # Saving
    print(results)
    mngs.io.save(results, "stats.csv")

    H0_values = results.index.get_level_values("H0")
    for h0 in np.unique(H0_values):
        results_h0 = results[H0_values == h0]
        mngs.io.save(results_h0, f"stats_{h0}.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
