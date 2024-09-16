#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-16 10:26:35 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/stats.py

"""
This script performs statistical tests.

Null hypotheses are as follows:
H0-1. The direction (radians/cosines) of eSWR+ and eSWR- are the same
H0-2. The direction (radians/cosines) of rSWR+ and rSWR- are the same
H0-3. The direction (radians/cosines) of eSWR+ and rSWR+ are the same
H0-4. The direction (radians/cosines) of eSWR+ and ER are the same
H0-5. The direction (radians/cosines) of rSWR+ and ER are the same

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


def run_ks_test(data1, data2):
    # Perform a two-sample Kolmogorov-Smirnov test
    statistic, p_value = stats.ks_2samp(data1, data2)
    return statistic, p_value


def perform_tests(df, hypotheses):
    results = {}
    for hypothesis, comparison, type1, type2 in hypotheses:
        data1 = df[
            df.columns[df.columns.str.contains(f"{type1}_{comparison}")]
        ]
        data2 = df[
            df.columns[df.columns.str.contains(f"{type2}_{comparison}")]
        ]
        statistic, p_value = run_ks_test(
            data1.values.flatten(), data2.values.flatten()
        )
        results[hypothesis] = {"statistic": statistic, "p_value": p_value}
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

    for def_num in [1, 2]:
        for measure in ["radian", "cosine"]:
            for set_size in ["all", 4, 6, 8]:
                for control in [True, False]:
                    file_pattern = f"{base_path}/kde_vSWR_def{def_num}/{measure}/set_size_{set_size}{'_control' if control else ''}.csv"

                    lpath = mngs.gen.glob(file_pattern)[0]
                    df = mngs.io.load(lpath)

                    print(
                        f"\nResults for vSWR_def{def_num}, {measure}, set_size_{set_size}, {'control' if control else 'non-control'}:"
                    )
                    for hypothesis, result in results.items():
                        print(
                            f"{hypothesis}: Statistic = {result['statistic']:.4f}, p-value = {result['p_value']:.4f}"
                        )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
