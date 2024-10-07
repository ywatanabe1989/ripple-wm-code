#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-03 16:24:18 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/clf/format_stats.py

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
mngs.pd.ignore_SettingWithCopyWarning()

"""Parameters"""

"""Functions & Classes"""


def resolve_condition(df):
    df = (
        df[df.condition != "geometric_median"]
        .reset_index()
        .drop(columns="index")
    )

    # Match
    df[["match", "set_size"]] = np.nan
    df[["match", "set_size"]] = df[["match", "set_size"]].astype(str)
    df.loc[mngs.gen.search("all", df.condition)[0], "match"] = "all"
    df.loc[mngs.gen.search("match-1.0", df.condition)[0], "match"] = 1
    df.loc[mngs.gen.search("match-2.0", df.condition)[0], "match"] = 2
    df.match = df.match.astype(str).replace(CONFIG.MATCHES_STR)

    # Set size
    df.loc[
        ~df.index.isin(mngs.gen.search("set_size", df.condition)[0]),
        "set_size",
    ] = "all"
    df.loc[mngs.gen.search("set_size-4.0", df.condition)[0], "set_size"] = 4
    df.loc[mngs.gen.search("set_size-6.0", df.condition)[0], "set_size"] = 6
    df.loc[mngs.gen.search("set_size-8.0", df.condition)[0], "set_size"] = 8
    df.set_size = df.set_size.astype(str).replace({"all": "ALL"})

    # Cleanup
    df = df.drop(columns="condition")

    return df




def resolve_ci(df):
    ci = df["bACC (95% CI)"].apply(eval)
    ci = np.vstack(ci)
    ci_range = np.array(df["bACC (mean)"])[:, np.newaxis] - ci
    ci_range = ci_range[:, 0] - ci_range[:, 1]
    df["bACC (95% CI)"] = ci_range
    df = df.rename(columns={"bACC (95% CI)": "bACC (95% CI width)"})
    return df


def resolve_n_samples(df):
    df["# of training samples"] /= df["n_folds-sum"]
    df["# of test samples"] /= df["n_folds-sum"]
    df = df.drop(columns=["n_folds-sum"])
    return df


def resolve_columns(df):
    df = df.rename(
        columns={
            "bACC_mean-mean": "bACC (mean)",
            "bACC_mean-agg_ci": "bACC (95% CI)",
            "n_samples_train-sum": "# of training samples",
            "n_samples_test-sum": "# of test samples",
            **{
                f"{col}-mean": col
                for col in [
                    "n_folds",
                    "w_statistic",
                    "dof",
                    "effsize",
                    "p_value",
                ]
            },
        }
    )
    # Stars
    df["stars"] = df.p_value.apply(mngs.stats.p2stars)

    # Sorting
    SORTED_COLS = [
        "classifier",
        "match",
        "set_size",
        "bACC (mean)",
        "bACC (95% CI)",
        "p_value",
        "stars",
        "w_statistic",
        "dof",
        "effsize",
        "# of training samples",
        "# of test samples",
        "n_folds-sum",
    ]
    df = df[SORTED_COLS]
    df = df.rename(
        columns={
            "classifier": "Classifier",
            "match": "Match",
            "set_size": "Set size",
            "p_value": "P value",
            "stars": "",
            "w_statistic": "w",
        }
    )
    return df


def resolve_clf(df):
    df["Classifier"] = df["Classifier"].replace(
        {"DummyClassifier": "Dummy", "LinearSVC": "SVM"}
    )
    return df


def main():
    lpath = "/mnt/ssd/ripple-wm-code/scripts/NT/clf/linearSVC/Encoding_Retrieval/metrics_all.csv"
    df = mngs.io.load(lpath)

    # Condition
    df = resolve_condition(df)

    # Columns
    df = resolve_columns(df)

    # Values
    df = resolve_clf(df)
    df = resolve_ci(df)
    df = resolve_n_samples(df)

    #
    df = df.sort_values(["Match", "Set size", "Classifier"])
    df = mngs.pd.round(df, 3)

    # Saving
    mngs.io.save(df, "formatted.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
