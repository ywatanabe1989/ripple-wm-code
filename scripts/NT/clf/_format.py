#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-27 03:47:37 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/clf/_format.py

"""This script does XYZ."""

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
from functools import partial
try:
    from scripts import utils
except:
    pass

"""CONFIG"""
CONFIG = mngs.gen.load_configs()

# def reorganize_conditional_metrics(df):
#     new_df = []
#     for index, row in df.iterrows():
#         for condition, metrics in row["conditional_metrics"].items():
#             new_row = {
#                 "condition": condition,
#                 "n": metrics["n"],
#                 "balanced_accuracy": metrics["balanced_accuracy"],
#                 "confusion_matrix": metrics["confusion_matrix"],
#             }
#             new_df.append(new_row)
#     return pd.DataFrame(new_df)


def format_metrics_all(metrics_all):
    metrics_all = metrics_all.reset_index().rename(
        columns={"index": "condition"}
    )
    metrics_all = metrics_all.set_index(
        ["sub", "session", "roi", "condition", "classifier"]
    )
    metrics_all = metrics_all.reset_index()
    metrics_all = metrics_all.groupby(["classifier", "condition"]).agg(
        {
            "bACC_mean": ["mean", "std"],
            "bACC_std": ["mean", "std"],
            "weights_mean": ["mean"],
            "weights_std": ["mean"],
            "bias_mean": ["mean"],
            "bias_std": ["mean"],
            "n_samples": "sum",
            "n_folds": "sum",
            "w_statistic": ["mean"],
            "p_value": ["mean"],
            "dof": ["mean"],
            "effsize": ["mean"],
        }
    )
    return metrics_all


def format_conf_mats_all(conf_mats_all):
    conf_mats_all = conf_mats_all.reset_index().rename(
        columns={"index": "condition"}
    )

    conf_mats_all = conf_mats_all.set_index(
        ["sub", "session", "roi", "condition", "classifier"]
    ).reset_index()

    def _my_calc(x, func, index, columns):
        return pd.DataFrame(
            func(np.stack(x.tolist()), axis=0), index=index, columns=columns
        )

    index = columns = list(conf_mats_all.iloc[0]["conf_mat"].index)
    my_sum = partial(_my_calc, func=np.nansum, index=index, columns=columns)

    conf_mats_all = conf_mats_all.groupby(["classifier", "condition"]).agg(
        {"conf_mat": [("sum", my_sum)]}
    )

    for col in ["sum"]:
        conf_mats_all[("conf_mat", col)] = conf_mats_all[
            ("conf_mat", col)
        ].apply(lambda x: pd.DataFrame(x, index=index, columns=columns))

    return conf_mats_all


def format_gs(GS, phases_tasks):
    GS = GS[mngs.gen.search(phases_tasks, GS.columns)[1]]
    X = np.array(GS).T
    # X = X[..., np.newaxis]
    # __import__("ipdb").set_trace()
    # n_points = CONFIG.PHASES.Fixation.mid_end - CONFIG.PHASES.Fixation.mid_start
    # X = np.repeat(X, n_points, axis=-1).reshape(len(X), -1)
    T = np.array(GS.columns)
    C = np.full(len(X), "geometric_median")
    indi_task = mngs.gen.search(phases_tasks, T)[0]
    return X[indi_task], T[indi_task], C[indi_task]
