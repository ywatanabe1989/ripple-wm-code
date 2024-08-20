#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-20 19:52:01 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/clf/SVC.py

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
from functools import partial
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
from scripts import utils
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import LinearSVC as SVC
from tqdm import tqdm

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

N_REPEAT = 100
N_CV = 10


def NT_to_X_T_C(NT, trials_info):
    """Extracts NT of the middle 1 second for each phase as X"""

    trials_info = mngs.pd.merge_columns(trials_info, "set_size", "match")

    X = {}
    T = {}
    C = {}

    for phase in CONFIG.PHASES.keys():
        X[phase] = NT[
            ..., CONFIG.PHASES[phase].mid_start : CONFIG.PHASES[phase].mid_end
        ]
        T[phase] = np.full(len(X[phase]), phase)
        assert len(X[phase]) == len(trials_info)
        C[phase] = trials_info["set_size_match"]

    X = np.stack(list(X.values()), axis=1)
    T = np.stack(list(T.values()), axis=1)
    X = X.reshape(X.shape[0] * X.shape[1], -1)
    T = T.reshape(-1)
    C = np.stack(list(C.values()), axis=-1).reshape(-1)

    return X, T, C


def train_and_eval_SVC(clf, rskf, X, T, C, trials_info):
    conditions_uq = list(np.unique(C))
    conditional_metrics = mngs.gen.listed_dict()

    for train_index, test_index in rskf.split(X, y=T):
        # Runs a fold
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        _, C_test = C[train_index], C[test_index]

        # Trains the classifier with full training data
        clf.fit(X_train, T_train)
        T_pred = clf.predict(X_test)

        # Calculates conditional bACC and Confusion Matrix
        for condition in conditions_uq + ["all"]:
            indi = (
                C_test == condition
                if condition != "all"
                else np.full(len(C_test), True)
            )

            bACC_condi = balanced_accuracy_score(T_test[indi], T_pred[indi])

            _, conf_mat_condi = mngs.ml.plt.conf_mat(
                plt,
                y_true=T_test[indi],
                y_pred=T_pred[indi],
                labels=clf.classes_,
                sorted_labels=[
                    "Fixation",
                    "Encoding",
                    "Maintenance",
                    "Retrieval",
                ],
            )
            plt.close()
            conditional_metrics[condition].append(
                {
                    "bACC_fold": bACC_condi,
                    "conf_mat_fold": conf_mat_condi,
                    "n_samples": indi.sum(),
                }
            )

    return conditional_metrics


def to_metrics_df(conditional_metrics, dummy):
    # Reorganize metrics
    out = {}
    for cc, mm in conditional_metrics.items():
        df = pd.concat([pd.DataFrame(pd.Series(_mm)).T for _mm in mm])
        n_samples = df["n_samples"].sum()
        bACCs = df["bACC_fold"]
        bACC_mean = bACCs.mean()
        bACC_std = bACCs.std()
        conf_mat = df["conf_mat_fold"].sum().astype(int)
        out[cc] = {
            "n_samples": n_samples,
            "n_folds": len(bACCs),
            "bACCs": [bACCs],  # for statistical tests
            "bACC_mean": bACC_mean,
            "bACC_std": bACC_std,
            "conf_mat": conf_mat,
        }
    df = pd.DataFrame(out).T
    df["classifier"] = "dummy" if dummy else "SVC"
    return df


def calc_folds(NT, trials_info, dummy):
    # Loading
    X, T, C = NT_to_X_T_C(NT, trials_info)

    # CV maintener
    rskf = RepeatedStratifiedKFold(
        n_splits=N_CV, n_repeats=N_REPEAT, random_state=42
    )

    # Classifier
    clf = SVC() if not dummy else DummyClassifier(strategy="stratified")

    # Train and Evaluate SVC
    conditional_metrics = train_and_eval_SVC(clf, rskf, X, T, C, trials_info)

    # Organize data
    metrics_df = to_metrics_df(conditional_metrics, dummy)

    return metrics_df


def main_NT(NT, trials_info):

    metrics = []
    for dummy in [True, False]:
        _metrics = calc_folds(NT, trials_info, dummy)
        metrics.append(_metrics)
    metrics = pd.concat(metrics)

    # Separate conf_mats
    conf_mats = metrics[["conf_mat", "classifier"]]
    metrics = metrics.drop(columns=["conf_mat"])

    # Statistical Tests
    metrics[["w_statistic", "p_value", "dof", "effsize"]] = "NaN"
    for indi in metrics.index.unique():
        df = metrics.loc[indi]

        bm_stats = mngs.stats.brunner_munzel_test(
            df["bACCs"].iloc[0][0], df["bACCs"].iloc[1][0]
        )

        for k, v in bm_stats.items():
            metrics.loc[indi, k] = v

    metrics = metrics.drop(columns=["bACCs"])

    return metrics, conf_mats


def reorganize_conditional_metrics(df):
    new_df = []
    for index, row in df.iterrows():
        for condition, metrics in row["conditional_metrics"].items():
            new_row = {
                "condition": condition,
                "n": metrics["n"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            new_df.append(new_row)
    return pd.DataFrame(new_df)


def format_metrics_all(metrics_all):
    metrics_all = metrics_all.reset_index().rename(
        columns={"index": "condition"}
    )
    metrics_all = metrics_all.set_index(
        ["sub", "session", "roi", "condition", "classifier"]
    )
    metrics_all = metrics_all.astype(float)
    metrics_all = metrics_all.reset_index()
    metrics_all = metrics_all.groupby(["classifier", "condition"]).agg(
        {
            "bACC_mean": ["mean", "std"],
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
    my_mean = partial(_my_calc, func=np.nanmean, index=index, columns=columns)
    my_std = partial(_my_calc, func=np.nanstd, index=index, columns=columns)

    conf_mats_all = conf_mats_all.groupby(["classifier", "condition"]).agg(
        {"conf_mat": [("sum", my_sum), ("mean", my_mean), ("std", my_std)]}
    )

    for col in ["sum", "mean", "std"]:
        conf_mats_all[("conf_mat", col)] = conf_mats_all[
            ("conf_mat", col)
        ].apply(lambda x: pd.DataFrame(x, index=index, columns=columns))

    return conf_mats_all


def main():
    metrics_all = []
    conf_mats_all = []

    # Calculation
    for ca1 in CONFIG.ROI.CA1:
        NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
        trials_info = mngs.io.load(
            mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        )
        metrics, conf_mats = main_NT(NT, trials_info)

        for k, v in ca1.items():
            metrics[k] = v
            conf_mats[k] = v

        # Buffering
        metrics_all.append(metrics)
        conf_mats_all.append(conf_mats)

    # Summary
    metrics_all = pd.concat(metrics_all)
    metrics_all = format_metrics_all(metrics_all)

    conf_mats_all = pd.concat(conf_mats_all)
    conf_mats_all = format_conf_mats_all(conf_mats_all)

    # Cache
    metrics_all, conf_mats_all = mngs.io.cache(
        "id_31024987",
        "metrics_all",
        "conf_mats_all",
    )

    # Saving

    # Metrics
    mngs.io.save(
        metrics_all, "./data/NT/LinearSVC/metrics_all.csv", from_cwd=True
    )

    # Confusioon Matrices
    for ii, row in conf_mats_all.iterrows():
        string_base = "_".join(ii)
        conf_mats = conf_mats_all.loc[ii]
        for iic in conf_mats.index:
            string = string_base + "-" + "_".join(iic)
            cm = conf_mats.loc[iic]
            fig, cm = mngs.ml.plt.conf_mat(plt, cm=cm, title=string)
            mngs.io.save(
                fig,
                f"./data/NT/LinearSVC/conf_mat/figs/{string}.jpg",
                from_cwd=True,
            )
            mngs.io.save(
                cm,
                f"./data/NT/LinearSVC/conf_mat/csv/{string}.csv",
                from_cwd=True,
            )
            plt.close()


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
