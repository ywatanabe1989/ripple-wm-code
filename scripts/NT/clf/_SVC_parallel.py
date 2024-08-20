#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-19 08:28:18 (ywatanabe)"
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
from joblib import Parallel, delayed
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
from sklearn.svm import SVC
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

N_REPEAT = 10
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

    def process_fold(train_index, test_index):
        print("process fold")
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
                }
            )
            return conditional_metrics

    conditions_uq = list(np.unique(C))
    conditional_metrics = mngs.gen.listed_dict()

    # for train_index, test_index in rskf.split(X, y=T):
    results = Parallel(n_jobs=12)(
        delayed(process_fold)(train_index, test_index)
        for train_index, test_index in rskf.split(X, y=T)
    )

    # Combine results
    combined_metrics = mngs.gen.listed_dict()
    for result in results:
        for condition, metrics in result.items():
            combined_metrics[condition].extend(metrics)

    return combined_metrics


def to_metrics_df(conditional_metrics, dummy):
    # Reorganize metrics
    out = {}
    for cc, mm in conditional_metrics.items():
        df = pd.concat([pd.DataFrame(pd.Series(_mm)).T for _mm in mm])
        bACCs = df["bACC_fold"]
        bACC_mean = bACCs.mean()
        bACC_std = bACCs.std()
        conf_mat = df["conf_mat_fold"].sum().astype(int)
        out[cc] = {
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


def process_ca1(ca1):
    NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
    trials_info = mngs.io.load(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1))
    metrics, conf_mats = main_NT(NT, trials_info)
    for k, v in ca1.items():
        metrics[k] = v
        conf_mats[k] = v
    return metrics, conf_mats


# def main():
#     results = Parallel(n_jobs=-1)(delayed(process_ca1)(ca1) for ca1 in CONFIG.ROI.CA1)
#     metrics_all = pd.concat([r[0] for r in results])
#     conf_mats_all = pd.concat([r[1] for r in results])


def main():
    metrics_all = []
    conf_mats_all = []

    # Calculation
    for ca1 in CONFIG.ROI.CA1:
        metrics, conf_mats = process_ca1(ca1)
        # NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
        # trials_info = mngs.io.load(
        #     mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        # )
        # metrics, conf_mats = main_NT(NT, trials_info)

        # for k, v in ca1.items():
        #     metrics[k] = v
        #     conf_mats[k] = v

        # Buffering
        metrics_all.append(metrics)
        conf_mats_all.append(conf_mats)

    # Summary
    metrics_all = pd.concat(metrics_all)
    conf_mats_all = pd.concat(conf_mats_all)

    # Saving
    df_svc = metrics_all[metrics_all.classifier == "SVC"].loc["all"]
    df_dummy = metrics_all[metrics_all.classifier == "dummy"].loc["all"]
    conf_mats_svc = conf_mats_all[conf_mats_all.classifier == "SVC"].loc["all"]
    conf_mats_dummy = conf_mats_all[conf_mats_all.classifier == "dummy"].loc[
        "all"
    ]

    __import__("ipdb").set_trace()

    # mngs.io.save(stats, "./data/NT/clf/SVC_condition_stats.csv", from_cwd=True)

    # folds_scores_global = np.vstack(df.folds_scores_raw)
    # dummy_scores_global = np.vstack(df.dummy_scores_raw)
    # folds_scores_global_str = f"{folds_scores_global.mean().round(3)} +/- {folds_scores_global.std().round(3)}"
    # dummy_scores_global_str = f"{dummy_scores_global.mean().round(3)} +/- {dummy_scores_global.std().round(3)}"

    # df = pd.concat(
    #     [
    #         df,
    #         pd.DataFrame(
    #             {
    #                 "folds_scores": [folds_scores_global_str],
    #                 "dummy_scores": [dummy_scores_global_str],
    #             }
    #         ),
    #     ],
    #     ignore_index=True,
    # )

    # df_clf = df.drop(["folds_scores_raw", "dummy_scores_raw"], axis=1)
    # # df_confmat = pd.DataFrame(global_conf_matrix).astype(int)

    # # fig, _ = mngs.ml.plt.conf_mat(plt, cm=df_confmat)

    # # Saving
    # mngs.io.save(df_clf, "./data/NT/clf/SVC_clf.csv", from_cwd=True)
    # # mngs.io.save(df_confmat, "./data/NT/clf/SVC_confmat.csv", from_cwd=True)
    # # mngs.io.save(fig, "./data/NT/clf/SVC_confmat.jpg", from_cwd=True)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
