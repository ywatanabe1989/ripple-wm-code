#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-18 15:37:41 (ywatanabe)"
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
from natsort import natsorted
from scripts import utils
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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


def NT_to_X_T_C(NT, trials_info):
    NT_mid = {}
    T = {}
    C = {}
    trials_info = mngs.pd.merge_columns(trials_info, ["set_size", "match"])

    for phase in CONFIG.PHASES.keys():
        NT_mid[phase] = NT[
            ..., CONFIG.PHASES[phase].mid_start : CONFIG.PHASES[phase].mid_end
        ]
        T[phase] = np.full(NT_mid[phase].shape, phase)
        C[phase] = trials_info.set_size_match

    X = np.vstack(list(NT_mid.values()))
    T = np.vstack(list(T.values()))
    C = np.hstack(list(C.values()))
    return X.reshape(len(X), -1), T.reshape(len(X), -1)[..., 0], C.astype(int)


def main_NT(NT, trials_info):
    # To X and T
    X, T, C = NT_to_X_T_C(NT, trials_info)

    # # Split data
    # X_train, X_test, T_train, T_test, S_train, S_test = train_test_split(
    #     X, T, S, test_size=0.2, random_state=42
    # )

    # SVC
    clf = SVC()
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    cv_scores = []
    all_T_test = []
    all_T_pred = []
    all_C_test = []

    for train_index, test_index in rskf.split(X, T):
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        _C_train, C_test = C[train_index], C[test_index]

        clf.fit(X_train, T_train)
        T_pred = clf.predict(X_test)

        cv_scores.append(accuracy_score(T_test, T_pred))
        all_T_test.extend(T_test)
        all_T_pred.extend(T_pred)
        all_C_test.extend(C_test)

    cv_scores = np.array(cv_scores)
    cv_scores_str = f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"

    def calc_conditional_metrics(all_C_test, all_T_test, all_T_pred):
        # Calculate metrics and confusion matrices for each condition
        conditions = np.unique(all_C_test)
        condition_metrics = {}

        for condition in conditions:
            condition_mask = np.array(all_C_test) == condition
            T_test_cond = np.array(all_T_test)[condition_mask]
            T_pred_cond = np.array(all_T_pred)[condition_mask]

            accuracy = accuracy_score(T_test_cond, T_pred_cond)
            # conf_mat = confusion_matrix(T_test_cond, T_pred_cond, labels=clf.classes_)

            _, conf_mat = mngs.ml.plt.conf_mat(
                plt,
                y_true=T_test_cond,
                y_pred=T_pred_cond,
                labels=clf.classes_,
                sorted_labels=[
                    "Fixation",
                    "Encoding",
                    "Maintenance",
                    "Retrieval",
                ],
            )

            condition_metrics[condition] = {
                "accuracy": accuracy,
                "confusion_matrix": conf_mat,
            }

    condition_metrics = calc_conditional_metrics(
        all_C_test, all_T_test, all_T_pred
    )

    # Overall confusion matrix
    _, overall_conf_mat = mngs.ml.plt.conf_mat(
        plt,
        y_true=all_T_test,
        y_pred=all_T_pred,
        labels=clf.classes_,
        sorted_labels=["Fixation", "Encoding", "Maintenance", "Retrieval"],
    )

    # Dummy
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_scores = cross_val_score(dummy_clf, X, T, cv=rskf)
    dummy_scores_str = (
        f"{dummy_scores.mean():.3f} +/- {dummy_scores.std():.3f}"
    )

    # Brunner Munzel test
    w_statistic, p_value, dof, effsize = mngs.stats.brunner_munzel_test(
        cv_scores, dummy_scores
    )

    return {
        "cv_scores_raw": cv_scores,
        "cv_scores": cv_scores_str,
        "dummy_scores_raw": dummy_scores,
        "dummy_scores": dummy_scores_str,
        "w_statistic": round(w_statistic, 3),
        "p_value": round(p_value, 3),
        "dof": round(dof, 3),
        "effsize": round(effsize, 3),
        "confusion_matrix": overall_conf_mat,
    }


def main():
    df = []
    global_conf_matrix = np.zeros((len(CONFIG.PHASES), len(CONFIG.PHASES)))

    for ca1 in CONFIG.ROI.CA1:
        NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
        trials_info = mngs.io.load(
            mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        )
        _dict = main_NT(NT, trials_info)
        global_conf_matrix += _dict["confusion_matrix"]
        _dict.update(ca1)
        df.append(_dict)

    df = pd.DataFrame(df)
    cv_scores_global = np.vstack(df.cv_scores_raw)
    dummy_scores_global = np.vstack(df.dummy_scores_raw)
    cv_scores_global_str = f"{cv_scores_global.mean().round(3)} +/- {cv_scores_global.std().round(3)}"
    dummy_scores_global_str = f"{dummy_scores_global.mean().round(3)} +/- {dummy_scores_global.std().round(3)}"

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "cv_scores": [cv_scores_global_str],
                    "dummy_scores": [dummy_scores_global_str],
                }
            ),
        ],
        ignore_index=True,
    )

    df_clf = df.drop(["cv_scores_raw", "dummy_scores_raw"], axis=1)
    df_confmat = pd.DataFrame(global_conf_matrix).astype(int)

    fig, _ = mngs.ml.plt.conf_mat(plt, cm=df_confmat)

    # Saving
    mngs.io.save(df_clf, "./data/NT/clf/SVC_clf.csv", from_cwd=True)
    mngs.io.save(df_confmat, "./data/NT/clf/SVC_confmat.csv", from_cwd=True)
    mngs.io.save(fig, "./data/NT/clf/SVC_confmat.jpg", from_cwd=True)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
