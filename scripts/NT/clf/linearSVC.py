#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-27 18:32:50 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/clf/linearSVC.py

"""This script classifies neural trajectory of phases using SVC to check the existence of states in the NT space."""

"""Imports"""
import sys
import warnings


import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd

from scripts.utils import sort_phases
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from tqdm import tqdm
from scripts.NT.clf._helpers import (
    format_metrics_all,
    format_conf_mats_all,
    format_gs,
    filter_X_T_C_by_condition,
    get_wandb,
    NT_to_X_T_C,
    aggregate_conditional_metrics,
)


"""Warnings"""
warnings.simplefilter("ignore", RuntimeWarning)

"""Functions & Classes"""


def main(phases_tasks):
    # Params
    CONFIG.N_REPEAT = 100
    CONFIG.N_CV = 100
    N_FACTORS = 3
    sdir = f"./{'_'.join(phases_tasks)}/"

    # Training and evaluation by each session in CA1 regions
    metrics_all, conf_mats_all = process_CA1_regions(
        CONFIG, N_FACTORS, phases_tasks
    )

    # Saving
    save(metrics_all, conf_mats_all, sdir)


def process_CA1_regions(CONFIG, N_FACTORS, phases_tasks):
    metrics_all = []
    conf_mats_all = []

    for conditions in [["match"], ["match", "set_size"]]:
        for ca1 in tqdm(CONFIG.ROI.CA1):
            NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))[
                :, :N_FACTORS, :
            ]
            GS = mngs.io.load(
                mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1)
            ).iloc[:N_FACTORS]
            TI = mngs.io.load(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1))
            # mngs.pd.merge_cols(TI, "match", name="condition")
            mngs.pd.merge_cols(TI, *conditions, name="condition")

            metrics, conf_mats = process_NT(NT, TI, GS, phases_tasks)

            for k, v in ca1.items():
                metrics[k] = v
                conf_mats[k] = v

            metrics_all.append(metrics)
            conf_mats_all.append(conf_mats)

    metrics_all = format_metrics_all(pd.concat(metrics_all))
    conf_mats_all = format_conf_mats_all(pd.concat(conf_mats_all))

    return metrics_all, conf_mats_all


def process_NT(NT, TI, GS, phases_tasks):
    # Stratified K-fold Cross Validation
    metrics = pd.concat(
        [
            calc_folds(NT, TI, GS, dummy, phases_tasks)
            for dummy in [False, True]
        ]
    )

    # Separate conf_mats
    conf_mats = metrics[["conf_mat", "classifier"]]
    metrics = metrics.drop(columns=["conf_mat"])

    # Statistical Test
    metrics = run_stats(metrics)

    return metrics, conf_mats


def calc_folds(NT, TI, GS, dummy, phases_tasks):
    # Loading
    X, T, C = NT_to_X_T_C(NT, TI, phases_tasks)

    # CV maintener
    rskf = RepeatedStratifiedKFold(
        n_splits=CONFIG.N_CV, n_repeats=CONFIG.N_REPEAT, random_state=42
    )

    # Classifier
    clf_name = "DummyClassifier" if dummy else "LinearSVC"

    # Train and Evaluate SVC
    conditional_metrics = train_and_eval_SVC(
        clf_name, rskf, X, T, C, TI, GS, phases_tasks
    )

    # Organize data
    metrics_df = aggregate_conditional_metrics(conditional_metrics, dummy)

    return metrics_df


def train_and_eval_SVC(clf_name, rskf, X, T, C, TI, GS, phases_tasks):
    is_dummy = clf_name == "DummyClassifier"

    conditions_uq = list(np.unique(C))
    agg_conditional_metrics = mngs.gen.listed_dict()
    for condition in conditions_uq + ["geometric_median", "all"]:

        # Fold
        for train_index, test_index in rskf.split(X, y=T):
            clf = {
                "DummyClassifier": DummyClassifier(
                    strategy="stratified", random_state=42
                ),
                "LinearSVC": LinearSVC(C=1.0, random_state=42),
            }[clf_name]

            # Train-test splitting
            X_train, X_test = X[train_index], X[test_index]
            T_train, T_test = T[train_index], T[test_index]
            C_train, C_test = C[train_index], C[test_index]

            # Adds geometric medians to test data
            X_GS, T_GS, C_GS = format_gs(GS, phases_tasks)
            X_test = np.vstack([X_test, X_GS])
            T_test = np.hstack([T_test, T_GS])
            C_test = np.hstack([C_test, C_GS])

            # Filters by condition
            X_train, T_train, C_train = filter_X_T_C_by_condition(
                condition, X_train, T_train, C_train, True
            )
            X_test, T_test, C_test = filter_X_T_C_by_condition(
                condition, X_test, T_test, C_test, False
            )

            if (len(X_train) == 0) or (len(X_test) == 0):
                continue

            # Train classifier
            clf.fit(X_train, T_train)

            # Prediction
            T_pred = clf.predict(X_test)

            with mngs.gen.quiet():
                bACC = balanced_accuracy_score(T_test, T_pred)
                _, conf_mat = mngs.ml.plt.conf_mat(
                    plt,
                    y_true=T_test,
                    y_pred=T_pred,
                    labels=clf.classes_,
                    sorted_labels=sort_phases(clf.classes_),
                )
                if not is_dummy:
                    print(round(bACC, 3))

                plt.close()

            weights, bias = get_wandb(
                clf, is_dummy, condition, X_train, T_train
            )

            agg_conditional_metrics[condition].append(
                {
                    "classifier": clf_name,
                    "bACC_fold": bACC,
                    "weights_fold": weights,
                    "bias_fold": bias,
                    "conf_mat_fold": conf_mat,
                    "n_samples_train": len(X_train),
                    "n_samples_test": len(X_test),
                }
            )
    return agg_conditional_metrics


def run_stats(metrics):

    metrics[["w_statistic", "p_value", "dof", "effsize"]] = "NaN"
    for indi in metrics.index.unique():
        df = metrics.loc[indi]

        with mngs.gen.suppress_output():
            bm_stats = mngs.stats.brunner_munzel_test(
                df["bACCs"].iloc[1][0], df["bACCs"].iloc[0][0]
            )

        for k, v in bm_stats.items():
            metrics.loc[indi, k] = v

    metrics = metrics.drop(columns=["bACCs"])
    return metrics


def save(metrics_all, conf_mats_all, sdir):
    # Metrics
    mngs.io.save(
        metrics_all,
        sdir + f"metrics_all.csv",
    )

    # weights and biases
    cols_wb = [
        ("weights_mean", "mean"),
        ("weights_std", "mean"),
        ("bias_mean", "mean"),
        ("bias_std", "mean"),
    ]
    weights_and_biases = metrics_all[cols_wb].loc[("LinearSVC", "all")]

    mngs.io.save(
        weights_and_biases,
        sdir + "weights_and_biases.pkl",
    )

    # Confusioon Matrices
    for ii, row in conf_mats_all.iterrows():
        model_condi = "_".join(ii)
        conf_mats = conf_mats_all.loc[ii]
        for iic in conf_mats.index:
            model_condi_conf_mat_sum = model_condi + "-" + "_".join(iic)
            cm = conf_mats.loc[iic]
            fig, cm = mngs.ml.plt.conf_mat(
                plt,
                cm=cm,
                title=model_condi_conf_mat_sum,
            )
            mngs.io.save(
                fig,
                sdir + f"conf_mat/figs/{model_condi_conf_mat_sum}.jpg",
            )
            mngs.io.save(
                mngs.pd.to_xyz(cm),
                sdir + f"conf_mat/csv/xyz/{model_condi_conf_mat_sum}.csv",
            )
            plt.close()


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    for phases_tasks in [
        ["Encoding", "Retrieval"],
        ["Fixation", "Encoding", "Maintenance", "Retrieval"],
    ]:
        main(phases_tasks)
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
