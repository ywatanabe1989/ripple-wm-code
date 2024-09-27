#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-27 14:34:32 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/clf/linearSVC.py

"""This script classifies neural trajectory of phases using SVC to check the existence of states in the NT space."""

"""Imports"""
import sys
import warnings


import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
# from imblearn.under_sampling import RandomUnderSampler
# from scripts.utils import sort_phases
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from tqdm import tqdm
from scripts.NT.clf._format import (
    format_metrics_all,
    format_conf_mats_all,
    format_gs,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


"""Warnings"""
warnings.simplefilter("ignore", RuntimeWarning)

"""Functions & Classes"""

def NT_to_X_T_C(NT, TI, phases_tasks):
    """
    Extracts NT of the middle 1 second for each phase as X.

    Parameters
    ----------
    NT : array-like
        Neural trajectory data
    TI : pandas.DataFrame
        Trial information
    phases_tasks : list
        List of phases to consider

    Returns
    -------
    X : array-like
        Features
    T : array-like
        Targets (phases)
    C : array-like
        Conditions (match & set size)

    Example
    -------
    X, T, C = NT_to_X_T_C(NT, TI, ['Encoding', 'Retrieval'])
    """
    X, T, C = {}, {}, {}

    for phase in CONFIG.PHASES.keys():
        X[phase] = NT[
            ..., CONFIG.PHASES[phase].mid-10 : CONFIG.PHASES[phase].mid+10
        ]
        T[phase] = np.full(len(X[phase]), phase)
        assert len(X[phase]) == len(TI)
        C[phase] = TI["match_set_size"]

    X = np.stack(list(X.values()), axis=1) # (50, 4, 3, 20)
    T = np.stack(list(T.values()), axis=1) # (50, 4)
    C = np.stack(list(C.values()), axis=1) # (50, 4)

    # # Squeeze the time dimension
    # with mngs.gen.suppress_output():
    #     X = mngs.linalg.geometric_median(X)

    # Reshape to (n_trials, n_bins, n_phases, n_factors)
    X = X.transpose(0,3,1,2)
    T = T[:,np.newaxis,:].repeat(X.shape[1], 1)
    C = C[:,np.newaxis,:].repeat(X.shape[1], 1)

    # Each bin as sample
    n_samples = int(X.shape[0] * X.shape[1] * X.shape[2])
    X = X.reshape(n_samples, -1).squeeze()
    T = T.reshape(n_samples,)
    C = C.reshape(n_samples,)

    indi_task = mngs.gen.search(phases_tasks, T)[0]

    return X[indi_task], T[indi_task], C[indi_task]


def train_and_eval_SVC(clf_name, rskf, X, T, C, TI, GS, phases_tasks):
    """
    Train and evaluate SVC classifier.

    Parameters
    ----------
    clf_class : class
        Classifier class to be used
    is_dummy : bool
        Whether the classifier is a dummy classifier
    rskf : RepeatedStratifiedKFold
        Cross-validation splitter
    X : array-like
        Features
    T : array-like
        Targets
    C : array-like
        Conditions
    TI : pandas.DataFrame
        Trial information
    GS : array-like
        Geometric median data
    phases_tasks : list
        List of phases to consider

    Returns
    -------
    dict
        Conditional metrics for each fold and condition

    Example
    -------
    metrics = train_and_eval_SVC(LinearSVC, False, rskf, X, T, C, TI, GS, ['Encoding', 'Retrieval'])
    """
    is_dummy = clf_name == "DummyClassifier"

    conditions_uq = list(np.unique(C))
    conditional_metrics = mngs.gen.listed_dict()

    for train_index, test_index in rskf.split(X, y=T):
        if clf_name == "DummyClassifier":
            clf = DummyClassifier(strategy="stratified", random_state=42)
        elif clf_name == "LinearSVC":
            clf = LinearSVC(C=1.0, random_state=42)
        else:
            pass

        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        _, C_test = C[train_index], C[test_index]

        X_GS, T_GS, C_GS = format_gs(GS, phases_tasks)
        X_test = np.vstack([X_test, X_GS])
        T_test = np.hstack([T_test, T_GS])
        C_test = np.hstack([C_test, C_GS])

        clf.fit(X_train, T_train)
        T_pred = clf.predict(X_test)

        for condition in conditions_uq + ["geometric_median", "all"]:
            indi_condi = (
                C_test != "geometric_median"
                if condition == "all"
                else C_test == condition
            )
            T_test_condi = T_test[indi_condi]
            T_pred_condi = T_pred[indi_condi]

            with mngs.gen.quiet():
                # bACC_condi = balanced_accuracy_score(
                #     T_test_condi, T_pred_condi
                # )
                acc = (T_test_condi == T_pred_condi).mean()
                bACC_condi = acc
                _, conf_mat_condi = mngs.ml.plt.conf_mat(
                    plt,
                    y_true=T_test_condi,
                    y_pred=T_pred_condi,
                    labels=clf.classes_,
                    # sorted_labels=sort_phases(le.classes_),
                )
                plt.close()

            weights = (
                clf.coef_
                if (not is_dummy) and (condition == "all")
                else np.full(
                    (len(np.unique(T_train)), X_train.shape[-1]), np.nan
                )
            )
            bias = (
                clf.intercept_
                if (not is_dummy) and (condition == "all")
                else np.full(X_train.shape[-1], np.nan)
            )

            conditional_metrics[condition].append(
                {
                    "classifier": clf_name,
                    "bACC_fold": bACC_condi,
                    "weights_fold": weights,
                    "bias_fold": bias,
                    "conf_mat_fold": conf_mat_condi,
                    "n_samples": indi_condi.sum(),
                }
            )
    return conditional_metrics


# def train_and_eval_SVC(clf_class, is_dummy, rskf, X, T, C, TI, GS, phases_tasks):
#     clf_name = clf_class.__name__

#     conditions_uq = list(np.unique(C))
#     conditional_metrics = mngs.gen.listed_dict()

#     # Initialize LabelEncoder
#     le = LabelEncoder()
#     T_encoded = le.fit_transform(T)

#     for train_index, test_index in rskf.split(X, y=T):
#         # Initialize classifier for each fold
#         clf = clf_class()

#         # Runs a fold
#         X_train, X_test = X[train_index], X[test_index]
#         T_train, T_test = T_encoded[train_index], T_encoded[test_index]

#         _, C_test = C[train_index], C[test_index]
#         # 30 features; 3 factors * 10 bins

#         # # Undersamples training data
#         # rus = RandomUnderSampler(random_state=42)
#         # X_train, T_train = rus.fit_resample(X_train, T_train)

#         # Adds GS to the last of test data
#         X_GS, T_GS, C_GS = format_gs(GS, phases_tasks)
#         X_test = np.vstack([X_test, X_GS])
#         T_test = np.hstack([T_test, le.transform(T_GS)])
#         C_test = np.hstack([C_test, C_GS])

#         # Trains the classifier "with full training data"
#         clf.fit(X_train, T_train)

#         # Predicting all the test data
#         T_pred = clf.predict(X_test)
#         # T_pred = T_test # for debugging

#         # Calculates conditional metrics
#         for condition in conditions_uq + ["geometric_median", "all"]:

#             if condition == "all":
#                 indi_condi = C_test != "geometric_median"
#             else:
#                 indi_condi = C_test == condition

#             T_test_condi = T_test[indi_condi]
#             T_pred_condi = T_pred[indi_condi]

#             with mngs.gen.suppress_output():
#                 bACC_condi = balanced_accuracy_score(T_test_condi, T_pred_condi)
#                 _, conf_mat_condi = mngs.ml.plt.conf_mat(
#                     plt,
#                     y_true=le.inverse_transform(T_test_condi),
#                     y_pred=le.inverse_transform(T_pred_condi),
#                     labels=le.classes_,
#                     sorted_labels=sort_phases(le.classes_),
#                 )
#                 plt.close()

#             if (not is_dummy) and (condition == "all"):
#                 weights = clf.coef_
#                 bias = clf.intercept_
#             else:
#                 weights = np.full(
#                     (len(np.unique(T_train)), X_train.shape[-1]), np.nan
#                 )
#                 bias = np.full(X_train.shape[-1], np.nan)

#             conditional_metrics[condition].append(
#                 {
#                     "classifier": clf_name,
#                     "bACC_fold": bACC_condi,
#                     "weights_fold": weights,
#                     "bias_fold": bias,
#                     "conf_mat_fold": conf_mat_condi,
#                     "n_samples": indi_condi.sum(),
#                 }
#             )
#     return conditional_metrics


def aggregate_conditional_metrics(conditional_metrics, dummy):
    agg = {}
    for cc, mm in conditional_metrics.items():

        df = pd.concat([pd.DataFrame(pd.Series(_mm)).T for _mm in mm])
        assert len(np.unique(df["classifier"])) == 1

        with mngs.gen.suppress_output():
            agg[cc] = {
                "n_samples": df["n_samples"].sum(),
                "n_folds": len(df["bACC_fold"]),
                "weights_mean": np.nanmean(
                    np.stack(df["weights_fold"]), axis=0
                ),
                "weights_std": np.nanstd(np.stack(df["weights_fold"]), axis=0),
                "bias_mean": np.nanmean(np.stack(df["bias_fold"]), axis=0),
                "bias_std": np.nanstd(np.stack(df["bias_fold"]), axis=0),
                "bACCs": [df["bACC_fold"]],
                "bACC_mean": df["bACC_fold"].mean(),
                "bACC_std": df["bACC_fold"].std(),
                "conf_mat": df["conf_mat_fold"].sum().astype(int),
                "classifier": df["classifier"].iloc[0],
            }
    df = pd.DataFrame(agg).T

    return df


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


def main_NT(NT, TI, GS, phases_tasks):

    metrics = []
    for dummy in [False, True]:
        _metrics = calc_folds(NT, TI, GS, dummy, phases_tasks)
        metrics.append(_metrics)
    metrics = pd.concat(metrics)

    # Separate conf_mats
    conf_mats = metrics[["conf_mat", "classifier"]]
    metrics = metrics.drop(columns=["conf_mat"])

    # Statistical Tests
    metrics[["w_statistic", "p_value", "dof", "effsize"]] = "NaN"
    for indi in metrics.index.unique():
        df = metrics.loc[indi]

        with mngs.gen.suppress_output():
            bm_stats = mngs.stats.brunner_munzel_test(
                df["bACCs"].iloc[0][0], df["bACCs"].iloc[1][0]
            )

        for k, v in bm_stats.items():
            metrics.loc[indi, k] = v

    metrics = metrics.drop(columns=["bACCs"])

    return metrics, conf_mats


def main(phases_tasks):
    # Params ----------------------------------------
    CONFIG.N_REPEAT = 2
    CONFIG.N_CV = 10
    # CONFIG.N_REPEAT = 3
    # CONFIG.N_CV = 2
    N_FACTORS = 3
    sdir = f"./{'_'.join(phases_tasks)}/"

    # Calculation ----------------------------------------
    metrics_all = []
    conf_mats_all = []
    for ca1 in tqdm(CONFIG.ROI.CA1):
        NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))[
            :, :N_FACTORS, :
        ]
        GS = mngs.io.load(
            mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1)
        ).iloc[:N_FACTORS]
        TI = mngs.io.load(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1))
        mngs.pd.merge_cols(TI, "match", "set_size", name="match_set_size")
        metrics, conf_mats = main_NT(
            NT,
            TI,
            GS,
            phases_tasks,
        )

        for k, v in ca1.items():
            metrics[k] = v
            conf_mats[k] = v

        # Buffering
        metrics_all.append(metrics)
        conf_mats_all.append(conf_mats)

    # Summary
    metrics_all = format_metrics_all(pd.concat(metrics_all))
    conf_mats_all = format_conf_mats_all(pd.concat(conf_mats_all))

    # Saving ----------------------------------------
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
        # ["Fixation", "Encoding", "Maintenance", "Retrieval"],
    ]:
        main(phases_tasks)
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
