#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-08 17:04:46 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/clf/SVC.py

"""
This script does XYZ.
"""

"""
Imports
"""
import sys
import warnings
from functools import partial

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from scripts.utils import sort_phases
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC as SVC
from tqdm import tqdm

# Params
PHASES_TASKS = ["Encoding", "Retrieval"]
# PHASES_TASKS = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
"""
Warnings
"""
warnings.simplefilter("ignore", RuntimeWarning)

"""
Functions & Classes
"""


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
        C[phase] = trials_info["merged"]

    X = np.stack(list(X.values()), axis=1)
    T = np.stack(list(T.values()), axis=1)
    X = X.reshape(X.shape[0] * X.shape[1], -1)  # 3 * 20
    T = T.reshape(-1)
    C = np.stack(list(C.values()), axis=-1).reshape(-1)

    indi_task = mngs.gen.search(CONFIG.PHASES_TASK, T)[0]

    X = X[indi_task]
    T = T[indi_task]
    C = C[indi_task]

    return X, T, C


def train_and_eval_SVC(clf, rskf, X, T, C, trials_info, GS):
    clf_name = clf.__class__.__name__
    is_dummy = clf_name == "DummyClassifier"

    conditions_uq = list(np.unique(C))
    conditional_metrics = mngs.gen.listed_dict()

    for train_index, test_index in rskf.split(X, y=T):
        # Runs a fold
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        _, C_test = C[train_index], C[test_index]
        # 60 features; 3 factors * 20 bins

        # Undersamples training data
        rus = RandomUnderSampler(random_state=42)
        X_train, T_train = rus.fit_resample(X_train, T_train)

        # Adds GS to the last of test data
        X_GS, T_GS, C_GS = format_gs(GS)
        X_test = np.vstack([X_test, X_GS])
        T_test = np.hstack([T_test, T_GS])
        C_test = np.hstack([C_test, C_GS])

        # Trains the classifier "with full training data"
        clf.fit(X_train, T_train)

        # Predicting all the test data
        T_pred = clf.predict(X_test)

        # Calculates conditional metrics
        for condition in conditions_uq + ["geometric_median", "all"]:

            indi = (
                C_test != "geometric_median"
                if condition == "all"
                else C_test == condition
            )

            bACC_condi = balanced_accuracy_score(T_test[indi], T_pred[indi])

            _, conf_mat_condi = mngs.ml.plt.conf_mat(
                plt,
                y_true=T_test[indi],
                y_pred=T_pred[indi],
                labels=clf.classes_,
                sorted_labels=sort_phases(clf.classes_),
            )
            plt.close()

            if (not is_dummy) and (condition == "all"):
                weights = clf.coef_
                bias = clf.intercept_
            else:
                weights = np.full(
                    (len(np.unique(T_train)), X_train.shape[-1]), np.nan
                )
                bias = np.full(X_train.shape[-1], np.nan)

            conditional_metrics[condition].append(
                {
                    "classifier": clf_name,
                    "bACC_fold": bACC_condi,
                    "weights_fold": weights,
                    "bias_fold": bias,
                    "conf_mat_fold": conf_mat_condi,
                    "n_samples": indi.sum(),
                }
            )

    return conditional_metrics


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


def calc_folds(NT, trials_info, GS, dummy):
    # Loading
    X, T, C = NT_to_X_T_C(NT, trials_info)

    # CV maintener
    rskf = RepeatedStratifiedKFold(
        n_splits=CONFIG.N_CV, n_repeats=CONFIG.N_REPEAT, random_state=42
    )

    # Classifier
    clf = SVC() if not dummy else DummyClassifier(strategy="stratified")

    # Train and Evaluate SVC
    conditional_metrics = train_and_eval_SVC(
        clf, rskf, X, T, C, trials_info, GS
    )

    # Organize data
    metrics_df = aggregate_conditional_metrics(conditional_metrics, dummy)

    return metrics_df


def main_NT(NT, trials_info, GS):

    metrics = []
    for dummy in [True, False]:
        _metrics = calc_folds(NT, trials_info, GS, dummy)
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


def format_gs(GS):
    X = np.array(GS).T
    X = X[..., np.newaxis]
    X = np.repeat(X, 20, axis=-1).reshape(len(X), -1)
    T = np.array(GS.columns)
    C = np.full(len(X), "geometric_median")
    indi_task = mngs.gen.search(CONFIG.PHASES_TASK, T)[0]
    return X[indi_task], T[indi_task], C[indi_task]


def main():
    # Params ----------------------------------------
    CONFIG.N_REPEAT = 100
    CONFIG.N_CV = 10
    # CONFIG.N_REPEAT = 2
    # CONFIG.N_CV = 2
    CONFIG.PHASES_TASK = PHASES_TASKS
    CONFIG.SPATH_PREFFIX = f"./data/CA1/svc/{'_'.join(CONFIG.PHASES_TASK)}/"

    # Calculation ----------------------------------------
    metrics_all = []
    conf_mats_all = []
    for ca1 in tqdm(CONFIG.ROI.CA1):
        NT = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
        GS = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_GS, ca1))
        trials_info = mngs.io.load(
            mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        )
        metrics, conf_mats = main_NT(
            NT,
            trials_info,
            GS,
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

    # Cache ----------------------------------------
    metrics_all, conf_mats_all = mngs.io.cache(
        "id_31024987",
        "metrics_all",
        "conf_mats_all",
    )

    # Saving ----------------------------------------
    # Metrics
    mngs.io.save(
        metrics_all,
        CONFIG.SPATH_PREFFIX + f"metrics_all.csv",
        from_cwd=True,
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
        CONFIG.SPATH_PREFFIX + "weights_and_biases.pkl",
        from_cwd=True,
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
                CONFIG.SPATH_PREFFIX + f"conf_mat/figs/{string}.jpg",
                from_cwd=True,
            )
            mngs.io.save(
                cm,
                CONFIG.SPATH_PREFFIX + f"conf_mat/csv/{string}.csv",
                from_cwd=True,
            )
            mngs.io.save(
                mngs.pd.to_xyz(cm),
                CONFIG.SPATH_PREFFIX + f"conf_mat/csv/xyz/{string}_xyz.csv",
                from_cwd=True,
            )

            plt.close()


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
