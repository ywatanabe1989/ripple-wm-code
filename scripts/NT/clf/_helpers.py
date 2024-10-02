#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-30 21:50:36 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/clf/_format.py

"""This script does XYZ."""

"""Imports"""

import mngs

import numpy as np
import pandas as pd
from functools import partial
from scipy import stats

"""CONFIG"""
CONFIG = mngs.gen.load_configs()


def agg_ci(x):
    x = pd.to_numeric(x, errors='coerce')
    x = x.dropna()
    if len(x) < 2:
        return np.nan, np.nan
    mean = x.mean()
    std = x.std()
    n = len(x)
    se = std / np.sqrt(n)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    return ci

def agg_n(x):
    x = pd.to_numeric(x, errors='coerce')
    return x.count()

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
            "bACC_mean": ["mean", agg_ci, agg_n],
            "weights_mean": ["mean"],
            "weights_std": ["mean"],
            "bias_mean": ["mean"],
            "bias_std": ["mean"],
            "n_samples_train": "sum",
            "n_samples_test": "sum",
            "n_folds": "sum",
            "w_statistic": ["mean"],
            "p_value": ["mean"],
            "dof": ["mean"],
            "effsize": ["mean"],
        }
    )
    metrics_all.columns = ["-".join(col) for col in metrics_all.columns]
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
            ..., CONFIG.PHASES[phase].mid_start : CONFIG.PHASES[phase].mid_end
        ]
        T[phase] = np.full(len(X[phase]), phase)
        assert len(X[phase]) == len(TI)
        C[phase] = TI["condition"]

    X = np.stack(list(X.values()), axis=1) # (50, 4, 3, 20)
    T = np.stack(list(T.values()), axis=1) # (50, 4)
    C = np.stack(list(C.values()), axis=1) # (50, 4)

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

def filter_X_T_C_by_condition(condition, X_all, T_all, C_all, is_train):
    if condition == "all":
        indi_condi = C_all != "geometric_median"
    elif (condition == "geometric_median") and is_train:
        indi_condi = np.arange(len(C_all))
    else:
        indi_condi = C_all == condition

    X = X_all[indi_condi]
    T = T_all[indi_condi]
    C = C_all[indi_condi]
    return X, T, C

def aggregate_conditional_metrics(conditional_metrics, dummy):
    agg = {}
    for cc, mm in conditional_metrics.items():

        df = pd.concat([pd.DataFrame(pd.Series(_mm)).T for _mm in mm])
        assert len(np.unique(df["classifier"])) == 1

        with mngs.gen.suppress_output():
            agg[cc] = {
                "n_samples_train": df["n_samples_train"].sum(),
                "n_samples_test": df["n_samples_test"].sum(),
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
                "bACC_ci": 1.96 * df["bACC_fold"].std() / np.sqrt(len(df["bACC_fold"])),
                "bACC_nn": len(df["bACC_fold"]),
                "conf_mat": df["conf_mat_fold"].sum().astype(int),
                "classifier": df["classifier"].iloc[0],
            }
    df = pd.DataFrame(agg).T

    return df

def get_wandb(clf, is_dummy, condition, X_train, T_train):
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
    return weights, bias
