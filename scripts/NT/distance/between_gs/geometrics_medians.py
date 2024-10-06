#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 20:37:23 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/geometrics_medians.py

"""
Functionality:
    - Calculates geometric medians for neural trajectories (NT) across trials, sessions, and conditions
Input:
    - Neural trajectory data files specified in CONFIG.PATH.NT_Z
    - Trial information files
Output:
    - Trial-wise geometric medians saved as .pkl files
    - Session-wise geometric medians saved as .csv files
    - Condition-wise (match x set_size) geometric medians saved as .csv files
Prerequisites:
    - mngs package, numpy, pandas, xarray, matplotlib, scipy
"""

"""Imports"""
import sys
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr

import utils

"""Functions & Classes"""


def calculate_gs(
    neural_trajectory: np.ndarray,
    trial_info: pd.DataFrame,
    num_factors: int = 3,
) -> Dict[str, Any]:
    (
        gs_trial,
        gs_session,
        gs_condition,
    ) = ({}, {}, {})
    for phase, data in CONFIG.PHASES.items():
        NT_phase = neural_trajectory[
            :, :num_factors, data.mid_start : data.mid_end
        ].transpose(1, 0, 2)
        n_factors, n_trials, _ = NT_phase.shape

        gs_trial[phase] = pd.DataFrame(
            np.vstack(
                [
                    mngs.linalg.geometric_median(
                        NT_phase[:, trial_idx, :], axis=-1
                    )
                    for trial_idx in range(n_trials)
                ]
            ),
            columns=[
                [f"factor_{factor_idx+1}" for factor_idx in range(n_factors)]
            ],
        )

        NT_phase_flatten = NT_phase.reshape(n_factors, -1)
        gs_session[phase] = mngs.linalg.geometric_median(
            NT_phase_flatten, axis=-1
        )

        gs_condition[phase] = calculate_condition_medians(
            NT_phase, trial_info, n_factors
        )

    gs_condition = pd.concat(
        gs_condition, axis=1
    )
    gs_condition.columns = [
        "-".join(cols) for cols in gs_condition.columns
    ]

    return {
        "trial": gs_trial,
        "session": gs_session,
        "condition": gs_condition,
    }


def calculate_condition_medians(
    NT_phase: np.ndarray, trial_info: pd.DataFrame, n_factors: int
) -> pd.DataFrame:
    condition_medians = {}
    for match in CONFIG.MATCHES:
        for set_size in CONFIG.SET_SIZES:
            condition_key = f"match_{match}_set_size_{set_size}"
            condition_mask = (trial_info.match == match) & (
                trial_info.set_size == set_size
            )
            NT_phase_condition = NT_phase[:, condition_mask, :]
            NT_phase_condition_flatten = NT_phase_condition.reshape(
                n_factors, -1
            )
            condition_medians[condition_key] = mngs.linalg.geometric_median(
                NT_phase_condition_flatten, axis=-1
            )
    return pd.DataFrame(condition_medians)


def save_gs(
    gs_data: Dict[str, Any], lpath_NT: str
) -> None:
    save_trial_medians(gs_data["trial"], lpath_NT)
    save_session_medians(gs_data["session"], lpath_NT)
    save_condition_medians(gs_data["condition"], lpath_NT)


def save_trial_medians(
    gs_trial: Dict[str, pd.DataFrame], lpath_NT: str
) -> None:
    gs_trial_xr = (
        xr.Dataset.from_dataframe(pd.concat(gs_trial))
        .to_array()
        .rename({"variable": "factor", "level_0": "phase", "level_1": "trial"})
    )
    gs_trial_xr["factor"] = [
        factor[0] for factor in gs_trial_xr["factor"].values
    ]
    gs_trial_xr = gs_trial_xr.transpose(
        "trial", "factor", "phase"
    )
    spath_trial = mngs.gen.replace(
        CONFIG.PATH.NT_GS_TRIAL,
        utils.parse_lpath(lpath_NT),
    )
    mngs.io.save(
        gs_trial_xr,
        spath_trial,
        from_cwd=True,
    )


def save_session_medians(
    gs_session: Dict[str, np.ndarray], lpath_NT: str
) -> None:
    n_factors = len(
        gs_session[list(gs_session.keys())[0]]
    )
    gs_session_df = pd.DataFrame(
        gs_session,
        index=[f"factor_{factor_idx+1}" for factor_idx in range(n_factors)],
    )
    spath_session = mngs.gen.replace(
        CONFIG.PATH.NT_GS_SESSION,
        utils.parse_lpath(lpath_NT),
    )
    mngs.io.save(
        gs_session_df,
        spath_session,
        from_cwd=True,
    )


def save_condition_medians(
    gs_condition: pd.DataFrame, lpath_NT: str
) -> None:
    spath_condition = mngs.gen.replace(
        CONFIG.PATH.NT_GS_MATCH_SET_SIZE,
        utils.parse_lpath(lpath_NT),
    )
    mngs.io.save(
        gs_condition,
        spath_condition,
        from_cwd=True,
    )


def main():
    LPATHS_NT = mngs.gen.glob(CONFIG.PATH.NT_Z)

    for lpath_NT in LPATHS_NT:
        NT = mngs.io.load(lpath_NT)
        TI = mngs.io.load(
            mngs.gen.replace(
                CONFIG.PATH.TRIALS_INFO,
                utils.parse_lpath(lpath_NT),
            )
        )
        gs_data = calculate_gs(NT, TI)
        save_gs(gs_data, lpath_NT)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
