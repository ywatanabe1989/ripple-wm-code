#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-22 13:27:33 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/geometrics_medians.py


"""This script does XYZ."""


"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr

"""Functions & Classes"""

import numpy as np
import torch
from scipy.optimize import minimize


def geometric_median(X, eps=1e-5):
    y = torch.mean(X, dim=0)

    while True:
        D = torch.sqrt(torch.sum((X - y) ** 2, dim=1))
        nonzeros = D != 0
        y_new = (X[nonzeros] / D[nonzeros][:, None]).sum(
            dim=0
        ) / nonzeros.sum()
        if torch.norm(y - y_new) < eps:
            return y_new
        y = y_new


# def geometric_median(X, eps=1e-5):
#     y = np.mean(X, 0)

#     while True:
#         D = np.sqrt(((X - y)**2).sum(axis=1))
#         nonzeros = (D != 0)
#         y_new = (X[nonzeros]/D[nonzeros][:, None]).sum(axis=0) / nonzeros.sum()
#         if np.linalg.norm(y - y_new) < eps:
#             return y_new
#         y = y_new


def main():
    LPATHS_NT = mngs.gen.glob(CONFIG.PATH.NT_Z)

    for lpath_NT in LPATHS_NT:
        NT = mngs.io.load(lpath_NT)

        gs_trial = {}
        gs_session = {}
        for phase, data in CONFIG.PHASES.items():
            NT_phase = NT[..., data.mid_start : data.mid_end].transpose(
                1, 0, 2
            )

            n_factors = NT_phase.shape[0]

            # Trial
            n_trials = NT_phase.shape[1]
            gs_trial[phase] = pd.DataFrame(
                np.vstack(
                    [
                        mngs.linalg.geometric_median(
                            NT_phase[:, i_trial, :], axis=-1
                        )
                        for i_trial in range(n_trials)
                    ]
                ),
                columns=[[f"factor_{ii+1}" for ii in range(n_factors)]],
            )

            # Session
            NT_phase_flatten = NT_phase.reshape(len(NT_phase), -1)
            gs_session[phase] = mngs.linalg.geometric_median(
                NT_phase_flatten, axis=-1
            )

        # Trial
        gs_trial = (
            xr.Dataset.from_dataframe(pd.concat(gs_trial))
            .to_array()
            .rename(
                {
                    "variable": "factor",
                    "level_0": "phase",
                    "level_1": "trial",
                }
            )
        )
        gs_trial["factor"] = [f[0] for f in gs_trial["factor"].values]
        gs_trial = gs_trial.transpose("trial", "factor", "phase")

        mngs.io.save(
            gs_trial,
            lpath_NT.replace(".npy", "/gs_trial.pkl"),
            from_cwd=True,
        )

        # Session
        gs_session = pd.DataFrame(
            gs_session, index=[f"factor_{ii+1}" for ii in range(n_factors)]
        )
        mngs.io.save(
            gs_session,
            lpath_NT.replace(".npy", "/gs_session.csv"),
            from_cwd=True,
        )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
