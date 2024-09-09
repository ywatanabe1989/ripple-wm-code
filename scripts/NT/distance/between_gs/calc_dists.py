#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-29 13:26:29 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA.py


"""This script does XYZ."""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr

"""Functions & Classes"""

# PHASES_TO_PLOT = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
PHASES_TO_PLOT = ["Encoding", "Retrieval"]
MATCH_CONDI = ["All", "Match IN", "Mismatch OUT"]


def balance_phase(NT):
    NTs_pp = {
        phase: NT[..., data.mid_start : data.mid_end]
        for phase, data in CONFIG.PHASES.items()
    }
    assert np.all([v.shape[-1] for v in list(NTs_pp.values())])
    NTs_pp = np.stack(list(NTs_pp.values()), axis=1)
    return NTs_pp


# Distances
def calc_dists(NT, GS, TI, ca1):
    # To the shape of ("factor", "phase", ...)
    NT = mngs.gen.transpose(
        NT,
        ["i_trial", "i_phase", "i_factor", "i_bin_in_phase"],
        ["i_factor", "i_phase", "i_trial", "i_bin_in_phase"],
    )
    GS = np.array(GS)

    # Euclidean distances
    dists_arr = mngs.linalg.edist(NT, GS)
    dists_xr = xr.DataArray(
        dists_arr,
        dims=["phase_NT", "match", "i_bin", "phase_g"],
        coords={
            "phase_NT": list(CONFIG.PHASES.keys()),
            "match": TI.match,
            "i_bin": np.arange(dists_arr.shape[2]),
            "phase_g": list(CONFIG.PHASES.keys()),
        },
    )

    # To dist_df
    dist_df = []
    for i_phase_nt, phase_nt in enumerate(CONFIG.PHASES.keys()):
        for i_phase_g, phase_g in enumerate(CONFIG.PHASES.keys()):
            if (phase_nt in PHASES_TO_PLOT) and (phase_g in PHASES_TO_PLOT):
                _dist = dists_xr[i_phase_nt, ..., i_phase_g]
                match_flatten = np.array(_dist.match).repeat(_dist.shape[-1])
                dist_flatten = np.array(_dist).flatten()
                _dist_df = pd.DataFrame(
                    {
                        "phase_g": [phase_g for _ in range(len(dist_flatten))],
                        "phase_nt": [
                            phase_nt for _ in range(len(dist_flatten))
                        ],
                        "dist": dist_flatten,
                        "match": match_flatten,
                    }
                )
                dist_df.append(_dist_df)
    dist_df = pd.concat(dist_df)

    mngs.pd.merge_cols(dist_df, "phase_g", "phase_nt")
    dist_df = dist_df.rename(columns={"merged": "group"})
    dist_df["group"] = dist_df["group"].apply(replace_group)

    for k, v in ca1.items():
        dist_df[k] = v

    return dist_df


def extract_conditions(df):
    conditions = {
        k: df[k].unique().tolist() for k in ["phase_g", "phase_nt", "match"]
    }
    conditions = list(mngs.gen.yield_grids(conditions))
    conditions = pd.concat(
        [pd.DataFrame(pd.Series(cc)).T for cc in conditions]
    )

    sorted_conditions = mngs.pd.sort(
        conditions,
        orders={
            "match": [1, 2],
            "phase_g": ["Encoding", "Retrieval"],
            "phase_nt": ["Encoding", "Retrieval"],
        },
    )

    return sorted_conditions.apply(dict, axis=1).tolist()


def replace_group(text):
    replaced = (
        text.replace("_phase_nt-", "-NT")
        .replace("phase_g-", "g")
        .replace("Encoding", "_E")
        .replace("Retrieval", "_R")
    )
    tex = f"${replaced}$"

    return tex


def main():
    dfs = []
    for i_ca1, ca1 in enumerate(CONFIG.ROI.CA1):

        lpath_NT = mngs.gen.replace(CONFIG.PATH.NT_Z, ca1)
        lpath_GS = mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1)
        lpath_TI = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        # spath_base = eval(lpath_NT.replace(".npy", "/"))

        # NT, G
        NT = mngs.io.load(lpath_NT)
        GS = mngs.io.load(lpath_GS)
        TI = mngs.io.load(lpath_TI)

        NT = balance_phase(NT)

        # N Samples
        df = calc_dists(NT, GS, TI, ca1)

        dfs.append(df)

    df = pd.concat(dfs)

    mngs.io.save(df, "./data/CA1/dist.csv", from_cwd=True)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
