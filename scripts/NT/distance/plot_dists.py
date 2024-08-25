#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-26 08:08:41 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA.py


"""
This script does XYZ.
"""

# I would like to check which combinations of condition is smaller in distance; match 1 or 2; g E or R; and NT E or R
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
from persim import plot_diagrams
from ripser import ripser
from scipy.spatial.distance import cdist

# sys.path = ["."] + sys.path
from scripts import load, utils
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
def calc_distances(NT, GS, TI):
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
    return dist_df


# def count(dist_df, radii):

#     # Count
#     dist_df["n_bin"] = 1
#     all_combinations = pd.MultiIndex.from_product(
#         [
#             dist_df["phase_g"].unique(),
#             dist_df["phase_nt"].unique(),
#             dist_df["match"].unique(),
#         ],
#         names=["phase_g", "phase_nt", "match"],
#     )

#     df = []
#     for rr in radii:
#         r_df = (
#             dist_df[dist_df["dist"] < rr]
#             .groupby(["phase_g", "phase_nt", "match"])
#             .agg({"n_bin": "sum"})
#             .reindex(all_combinations, fill_value=0)
#             .reset_index()
#         )
#         r_df["radius"] = rr
#         df.append(r_df)
#     df = pd.concat(df, ignore_index=True)
#     return df


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
            "phase_g": ["Encoding", "Retrieval"],
        },
    )

    return sorted_conditions.apply(dict, axis=1).tolist()


def prepare_heatmap_data(df, conditions):
    heatmap_data = []

    for cc in conditions:
        df_cc = mngs.pd.slice(df, cc)
        df_cc = df_cc.sort_values(["radius"])
        yy = df_cc.n_bin
        yy = yy / yy.max() * 100
        heatmap_data.append(yy)

    heatmap_data = np.vstack(heatmap_data)
    return heatmap_data


def plot_dists(NT, GS, TI, ca1, spath_base):
    mngs.plt.configure_mpl(plt, verbose=False)

    # Distance
    dist_df = calc_distances(NT, GS, TI)

    # # Params
    # radii = np.logspace(np.log10(0.1), np.log10(dist_df.dist.max()), 100)

    # # Count
    # df = count(dist_df, radii)

    # # Conditions
    # conditions = extract_conditions(df)

    # __import__("ipdb").set_trace()

    # # Heatmap data
    # heatmap_data = prepare_heatmap_data(df, conditions)

    # Plotting
    fig, ax = mngs.plt.subplots(figsize=(12, 8))

    mngs.pd.merge_cols(dist_df, "phase_g", "phase_nt")

    dist_df = dist_df.rename(columns={"phase_g_phase_nt": "group"})

    def replace_group(text):
        replaced = (
            text.replace("_phase_nt-", "-NT")
            .replace("phase_g-", "g")
            .replace("Encoding", "_E")
            .replace("Retrieval", "_R")
        )
        tex = f"${replaced}$"

        return tex

    dist_df["group"] = dist_df["group"].apply(replace_group)

    hue_order = ["$g_E-NT_E$", "$g_E-NT_R$", "$g_R-NT_E$", "$g_R-NT_R$"]
    hue_colors = [CC[cc] for cc in ["blue", "light_blue", "pink", "red"]]

    ax.sns_boxplot(
        data=dist_df,
        x="match",
        y="dist",
        hue="group",
        hue_order=hue_order,
        palette=hue_colors,
        showfliers=False,
    )

    ax.set_yscale("log")

    # Saving
    out_df = ax.to_sigma()
    for k, v in ca1.items():
        out_df[k] = v
    mngs.io.save(fig, "./" + mngs.gen.title2path(ca1) + ".jpg", from_cwd=True)
    mngs.io.save(
        out_df, "./" + mngs.gen.title2path(ca1) + ".csv", from_cwd=True
    )


def main():
    for i_ca1, ca1 in enumerate(CONFIG.ROI.CA1):

        lpath_NT = mngs.gen.replace(CONFIG.PATH.NT_Z, ca1)
        lpath_GS = mngs.gen.replace(CONFIG.PATH.NT_GS, ca1)
        lpath_TI = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        spath_base = eval(lpath_NT.replace(".npy", "/"))

        # NT, G
        NT = mngs.io.load(lpath_NT)
        GS = mngs.io.load(lpath_GS)
        TI = mngs.io.load(lpath_TI)

        NT = balance_phase(NT)

        # N Samples
        plot_dists(NT, GS, TI, ca1, spath_base)


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
