#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-24 18:35:55 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA.py


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


# def slice_NT_by_match(NT, TI, match):
#     assert match in MATCH_CONDI
#     indi = {
#         "All": np.full(len(TI), True),
#         "Match IN": TI.match == 1,
#         "Mismatch OUT": TI.match == 2,
#     }[match]
#     return NT[indi]


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

    # To df
    df = []
    for i_phase_nt, phase_nt in enumerate(CONFIG.PHASES.keys()):
        for i_phase_g, phase_g in enumerate(CONFIG.PHASES.keys()):
            if (phase_nt in PHASES_TO_PLOT) and (phase_g in PHASES_TO_PLOT):
                _dist = dists_xr[i_phase_nt, ..., i_phase_g]
                match_flatten = np.array(_dist.match).repeat(_dist.shape[-1])
                dist_flatten = np.array(_dist).flatten()
                _df = pd.DataFrame(
                    {
                        "phase_g": [phase_g for _ in range(len(dist_flatten))],
                        "phase_nt": [
                            phase_nt for _ in range(len(dist_flatten))
                        ],
                        "dist": dist_flatten,
                        "match": match_flatten,
                    }
                )
                df.append(_df)
    df = pd.concat(df)
    return df


# # N Samples
# def calc_n_samples_in_spheres(distances, radii):

#     def _calc_n_samples_in_spheres(distances, radius):
#         return (distances <= radius).sum()

#     # Calculate the values
#     df_new = pd.DataFrame(
#         {
#             f"radius-{rr}": distances.apply(
#                 lambda x: _calc_n_samples_in_spheres(x, rr)
#             )
#             for rr in radii
#         }
#     )

#     # To probability
#     df_new /= np.array(df_new).max(axis=1, keepdims=True)

#     return df_new


# def plot_n_samples(NT, GS, TI, ca1, spath_base):
#     mngs.plt.configure_mpl(plt, verbose=False)

#     fig, axes = mngs.plt.subplots(
#         nrows=len(PHASES_TO_PLOT), sharex=True, sharey=True
#     )

#     # Distance
#     dist_df = calc_distances(NT, GS, TI)

#     # Radii to count
#     radii = np.linspace(0, dist_df.dist.max(), 100)

#     # Count
#     dist_df["n_bin"] = 1
#     df = []
#     for rr in radii:
#         r_df = (
#             dist_df[dist_df["dist"] < rr]
#             .groupby(["phase_g", "phase_nt", "match"])
#             .agg({"n_bin": "sum"})
#             .reset_index()
#         )
#         if len(r_df) > 0:
#             r_df.loc[:, "radius"] = rr
#         df.append(r_df)
#     df = pd.concat(df)

#     conditions = {
#         k: df[k].unique().tolist() for k in ["phase_g", "phase_nt", "match"]
#     }
#     conditions = list(mngs.gen.yield_grids(conditions))

#     fig, ax = mngs.plt.subplots()
#     for cc in conditions:
#         df_cc = mngs.pd.slice(df, cc)
#         df_cc = df_cc.sort_values(["radius"])
#         xx = df_cc.radius
#         yy = df_cc.n_bin
#         yy = yy / yy.max() * 100
#         ax.plot(xx, yy, label=str(cc))
#         ax.legend()
#         ax.set_xscale("log")
#         ax.set_xlim(0.1, 10)
#         ax.set_ylim(0, 100)

#     fig.supxyt(
#         "Radius",
#         "Sample count [%]",
#         str(ca1),
#     )

#     return fig


def plot_n_samples(NT, GS, TI, ca1, spath_base):
    mngs.plt.configure_mpl(plt, verbose=False)

    # Distance
    dist_df = calc_distances(NT, GS, TI)

    # Radii to count
    radii = np.logspace(np.log10(0.1), np.log10(10), 100)


    # Count
    dist_df["n_bin"] = 1
    all_combinations = pd.MultiIndex.from_product([
        dist_df["phase_g"].unique(),
        dist_df["phase_nt"].unique(),
        dist_df["match"].unique()
    ], names=["phase_g", "phase_nt", "match"])

    df = []
    for rr in radii:
        r_df = (
            dist_df[dist_df["dist"] < rr]
            .groupby(["phase_g", "phase_nt", "match"])
            .agg({"n_bin": "sum"})
            .reindex(all_combinations, fill_value=0)
            .reset_index()
        )
        r_df["radius"] = rr
        df.append(r_df)
    df = pd.concat(df, ignore_index=True)

    # # Count
    # dist_df["n_bin"] = 1
    # df = []
    # for rr in radii:
    #     r_df = (
    #         dist_df[dist_df["dist"] < rr]
    #         .groupby(["phase_g", "phase_nt", "match"])
    #         .agg({"n_bin": "sum"})
    #         .reset_index()
    #     )
    #     if len(r_df) > 0:
    #         r_df.loc[:, "radius"] = rr
    #     df.append(r_df)
    # df = pd.concat(df)

    conditions = {
        k: df[k].unique().tolist() for k in ["phase_g", "phase_nt", "match"]
    }
    conditions = list(mngs.gen.yield_grids(conditions))

    fig, ax = mngs.plt.subplots(figsize=(12, 8))

    heatmap_data = []
    condition_labels = []
    for cc in conditions:
        df_cc = mngs.pd.slice(df, cc)
        df_cc = df_cc.sort_values(["radius"])
        yy = df_cc.n_bin
        yy = yy / yy.max() * 100
        heatmap_data.append(yy)
        condition_labels.append(str(cc))

    heatmap_data = np.vstack(heatmap_data)

    sorted_indices = natsorted(range(len(condition_labels)), key=lambda i: condition_labels[i])

    heatmap_data = heatmap_data[sorted_indices]
    condition_labels = np.array(condition_labels)[sorted_indices].tolist()

    ax.imshow2d(heatmap_data.T)

    condition_labels = [str(cl).replace(", ", ",\n") for cl in condition_labels]
    ax.set_ticks(
        yvals=np.arange(len(condition_labels)),
        yticks=np.array(condition_labels),
    )

    mngs.io.save(fig, "/tmp/tmp.jpg")

    # im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', extent=[np.log10(0.1), np.log10(10), 0, 8])

    # ax.set_yticks(np.arange(8) + 0.5)
    # ax.set_yticklabels(condition_labels)
    # ax.set_xticks(np.log10([0.1, 1, 10]))
    # ax.set_xticklabels([0.1, 1, 10])

    # cbar = fig.colorbar(im, ax=ax, label='Sample count [%]')

    # ax.set_xlabel('Radius')
    # ax.set_ylabel('Conditions')
    # ax.set_title(str(ca1))

    # return fig


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
        plot_n_samples(NT, GS, TI, ca1, spath_base)



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
