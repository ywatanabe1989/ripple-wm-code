#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-05 22:04:23 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/from_O_of_MTL_regions.py

"""This script does XYZ."""


"""Imports"""
import itertools
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import norm
import utils


"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""

# def load_gs(N_FACTORS=3):

#     def load_corresponding_TI(lpath):
#         # Loading the corresponding trials info
#         parsed = utils.parse_lpath(lpath)
#         lpath_TI = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
#         TI = mngs.io.load(lpath_TI)
#         return TI

#     LPATHS_GS = mngs.io.glob(CONFIG.PATH.NT_GS_TRIAL)
#     LPATHS_GS = mngs.gen.search(["Session_01", "Session_02"], LPATHS_GS)[1]
#     gs_agg = {}
#     for mtl in CONFIG.ROI.MTL.keys():
#         lpaths_gs_mtl = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS_GS)[1]
#         gss = []
#         for lpath_gs in lpaths_gs_mtl:
#             gs = mngs.io.load(lpath_gs)
#             gs = gs[:, :N_FACTORS, :]

#             # Loading the corresponding trials info
#             TI = load_corresponding_TI(lpath_gs)

#             # Adding set size information to gs gsta
#             gs = gs.swap_dims({'trial': 'set_size'})
#             gs = gs.assign_coords(set_size=("set_size", TI["set_size"]))

#             # Aggregation
#             gss.append(gs)

#         gs_agg[mtl] = gss

#         print(f"n_lpaths_mtl: {mtl, len(lpaths_gs_mtl)}")
#     return gs_agg


def plot_box(dist):
    fig, ax = mngs.plt.subplots()
    dist = dist[~dist.isna().any(axis=1)]

    labels = ["HIP", "EC", "AMY"]
    data = [dist[dist.MTL == ll].distance for ll in labels]
    print([len(dd) for dd in data])

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
    )
    ax.set_yscale("log")
    return fig


# def gs2dists(gs):
#     dists = {}
#     for p1, p2 in combinations(CONFIG.PHASES.keys(), 2):
#         for set_size in CONFIG.SET_SIZES:
#             gs_1 = gs[..., gs.phase == p1].squeeze()
#             gs_2 = gs[..., gs.phase == p2].squeeze()

#             indi_ss_1 = gs_1.set_size == set_size
#             indi_ss_2 = gs_2.set_size == set_size
#             assert (indi_ss_1 == indi_ss_2).all()
#             indi_ss = indi_ss_1

#             gs_1_ss = gs_1[indi_ss]
#             gs_2_ss = gs_2[indi_ss]

#             dists[f"{p1[0]}{p2[0]}-set_size_{set_size}"] = mngs.linalg.nannorm(gs_1_ss - gs_2_ss)

#     dists = mngs.pd.force_df(dists)
#     return dists


# def gs_all_to_dists(gs_all):
#     dists = mngs.gen.listed_dict()
#     for mtl in CONFIG.ROI.MTL.keys():
#         gs_mtl = gs_all[mtl]
#         for gs in gs_mtl:
#             __import__("ipdb").set_trace()
#             dist = gs2dists(gs)
#             dists[mtl].append(dist)

#     dists = {k: pd.concat(v) for k,v in dists.items()}
#     return dists

def main():
    __import__("ipdb").set_trace()
    LPATHS_GS_TRIAL = mngs.gen.glob(CONFIG.PATH.NT_DIST_BETWEEN_GS_TRIAL)
    # # Loading
    # gs_all = load_gs()

    # # Distances between geometric medians
    # dists = gs_all_to_dists(gs_all)

    # # Plotting
    # fig, axes = mngs.plt.subplots(ncols=len(dists.keys()))
    # for ax, mtl in zip(axes, dists.keys()):
    #     df = dists[mtl].melt()
    #     df = df.rename(columns={
    #         "variable": "phase_combi_set_size",
    #         "value": "distance",
    #     })
    #     ax.sns_boxplot(
    #         data=df,
    #         x="phase_combi_set_size",
    #         y="distance",
    #         showfliers=False,
    #         id=mtl,
    #     )
    #     ax.set_xyt(None, None, mtl)
    #     ax.rotate_labels(x=90, y=0)
    #     ax.extend(y_ratio=0.5)
    #     ax.set_yscale("log")

    # mngs.io.save(fig, "box.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
