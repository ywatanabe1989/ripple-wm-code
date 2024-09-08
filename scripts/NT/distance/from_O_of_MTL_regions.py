#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-09 07:17:41 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/from_O_of_MTL_regions.py

"""This script does XYZ."""


"""Imports"""
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scipy.linalg import norm

sys.path = ["."] + sys.path
try:
    from scripts import utils
except Exception as e:
    pass

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def load_trajs():
    LPATHS = mngs.io.glob(CONFIG.PATH.NT_Z)
    LPATHS = mngs.gen.search(["Session_01", "Session_02"], LPATHS)[1]
    traj = {}
    for mtl in CONFIG.ROI.MTL.keys():
        lpaths_mtl = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS)[1]
        traj[mtl] = np.vstack([mngs.io.load(lpath) for lpath in lpaths_mtl])
    return traj


def calc_dist(traj_MTL_region):
    norm_nonnan_MTL_region = {}
    for i_bin in range(traj_MTL_region.shape[-1]):  # i_bin = 0
        traj_MTL_region_i_bin = traj_MTL_region[..., i_bin]
        norm_MTL_region_i_bin = norm(
            traj_MTL_region_i_bin[
                ~np.isnan(traj_MTL_region_i_bin).any(axis=1)
            ],
            axis=-1,
        )
        norm_nonnan_MTL_region[i_bin] = norm_MTL_region_i_bin
    return mngs.pd.force_df(norm_nonnan_MTL_region)


def calc_mean_and_ci(dist):
    dist_mm = np.nanmean(dist, axis=0)
    dist_sd = np.nanstd(dist, axis=0)
    dist_nn = (~np.isnan(dist)).astype(int).sum(axis=0)
    dist_ci = 1.96 * dist_mm / (dist_sd * dist_nn)
    return dist_mm, dist_ci


def plot_line(dist):
    mm_ci = {k: calc_mean_and_ci(v) for k, v in dist.items()}

    xx = np.linspace(
        0,
        CONFIG.TRIAL.DUR_SEC,
        int(CONFIG.TRIAL.DUR_SEC / CONFIG.GPFA.BIN_SIZE_MS * 1e3),
    )
    fig, ax = mngs.plt.subplots()
    [
        ax.plot_with_ci(xx, mm_ci[k][0], mm_ci[k][1], label=k, id=k)
        for k, v in mm_ci.items()
    ]
    return fig


def plot_box(dist):
    dist_df = pd.concat(
        {k: pd.Series(np.array(dist[k]).reshape(-1)) for k, v in dist.items()}
    ).reset_index()
    dist_df = dist_df.drop(columns="level_1")
    dist_df = dist_df.rename(columns={"level_0": "MTL"})
    dist_df = dist_df.rename(columns={0: "dist_from_O"})

    fig, ax = mngs.plt.subplots()

    ax.sns_boxplot(
        data=dist_df,
        x="MTL",
        y="dist_from_O",
        showfliers=False,
        order=["HIP", "EC", "AMY"],
    )
    ax.set_yscale("log")
    return fig


def main():
    traj = load_trajs()

    # Distance from O
    dist = {k: calc_dist(v) for k, v in traj.items()}

    fig = plot_line(dist)
    mngs.io.save(
        fig, "./data/NT/distance/from_O_of_MTL_regions_line.jpg", from_cwd=True
    )
    mngs.io.save(
        fig.to_sigma(),
        "./data/NT/distance/from_O_of_MTL_regions_line.csv",
        from_cwd=True,
    )

    fig = plot_box(dist)
    mngs.io.save(
        fig, "./data/NT/distance/from_O_of_MTL_regions_box.jpg", from_cwd=True
    )
    mngs.io.save(
        fig.to_sigma(),
        "./data/NT/distance/from_O_of_MTL_regions_box.csv",
        from_cwd=True,
    )

    # Distance between phases
    tt = traj["HIP"]
    nan_mask = np.isnan(tt).any(axis=(-2, -1))
    tt = tt[~nan_mask]

    gs = np.stack(
        [
            mngs.linalg.geometric_median(
                tt[..., v.mid_start : v.mid_end], axis=-1
            )
            for k, v in CONFIG.PHASES.items()
        ],
        axis=-1,
    )

    from itertools import combinations

    from IPython import embed

    embed()

    count = 0
    for ii, jj in combinations(np.arange(gs.shape[-1]), 2):  # 12 patterns
        count += 1
        print(count)
        dist_between_gs = mngs.linalg.cdist(gs[..., ii], gs[..., jj])

    # gs = load_gs()
    __import__("ipdb").set_trace()


if __name__ == "__main__":
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)


#     # Distance between gs
#     dist_between_gs_Hipp_all = np.hstack(
#         [val for val in calc_dist_between_gs(traj_Hipp).values()]
#     )
#     dist_between_gs_EC_all = np.hstack(
#         [val for val in calc_dist_between_gs(traj_EC).values()]
#     )
#     dist_between_gs_Amy_all = np.hstack(
#         [val for val in calc_dist_between_gs(traj_Amy).values()]
#     )
#     df_dist_between_gs_all = mngs.gen.force_dataframe(
#         {
#             "Hipp.": dist_between_gs_Hipp_all,
#             "EC": dist_between_gs_EC_all,
#             "Amy.": dist_between_gs_Amy_all,
#         }
#     )
#     # mngs.io.save(df_dist_between_gs_all, "./res/figs/box/MTL_regions/dist_between_gs_all.csv")
#     # # mngs.io.save(df_dist_all, "./res/figs/box/MTL_regions/dist_from_O_all.csv")

#     # mngs.io.save(pd.DataFrame(dist_between_gs_Hipp), "./res/figs/box/MTL_regions/dist_between_gs_Hipp.csv")
#     # mngs.io.save(pd.DataFrame(dist_between_gs_EC), "./res/figs/box/MTL_regions/dist_between_gs_EC.csv")
#     # mngs.io.save(pd.DataFrame(dist_between_gs_Amy), "./res/figs/box/MTL_regions/dist_between_gs_Amy.csv")
