#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 22:06:31 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/speed/from_O/MTL_regions.py

"""This script calculates speed per 50 ms from neural trajectory"""


"""Imports"""
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scipy.linalg import norm

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""
import numpy as np
from scripts.NT.distance.from_O.MTL_regions import load_trajs


def calc_speed(trajectory):
    # Calculate displacements between consecutive time bins
    displacements = np.diff(trajectory, axis=2)

    # Calculate distances
    distances = np.linalg.norm(displacements, axis=1)

    # Calculate speed (assuming 50ms time bins)
    speeds = distances / 0.05  # 50ms = 0.05s

    return speeds


def plot_line(speed):
    described = {
        k: pd.DataFrame(mngs.gen.describe(v, "mean_ci"))
        for k, v in speed.items()
    }
    xx = eval(CONFIG.NT.TIME_AXIS)[:-1]
    fig, ax = mngs.plt.subplots()
    [
        ax.mplot(xx=xx, **vv, label=kk, color=CC[CONFIG.COLORS[kk]])
        for kk, vv in described.items()
    ]
    ax.legend(loc="upper left")
    ax.set_xyt("Time [s]", "Speed [a.u.]", None)
    return fig


def main():
    # Loading
    traj = load_trajs()

    # Speeds betwen consective bins
    speed = {k: calc_speed(v) for k, v in traj.items()}

    # Line
    fig = plot_line(speed)
    mngs.io.save(fig, "line.jpg", from_cwd=True)

    # fig = plot_line(dist)
    # mngs.io.save(
    #     fig, "./data/NT/distance/from_O_of_MTL_regions_line.jpg", from_cwd=True
    # )
    # mngs.io.save(
    #     fig.to_sigma(),
    #     "./data/NT/distance/from_O_of_MTL_regions_line.csv",
    #     from_cwd=True,
    # )

    # fig = plot_box(dist)
    # mngs.io.save(
    #     fig, "./data/NT/distance/from_O_of_MTL_regions_box.jpg", from_cwd=True
    # )
    # mngs.io.save(
    #     fig.to_sigma(),
    #     "./data/NT/distance/from_O_of_MTL_regions_box.csv",
    #     from_cwd=True,
    # )


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


# def calc_dist(traj_MTL_region):
#     norm_nonnan_MTL_region = {}
#     for i_bin in range(traj_MTL_region.shape[-1]):  # i_bin = 0
#         traj_MTL_region_i_bin = traj_MTL_region[..., i_bin]
#         norm_MTL_region_i_bin = norm(
#             traj_MTL_region_i_bin[
#                 ~np.isnan(traj_MTL_region_i_bin).any(axis=1)
#             ],
#             axis=-1,
#         )
#         norm_nonnan_MTL_region[i_bin] = norm_MTL_region_i_bin
#     return mngs.pd.force_df(norm_nonnan_MTL_region)


# def calc_mean_and_ci(dist):
#     dist_mm = np.nanmean(dist, axis=0)
#     dist_sd = np.nanstd(dist, axis=0)
#     dist_nn = (~np.isnan(dist)).astype(int).sum(axis=0)
#     dist_ci = 1.96 * dist_mm / (dist_sd * dist_nn)
#     return dist_mm, dist_ci


# def plot_line(dist):
#     mm_ci = {k: calc_mean_and_ci(v) for k, v in dist.items()}

#     xx = np.linspace(
#         0,
#         CONFIG.TRIAL.DUR_SEC,
#         int(CONFIG.TRIAL.DUR_SEC / CONFIG.GPFA.BIN_SIZE_MS * 1e3),
#     )
#     fig, ax = mngs.plt.subplots()
#     [
#         ax.plot_with_ci(xx, mm_ci[k][0], mm_ci[k][1], label=k, id=k)
#         for k, v in mm_ci.items()
#     ]
#     return fig


# def plot_box(dist):
#     dist_df = pd.concat(
#         {k: pd.Series(np.array(dist[k]).reshape(-1)) for k, v in dist.items()}
#     ).reset_index()
#     dist_df = dist_df.drop(columns="level_1")
#     dist_df = dist_df.rename(columns={"level_0": "MTL"})
#     dist_df = dist_df.rename(columns={0: "dist_from_O"})

#     fig, ax = mngs.plt.subplots()

#     ax.sns_boxplot(
#         data=dist_df,
#         x="MTL",
#         y="dist_from_O",
#         showfliers=False,
#         order=["HIP", "EC", "AMY"],
#     )
#     ax.set_yscale("log")
#     return fig
