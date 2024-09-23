#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-24 02:03:01 (ywatanabe)"
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

def plot_line(dist_dict):
    xx = eval(CONFIG.NT.TIME_AXIS)
    fig, ax = mngs.plt.subplots()

    for roi, dist in dist_dict.items():
        described = mngs.gen.describe(dist_dict[roi], method="mean_ci")
        ax.plot_(xx=xx, **described, label=roi, id=roi)
    ax.legend()
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

    # Line plot
    fig = plot_line(dist)
    mngs.io.save(fig, "line.jpg")

    # Box plot
    fig = plot_box(dist)
    mngs.io.save(fig, "box.jpg")


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
