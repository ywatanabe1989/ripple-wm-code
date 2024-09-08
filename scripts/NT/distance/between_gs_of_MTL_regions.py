#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-09 08:12:41 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/from_O_of_MTL_regions.py

"""This script does XYZ."""


"""Imports"""
import sys
from itertools import combinations

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
from scripts.NT.distance.from_O_of_MTL_regions import load_trajs

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def calc_dist_between_gs(traj_MTL_region):
    nan_mask = np.isnan(traj_MTL_region).any(axis=(-2, -1))
    traj_MTL_region = traj_MTL_region[~nan_mask]

    gs = np.stack(
        [
            # np.median(
            mngs.linalg.geometric_median(
                traj_MTL_region[..., v.mid_start : v.mid_end], axis=-1
            )
            for k, v in CONFIG.PHASES.items()
        ],
        axis=-1,
    )

    dist_between_gs = []
    for ii, jj in combinations(np.arange(gs.shape[-1]), 2):  # 6 patterns
        v1 = gs[..., ii]
        v2 = gs[..., jj]
        _dist_between_gs = norm(v1 - v2, axis=-1)
        dist_between_gs.append(_dist_between_gs)
    return np.hstack(dist_between_gs)


def plot_box(dist):
    fig, ax = mngs.plt.subplots()
    ax.sns_boxplot(
        data=dist,
        x="MTL",
        y="distance",
        showfliers=False,
        order=["HIP", "EC", "AMY"],
    )
    ax.set_yscale("log")
    return fig


def main():
    traj = load_trajs()

    dist = {k: calc_dist_between_gs(v) for k, v in traj.items()}
    dist = mngs.pd.force_df(dist).melt()
    dist = dist.rename(columns={"variable": "MTL", "value": "distance"})

    fig = plot_box(dist)
    mngs.io.save(
        fig,
        "./data/NT/distance/between_gs_of_MTL_regions_box.jpg",
        from_cwd=True,
    )


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
