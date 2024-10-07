#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 18:13:34 (ywatanabe)"
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

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def load_gs():
    LPATHS = mngs.io.glob(CONFIG.PATH.NT_GS_TRIAL)
    LPATHS = mngs.gen.search(["Session_01", "Session_02"], LPATHS)[1]
    gs = {}
    for mtl in CONFIG.ROI.MTL.keys():
        lpaths_mtl = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS)[1]
        das = [mngs.io.load(lpath) for lpath in lpaths_mtl]
        gs[mtl] = xr.concat(das, dim="session", coords="minimal")
        print(f"n_lpaths_mtl: {mtl, len(lpaths_mtl)}")
    return gs


def calc_distances_between_gs(gs_mtl):
    phases = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    phase_combinations = list(itertools.combinations(phases, 2))

    distances = []
    for phase1, phase2 in phase_combinations:

        v1 = np.array(gs_mtl.sel(phase=phase1))
        v2 = np.array(gs_mtl.sel(phase=phase2))

        v1 = v1.reshape(-1, v1.shape[-1])
        v2 = v2.reshape(-1, v2.shape[-1])

        mask = ~(np.isnan(v1).any(axis=-1) + np.isnan(v2).any(axis=-1))

        distance = norm(v1[mask] - v2[mask], axis=-1)

        distances.append(distance)

    return np.hstack(distances)

def plot_box(dist):
    fig, ax = mngs.plt.subplots()
    dist = dist[~dist.isna().any(axis=1)]

    labels = ["HIP", "EC", "AMY"]
    data = [dist[dist.MTL == ll].distance for ll in labels]
    print([len(dd) for dd in data])  # [7338, 5016, 6198]

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
    )
    ax.set_yscale("log")
    return fig


def main():
    gs = load_gs()

    ################################################################################
    # checking
    ################################################################################
    gs_HIP = np.array(gs["HIP"])
    gs_HIP_F = gs_HIP[..., -1]
    gs_HIP_F = gs_HIP_F.reshape(-1, gs_HIP_F.shape[-1])
    gs_HIP_F = gs_HIP_F[~np.isnan(gs_HIP_F).any(axis=-1)]
    ################################################################################

    dists = {k: calc_distances_between_gs(gs_mtl) for k, gs_mtl in gs.items()}
    dists = {k: np.hstack(v) for k, v in dists.items()}
    dists = (
        mngs.pd.force_df(dists)
        .melt()
        .rename(columns={"variable": "MTL", "value": "distance"})
    )
    dists = dists[~dists.isna().any(axis=1)]

    fig = plot_box(dists)
    mngs.io.save(
        fig,
        "box.jpg",
    )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
