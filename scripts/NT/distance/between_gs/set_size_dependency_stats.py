#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-05 18:45:41 (ywatanabe)"
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

# # This can add set_size to gs
# MTL = "HIP"
# lpaths_mtl = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS)[1]
# LPATHS_GS = mngs.io.glob(CONFIG.PATH.NT_GS_TRIAL)
# LPATHS_GS = mngs.gen.search(["Session_01", "Session_02"], LPATHS)[1]
# lpath_gs = LPATHS_GS[0]
# parsed = utils.parse_lpath(lpath_gs)
# GS = mngs.io.load(lpath_gs)
# TI = mngs.io.load(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed))
# # GS is xr
# GS = GS.swap_dims({'trial': 'set_size'}).assign_coords(set_size=("set_size", TI["set_size"]))




# def load_gs():
#     LPATHS = mngs.io.glob(CONFIG.PATH.NT_GS_TRIAL)
#     LPATHS = mngs.gen.search(["Session_01", "Session_02"], LPATHS)[1]
#     gs = {}
#     for mtl in CONFIG.ROI.MTL.keys():
#         lpaths_mtl = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS)[1]
#         das = [mngs.io.load(lpath) for lpath in lpaths_mtl]
#         __import__("ipdb").set_trace()
#         gs[mtl] = xr.concat(das, dim="session", coords="minimal")
#         print(f"n_lpaths_mtl: {mtl, len(lpaths_mtl)}")
#     return gs


def load_gs(N_FACTORS=3):

    def load_corresponding_TI(lpath):
        # Loading the corresponding trials info
        parsed = utils.parse_lpath(lpath)
        lpath_TI = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
        TI = mngs.io.load(lpath_TI)
        return TI

    LPATHS_GS = mngs.io.glob(CONFIG.PATH.NT_GS_TRIAL)
    LPATHS_GS = mngs.gen.search(["Session_01", "Session_02"], LPATHS_GS)[1]
    gs_agg = {}
    for mtl in CONFIG.ROI.MTL.keys():
        lpaths_gs_mtl = mngs.gen.search(CONFIG.ROI.MTL[mtl], LPATHS_GS)[1]
        gss = []
        for lpath_gs in lpaths_gs_mtl:
            gs = mngs.io.load(lpath_gs)
            gs = gs[:, :N_FACTORS, :]

            # Loading the corresponding trials info
            TI = load_corresponding_TI(lpath_gs)

            # Adding set size information to gs gsta
            gs = gs.swap_dims({'trial': 'set_size'})
            gs = gs.assign_coords(set_size=("set_size", TI["set_size"]))

            # Aggregation
            gss.append(gs)

        gs_agg[mtl] = gss

        print(f"n_lpaths_mtl: {mtl, len(lpaths_gs_mtl)}")
    return gs_agg



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

from itertools import combinations

def gs2dists(gs):
    dists = {}
    for p1, p2 in combinations(CONFIG.PHASES.keys(), 2):
        for set_size in CONFIG.SET_SIZES:
            gs_1 = gs[..., gs.phase == p1].squeeze() # 50,3
            gs_2 = gs[..., gs.phase == p2].squeeze() # 50,3

            indi_ss_1 = gs_1.set_size == set_size
            indi_ss_2 = gs_2.set_size == set_size
            assert (indi_ss_1 == indi_ss_2).all()
            indi_ss = indi_ss_1

            gs_1_ss = gs_1[indi_ss]
            gs_2_ss = gs_2[indi_ss]

            dists[f"{p1[0]}{p2[0]}-set_size_{set_size}"] = mngs.linalg.nannorm(gs_1_ss - gs_2_ss)

    dists = mngs.pd.force_df(dists)
    return dists

def gs_all_to_dists(gs_all):
    dists = mngs.gen.listed_dict()
    for mtl in CONFIG.ROI.MTL.keys():
        gs_mtl = gs_all[mtl]
        for gs in gs_mtl:
            dist = gs2dists(gs)
            dists[mtl].append(dist)

    dists = {k: pd.concat(v) for k,v in dists.items()}
    return dists

def main():
    gs_all = load_gs()

    dists = gs_all_to_dists(gs_all)


    fig, axes = mngs.plt.subplots(ncols=len(dists.keys()))
    for ax, mtl in zip(axes, dists.keys()):
        df = dists[mtl].melt()
        df = df.rename(columns={
            "variable": "phase_combi_set_size",
            "value": "distance",
        })
        ax.sns_boxplot(
            data=df,
            x="phase_combi_set_size",
            y="distance",
            showfliers=False,
        )
        ax.set_xyt(None, None, mtl)
        ax.rotate_labels(x=90, y=0)
        ax.extend(y_ratio=0.5)
        ax.set_yscale("log")

    mngs.io.save(fig, "box.jpg")

    # df = mngs.io.load("/tmp/fake-ywatanabe/temp.csv")

    # __import__("ipdb").set_trace()


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
