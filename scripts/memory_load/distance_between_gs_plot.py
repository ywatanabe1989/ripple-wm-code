#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-12 04:03:43 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/memory-load/distance_between_gs.py

"""This script calculates and analyzes distances between phase states."""

import sys
from itertools import combinations

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scipy.spatial.distance import norm

sys.path = ["."] + sys.path
try:
    from scripts import utils
except ImportError:
    pass


def load_data(mtl_region, config):
    """Load and preprocess data for the specified MTL region."""
    lpaths_gs = mngs.gen.glob(config.PATH.NT_GS_TRIAL)
    lpaths_gs = mngs.gen.search(config.SESSION.FIRST_TWO, lpaths_gs)[1]
    lpaths_gs = mngs.gen.search(config.ROI.MTL[mtl_region], lpaths_gs)[1]

    gs_list, ti_list = [], []
    for lpath_gs in lpaths_gs:
        lpath_ti = mngs.gen.replace(
            config.PATH.TRIALS_INFO, utils.parse_lpath(lpath_gs)
        )
        gs_list.append(mngs.io.load(lpath_gs))
        ti_list.append(mngs.io.load(lpath_ti))

    gs_data = np.vstack(gs_list)
    ti_data = pd.concat(ti_list)
    mask = ~np.isnan(gs_data).any(axis=(1, 2))
    return gs_data[mask], ti_data[mask]


def calculate_distances(gs_data, config):
    """Calculate distances between phase states."""
    distances = {}
    for idx1, idx2 in combinations(range(len(config.PHASES)), 2):
        phase1, phase2 = (
            list(config.PHASES.keys())[idx1],
            list(config.PHASES.keys())[idx2],
        )
        vec1, vec2 = gs_data[..., idx1], gs_data[..., idx2]
        mask = ~(np.isnan(vec1).any(axis=-1) | np.isnan(vec2).any(axis=-1))
        distances[f"{phase1[0]}{phase2[0]}"] = pd.Series(
            norm(vec1[mask] - vec2[mask], axis=-1)
        )
    return pd.concat(distances, axis=1)


def analyze_mtl_region(mtl_region, use_log, config):
    """Analyze data for a specific MTL region."""
    gs_data, ti_data = load_data(mtl_region, config)
    distances = calculate_distances(gs_data, config)

    distances_reset = distances.reset_index(drop=True)
    if use_log:
        distances_reset = np.log10(distances_reset)
    ti_reset = ti_data.reset_index(drop=True)

    distances_reset = distances_reset.melt().rename(
        columns={
            "variable": "phase_combination",
            "value": "distance",
        }
    )
    repeated_ti = pd.concat([ti_reset] * 6, ignore_index=True)
    merged_df = pd.concat(
        [
            distances_reset,
            repeated_ti[["correct", "response_time", "set_size"]],
        ],
        axis=1,
    )

    return merged_df


def main():
    mtl_regions = CONFIG.ROI.MTL.keys()
    all_results = []

    for use_log in [False, True]:
        fig, axes = mngs.plt.subplots(
            1, len(mtl_regions), figsize=(8 * len(mtl_regions), 8)
        )

        for idx, mtl_region in enumerate(mtl_regions):
            merged_df = analyze_mtl_region(mtl_region, use_log, CONFIG)

            # Plotting
            fig, ax = mngs.plt.subplots()
            ax.sns_boxplot(
                data=merged_df,
                x="phase_combination",
                y="distance",
                hue="set_size",
                showfliers=False,
            )
            title = f"{mtl_region}_{'log' if use_log else 'linear'}"
            ax.set_xyt("Phase Combination", "Distance [a.u.]", title)
            mngs.io.save(fig, f"./memory_load/distance_between_gs/{title}.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
