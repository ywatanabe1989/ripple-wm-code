#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-12 03:54:32 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/memory-load/distance_between_gs.py

"""This script calculates and analyzes distances between phase states."""

import sys
from itertools import combinations

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scipy import stats
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


def perform_statistical_tests(merged_df):
    """Perform statistical tests on the merged data."""
    phase_combinations = ["FE", "FM", "FR", "EM", "ER", "MR"]
    results = {}

    np.random.seed(42)
    for phase in phase_combinations:
        corr, p_value = stats.pearsonr(merged_df[phase], merged_df["set_size"])
        sample_size = len(merged_df[phase])
        effect_size = abs(corr)

        shuffled_corrs = [
            stats.pearsonr(
                merged_df[phase], np.random.permutation(merged_df["set_size"])
            )[0]
            for _ in range(1000)
        ]
        p_value_surrogate = np.mean(np.abs(shuffled_corrs) >= np.abs(corr))

        groups = [
            merged_df[merged_df["set_size"] == ss][phase] for ss in [4, 6, 8]
        ]
        h_statistic, p_value_kw = stats.kruskal(*groups)

        bm_results = []
        for ii, jj in combinations(range(3), 2):
            bm_test_output = mngs.stats.brunner_munzel_test(
                groups[ii], groups[jj]
            )
            bm_results.append(
                (f"{[4,6,8][ii]} vs {[4,6,8][jj]}", bm_test_output)
            )

        results[phase] = {
            "correlation": (corr, p_value, f"{sample_size:,}", effect_size),
            "surrogate": p_value_surrogate,
            "kruskal_wallis": (h_statistic, p_value_kw),
            "brunner_munzel": bm_results,
        }

    return results


def analyze_mtl_region(mtl_region, use_log, config):
    """Analyze data for a specific MTL region."""
    gs_data, ti_data = load_data(mtl_region, config)
    distances = calculate_distances(gs_data, config)

    distances_reset = distances.reset_index(drop=True)
    if use_log:
        distances_reset = np.log10(distances_reset)
    ti_reset = ti_data.reset_index(drop=True)
    merged_df = pd.concat(
        [distances_reset, ti_reset[["correct", "response_time", "set_size"]]],
        axis=1,
    )

    correlations = merged_df[
        ["FE", "FM", "FR", "EM", "ER", "MR", "correct", "response_time"]
    ].corrwith(merged_df["set_size"])
    statistical_results = perform_statistical_tests(merged_df)

    return merged_df, correlations, statistical_results


def main():

    # import os
    # from pathlib import Path

    # print(os.getcwd())
    # print(__file__)

    # current_dir = Path(os.getcwd())
    # file_path = Path(__file__)
    # diff = file_path.resolve().relative_to(current_dir)

    # print(f"Relative path: {diff}")

    # /mnt/ssd/ripple-wm-code
    # /mnt/ssd/ripple-wm-code/scripts/memory-load/distance_between_gs_stats.py
    # Relative path: scripts/memory-load/distance_between_gs_stats.py

    mtl_regions = CONFIG.ROI.MTL.keys()
    all_results = []

    for use_log in [False, True]:
        fig, axes = mngs.plt.subplots(
            1, len(mtl_regions), figsize=(8 * len(mtl_regions), 8)
        )

        for idx, mtl_region in enumerate(mtl_regions):
            merged_df, correlations, statistical_results = analyze_mtl_region(
                mtl_region, use_log, CONFIG
            )

            ax = axes[idx] if len(mtl_regions) > 1 else axes
            vlim = 0.275
            ax.sns_heatmap(
                correlations.to_frame().T,
                cmap="coolwarm",
                center=0,
                fmt=".3f",
                vmax=vlim,
                vmin=-vlim,
            )

            for ii, col in enumerate(
                [
                    "FE",
                    "FM",
                    "FR",
                    "EM",
                    "ER",
                    "MR",
                    "correct",
                    "response_time",
                ]
            ):
                xx = ii + 0.5
                yy = 1.0
                dyy = 0.2
                rotation = 20
                corr = correlations[col]

                ax.text(
                    xx,
                    yy - dyy,
                    f"Corr: {corr:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    rotation=rotation,
                    style="italic",
                )

                if col in statistical_results:
                    corr, p_value, sample_size, effect_size = (
                        statistical_results[col]["correlation"]
                    )
                    corr_stars = mngs.stats.p2stars(p_value)
                    ax.text(
                        xx,
                        yy - dyy * 2,
                        f"Corr p: {p_value:.3f} {corr_stars}\n(n={sample_size}, eff={effect_size:.3f})",
                        ha="center",
                        va="center",
                        fontsize=8,
                        rotation=rotation,
                        style="italic",
                    )

                    surr_stars = mngs.stats.p2stars(
                        statistical_results[col]["surrogate"]
                    )
                    ax.text(
                        xx,
                        yy - dyy * 3,
                        f"Surr p: {statistical_results[col]['surrogate']:.3f} {surr_stars}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        rotation=rotation,
                        style="italic",
                    )

                    kw_stars = mngs.stats.p2stars(
                        statistical_results[col]["kruskal_wallis"][1]
                    )
                    ax.text(
                        xx,
                        yy - dyy * 4,
                        f"KW p: {statistical_results[col]['kruskal_wallis'][1]:.3f} {kw_stars}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        rotation=rotation,
                        style="italic",
                    )

            ax.set_title(
                f"{mtl_region} ({'Log' if use_log else 'Linear'} scale)"
            )

            ax.set_ylim(-1, 1)

            for phase, results in statistical_results.items():
                all_results.append(
                    {
                        "MTL": mtl_region,
                        "Scale": "Log" if use_log else "Linear",
                        "Phase": phase,
                        "Correlation": f"{results['correlation'][0]:.3f}",
                        "Correlation_p": f"{results['correlation'][1]:.3f}",
                        "Sample_size": f"{results['correlation'][2]}",
                        "Effect_size": f"{results['correlation'][3]:.3f}",
                        "Surrogate_p": f"{results['surrogate']:.3f}",
                        "KW_H": f"{results['kruskal_wallis'][0]:.3f}",
                        "KW_p": f"{results['kruskal_wallis'][1]:.3f}",
                        "BM_4v6_w": f"{results['brunner_munzel'][0][1]['w_statistic']:.3f}",
                        "BM_4v6_p": f"{results['brunner_munzel'][0][1]['p_value']:.3f}",
                        "BM_4v8_w": f"{results['brunner_munzel'][1][1]['w_statistic']:.3f}",
                        "BM_4v8_p": f"{results['brunner_munzel'][1][1]['p_value']:.3f}",
                        "BM_6v8_w": f"{results['brunner_munzel'][2][1]['w_statistic']:.3f}",
                        "BM_6v8_p": f"{results['brunner_munzel'][2][1]['p_value']:.3f}",
                    }
                )

        plt.tight_layout()
        mngs.io.save(
            fig,
            f"./data/memory_load/summary_plot_{'log' if use_log else 'linear'}.jpg",
        )

    mngs.io.save(
        pd.DataFrame(all_results), "./data/memory_load/summary_stats.csv"
    )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
