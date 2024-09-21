#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-21 09:17:32 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/stats/vector_direction_test.py

"""This script compares the significance of (absolute) cosine similarity between eSWR, rSWR, and vER."""


"""Imports"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Parameters"""
COMPARISONS_P = ["eSWR+_vs_rSWR+",
                 "eSWR+_vs_vER",
                 "rSWR+_vs_vER",
                 ]
COMPARISONS_M = ["eSWR-_vs_rSWR-",
                 "eSWR-_vs_vER",
                 "rSWR-_vs_vER",
                 ]
MATCH = "Match ALL"
SET_SIZE = "all"

"""Functions & Classes"""
def find_cols(df):
    labels = {}
    for comparisons in [COMPARISONS_P, COMPARISONS_M]:
        for comparison in comparisons:
            v1, v2 = comparison.split("_vs_")
            cols = mngs.gen.search(
                f"{MATCH}-{SET_SIZE}-{comparison.replace('SWR+', 'SWR\+')}",
                df.columns)[1]
            assert len(cols) == 1
            col = cols[0]
            labels[col] = comparison.replace('SWR\+', 'SWR+').replace("_vs_", " and ")
    return labels

def perform_bm_test(df, take_abs):
    labels = find_cols(df)
    results = []
    for col in labels.keys():
        data = df[col]
        data = np.array(data[~data.isna()])
        uniform = np.random.uniform(low=-1, high=1, size=len(data))
        if take_abs:
            data = np.abs(data)
            uniform = np.abs(uniform)
        result = mngs.stats.brunner_munzel_test(uniform, data)
        result["vectors"] = labels[col]
        results.append(result)
    results = pd.DataFrame(results)
    results["stars"] = results["p_value"].apply(mngs.stats.p2stars)
    print(results)

    # mngs.io.save(results, "stats.csv")
    """
       w_statistic  p_value        dof  effsize vectors to calculate absolute cosine value stars
    0       21.929      0.0  33098.106    0.567                            eSWR+ and rSWR+   ***
    1        5.107      0.0    813.390    0.599                              eSWR+ and vER   ***
    2        4.950      0.0    726.651    0.602                              rSWR+ and vER   ***
    3       21.415      0.0  34740.429    0.565                            eSWR- and rSWR-   ***
    4        4.722      0.0    808.790    0.592                              eSWR- and vER   ***
    5        5.008      0.0    737.430    0.601                              rSWR- and vER   ***


    This indicates that all periods, especially baseline (vER), show more consistent directional relationships than would be expected by chance, with SWRs during retrieval showing stronger alignment than SWRs during encoding.

    baseline is e/rSWR-
    """

    return results

def main():
    stats_all = []
    for vSWR_def in CONFIG.RIPPLE.DIRECTIONS:
        for take_abs in [True, False]:
            mngs.gen.print_block(
                f"{vSWR_def} - abs: {take_abs}"
            )
            lpath = f"./scripts/ripple/NT/direction/kde_plot/kde_{vSWR_def}/cosine/set_size_all_box.csv"
            df = mngs.io.load(lpath)
            stats = perform_bm_test(df, take_abs)
            stats["vSWR_def"] = vSWR_def
            stats["is_abs"] = take_abs
            stats_all.append(stats)
    stats_all = pd.concat(stats_all)
    mngs.io.save(stats_all, "stats_all.csv")

if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
