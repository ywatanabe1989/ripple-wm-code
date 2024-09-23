#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-21 20:38:01 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/stats/SWR+_vs_SWR-.py

"""This script compares (absolute) cosine of (rSWR+, vER) and (rSWR+, vOR)."""

"""Imports"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from itertools import product, combinations

"""Parameters"""
COMPARISONS = [
    "rSWR+_vs_vER",
    "rSWR+_vs_vOR",
    "rSWR-_vs_vER",
    "rSWR-_vs_vOR",
]


"""Functions & Classes"""
def find_cols(df, match, set_size):
    labels = {}
    for comparison in COMPARISONS:
        exp = f"{CONFIG.MATCHES_STR[str(match)]}-{set_size}-{comparison.replace('SWR+', 'SWR\+')}"
        col = mngs.gen.search(exp, df.columns, ensure_one=True)[1][0]
        labels[col] = comparison.replace("_vs_", r" - ")
    return labels

def perform_bm_test(df, match, set_size, take_abs):
    labels = find_cols(df, match, set_size)
    results = []

    for col_1, col_2 in combinations(labels.keys(), 2):
        data_1 = df[col_1]
        data_2 = df[col_2]

        data_1 = data_1[~data_1.isna()]
        data_2 = data_2[~data_2.isna()]

        if take_abs:
            data_1 = np.abs(data_1)
            data_2 = np.abs(data_2)

        result = mngs.stats.brunner_munzel_test(data_2, data_1)
        result["v1"] = f"{labels[col_1]}"
        result["v2"] = f"{labels[col_2]}"
        results.append(result)
    results = pd.DataFrame(results)
    results["stars"] = results["p_value"].apply(mngs.stats.p2stars)
    print(results)
    return results

def main():
    stats_all = []
    for take_abs in [True, False]:
        for match in ["all"] + CONFIG.MATCHES:
            for set_size in ["all"]:# + CONFIG.SET_SIZES:
                for measure in ["cosine"]:
                    for vSWR_def in CONFIG.RIPPLE.DIRECTIONS:
                        mngs.gen.print_block(
                            f"vSWR_def: {vSWR_def}, {measure}, "
                            f"{CONFIG.MATCHES_STR[str(match)]}, Set size {set_size}"
                        )

                        # Loading
                        lpath = (
                            f"./scripts/ripple/NT/direction/kde_plot/kde_{vSWR_def}/"
                            f"{measure}/set_size_{set_size}_box.csv"
                        )
                        df = mngs.io.load(lpath)

                        # Statistical test
                        stats = perform_bm_test(df, match, set_size, take_abs)

                        # Aggregation
                        stats["match"] = match
                        stats["set_size"] = set_size
                        stats["measure"] = measure
                        stats["vSWR_def"] = vSWR_def
                        stats["is_abs"] = str(take_abs)
                        stats_all.append(stats)

    stats_all = pd.concat(stats_all)
    mngs.io.save(stats_all, "stats_all.csv")
    mngs.io.save(stats_all[stats_all.p_value < 0.05], "stats_significant.csv")

if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
