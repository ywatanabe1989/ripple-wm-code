#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-20 19:10:31 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/stats/SWR+_vs_SWR-.py

"""This script compares the significance of (absolute) cosine similarity between SWR+ and SWR-."""

# 	w_statistic	p_value	dof	effsize	v1	v2	stars	match	set_size	measure	vSWR_def	is_abs
# 2	2.008	0.047	125.221	0.602	rSWR+ - vER	rSWR- - vER	*	1	8	cosine	1	TRUE
# 0	2.212	0.027	909.349	0.541	eSWR+ - rSWR+	eSWR- - rSWR-	*	2	4	cosine	1	TRUE
# 1	-2.131	0.035	141.814	0.399	eSWR+ - vER	eSWR- - vER	*	2	6	cosine	2	FALSE
# 1	-2.04	0.044	124.257	0.397	eSWR+ - vER	eSWR- - vER	*	2	8	cosine	1	FALSE
# 1	2.837	0.005	120.431	0.643	eSWR+ - vER	eSWR- - vER	**	1	6	cosine	2	FALSE
# 0	2.794	0.005	7886.889	0.518	eSWR+ - rSWR+	eSWR- - rSWR-	**	2	all	cosine	2	FALSE
# 0	3.222	0.001	4863.628	0.526	eSWR+ - rSWR+	eSWR- - rSWR-	***	all	4	cosine	1	TRUE
# 0	-3.367	0.001	10517.165	0.481	eSWR+ - rSWR+	eSWR- - rSWR-	***	1	all	cosine	2	TRUE
# 0	3.381	0.001	35707.651	0.51	eSWR+ - rSWR+	eSWR- - rSWR-	***	all	all	cosine	2	FALSE

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


"""Functions & Classes"""
def find_cols(df, match, set_size):
    labels_p = {}
    labels_m = {}
    for comparison in COMPARISONS_P:
        exp_p = f"{CONFIG.MATCHES_STR[str(match)]}-{set_size}-{comparison.replace('SWR+', 'SWR\+')}"
        exp_m = exp_p.replace("SWR\+", "SWR-")

        col_p = mngs.gen.search(exp_p, df.columns, ensure_one=True)[1][0]
        col_m = mngs.gen.search(exp_m, df.columns, ensure_one=True)[1][0]

        labels_p[col_p] = comparison.replace("_vs_", r" - ")
        labels_m[col_m] = comparison.replace('SWR+', 'SWR-').replace("_vs_", r" - ")

    assert len(labels_p) == len(labels_m)
    return labels_p, labels_m

def perform_bm_test(df, match, set_size, take_abs):
    labels_p, labels_m = find_cols(df, match, set_size)
    results = []
    for col_p, col_m in zip(labels_p.keys(), labels_m.keys()):
        data_p = df[col_p]
        data_m = df[col_m]

        data_p = data_p[~data_p.isna()]
        data_m = data_m[~data_m.isna()]

        if take_abs:
            data_p = np.abs(data_p)
            data_m = np.abs(data_m)

        result = mngs.stats.brunner_munzel_test(data_m, data_p)
        result["v1"] = f"{labels_p[col_p]}"
        result["v2"] = f"{labels_m[col_m]}"
        results.append(result)
    results = pd.DataFrame(results)
    results["stars"] = results["p_value"].apply(mngs.stats.p2stars)
    print(results)
    return results

def main():
    stats_all = []
    for take_abs in [True, False]:
        for match in ["all"] + CONFIG.MATCHES:
            for set_size in ["all"] + CONFIG.SET_SIZES:
                for measure in ["cosine"]:
                    for vSWR_def in [1,2]:
                        mngs.gen.print_block(
                            f"vSWR_def{vSWR_def}, {measure}, "
                            f"{CONFIG.MATCHES_STR[str(match)]}, Set size {set_size}"
                        )

                        # Loading
                        lpath = (
                            f"./scripts/ripple/NT/direction/kde_plot/kde_vSWR_def{vSWR_def}/"
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
                        # mngs.io.save(stats, f"{measure}_vSWR_def{vSWR_def}-match-{str(match)}-set_size-{set_size}.csv")

    stats_all = pd.concat(stats_all)
    mngs.io.save(stats_all, "stats_all.csv")

    mngs.io.save(stats_all[stats_all.p_value < 0.05], "stats_significant.csv")

if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
