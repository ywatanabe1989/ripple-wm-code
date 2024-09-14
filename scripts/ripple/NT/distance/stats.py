#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 20:39:57 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts import utils
from scripts.ripple.NT.distance.from_O_lineplot import calc_dist_by_condi

"""Functions & Classes"""


def bm(data):
    from itertools import combinations

    results = {}
    for swr_type in ["SWR-", "SWR+"]:
        for match in CONFIG.MATCHES:
            for phase in ["Encoding", "Retrieval"]:
                c = {
                    "SWR_type": swr_type,
                    "phase": phase,
                    "match": match,
                }
                for p1, p2 in combinations(CONFIG.RIPPLE.BINS.keys(), 2):
                    c1 = c.copy()
                    c2 = c.copy()
                    c1["period"] = p1
                    c2["period"] = p2

                    d1 = mngs.pd.slice(data, c1).distances.iloc[0]
                    d2 = mngs.pd.slice(data, c2).distances.iloc[0]

                    results[(swr_type, match, phase, f"{p1} vs. {p2}")] = (
                        pd.DataFrame(
                            pd.Series(mngs.stats.brunner_munzel_test(d1, d2))
                        ).T
                    )

    results_df = pd.concat(results, axis=0)
    results_df.index.names = [
        "SWR_type",
        "match",
        "phase",
        "comparison",
        "delete_me",
    ]
    results_df = results_df.reset_index()
    del results_df["delete_me"]

    results_df["stars"] = results_df["p_value"].apply(mngs.stats.p2stars)

    results_df["match"] = results_df["match"].replace(
        {1: "Match IN", 2: "Mismatch OUT"}
    )

    results_df = results_df.set_index(
        ["SWR_type", "match", "phase", "comparison"]
    )

    return results_df


def main():
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    data = calc_dist_by_condi(swr_p_all, swr_m_all)

    bm_stats = bm(data)
    print(bm_stats)
    mngs.io.save(bm_stats, "bm_stats.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        fig_scale=2,
        font_size_legend=3,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
