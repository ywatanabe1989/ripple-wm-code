#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-21 16:07:55 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/stats/vector_direction_test.py

"""This script tests the directions of eSWR/rSWR in match conditions."""


"""Imports"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts import utils
from scripts.ripple.NT.direction.kde_plot import calc_measure
from itertools import product

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Parameters"""
VECTORS = [
    "eSWR+",
    "rSWR+",
    "eSWR-",
    "rSWR-",
    "vER",
    ]
MATCH = "Match ALL"
SET_SIZE = "all"

"""Functions & Classes"""
def perform_bm_test(data, uniform):
    result = mngs.stats.brunner_munzel_test(uniform, data)
    result["stars"] = mngs.stats.p2stars(result["p_value"])
    return result

def _compare_with_uniform(similarity):
    results = []

    data = similarity.dropna().values
    uniform = np.random.uniform(low=-1, high=1, size=len(data))

    # Regular comparison
    result = perform_bm_test(data, uniform)
    result["comparison"] = f"vs Uniform"
    result["abs"] = False
    results.append(result)

    # Absolute value comparison
    abs_data = np.abs(data)
    abs_uniform = np.abs(uniform)
    abs_result = perform_bm_test(abs_data, abs_uniform)
    abs_result["comparison"] = f"vs Uniform"
    abs_result["abs"] = True
    results.append(abs_result)

    out = pd.DataFrame(results)
    return out

def compare_with_uniform(df):
    results = []
    df = mngs.pd.merge_cols(df, list(df.columns[1:]), sep2="--")

    for ii, group in enumerate(df.merged.unique()):
        indi = df.merged == group
        _df = df[indi]
        result = _compare_with_uniform(_df["similarity"])

        meta = _df[_df.columns[1:-1]].drop_duplicates()
        assert len(meta) == 1

        result = pd.concat([result, meta], axis=1).ffill()

        results.append(result)
    stats = pd.concat(results)
    return stats


def calc_similarity(swr_all):
    dfs = []
    for vSWR_def in CONFIG.RIPPLE.DIRECTIONS:
        for swr_type_1, swr_type_2 in product(CONFIG.RIPPLE.TYPES, CONFIG.RIPPLE.TYPES):
            for ca1 in CONFIG.ROI.CA1:
                for phase_1, phase_2 in product(["Encoding", "Retrieval"], ["Encoding", "Retrieval"]):
                    # CA1
                    swr_all["sub"] = swr_all["subject"]
                    swr_ca1 = mngs.pd.slice(swr_all, ca1)

                    # Vectors for Match IN and Mismatch OUT
                    condi_in = {
                        "match": 1,
                        "phase": phase_1,
                        "swr_type": swr_type_1,
                    }
                    condi_out = {
                        "match": 2,
                        "phase": phase_2,
                        "swr_type": swr_type_2,
                    }
                    v_in = mngs.pd.slice(swr_ca1, condi_in)
                    v_out = mngs.pd.slice(swr_ca1, condi_out)

                    # Main
                    similarity = calc_measure(np.vstack(v_in[vSWR_def]), np.vstack(v_out[vSWR_def]), "cosine")
                    # Aggregation
                    df = pd.DataFrame({"similarity": similarity})

                    for k,v in condi_in.items():
                        df[f"{k}_in"] = v

                    for k,v in condi_out.items():
                        df[f"{k}_out"] = v

                    df["vSWR_def"] = vSWR_def
                    dfs.append(df)
    df = pd.concat(dfs)
    return df

def plot_similarity(df):
    # Grouping
    df = mngs.pd.merge_cols(df, list(df.columns[1:]), sep2="--")

    # Main
    fig, ax = mngs.plt.subplots()

    # Convert 'similarity' to numeric, coercing errors to NaN
    df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')

    # Remove rows with NaN in 'similarity'
    df = df.dropna(subset=['similarity'])

    ax.sns_violinplot(
        data=df[["similarity", "merged"]],
        x="merged",
        y="similarity",
    )

    del df["merged"]
    return fig

def calc_stats(df):
    stats = df.groupby(list(df.columns[1:])).agg({
        "similarity": [
            "mean",
            "median",
            "std",
            lambda x: np.percentile(x, [2.5, 97.5]).tolist(),
            lambda x: (np.percentile(x, 75) - np.percentile(x, 25)),
            "count"
        ]
    })
    stats.columns = ["mean", "median", "std", "ci", "iqr", "n"]
    return stats

def calc_stats_abs(df):
    # Calculate stats for absolute similarity
    df["abs_similarity"] = df["similarity"].abs()
    abs_stats = df.groupby(list(df.columns[1:-1])).agg({
        "abs_similarity": [
            "mean",
            "median",
            "std",
            lambda x: np.percentile(x, [2.5, 97.5]).tolist(),
            lambda x: (np.percentile(x, 75) - np.percentile(x, 25)),
            "count"
        ]
    })
    abs_stats.columns = ["mean", "median", "std", "ci", "iqr", "n"]
    return abs_stats

def main():
    # Loading
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    swr_p_all["swr_type"] = "SWR+"
    swr_m_all["swr_type"] = "SWR-"
    swr_all = pd.concat([swr_p_all, swr_m_all])

    # Main
    df = calc_similarity(swr_all)


    # Plotting
    fig = plot_similarity(df)
    mngs.io.save(fig, "fig.jpg")

    # mean, median, std, ci, iqr, n
    stats = calc_stats(df)
    mngs.io.save(stats, "stats.csv")

    # vs. Uniform distribution
    stats_vs_uniform = compare_with_uniform(df)
    mngs.io.save(stats_vs_uniform, "stats_vs_uniform.csv")

    abs_stats = calc_stats_abs(df)
    mngs.io.save(abs_stats, "abs_stats.csv")



if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
