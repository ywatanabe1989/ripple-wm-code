#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 17:14:47 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import importlib
import logging
import os
import re
import sys
import warnings
from bisect import bisect_left
from glob import glob
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from scripts import utils
from tqdm import tqdm

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def add_phase(xx_all):
    xx_all["phase"] = str(np.nan)
    for phase, phase_data in CONFIG.PHASES.items():
        indi_phase = (phase_data.start <= xx_all.peak_i) * (
            xx_all.peak_i < phase_data.end
        )
        xx_all.loc[indi_phase, "phase"] = phase
    return xx_all


def calc_distances(xx_all):
    # NT to distance
    nt_xx = np.stack(xx_all.NT, axis=0)
    dd_xx = np.sqrt((nt_xx**2).sum(axis=1))
    return dd_xx


def prepare_data(xxp_all, xxm_all):
    PHASES = ["Encoding", "Retrieval"]

    condi_all = {
        "match": CONFIG.MATCHES,
        "phase": PHASES,
    }

    data = {}
    for xx_all, swr_type in zip([xxp_all, xxm_all], ["SWR+", "SWR-"]):
        for condi in mngs.gen.yield_grids(condi_all):
            key = [swr_type, condi["match"], condi["phase"]]
            df = mngs.pd.slice(xx_all, condi)
            dd_xx = calc_distances(df)
            data[tuple(key + ["all"])] = dd_xx
            for period in ["pre", "mid", "post"]:
                start, end = CONFIG.RIPPLE.BINS[period]
                dd_period = dd_xx[:, start:end]
                data[tuple(key + [period])] = dd_xx
    return data


def plot_line(data):
    df = pd.DataFrame(
        data.keys(),
        columns=["SWR_type", "match", "phase", "period"],
    )

    conditions = list(
        (
            df[["SWR_type", "match", "phase"]].drop_duplicates().T.to_dict()
        ).values()
    )

    fig, axes = mngs.plt.subplots(
        nrows=1,
        ncols=len(conditions) // 2,
    )

    PHASES = list(df.phase.unique())

    for condi in conditions:
        i_ax = (condi["match"] - 1) * len(PHASES) + mngs.gen.search(
            condi["phase"], PHASES
        )[0][0]
        ax = axes[i_ax]

        dd = data[(*condi.values(), "all")]

        described = mngs.gen.describe(dd, "mean_ci", axis=0)

        tt = np.arange(len(described["mean"])) * CONFIG.GPFA.BIN_SIZE_MS
        tt -= int(tt.mean())

        ax.mplot(
            xx=tt,
            **described,
            label=condi["SWR_type"],
            id=mngs.gen.dict2str(condi),
            alpha=0.1,
            color=CC["purple" if condi["SWR_type"] == "SWR+" else "yellow"],
        )

        ax.set_xyt(
            None, None, {1: "Match IN", 2: "Mismatch OUT"}[condi["match"]]
        )

    fig.supxyt("Time from SWR peak [ms]", "Distance from O [a.u.]", None)
    return fig


def run_statistical_tests(data):
    # Implement your statistical tests here
    # For example, comparing SWR+ vs SWR- for each condition
    results = {}
    for match in CONFIG.MATCHES:
        for phase in PHASES:
            swrp = data[("SWR+", match, phase)]
            swrm = data[("SWR-", match, phase)]
            # Perform your statistical test here
            # results[(match, phase)] = your_statistical_test(swrp, swrm)
    return results


def main():
    xxp_all, xxm_all = utils.load_ripples(with_NT=True)
    xxp_all, xxm_all = add_phase(xxp_all), add_phase(xxm_all)

    data = prepare_data(xxp_all, xxm_all)

    # Line Plot
    fig = plot_line(data)
    mngs.io.save(fig, "./SWR-triggered_distance_from_O.jpg")


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
