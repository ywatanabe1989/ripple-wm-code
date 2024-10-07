#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-05 12:15:08 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/figures/01.py


"""This script does XYZ."""


"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import pandas as pd
from scripts import utils

"""Config"""
CONFIG = mngs.gen.load_configs()


"""Functions & Classes"""


def _load_data(correct_or_response_time):
    """
    Load and process data for a given measure.

    Example
    -------
    df = load_data("correct")

    Parameters
    ----------
    measure : str
        Either "correct" or "response_time"

    Returns
    -------
    pandas.DataFrame
        Processed data with mean and std per set size
    """
    LPATHS = mngs.gen.glob(CONFIG.PATH.TRIALS_INFO)
    dfs = []
    for lpath in LPATHS:
        parsed = utils.parse_lpath(lpath)

        if int(parsed["sub"]) > CONFIG.SESSION.THRES:
            continue

        df = mngs.io.load(lpath)
        for k, v in parsed.items():
            df[k] = v
        dfs.append(df)
    df = pd.concat(dfs)
    df = (
        df.groupby(["sub", "set_size"])[correct_or_response_time]
        .mean()
        .groupby("set_size")
        .agg(["mean", "std"])
    ).reset_index()
    return df


def A():
    df = _load_data("correct")
    fig, ax = mngs.plt.subplots()
    ax.bar(
        df["set_size"],
        df["mean"],
        yerr=df["std"],
    )
    ax.set_xyt("Set size", "Correct rate", None)


def B():
    df = _load_data("response_time")

    fig, ax = mngs.plt.subplots()
    ax.bar(
        df["set_size"],
        df["mean"],
        yerr=df["std"],
    )
    ax.set_xyt("Set size", "Response time [s]", None)


def C():
    from scripts.NT.distance.from_O.MTL_regions import main

    main()


def D():
    from scripts.NT.distance.from_O.MTL_regions import main

    main()


def E():
    from scripts.NT.distance.between_gs.MTL_regions.py import main

    main()


def main():
    """Executes main analysis and plotting routines."""
    fig_A = A()
    fig_B = B()
    fig_C = C()
    fig_D = D()
    fig_E = E()
    plt.show()


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        # agg=True,
        font_size_axis_label=6,
        font_size_title=6,
        alpha=0.5,
        dpi_display=100,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
