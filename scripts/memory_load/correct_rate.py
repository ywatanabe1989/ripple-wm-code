#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-08 17:58:35 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/figures/01.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
import pandas as pd
from scripts import utils  # , load

"""
Config
"""
# CONFIG = mngs.gen.load_configs()
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def _load_data(correct_or_response_time):
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
        df.groupby(["sub", "session", "set_size"])[correct_or_response_time]
        .mean()
        .groupby("set_size")
        .agg(["mean", "std"])
    ).reset_index()
    return df


def main():
    df = _load_data("correct")
    fig, ax = mngs.plt.subplots()
    ax.bar(
        df["set_size"],
        df["mean"],
        yerr=df["std"],
    )
    ax.set_xyt("Set size", "Correct rate", None)

    mngs.io.save(fig, "./data/memory_load/correct_rate.jpg", from_cwd=True)
    mngs.io.save(
        fig.to_sigma(), "./data/memory_load/correct_rate.csv", from_cwd=True
    )
    return fig


if __name__ == "__main__":
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        font_size_axis_label=6,
        font_size_title=6,
        alpha=0.5,
        dpi_display=100,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
