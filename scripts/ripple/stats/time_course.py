#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-12 21:54:38 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/stats/time_course.py

"""This script does XYZ."""

"""Imports"""
# import os
# import re
import sys

# import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
# from scripts.ripple.stats.duration_amplitude import _load_ripples
from scripts.utils import load_ripples

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""

#     # under = gaussian_filter1d(under, truncate=1, sigma=4, mode="constant")
#     # middle = gaussian_filter1d(middle, truncate=1, sigma=4, mode="constant")
#     # upper = gaussian_filter1d(upper, truncate=1, sigma=4, mode="constant")


def main():
    pp, mm = load_ripples()
    tt = np.linspace(
        0,
        CONFIG.TRIAL.DUR_SEC,
        int(CONFIG.TRIAL.DUR_SEC / CONFIG.GPFA.BIN_SIZE_MS / 1e-3),
    )

    # Raster plot
    fig, ax = mngs.plt.subplots()
    for ii, (xx, label) in enumerate(zip([pp, mm], ["SWR+", "SWR-"])):
        xx = xx.copy()
        xx_raster = mngs.pd.merge_cols(
            xx.reset_index(), "subject", "session", "roi", "trial_number"
        )
        ax.raster(
            [
                np.array(xx_raster.loc[xx_raster.merged == mm, "peak_s"])
                for mm in xx_raster.merged.unique()
            ],
            time=tt,
            label=label,
            id=label,
        )
    mngs.io.save(fig, "raster.jpg")

    # Time course; smoothed instant frequency
    df = fig.to_sigma()
    df_p = df[mngs.gen.search(r"SWR\+", df.columns)[1]]
    df_m = df[mngs.gen.search(r"SWR\-", df.columns)[1]]

    fig, ax = mngs.plt.subplots()
    for ii, (xx, label) in enumerate(zip([df_p, df_m], ["SWR+", "SWR-"])):
        xx = xx.copy()
        digi = (~xx.isna()).astype(int).T
        filt = mngs.dsp.filt.gauss(digi, sigma=1).squeeze()
        nn = len(digi)
        mm = filt.mean(axis=0)
        ss = filt.std(axis=0)
        ci = 1.96 * ss / np.sqrt(nn)
        ax.plot_with_ci(tt, mm, ci, n=nn, label=label, id=label)
        ax.legend()
    mngs.io.save(fig, "time_course.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
