#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-12 21:53:59 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/check_SWR.py


"""This script checks the duration and peak ripple amplitude of SWR+/-."""


"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import pandas as pd
import xarray as xr
from scripts.utils import load_ripples, parse_lpath

"""Config"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def plot_duration(pp, mm):
    # Duration
    fig, ax = mngs.plt.subplots()
    ax.hist(pp.duration_s * 1e3, label="SWR+")
    ax.hist(mm.duration_s * 1e3, label="SWR-")
    ax.set_xyt("Duration [ms]", "SWR count")
    ax.set_xscale("log")
    return fig


def plot_amplitude(pp, mm):
    fig, ax = mngs.plt.subplots()
    ax.hist(pp.peak_amp_sd, label="SWR+", id="SWR+")
    ax.hist(mm.peak_amp_sd, label="SWR-", id="SWR-")
    ax.set_xyt("Ripple band peak amplitude [SD of baseline]", "SWR count")
    ax.set_xscale("log")
    return fig


def main():
    # Loading
    pp, mm = load_ripples()

    # Duration
    fig = plot_duration(pp, mm)
    mngs.io.save(fig, "duration.jpg")

    # Amplitude
    fig = plot_amplitude(pp, mm)
    mngs.io.save(fig, "ripple_band_peak_amplitude.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
