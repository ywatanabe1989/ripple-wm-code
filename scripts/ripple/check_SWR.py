#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-30 12:15:56 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/check_SWR.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import seaborn as sns

mngs.gen.reload(mngs)
import warnings
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from scripts.utils import parse_lpath
from tqdm import tqdm

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def ax_fill_vline(ax, start, end, color):
    """
    Fill a vertical span between `start` and `end` on x-axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to draw.
    start (float): The starting x value.
    end (float): The ending x value.
    color (str): Color of the filled area.
    """
    # Get y-axis limits
    ymin, ymax = ax.get_ylim()

    # Fill between the vertical lines
    ax.fill_betweenx([ymin, ymax], start, end, color=color, alpha=0.5)


# def main():
lpath_ripple = mngs.gen.natglob(CONFIG["PATH_RIPPLE"])[0]
lpath_iEEG_ripple_band = mngs.gen.natglob(CONFIG["PATH_iEEG_RIPPLE_BAND"])[0]
assert parse_lpath(lpath_ripple) == parse_lpath(lpath_iEEG_ripple_band)

# Loading
rr = mngs.io.load(lpath_ripple)
xr, fsr = mngs.io.load(lpath_iEEG_ripple_band)

i_trial = 0
rri = rr.loc[i_trial]
xri = xr[i_trial]

time = mngs.dsp.time(0, xri.shape[-1] / fsr, fsr)


fig, ax = mngs.plt.subplots()
ax.plot(xri.T, label="iEEG ripple-band signal")
for _, row in rr.iterrows():
    ax_fill_vline(
        ax, xri[int(row.start_s * fsr) : int(row.end_s * fsr)], CC["red"]
    )
plt.show()


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
