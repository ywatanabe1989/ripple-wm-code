#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-05 07:02:10 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/plot_SWR_p.py


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


def main():
    rep = eval(CONFIG["REPRESENTATIVE"])
    trial_number = int(rep["trial"])

    # Loading
    # Ripple
    lpath_rip = eval(mngs.gen.replace(CONFIG["PATH_RIPPLE"], rep))
    rip = mngs.io.load(lpath_rip).loc[trial_number]

    # iEEG
    lpath_iEEG = eval(mngs.gen.replace(CONFIG["PATH_iEEG"], rep))
    xx = mngs.io.load(lpath_iEEG)[trial_number]

    # Plotting
    rip_starts = np.array(rip["start_s"] * CONFIG["FS_iEEG"])
    rip_ends = np.array(rip["end_s"] * CONFIG["FS_iEEG"])

    fig, ax = mngs.plt.subplots()
    ax.fillv(rip_starts, rip_ends, color="red")
    ax.plot(np.array(xx).T)

    # Saving
    mngs.io.save(fig, "plot.jpg")
    mngs.io.save(ax.to_sigma(), "./plot.csv")

    # lpath = CONFIG["PATH_RIPPLE"]
    # for k, v in representative.items():
    #     lpath = lpath.replace("{" + k + "}", v)

    # mngs.gen.natglob(
    #     [
    #         (CONFIG["PATH_RIPPLE"]).replace(k, v)
    #         for k, v in representative.items()
    #     ]
    # )


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)


# EOF
