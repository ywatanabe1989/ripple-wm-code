#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-06 14:55:29 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/demographic/electrode_positions.py


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
from collections import OrderedDict
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
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""
import os
import re
from collections import defaultdict


def main():
    out = mngs.gen.listed_dict()

    for sub, roi in CONFIG["PUTATIVE_CA1"].items():
        for session in CONFIG["SESSIONS_0102"]:
            df = mngs.io.load(eval(CONFIG["PATH_RIPPLE"]))
            trials_info = mngs.io.load(eval(CONFIG["PATH_TRIALS_INFO"]))

            n_ripples = len(df)
            n_trials = len(trials_info)

            out["sub"].append(sub)
            out["roi"].append(roi)
            out["n_ripples"].append(n_ripples)
            out["n_trials"].append(n_trials)

    df = pd.DataFrame(out)
    df = df.groupby(["sub", "roi"]).agg(
        {
            "n_trials": "sum",
            "n_ripples": "sum",
        }
    )
    df["indidence_hz"] = (
        df["n_ripples"] / (df["n_trials"] * CONFIG["TRIAL_SEC"])
    ).round(3)

    # Saving
    mngs.io.save(df, "./data/demographic/ripple_count.csv", from_cwd=True)


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
