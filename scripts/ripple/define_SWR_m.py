#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-28 08:42:57 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/define_SWR-.py


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
    LPATHS_RIPPLE = mngs.gen.natglob(CONFIG["PATH_RIPPLE"])
    lpath = LPATHS_RIPPLE[0]
    df = mngs.io.load(lpath)

    __import__("ipdb").set_trace()


    print(LPATHS_RIPPLE)
    for lpath in LPATHS_RIPPLE:
        df_plus = mngs.io.load(lpath)

        df_plus[["subject", "session"]].drop_duplicates()


        df_minus = df_plus.copy()
        cols_to_del = ["peak_s", "rel_peak_pos", "peak_amp_sd", "incidence_hz", "set_size", "match"]
        df_minus["peak_s"] = np.nan
        df_minus["rel_peak_pos"] = np.nan

        print()

    __import__("ipdb").set_trace()
    # rips_df = mngs.io.load(f"./data/rips_df/{iEEG_roi_connected}.pkl")

    pass


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
