#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-22 13:27:33 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/geometrics_medians.py


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
import seaborn as sns
import importlib

import mngs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from glob import glob
from pprint import pprint
import warnings
import logging
from tqdm import tqdm
import xarray as xr

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


def main():
    LPATHS_NT = mngs.gen.natglob(CONFIG.PATH.NT_Z)

    for lpath_NT in LPATHS_NT:
        NT = mngs.io.load(lpath_NT)

        gs = {}
        for phase, data in CONFIG.PHASES.items():
            NT_phase = NT[..., data.mid_start : data.mid_end].transpose(
                1, 0, 2
            )
            NT_phase = NT_phase.reshape(len(NT_phase), -1)
            gs[phase] = mngs.linalg.geometric_median(NT_phase)
        gs = pd.DataFrame(gs)
        gs.index = [f"factor_{ii+1}" for ii in range(len(gs))]

        mngs.io.save(gs, lpath_NT.replace(".npy", "/gs.csv"), from_cwd=True)


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
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
