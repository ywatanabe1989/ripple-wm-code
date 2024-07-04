#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-03 01:31:53 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/define_putative_CA1_using_UMAP.py


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
from sklearn.metrics import silhouette_score
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
# lpath_ripple_p = mngs.gen.natglob(CONFIG["PATH_RIPPLE"])[0]
# # lpath_ripple_m = mngs.gen.natglob(CONFIG["PATH_RIPPLE_MINUS"])[0]


def main_lpath(lpath_ripple_p):
    # SWR+
    pp = mngs.io.load(lpath_ripple_p)
    X_pp = np.vstack(pp.firing_pattern)
    T_pp = np.ones(len(pp))  # ["SWR+" for _ in range(len(pp))]

    # SWR-
    lpath_ripple_m = lpath_ripple_p.replace("SWR_p", "SWR_m")
    mm = mngs.io.load(lpath_ripple_m)
    X_mm = np.vstack(mm.firing_pattern)
    T_mm = np.zeros(len(mm))
    # T_mm = ["SWR-" for _ in range(len(mm))]

    # When firing patterns are not available
    if X_pp.size == X_mm.size == 0:
        return np.nan

    assert len(pp) == len(mm)

    # UMAP clustering
    fig, legend_figs, _umap = mngs.ml.clustering.umap(
        [X_pp, X_mm],
        [T_pp, T_mm],
    )
    plt.close()
    U_pp = _umap.transform(X_pp)
    U_mm = _umap.transform(X_mm)

    # Silhouette score
    sil_score = silhouette_score(
        np.vstack([U_pp, U_mm]), np.hstack([T_pp, T_mm])
    )
    return round(sil_score, 3)


def main():
    out = mngs.gen.listed_dict()
    for lpath_ripple_p in tqdm(mngs.gen.natglob(CONFIG["PATH_RIPPLE"])):
        parsed = parse_lpath(lpath_ripple_p)
        sil_score = main_lpath(lpath_ripple_p)

        out["sub"].append(parsed["sub"])
        out["session"].append(parsed["session"])
        out["roi"].append(parsed["roi"])
        out["silhouette_score"].append(sil_score)
        out["lpath_ripple"].append(lpath_ripple_p)

    # Saving
    out = pd.DataFrame(out)
    mngs.io.save(out, "./data/silhouette_scores.csv", from_cwd=True)


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
