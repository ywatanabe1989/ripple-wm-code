#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 13:39:23 (ywatanabe)"
# set_size_dependency_stats.py

"""
Functionality:
    - Analyzes the relationship between set size and distance in MTL regions
Input:
    - Neural trajectory distance data between ground states
Output:
    - Statistical test results for set size dependency
Prerequisites:
    - mngs package, scipy, pandas, numpy
"""

"""Imports"""
import os
import re
import sys
import itertools
from typing import List, Tuple, Dict, Any, Union, Sequence, Optional, Literal
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import xarray as xr
from itertools import combinations
from copy import deepcopy
import mngs
from functools import partial

try:
    from scripts import utils
except ImportError:
    pass

ArrayLike = Union[
    List,
    Tuple,
    np.ndarray,
    pd.Series,
    pd.DataFrame,
    xr.DataArray,
    torch.Tensor,
]

"""Parameters"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def main():
    df = mngs.io.load(
        "./scripts/NT/distance/between_gs/set_size_dependency_stats/stats_corr-pearson.csv"
    )
    surrogate = mngs.io.load(
        "./scripts/NT/distance/between_gs/set_size_dependency_stats/surrogate_pearson.npy"
    )

    fig, ax = mngs.plt.subplots()
    ax.kde(surrogate[0])
    mngs.io.save(fig, "kde.jpg")


    observed_corr = df["corr"]

    df = mngs.pd.merge_cols(df, ["MTL", "match", "phase_combination"])
    df_surrogate = pd.concat([df["merged"], pd.DataFrame(surrogate)], axis=1)
    df_surrogate = mngs.pd.melt_cols(
        df_surrogate, list(set(df_surrogate.columns) - {"merged"})
    )

    fig, ax = mngs.plt.subplots()
    ax.sns_violinplot(data=df_surrogate, x="merged", y="value", half=True)
    ax.scatter(
        data=df,
        x="merged",
        y="corr",
        label="obeserved",
        c=CC["red"],
    )

    ax.extend(x_ratio=10)
    mngs.io.save(ax, "violin.jpg")


# Index(['MTL', 'match', 'phase_combination', 'p_value', 'p_value_stars',
#        'stars', 'effsize', 'corr', 'n', 'test_name', 'statistic', 'H0',
#        'p_value_fdr', 'p_value_fdr_stars'],
#       dtype='object')

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True, np=np
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
