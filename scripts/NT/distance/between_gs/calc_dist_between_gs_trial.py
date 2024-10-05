#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-05 21:49:36 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/calc_dist_between_gs_trial.py

"""
1. Functionality:
   - Calculates distances between geometric medians (GS) for different trial phases
2. Input:
   - geometric median (GS) data and corresponding trial information
3. Output:
   - Distances between GS trials for different phase combinations
4. Prerequisites:
   - mngs package, numpy, pandas, xarray
"""

import sys
from itertools import combinations
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr

try:
    from scripts import utils
except ImportError:
    pass

def load_gs(lpath_gs: str, n_factors: int = 3) -> xr.DataArray:
    """
    Load geometric median (GS) data and add set size information.

    Example
    -------
    >>> gs = load_gs("/path/to/gs_data.nc", n_factors=3)
    >>> print(gs.dims)
    ('set_size', 'factor', 'phase')

    Parameters
    ----------
    lpath_gs : str
        Path to the GS data file
    n_factors : int, optional
        Number of factors to consider (default is 3)

    Returns
    -------
    xr.DataArray
        geometric median data with set size information
    """
    gs = mngs.io.load(lpath_gs)
    gs = gs[:, :n_factors, :]

    parsed = utils.parse_lpath(lpath_gs)
    lpath_ti = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
    ti = mngs.io.load(lpath_ti)

    gs = gs.swap_dims({'trial': 'set_size'})
    gs = gs.assign_coords(set_size=("set_size", ti["set_size"]))
    return gs

def calc_dists_between_gs_trial(gs: xr.DataArray) -> pd.DataFrame:
    """
    Calculate distances between geometric medians for different phase combinations.

    Example
    -------
    >>> gs = load_gs("/path/to/gs_data.nc")
    >>> dists = calc_dists_between_gs_trial(gs)
    >>> print(dists.columns)
    ['ec', 'em', 'cm']

    Parameters
    ----------
    gs : xr.DataArray
        geometric median data

    Returns
    -------
    pd.DataFrame
        Distances between GS trials for different phase combinations
    """
    dists_between_gs_trial = {}
    for p1, p2 in combinations(CONFIG.PHASES.keys(), 2):
        gs_p1 = gs.sel(phase=p1).squeeze()
        gs_p2 = gs.sel(phase=p2).squeeze()
        dists_between_gs_trial[f"{p1[0]}{p2[0]}"] = mngs.linalg.nannorm(gs_p1 - gs_p2)
    return mngs.pd.force_df(dists_between_gs_trial)

def process_gs_file(lpath_gs: str, n_factors: int) -> Tuple[pd.DataFrame, str]:
    """
    Process a single GS file and calculate distances.

    Parameters
    ----------
    lpath_gs : str
        Path to the GS data file
    n_factors : int
        Number of factors to consider

    Returns
    -------
    Tuple[pd.DataFrame, str]
        Processed distances and the save path for the results
    """
    gs = load_gs(lpath_gs, n_factors)
    dists_between_gs_trial = calc_dists_between_gs_trial(gs)

    parsed = utils.parse_lpath(lpath_gs)
    lpath_ti = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
    ti = mngs.io.load(lpath_ti)

    dists_between_gs_trial = pd.concat([ti, dists_between_gs_trial], axis=1)
    spath_dists_between_gs_trial = mngs.gen.replace(CONFIG.PATH.NT_DIST_BETWEEN_GS_TRIAL, parsed)

    return dists_between_gs_trial, spath_dists_between_gs_trial

def main(n_factors: int = 3) -> None:
    """
    Main function to process all GS files and save results.

    Parameters
    ----------
    n_factors : int, optional
        Number of factors to consider (default is 3)
    """
    lpaths_gs = mngs.io.glob(CONFIG.PATH.NT_GS_TRIAL)
    for lpath_gs in lpaths_gs:
        dists_between_gs_trial, spath_dists_between_gs_trial = process_gs_file(lpath_gs, n_factors)
        mngs.io.save(dists_between_gs_trial, spath_dists_between_gs_trial)

if __name__ == '__main__':
    np.random.seed(42)
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
