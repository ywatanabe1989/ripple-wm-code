#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-05 22:33:48 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/calc_dist_between_gs_session.py

"""
1. Functionality:
   - Calculates distances between geometric medians (GS) for different session phases
2. Input:
   - Geometric median (GS) data files
3. Output:
   - Distances between GS sessions for different phase combinations
4. Prerequisites:
   - mngs package, numpy, pandas
"""

import sys
from itertools import combinations
from typing import Dict

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd

try:
    from scripts import utils
except ImportError:
    pass

def calc_dists_between_gs_session(gs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distances between geometric medians for different phase combinations.

    Example
    -------
    >>> gs = pd.DataFrame({'encoding': [1, 2, 3], 'maintenance': [2, 3, 4], 'cue': [3, 4, 5]})
    >>> dists = calc_dists_between_gs_session(gs)
    >>> print(dists.columns)
    ['em', 'ec', 'mc']

    Parameters
    ----------
    gs : pd.DataFrame
        Geometric median data for different phases

    Returns
    -------
    pd.DataFrame
        Distances between GS sessions for different phase combinations
    """
    dists_between_gs_session = {}
    for p1, p2 in combinations(CONFIG.PHASES.keys(), 2):
        gs_p1 = gs[p1]
        gs_p2 = gs[p2]
        dists_between_gs_session[f"{p1[0]}{p2[0]}"] = mngs.linalg.nannorm(gs_p1 - gs_p2)
    return mngs.pd.force_df(dists_between_gs_session)

def process_gs_file(lpath_gs: str) -> None:
    """
    Process a single GS file and save the calculated distances.

    Parameters
    ----------
    lpath_gs : str
        Path to the GS data file
    """
    try:
        gs = mngs.io.load(lpath_gs)
        dists = calc_dists_between_gs_session(gs)
        parsed = utils.parse_lpath(lpath_gs)
        spath = mngs.gen.replace(CONFIG.PATH.NT_DIST_BETWEEN_GS_SESSION, parsed)
        mngs.io.save(dists, spath, from_cwd=True)
    except Exception as e:
        print(f"Error processing file {lpath_gs}: {str(e)}")

def main() -> None:
    """
    Main function to process all GS files and save results.
    """
    lpaths_gs = mngs.io.glob(CONFIG.PATH.NT_GS_SESSION)
    for lpath_gs in lpaths_gs:
        process_gs_file(lpath_gs)

if __name__ == '__main__':
    np.random.seed(42)
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
