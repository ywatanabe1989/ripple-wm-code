#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 23:29:42 (ywatanabe)"
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
import sys
from mngs.typing import List, Tuple, Dict, Any, Union, Sequence, Literal, Optional, Iterable, ArrayLike

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from copy import deepcopy
import mngs
from functools import partial
from scripts import utils
from scripts.NT.distance.between_gs.set_size_dependency._stats_helper import (
    run_kruskal_wallis,
    run_brunner_munzel,
    run_corr_test,
    sort_columns,
)

"""Parameters"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def load_NT_dist_between_gs_match_set_size_all() -> pd.DataFrame:
    """
    Load neural trajectory distance data between ground states for all trials and MTL regions.

    Returns:
    --------
    pd.DataFrame
        Combined dataframe of neural trajectory distances for all MTL regions.
    """

    def roi2mtl(roi: str) -> str:
        for mtl, subregions in CONFIG.ROI.MTL.items():
            if roi in subregions:
                return mtl
        return None

    LPATHS = mngs.gen.glob(CONFIG.PATH.NT_DIST_BETWEEN_GS_MATCH_SET_SIZE)

    dfs = mngs.gen.listed_dict()
    for lpath in LPATHS:
        df = mngs.io.load(lpath)
        # # Z norm
        # df[phase_combinations] = mngs.gen.to_z(df[phase_combinations], axis=(0,1))
        parsed = utils.parse_lpath(lpath)
        if parsed["session"] not in CONFIG.SESSION.FIRST_TWO:
            continue
        mtl = roi2mtl(parsed["roi"])
        if mtl:
            dfs[mtl].append(df)

    for mtl, df_list in dfs.items():
        dfs[mtl] = pd.concat(df_list)
        dfs[mtl]["MTL"] = mtl

    df = pd.concat(dfs.values())
    df.match = df.match.replace(CONFIG.MATCHES_STR)
    return df


def run_stats(df: pd.DataFrame, scale: Optional[str] = "linear") -> Dict[str, pd.DataFrame]:
    """
    Run all statistical tests on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    scale : Optional[str], default "linear"
        Scale of the distance data. Can be "linear" or "log10".

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing results of all statistical tests.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'distance': np.random.rand(100),
    ...     'set_size': np.random.choice([4, 6, 8], 100),
    ...     'match': np.random.choice(['Match IN', 'Mismatch OUT'], 100),
    ...     'MTL': np.random.choice(['HIP', 'EC', 'AMY'], 100),
    ...     'phase_combination': np.random.choice(['FE', 'FM', 'FR', 'EM', 'ER', 'MR'], 100)
    ... })
    >>> results = run_stats(df)
    >>> print(list(results.keys()))
    ['kw', 'bm', 'corr']
    """
    df_copy = df.copy()
    if scale == "log10":
        df_copy["distance"] = np.log10(df_copy["distance"] + 1e-5)

    stats_agg = {}
    for test, run_fn in [
        ("kw", run_kruskal_wallis),
        ("bm", run_brunner_munzel),
        ("corr-spearman", partial(run_corr_test, test="spearman")),
        ("corr-pearson", partial(run_corr_test, test="pearson")),
    ]:

        # Stats
        _df = deepcopy(df_copy)
        stats, _surrogate = run_fn(_df)

        # stats = stats.dropna(subset=["p_value"])
        stats = mngs.stats.fdr_correction(stats)
        stats = mngs.stats.p2stars(stats)
        stats = mngs.pd.round(stats)

        # Sorting
        stats = sort_columns(stats)
        stats_agg[test] = stats

        # Surrogate
        if "corr" in test:
            mngs.io.save(_surrogate, f"surrogate_{test}_{scale}.npy")

    return stats_agg



def main():
    df = load_NT_dist_between_gs_match_set_size_all()

    # Run statistical tests
    linear_stats = run_stats(df, scale="linear")

    # Log-transform the distance and run tests again
    log10_stats = run_stats(df, scale="log10")

    # Save results
    for test_name, df in linear_stats.items():
        mngs.io.save(df, f"stats_{test_name}.csv")

    for test_name, df in log10_stats.items():
        mngs.io.save(df, f"stats_{test_name}_log10.csv")




if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True, np=np
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
