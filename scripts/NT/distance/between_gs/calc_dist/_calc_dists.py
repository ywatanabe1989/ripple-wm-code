#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 19:14:18 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/calc_dists.py

"""This script calculates distances between geometric medians for phases."""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr

"""Functions & Classes"""

MATCH_CONDI = ["All", "Match IN", "Mismatch OUT"]


def extract_10_mid_bins(NT):
    """Balance phases in neural trajectory data.

    Parameters
    ----------
    NT : numpy.ndarray
        Neural trajectory data.

    Returns
    -------
    numpy.ndarray
        Balanced neural trajectory data.

    Example
    -------
    balanced_nt = extract_10_mid_bins(neural_trajectory)
    """
    NTs_pp = {
        phase: NT[..., data.mid_start : data.mid_end]
        for phase, data in CONFIG.PHASES.items()
    }
    assert np.all([v.shape[-1] for v in list(NTs_pp.values())])
    NTs_pp = np.stack(list(NTs_pp.values()), axis=1)
    return NTs_pp


# Distances
def calc_dists(NT, GS, TI, ca1, phases_to_plot):
    """Calculate distances geometric medians.

    Parameters
    ----------
    NT : numpy.ndarray
        Neural trajectory data.
    GS : numpy.ndarray
        Geometric medians
    TI : pandas.DataFrame
        Trial information.
    ca1 : dict
        CA1 region information.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing distance information.

    Example
    -------
    dist_df = calc_dists(nt, gs, ti, ca1)
    """

    # To the shape of ("factor", "phase", ...)
    NT = mngs.gen.transpose(
        NT,
        ["i_trial", "i_phase", "i_factor", "i_bin_in_phase"],
        ["i_factor", "i_phase", "i_trial", "i_bin_in_phase"],
    )
    GS = np.array(GS)

    # Euclidean distances
    dists_arr = mngs.linalg.edist(NT, GS)
    dists_xr = xr.DataArray(
        dists_arr,
        dims=["phase_NT", "match", "i_bin", "phase_g"],
        coords={
            "phase_NT": list(CONFIG.PHASES.keys()),
            "match": TI.match,
            "i_bin": np.arange(dists_arr.shape[2]),
            "phase_g": list(CONFIG.PHASES.keys()),
        },
    )

    # To dist_df
    dist_df = []
    for i_phase_nt, phase_nt in enumerate(CONFIG.PHASES.keys()):
        for i_phase_g, phase_g in enumerate(CONFIG.PHASES.keys()):
            if (phase_nt in phases_to_plot) and (phase_g in phases_to_plot):
                _dist = dists_xr[i_phase_nt, ..., i_phase_g]
                match_flatten = np.array(_dist.match).repeat(_dist.shape[-1])
                dist_flatten = np.array(_dist).flatten()
                _dist_df = pd.DataFrame(
                    {
                        "phase_g": [phase_g for _ in range(len(dist_flatten))],
                        "phase_nt": [
                            phase_nt for _ in range(len(dist_flatten))
                        ],
                        "dist": dist_flatten,
                        "match": match_flatten,
                    }
                )
                dist_df.append(_dist_df)
    dist_df = pd.concat(dist_df)

    mngs.pd.merge_cols(dist_df, "phase_g", "phase_nt")
    dist_df = dist_df.rename(columns={"merged": "group"})
    dist_df["group"] = dist_df["group"].apply(replace_group)

    for k, v in ca1.items():
        dist_df[k] = v

    return dist_df


def extract_conditions(df):
    """Extract unique conditions from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    list
        List of condition dictionaries.

    Example
    -------
    conditions = extract_conditions(df)
    """
    conditions = {
        k: df[k].unique().tolist() for k in ["phase_g", "phase_nt", "match"]
    }
    conditions = list(mngs.gen.yield_grids(conditions))
    conditions = pd.concat(
        [pd.DataFrame(pd.Series(cc)).T for cc in conditions]
    )

    sorted_conditions = mngs.pd.sort(
        conditions,
        orders={
            "match": [1, 2],
            "phase_g": ["Encoding", "Retrieval"],
            "phase_nt": ["Encoding", "Retrieval"],
        },
    )

    return sorted_conditions.apply(dict, axis=1).tolist()


def replace_group(text):
    """Replace group text with LaTeX formatted string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        LaTeX formatted string.

    Example
    -------
    formatted_text = replace_group("phase_g-Encoding_phase_nt-Retrieval")
    """
    replaced = (
        text.replace("_phase_nt-", "-NT")
        .replace("phase_g-", "g")
        .replace("Fixation", "_F")
        .replace("Encoding", "_E")
        .replace("Maintenance", "_M")
        .replace("Retrieval", "_R")
    )
    tex = f"${replaced}$"

    return tex


def main():
    """Main function to process neural trajectory data."""
    PHASES_TO_PLOT = [
        ["Fixation", "Encoding", "Maintenance", "Retrieval"],
        ["Encoding", "Retrieval"],
    ]

    for phases_to_plot in PHASES_TO_PLOT:
        dfs = []
        for i_ca1, ca1 in enumerate(CONFIG.ROI.CA1):

            # Loading paths
            lpath_NT = mngs.gen.replace(CONFIG.PATH.NT_Z, ca1)
            lpath_GS = mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1)
            lpath_TI = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)

            # NT, G
            NT = mngs.io.load(lpath_NT)
            GS = mngs.io.load(lpath_GS)
            TI = mngs.io.load(lpath_TI)

            # Undersampling
            NT = extract_10_mid_bins(NT)

            # N Samples
            df = calc_dists(NT, GS, TI, ca1, phases_to_plot)

            dfs.append(df)

        # Saving
        df = pd.concat(dfs)
        mngs.io.save(df, f"{'_'.join(phases_to_plot)}/dist_ca1.csv", from_cwd=False)

        # Box plot
        fig, ax = mngs.plt.subplots()
        ax.sns_boxplot(
            data=df,
            x="group",
            y="dist",
            showfliers=False,
            )
        mngs.io.save(fig, f"{'_'.join(phases_to_plot)}/dist_ca1_box.jpg", from_cwd=False)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
