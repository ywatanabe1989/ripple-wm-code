#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-23 21:04:42 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA.py


"""
This script does XYZ.
"""


"""
Imports
"""
import importlib
import logging
import os
import re
import sys
import warnings
from glob import glob
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from persim import plot_diagrams
from ripser import ripser
from scipy.spatial.distance import cdist

# sys.path = ["."] + sys.path
from scripts import load, utils
from tqdm import tqdm


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

# PHASES_TO_PLOT = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
PHASES_TO_PLOT = ["Encoding", "Retrieval"]


# Bercode
def calc_persistent_homology(nt):
    result = ripser(nt, maxdim=2)
    diagrams = result["dgms"]
    return diagrams


def plot_barcodes(NTs, GS, ca1, condition, spath_base):
    mngs.plt.configure_mpl(plt, verbose=False)
    results = {}
    for phase, nt in NTs.items():
        if phase not in PHASES_TO_PLOT:
            continue
        nt_reshaped = nt.reshape(-1, nt.shape[-1])
        diagrams = calc_persistent_homology(nt_reshaped)
        results[phase] = {"diagrams": diagrams}

    fig, axes = mngs.plt.subplots(
        ncols=len(PHASES_TO_PLOT), sharex=True, sharey=True
    )

    for idx, (phase, data) in enumerate(results.items()):
        if phase in PHASES_TO_PLOT:
            ax = axes.flat[idx]
            plot_diagrams(
                data["diagrams"],
                ax=ax,
                show=False,
                size=20,
                # ax_color="blue",
                # colormap=mpl.style.available[1],
            )
            ax.set_title(f"{phase}")
            ax.set_n_ticks()

    plt.tight_layout()
    fig.supxyt(None, None, f"{condition}: {str(ca1)}")

    # Saving
    spath_csv = spath_base + f"TDA/barcode/{condition}.csv"
    # mngs.io.save(axes.to_sigma(), spath_csv, from_cwd=True)
    mngs.io.save(fig, spath_csv.replace(".csv", ".jpg"), from_cwd=True)


# Distances
def calc_distances(NTs, GS):
    def _calc_distances(gs, nt):
        nt = nt.transpose(1, 0, 2)
        nt = nt.reshape(len(nt), -1)  # (3, 1000)
        nt = nt.T

        distances = np.array(
            [
                mngs.linalg.euclidean_distance(gs, nt[ii])
                for ii in range(len(nt))
            ]
        )
        return np.array(distances)

    # Calculate distances from Gs to NT
    distances = []
    for phase_nt, nt in NTs.items():
        for phase_g in GS.columns:
            g = np.array(GS[phase_g])
            dists = _calc_distances(g, nt)
            distances.append(
                {"phase_NT": phase_nt, "phase_g": phase_g, "distances": dists}
            )
    df = pd.DataFrame(distances)

    return df


# Distances Sum
def calc_distances_sum_in_spheres(distances, radii):

    def _calc_distances_sum_in_spheres(distances, radius):
        return distances[(distances <= radius)].sum()

    # Calculate the values
    df_new = pd.DataFrame(
        {
            f"radius-{rr}": distances.apply(
                lambda x: _calc_distances_sum_in_spheres(x, rr)
            )
            for rr in radii
        }
    )

    # # To probability
    # df_new /= np.array(df_new).max(axis=1, keepdims=True)

    return df_new


def plot_distances(NTs, GS, ca1, condition, spath_base):
    mngs.plt.configure_mpl(plt, verbose=False)
    # Distances from Gs
    df_distances = calc_distances(NTs, GS)
    radii = np.linspace(0, np.hstack(df_distances.distances).max(), 100)

    # Distance Sum
    df_sum_distances = calc_distances_sum_in_spheres(
        df_distances.distances, radii
    )
    df = pd.concat([df_distances, df_sum_distances], axis=1)

    # Extract only PHASES_TO_PLOT
    df = df[df.phase_NT.isin(PHASES_TO_PLOT) * df.phase_g.isin(PHASES_TO_PLOT)]

    # Plotting
    cols_rad = mngs.gen.search("radius", df.columns)[1]
    fig, axes = mngs.plt.subplots(
        nrows=len(PHASES_TO_PLOT), sharex=True, sharey=True
    )

    i_ax = 0
    for i_g, phase_g in enumerate(df.phase_g.unique()):
        df_g = df[df.phase_g == phase_g]

        ax = axes[i_ax]

        for phase_nt in df.phase_NT.unique():

            if not (phase_nt in PHASES_TO_PLOT):
                continue

            x = radii + 1e-5
            y = np.array(df_g[df_g.phase_NT == phase_nt][cols_rad]).reshape(-1)
            ax.plot(
                x,
                y,
                label=phase_nt,
                color=CONFIG.PHASES[phase_nt].color,
            )
            ax.legend()
            ax.set_xyt(None, None, f"$g_{phase_g[0]}$")
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
            ax.set_ylim(0, 1)

        i_ax += 1
    fig.supxyt(
        "Radius", "Sum of Distances in Spheres", f"{condition}: {str(ca1)}"
    )

    # Saving
    spath_csv = spath_base + f"TDA/dist_from_g/{condition}.csv"
    mngs.io.save(axes.to_sigma(), spath_csv, from_cwd=True)
    mngs.io.save(fig, spath_csv.replace(".csv", ".jpg"), from_cwd=True)


# N Samples
def calc_n_samples_in_spheres(distances, radii):

    def _calc_n_samples_in_spheres(distances, radius):
        return (distances <= radius).sum()

    # Calculate the values
    df_new = pd.DataFrame(
        {
            f"radius-{rr}": distances.apply(
                lambda x: _calc_n_samples_in_spheres(x, rr)
            )
            for rr in radii
        }
    )

    # To probability
    df_new /= np.array(df_new).max(axis=1, keepdims=True)

    return df_new


def plot_n_samples(NTs, GS, ca1, condition, spath_base):
    mngs.plt.configure_mpl(plt, verbose=False)
    # Distances from Gs
    df_distances = calc_distances(NTs, GS)
    radii = np.linspace(0, np.hstack(df_distances.distances).max(), 100)

    # Distance Sum
    df_n_samples = calc_n_samples_in_spheres(df_distances.distances, radii)
    df = pd.concat([df_distances, df_n_samples], axis=1)

    # Extract only PHASES_TO_PLOT
    df = df[df.phase_NT.isin(PHASES_TO_PLOT) * df.phase_g.isin(PHASES_TO_PLOT)]

    # Plotting
    cols_rad = mngs.gen.search("radius", df.columns)[1]
    fig, axes = mngs.plt.subplots(
        nrows=len(PHASES_TO_PLOT), sharex=True, sharey=True
    )

    i_ax = 0
    for i_g, phase_g in enumerate(df.phase_g.unique()):
        df_g = df[df.phase_g == phase_g]

        if not (phase_g in PHASES_TO_PLOT):
            continue

        for phase_nt in df.phase_NT.unique():

            if not (phase_nt in PHASES_TO_PLOT):
                continue

            ax = axes[i_ax]
            x = radii + 1e-5
            y = np.array(df_g[df_g.phase_NT == phase_nt][cols_rad]).reshape(-1)
            ax.plot(
                x,
                y,
                label=phase_nt,
                color=CONFIG.PHASES[phase_nt].color,
            )
            ax.legend()
            ax.set_xyt(None, None, f"$g_{phase_g[0]}$")
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
            ax.set_ylim(0, 1)

        i_ax += 1

    fig.supxyt(
        "Radius",
        "Sample count [%]",
        f"{condition}: {str(ca1)}",
    )

    # Saving
    spath_csv = spath_base + f"TDA/n_samples/{condition}.csv"
    mngs.io.save(axes.to_sigma(), spath_csv, from_cwd=True)
    mngs.io.save(fig, spath_csv.replace(".csv", ".jpg"), from_cwd=True)


def main():

    for i_ca1, ca1 in enumerate(CONFIG.ROI.CA1):

        lpath_NT = mngs.gen.replace(CONFIG.PATH.NT_Z, ca1)
        lpath_GS = mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1)
        lpath_TI = mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1)
        spath_base = eval(lpath_NT.replace(".npy", "/"))

        # NT, G
        NT = mngs.io.load(lpath_NT)
        GS = mngs.io.load(lpath_GS)
        TI = mngs.io.load(lpath_TI)
        NTs = {
            phase: NT[..., data.mid_start : data.mid_end]
            for phase, data in CONFIG.PHASES.items()
        }

        assert np.all([v.shape[-1] for v in list(NTs.values())])

        conditions = ["All", "Match IN", "Mismatch OUT"]
        for cc in conditions:
            indi = {
                "All": np.full(len(TI), True),
                "Match IN": TI.match == 1,
                "Mismatch OUT": TI.match == 2,
            }[cc]

            NTs_cc = {k: v[indi] for k, v in NTs.items()}

            # Distances
            plot_distances(NTs_cc, GS, ca1, cc, spath_base)

            # N Samples
            plot_n_samples(NTs_cc, GS, ca1, cc, spath_base)

            # Barcode
            plot_barcodes(NTs_cc, GS, ca1, cc, spath_base)


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
