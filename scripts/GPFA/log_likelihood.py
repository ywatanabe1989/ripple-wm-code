#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-11 14:03:40 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/GPFA/log-likelihood.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib
import matplotlib.pyplot as plt
import importlib

import mngs

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import quantities as pq
from scripts.GPFA.calc_NT import spiketimes_to_spiketrains
from scripts.utils import parse_lpath
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from bisect import bisect_right
import logging
import concurrent.futures
import gc

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
# from contextlib import contextmanager, redirect_stdout, redirect_stderr
# import os

# @contextmanager
# def suppress_output():
#     """
#     A context manager that suppresses stdout and stderr.

#     Returns
#     -------
#     None
#     """
#     with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
#         try:
#             yield
#         finally:
#             pass


def get_n_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    bins = [
        bisect_right(bin_centers.rescale("s"), ts)
        for ts in np.array([times_sec])
    ]
    return bins


def main_lpath(
    lpath_spike_times,
):
    # Loading
    parsed = parse_lpath(lpath_spike_times)
    spike_times = mngs.io.load(lpath_spike_times)
    if np.vstack(spike_times).shape == (0, 0):
        logging.warn(
            f"No spikes are registered. Skipping {lpath_spike_times}."
        )
        return
    spike_trains = spiketimes_to_spiketrains(spike_times)

    # Parameters
    bin_size = CONFIG.GPFA.BIN_SIZE_MS * pq.ms
    n_cv = 3

    # NT calculation using GPFA
    cache = mngs.gen.listed_dict()
    for i_dim in range(1, 10):
        try:
            gpfa_cv = GPFA(bin_size=bin_size, x_dim=i_dim, verbose=False)
            cv_log_likelihoods = cross_val_score(
                gpfa_cv, spike_trains, cv=n_cv, n_jobs=-1, verbose=False
            )

        except Exception as e:
            logging.warn(e)
            cv_log_likelihoods = [np.nan for _ in range(n_cv)]

        cache[f"{i_dim}_dimensions"].append(cv_log_likelihoods)

    # Saving
    spath = lpath_spike_times.replace(
        "/spike_times/", "/GPFA/log_likelihoods/"
    )
    mngs.io.save(cache, spath)
    del cache
    gc.collect()


def main():
    LPATHS_SPIKE_TIMES = mngs.gen.natglob(CONFIG.PATH.SPIKE_TIMES)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        list(
            tqdm(
                executor.map(main_lpath, LPATHS_SPIKE_TIMES),
                total=len(LPATHS_SPIKE_TIMES),
            )
        )

    # for lpath_spike_times in tqdm(LPATHS_SPIKE_TIMES):
    #     main_lpath(lpath_spike_times)


# def plot_log_likelihood():
#     lpath = "./data/GPFA/log_likelihoods_all.pkl"
#     df = mngs.io.load(lpath)
#     # np.vstack([np.array(row).astype(float) for i_row, row in enumerate(df["log-likelihoods_mm"])])
#     mm = np.nanmean(np.vstack(df["log-likelihoods_mm"]), axis=0)
#     ss = np.nanstd(np.vstack(df["log-likelihoods_mm"]), axis=0)
#     nn = np.sum(~np.isnan(np.vstack(df["log-likelihoods_mm"])), axis=0)
#     ci = 1.96 * ss / np.sqrt(nn)

#     # sd = 2 # fixme
#     # se = sd / len(cv_log_likelihoods)
#     # ci = 1.96 * se

#     # is_nan = np.isnan(np.vstack(df["log-likelihoods_mm"])).any(axis=-1)
#     # ss = ss / (~is_nan).sum()

#     plt.errorbar(x=np.arange(len(mm)), y=mm, yerr=ci)
#     plt.errorbar(x=np.arange(len(mm)), y=mm, yerr=ss / 5)
#     plt.show()

#     out_df = pd.DataFrame(
#         {
#             "dim": np.arange(len(mm)),
#             "under": mm - ci,
#             "mean": mm,
#             "upper": mm + ci,
#         }
#     )
#     mngs.io.save(out_df, lpath.replace(".pkl", ".csv"))


# # plots
# fig, ax = plt.subplots()
# # ax.plot(i_dims, log_likelihoods_mm)
# ax.errorbar(x_dims, log_likelihoods_mm, yerr=log_likelihoods_ss)
# ax.set_xlabel("Dimensionality of latent variables for GPFA")
# ax.set_ylabel("Log-likelihood")
# # ax.plot(x_dims[np.argmax(log_likelihoods)], np.max(log_likelihoods), "x", markersize=10, color="r")
# mngs.io.save(
#     fig,
#     f"./tmp/figs/line/GPFA_log_likelihoods/Subject_{subject}_Session_{session}_ROI_{roi}.png",
# )
# plt.close()

# trajectories = gpfa.fit_transform(spike_trains)
# df = pd.DataFrame()
# df["log-likelihoods_mm"] = [np.array(log_likelihoods_mm)]
# df["log-likelihoods_ss"] = [np.array(log_likelihoods_ss)]
# df["subject"] = parsed["sub"]
# df["session"] = parsed["session"]
# df["roi"] = parsed["roi"]
# df = df[
#     [
#         "subject",
#         "session",
#         "roi",
#         "log-likelihoods_mm",
#         "log-likelihoods_ss",
#     ]
# ]

# return df


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
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
