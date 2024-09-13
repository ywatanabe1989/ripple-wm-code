#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 09:17:04 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import importlib
import logging
import os
import re
import sys
import warnings
from bisect import bisect_left
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
from scripts import utils
from tqdm import tqdm

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def add_phase(xx_all):
    xx_all["phase"] = str(np.nan)
    for phase, phase_data in CONFIG.PHASES.items():
        indi_phase = (phase_data.start <= xx_all.peak_i) * (
            xx_all.peak_i < phase_data.end
        )
        xx_all.loc[indi_phase, "phase"] = phase
    return xx_all


def main():
    xxp_all, xxm_all = utils.load_ripples(with_NT=True)
    xxp_all, xxm_all = add_phase(xxp_all), add_phase(xxm_all)

    ((xxp_all.NT).apply(np.sum) == 0).mean()

    xxp_all


#     I_BIN_LAST = int(CONFIG.TRIAL.DUR_SEC / (CONFIG.GPFA.BIN_SIZE_MS * 1e-3))
#     window = I_BIN_LAST // 2
#     TT = (
#         np.linspace(-window, window, 2 * window + 1)
#         * CONFIG.GPFA.BIN_SIZE_MS
#         * 1e-3
#     )

#     for ca1 in CONFIG.ROI.CA1:
#         nt = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
#         dd = pd.DataFrame(np.sqrt(np.sum(nt**2, axis=1)))
#         dd.index = pd.Series(np.arange(len(dd)), name="i_trial")
#         dd.columns = [f"bin_{ii:03d}" for ii in range(len(dd.columns))]

#         ca1["subject"] = ca1.pop("sub")
#         SWR_p = mngs.pd.slice(xxp_all, ca1)
#         SWR_m = mngs.pd.slice(xxm_all, ca1)

#         # Here, you should extract NT of each riSWR_ple
#         # NT = (i_trial, n_factors, n_bins)
#         SWR_p_aligned = create_aligned_data(
#             SWR_p, dd, I_BIN_LAST
#         )  # (210, 160), fixme
#         SWR_m_aligned = create_aligned_data(SWR_m, dd, I_BIN_LAST)

#         __import__("ipdb").set_trace()

#         # SWR_p_mean = np.nanmean(SWR_p_aligned, axis=0)
#         # SWR_p_ci = np.nanstd(SWR_p_aligned, axis=0) / np.sqrt(np.sum(~np.isnan(SWR_p_aligned), axis=0)) * 1.96
#         # SWR_m_mean = np.nanmean(SWR_m_aligned, axis=0)
#         # SWR_m_ci = np.nanstd(SWR_m_aligned, axis=0) / np.sqrt(np.sum(~np.isnan(SWR_m_aligned), axis=0)) * 1.96

#         # plt.figure(figsize=(10, 6))
#         # plt.plot(TT, SWR_p_mean, label='SWR_P')
#         # plt.fill_between(TT, SWR_p_mean - SWR_p_ci, SWR_p_mean + SWR_p_ci, alpha=0.3)
#         # plt.plot(TT, SWR_m_mean, label='SWR_M')
#         # plt.fill_between(TT, SWR_m_mean - SWR_m_ci, SWR_m_mean + SWR_m_ci, alpha=0.3)
#         # plt.xlabel('Time relative to riSWR_ple peak (s)')
#         # plt.ylabel('Distance from O')
#         # plt.legend()
#         # plt.title(f'RiSWR_ple-triggered distance from O - {ca1["subject"]}')
#         # plt.savefig(f'riSWR_ple_triggered_distance_{ca1["subject"]}.png')
#         # plt.close()


# # def find_indi(SWR_p, lim):
# #     "pre: peak_s -"
# #     for period in ["pre", "mid", "post"]:

# #         delta_s = np.array(CONFIG.RISWR_PLE.BINS[period]) * (
# #             CONFIG.GPFA.BIN_SIZE_MS * 1e-3
# #         )

# #         SWR_p[f"{period}_start_i"] = (
# #             np.array(
# #                 (SWR_p.peak_s + delta_s[0]) / (CONFIG.GPFA.BIN_SIZE_MS * 1e-3)
# #             )
# #             .astype(int)
# #             .clip(*lim)
# #         )
# #         SWR_p[f"{period}_end_i"] = (
# #             np.array(
# #                 (SWR_p.peak_s + delta_s[1]) / (CONFIG.GPFA.BIN_SIZE_MS * 1e-3)
# #             )
# #             .astype(int)
# #             .clip(*lim)
# #         )
# #     return SWR_p


# # def create_masks(SWR_p, I_BIN_LAST):
# #     masks = {
# #         period: np.zeros((len(SWR_p), I_BIN_LAST), dtype=bool)
# #         for period in ["pre", "mid", "post"]
# #     }
# #     for idx, row in SWR_p.iterrows():
# #         for period in ["pre", "mid", "post"]:
# #             masks[period][
# #                 idx, row[f"{period}_start_i"] : row[f"{period}_end_i"]
# #             ] = True
# #     return masks


# # def main():
# #     xxp_all, xxm_all = utils.load_riSWR_ples()
# #     I_BIN_LAST = int(CONFIG.TRIAL.DUR_SEC / (CONFIG.GPFA.BIN_SIZE_MS * 1e-3))
# #     TT = (
# #         np.linspace(-I_BIN_LAST // 2, I_BIN_LAST // 2, I_BIN_LAST)
# #         * CONFIG.GPFA.BIN_SIZE_MS
# #         * 1e-3
# #     )

# #     for ca1 in CONFIG.ROI.CA1:
# #         nt = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))
# #         dd = pd.DataFrame(np.sqrt(np.sum(nt**2, axis=1)))
# #         dd.index = pd.Series(np.arange(len(dd)), name="i_trial")
# #         dd.columns = [f"bin_{ii:03d}" for ii in range(len(dd.columns))]

# #         ca1["subject"] = ca1.pop("sub")
# #         SWR_p = mngs.pd.slice(xxp_all, ca1)
# #         SWR_m = mngs.pd.slice(xxm_all, ca1)

# #         SWR_p_aligned = create_aligned_data(SWR_p, dd, I_BIN_LAST)
# #         SWR_m_aligned = create_aligned_data(SWR_m, dd, I_BIN_LAST)

# #         SWR_p_mean = np.nanmean(SWR_p_aligned, axis=0)
# #         SWR_p_ci = (
# #             np.nanstd(SWR_p_aligned, axis=0)
# #             / np.sqrt(np.sum(~np.isnan(SWR_p_aligned), axis=0))
# #             * 1.96
# #         )
# #         SWR_m_mean = np.nanmean(SWR_m_aligned, axis=0)
# #         SWR_m_ci = (
# #             np.nanstd(SWR_m_aligned, axis=0)
# #             / np.sqrt(np.sum(~np.isnan(SWR_m_aligned), axis=0))
# #             * 1.96
# #         )

# #         plt.figure(figsize=(10, 6))
# #         plt.plot(TT, SWR_p_mean, label="SWR_P")
# #         plt.fill_between(TT, SWR_p_mean - SWR_p_ci, SWR_p_mean + SWR_p_ci, alpha=0.3)
# #         plt.plot(TT, SWR_m_mean, label="SWR_M")
# #         plt.fill_between(TT, SWR_m_mean - SWR_m_ci, SWR_m_mean + SWR_m_ci, alpha=0.3)
# #         plt.xlabel("Time relative to riSWR_ple peak (s)")
# #         plt.ylabel("Distance from O")
# #         plt.legend()
# #         plt.title(f'RiSWR_ple-triggered distance from O - {ca1["subject"]}')
# #         plt.savefig(f'riSWR_ple_triggered_distance_{ca1["subject"]}.png')
# #         plt.close()


# # def main():
# #     xxp_all, xxm_all = utils.load_riSWR_ples()
# #     I_BIN_LAST = int(CONFIG.TRIAL.DUR_SEC / (CONFIG.GPFA.BIN_SIZE_MS * 1e-3))
# #     TT = np.linspace(
# #         0,
# #         CONFIG.TRIAL.DUR_SEC,
# #         I_BIN_LAST,
# #     )
# #     for ca1 in CONFIG.ROI.CA1:
# #         # NT
# #         nt = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_Z, ca1))

# #         # Distance
# #         dd = pd.DataFrame(
# #             np.sqrt(np.sum(nt**2, axis=1)),
# #         )
# #         dd.index = pd.Series(np.arange(len(dd)), name="i_trial")
# #         dd.columns = [f"bin_{ii:03d}" for ii in range(len(dd.columns))]

# #         # RiSWR_ple
# #         ca1["subject"] = ca1.pop("sub")
# #         SWR_p = mngs.pd.slice(xxp_all, ca1)
# #         SWR_m = mngs.pd.slice(xxm_all, ca1)

# #         # Indices of start/end of pre-, mid-, or post-SWR
# #         SWR_p = find_indi(SWR_p, lim=(0, I_BIN_LAST))
# #         SWR_m = find_indi(SWR_m, lim=(0, I_BIN_LAST))

# #         # Create masks
# #         SWR_p_masks = create_masks(SWR_p, I_BIN_LAST)
# #         SWR_m_masks = create_masks(SWR_m, I_BIN_LAST)

# #         # Calculate distances
# #         results = {}
# #         for period in ["pre", "mid", "post"]:
# #             results[f"SWR_p_{period}"] = dd.values[SWR_p_masks[period]].mean()
# #             results[f"SWR_m_{period}"] = dd.values[SWR_m_masks[period]].mean()

# #         # # Calculates distance from O during pre-, mid-, or post-SWR
# #         # for _, row in SWR_p.iterrows():
# #         #     i_trial = row.name - 1
# #         #     _dd = dd[i_trial]
# #         #     _dd[row.pre_start_i:row.pre_end_i].mean()

# #         #         ss, ee = peak_s + np.array(CONFIG.RISWR_PLE.BINS[period]) * CONFIG.GPFA.BIN_SIZE_MS * 1e-3

# #         #     print(row.name, row.peak_s)

# #         # SWR_p.

# #         # CONFIG.RISWR_PLE.BINS.pre

# #         # __import__("ipdb").set_trace()

# #         # ss, ee = CONFIG.RISWR_PLE.BINS.pre

# #         # SWR_m = mngs.pd.slice(xxm_all, ca1)

# #         # SWR_p["start_i"] = [bisect_left(tt, ss) for ss in SWR_p.start_s]
# #         # SWR_p["end_i"] = [bisect_left(tt, ee) for ee in SWR_p.end_s]

# #         # SWR_m["start_i"] = [bisect_left(tt, ss) for ss in SWR_m.start_s]
# #         # SWR_m["end_i"] = [bisect_left(tt, ee) for ee in SWR_m.end_s]

# #         # CONFIG.RISWR_PLE.BINS.pre
# #         # CONFIG.RISWR_PLE.BINS.mid
# #         # CONFIG.RISWR_PLE.BINS.post
# #         # __import__("ipdb").set_trace()

# #         # nt.shape


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
