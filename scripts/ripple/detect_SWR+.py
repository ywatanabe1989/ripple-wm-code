#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-27 20:50:37 (ywatanabe)"
# detect_ripples.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
import itertools

import torch
from scripts import load

# from scripts.externals.ripple_detection.ripple_detection.detectors import (
#     Kay_ripple_detector,
# )
from tqdm import tqdm

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def calc_iou(a, b):
    """
    Calculate Intersection over Union
    a = [0, 10]
    b = [0, 3]
    calc_iou(a, b) # 0.3
    """
    (a_s, a_e) = a
    (b_s, b_e) = b

    a_len = a_e - a_s
    b_len = b_e - b_s

    abx_s = max(a_s, b_s)
    abx_e = min(a_e, b_e)

    abx_len = max(0, abx_e - abx_s)

    return abx_len / (a_len + b_len - abx_len)


def detect_ripples_roi(sub, session, sd, iEEG_roi):
    """
    1) take the common averaged signal of ROI
    2) detect ripples from the signals of ROI and the common averaged signal
    3) drop ROI ripples based on IoU
    """
    global iEEG, iEEG_ripple_band_passed

    trials_info = mngs.io.load(eval(CONFIG["PATH_TRIALS_INFO"]))

    # trials_info["set_size"]
    trials_info["correct"] = trials_info["correct"].replace(
        {0: False, 1: True}
    )

    # sync_z_session = mngs.io.load(
    #     f"./data/Sub_{sub}/Session_{session}/sync_z/{iEEG_roi}.npy"
    # )

    # koko
    iEEG, iEEG_common_ave = load.iEEG(
        sub, session, iEEG_roi, return_common_averaged_signal=True
    )
    # iEEG.shape # (50, 8, 16000)

    xx, xxa = iEEG, iEEG_common_ave  # aliases

    if xx.shape[1] == 0:  # For no channels data
        fake_df = pd.DataFrame(
            columns=["start_s", "end_s"],
            data=np.array([[np.nan, np.nan]]),
        )
        return fake_df

    # Bandpass filtering
    xx_r, xxa_r = [
        mngs.dsp.filt.bandpass(
            np.array(xx).astype(float),
            np.array(CONFIG["FS_iEEG"]).astype(float),
            CONFIG["RIPPLE_BANDS"],
        )
        .astype(np.float64)
        .squeeze(-2)
        for _xx in [xx, xxa]
    ]

    # Main
    time = mngs.dsp.time(0, CONFIG["TRIAL_SEC"], CONFIG["FS_iEEG"])
    time_iEEG = time
    # time_iEEG = np.arange(iEEG.shape[-1]) / CONFIG["FS_iEEG"]
    # speed = 0 * time_iEEG

    rip_df = mngs.dsp.detect_ripples(
        xx_r,
        CONFIG["FS_iEEG"],
        CONFIG["RIPPLE_LOW_HZ"],
        CONFIG["RIPPLE_HIGH_HZ"],
    )
    rip_df.index.name = "trial_number"

    # Transfer information to the dataframe
    transfer_keys = [
        "set_size",
        "match",
        "correct",
        "response_time",
    ]
    # Initialization
    for k in transfer_keys:
        rip_df[k] = str(np.nan)

    for ii in range(len(trials_info)):
        for k in transfer_keys:
            rip_df.loc[ii, k] = trials_info.loc[ii, k]

    rip_df["subject"] = sub
    rip_df["session"] = session

    # Drops rows with NaN values
    rip_df = rip_df[~rip_df.isna().any(axis=1)]

    return rip_df


def add_phase(rip_df):
    rip_df["center_time"] = (rip_df["start_s"] + rip_df["end_s"]) / 2
    rip_df["phase"] = None
    rip_df.loc[rip_df["center_time"] < 1, "phase"] = "Fixation"
    rip_df.loc[
        (1 < rip_df["center_time"]) & (rip_df["center_time"] < 3), "phase"
    ] = "Encoding"
    rip_df.loc[
        (3 < rip_df["center_time"]) & (rip_df["center_time"] < 6), "phase"
    ] = "Maintenance"
    rip_df.loc[6 < rip_df["center_time"], "phase"] = "Retrieval"
    return rip_df


# def plot_traces(FS_iEEG):
#     plt.close()
#     plot_start_sec = 0
#     plot_dur_sec = 8

#     plot_dur_pts = plot_dur_sec * FS_iEEG
#     plot_start_pts = plot_start_sec * FS_iEEG

#     i_ch = np.random.randint(iEEG_ripple_band_passed.shape[1])
#     i_trial = 0

#     fig, axes = plt.subplots(2, 1, sharex=True)
#     lw = 1

#     time_iEEG = (np.arange(iEEG.shape[-1]) / FS_iEEG) - 6

#     axes[0].plot(
#         time_iEEG[plot_start_pts : plot_start_pts + plot_dur_pts],
#         iEEG[0][i_ch, plot_start_pts : plot_start_pts + plot_dur_pts],
#         linewidth=lw,
#         label="Raw LFP",
#     )

#     axes[1].plot(
#         time_iEEG[plot_start_pts : plot_start_pts + plot_dur_pts],
#         iEEG_ripple_band_passed[0][
#             i_ch, plot_start_pts : plot_start_pts + plot_dur_pts
#         ],
#         linewidth=lw,
#         label="Ripple-band-passed LFP",
#     )
#     # fills ripple time
#     rip_plot_df = rip_df[
#         (rip_df["trial_number"] == i_trial + 1)
#         & (plot_start_sec < rip_df["start_s"])
#         & (rip_df["end_s"] < plot_start_sec + plot_dur_sec)
#     ]

#     for ax in axes:
#         for ripple in rip_plot_df.itertuples():
#             ax.axvspan(
#                 ripple.start_s - 6,
#                 ripple.end_s - 6,
#                 alpha=0.1,
#                 color="red",
#                 zorder=1000,
#             )
#             ax.axvline(x=-5, color="gray", linestyle="dotted")
#             ax.axvline(x=-3, color="gray", linestyle="dotted")
#             ax.axvline(x=0, color="gray", linestyle="dotted")

#     axes[-1].set_xlabel("Time from probe [sec]")

#     # plt.show()
#     mngs.io.save(
#         plt, "./tmp/ripple_repr_traces_sub_01_session_01_trial_01-50.png"
#     )


# def plot_hist(rip_df):
#     plt.close()
#     rip_df["dur_time"] = rip_df["end_s"] - rip_df["start_s"]
#     rip_df["dur_ms"] = rip_df["dur_time"] * 1000

#     plt.hist(rip_df["dur_ms"], bins=100)
#     plt.xlabel("Ripple duration [sec]")
#     plt.ylabel("Count of ripple events")
#     # plt.show()
#     mngs.io.save(plt, "./tmp/ripple_count_sub_01_session_01_trial_01-50.png")


def calc_rip_incidence_hz(rip_df):
    rip_df["n"] = 1
    rip_incidence_hz = pd.concat(
        [
            rip_df[rip_df["phase"] == "Fixation"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 1,
            rip_df[rip_df["phase"] == "Encoding"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 2,
            rip_df[rip_df["phase"] == "Maintenance"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 3,
            rip_df[rip_df["phase"] == "Retrieval"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 2,
        ],
        axis=1,
    ).fillna(0)
    rip_incidence_hz.columns = [
        "Fixation",
        "Encoding",
        "Maintenance",
        "Retrieval",
    ]
    return rip_incidence_hz


def detect_ripples_all():
    for iEEG_roi in CONFIG["iEEG_ROIS"]:
        mngs.gen.print_block(iEEG_roi)

        iEEG_roi_connected = mngs.general.connect_strs(
            [iEEG_roi]
        )  # For multiple ROIs

        rips_df = []
        for sub, session in itertools.product(
            CONFIG["SUBJECTS"], CONFIG["FIRST_TWO_SESSIONS"]
        ):
            _rip_df = add_phase(
                detect_ripples_roi(
                    sub=sub,
                    session=session,
                    sd=CONFIG["RIPPLE_SD"],
                    iEEG_roi=iEEG_roi_connected,
                )
            )
            rips_df.append(_rip_df)
        rips_df = pd.concat(rips_df)

        mngs.io.save(
            rips_df,
            f"./data/rips_df/{iEEG_roi_connected}.pkl",
            from_cwd=True,
        )


main = detect_ripples_all

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
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
