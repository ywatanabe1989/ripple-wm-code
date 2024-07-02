#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-29 16:15:56 (ywatanabe)"
# ./scripts/ripple/detect_SWR_p.py


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


def detect_ripples_roi(sub, session, sd, roi):
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

    iEEG, _ = load.iEEG(sub, session, roi, return_common_averaged_signal=True)
    xx = iEEG  # alias

    # Fake dataframe for no channels data
    if xx.shape[1] == 0:
        fake_df = pd.DataFrame(
            columns=["start_s", "end_s"],
            data=np.array([[np.nan, np.nan]]),
        )
        return fake_df

    # Main
    df_r, xx_r, fs_r = mngs.dsp.detect_ripples(
        xx,
        CONFIG["FS_iEEG"],
        CONFIG["RIPPLE_LOW_HZ"],
        CONFIG["RIPPLE_HIGH_HZ"],
        return_preprocessed_signal=True,
    )
    df_r.index.name = "trial_number"

    mngs.io.save(
        (xx_r, fs_r),
        eval(CONFIG["PATH_iEEG"]).replace("iEEG", "iEEG_ripple_preprocessed"),
        from_cwd=True,
    )

    df_r = transfer_metadata(df_r, trials_info)

    df_r["subject"] = sub
    df_r["session"] = session

    # Drops rows with NaN values
    df_r = df_r[~df_r.isna().any(axis=1)]

    return df_r


def transfer_metadata(df_r, trials_info):
    # Transfer information to the dataframe
    transfer_keys = [
        "set_size",
        "match",
        "correct",
        "response_time",
    ]
    # Initialization
    for k in transfer_keys:
        df_r[k] = str(np.nan)

    for ii in range(len(trials_info)):
        for k in transfer_keys:
            df_r.loc[ii, k] = trials_info.loc[ii, k]
    return df_r


def add_phase(df_r):
    df_r["center_time"] = (df_r["start_s"] + df_r["end_s"]) / 2
    df_r["phase"] = None
    df_r.loc[df_r["center_time"] < 1, "phase"] = "Fixation"
    df_r.loc[
        (1 < df_r["center_time"]) & (df_r["center_time"] < 3), "phase"
    ] = "Encoding"
    df_r.loc[
        (3 < df_r["center_time"]) & (df_r["center_time"] < 6), "phase"
    ] = "Maintenance"
    df_r.loc[6 < df_r["center_time"], "phase"] = "Retrieval"
    return df_r


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
#     rip_plot_df = df_r[
#         (df_r["trial_number"] == i_trial + 1)
#         & (plot_start_sec < df_r["start_s"])
#         & (df_r["end_s"] < plot_start_sec + plot_dur_sec)
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


# def plot_hist(df_r):
#     plt.close()
#     df_r["dur_time"] = df_r["end_s"] - df_r["start_s"]
#     df_r["dur_ms"] = df_r["dur_time"] * 1000

#     plt.hist(df_r["dur_ms"], bins=100)
#     plt.xlabel("Ripple duration [sec]")
#     plt.ylabel("Count of ripple events")
#     # plt.show()
#     mngs.io.save(plt, "./tmp/ripple_count_sub_01_session_01_trial_01-50.png")


# def calc_rip_incidence_hz(df_r):
#     df_r["n"] = 1
#     rip_incidence_hz = pd.concat(
#         [
#             df_r[df_r["phase"] == "Fixation"]
#             .pivot_table(columns=["trial_number"], aggfunc="sum")
#             .T["n"]
#             / 1,
#             df_r[df_r["phase"] == "Encoding"]
#             .pivot_table(columns=["trial_number"], aggfunc="sum")
#             .T["n"]
#             / 2,
#             df_r[df_r["phase"] == "Maintenance"]
#             .pivot_table(columns=["trial_number"], aggfunc="sum")
#             .T["n"]
#             / 3,
#             df_r[df_r["phase"] == "Retrieval"]
#             .pivot_table(columns=["trial_number"], aggfunc="sum")
#             .T["n"]
#             / 2,
#         ],
#         axis=1,
#     ).fillna(0)
#     rip_incidence_hz.columns = [
#         "Fixation",
#         "Encoding",
#         "Maintenance",
#         "Retrieval",
#     ]
#     return rip_incidence_hz


def detect_ripples_all():
    for roi in CONFIG["iEEG_ROIS"]:
        mngs.gen.print_block(roi)

        roi_connected = mngs.general.connect_strs([roi])  # For multiple ROIs

        rips_df = []
        for sub, session in itertools.product(
            CONFIG["SUBJECTS"], CONFIG["FIRST_TWO_SESSIONS"]
        ):
            _df_r = add_phase(
                detect_ripples_roi(
                    sub=sub,
                    session=session,
                    sd=CONFIG["RIPPLE_SD"],
                    roi=roi_connected,
                )
            )
            rips_df.append(_df_r)
        rips_df = pd.concat(rips_df)

        mngs.io.save(
            rips_df,
            f"./data/rips_df/{roi_connected}.pkl",
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
