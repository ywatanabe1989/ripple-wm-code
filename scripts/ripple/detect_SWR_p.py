#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-29 23:54:43 (ywatanabe)"
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


from scripts.load import load_iEEG
from scripts.utils import parse_lpath

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


# def detect_ripples_roi(sub, session, sd, roi):
#     """
#     1) take the common averaged signal of ROI
#     2) detect ripples from the signals of ROI and the common averaged signal
#     3) drop ROI ripples based on IoU
#     """
#     global iEEG, iEEG_ripple_band_passed

#     trials_info = mngs.io.load(eval(CONFIG["PATH_TRIALS_INFO"]))

#     # trials_info["set_size"]
#     trials_info["correct"] = trials_info["correct"].replace(
#         {0: False, 1: True}
#     )

#     iEEG, _ = load_iEEG(sub, session, roi, return_common_averaged_signal=True)
#     __import__("ipdb").set_trace()
#     xx = iEEG

#     # Fake dataframe for no channels data
#     if xx.shape[1] == 0:
#         fake_df = pd.DataFrame(
#             columns=["start_s", "end_s"],
#             data=np.array([[np.nan, np.nan]]),
#         )
#         return fake_df

#     # Main
#     df_r, xx_r, fs_r = mngs.dsp.detect_ripples(
#         xx,
#         CONFIG["FS_iEEG"],
#         CONFIG["RIPPLE_LOW_HZ"],
#         CONFIG["RIPPLE_HIGH_HZ"],
#         return_preprocessed_signal=True,
#     )
#     df_r.index.name = "trial_number"

#     mngs.io.save(
#         (xx_r, fs_r),
#         eval(CONFIG["PATH_iEEG"]).replace("iEEG", "iEEG_ripple_preprocessed"),
#         from_cwd=True,
#     )

#     df_r = transfer_metadata(df_r, trials_info)

#     df_r["subject"] = sub
#     df_r["session"] = session

#     # Drops rows with NaN values
#     df_r = df_r[~df_r.isna().any(axis=1)]

#     return df_r


def transfer_metadata(df_r, trials_info):
    df_r.index.name = "trial_number"

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


# def detect_ripples_all():
#     for roi in CONFIG["iEEG_ROIS"]:
#         mngs.gen.print_block(roi)

#         roi_connected = mngs.general.connect_strs([roi])

#         rips_df = []
#         for sub, session in itertools.product(
#             CONFIG["SUBJECTS"], CONFIG["FIRST_TWO_SESSIONS"]
#         ):
#             _df_r = add_phase(
#                 detect_ripples_roi(
#                     sub=sub,
#                     session=session,
#                     sd=CONFIG["RIPPLE_SD"],
#                     roi=roi_connected,
#                 )
#             )
#             rips_df.append(_df_r)
#         rips_df = pd.concat(rips_df)

#         mngs.io.save(
#             rips_df,
#             f"./data/rips_df/{roi_connected}.pkl",
#             from_cwd=True,
#         )


def main_lpath(lpath_iEEG):
    """
    LPATHS_iEEG = mngs.gen.natglob(CONFIG["PATH_iEEG"])
    lpath = LPATHS_iEEG[0]
    """
    # Parsing variables from lpath
    parsed = parse_lpath(lpath_iEEG)
    sub = parsed["sub"]
    session = parsed["session"]
    roi = parsed["roi"]

    # Loading
    iEEG = xx = mngs.io.load(lpath_iEEG)

    # Returns fake dataframe as ripple data for no channels data
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

    # Drops rows with NaN values due to the fake insertion
    df_r = df_r[~df_r.isna().any(axis=1)]

    # Adds metadata to the ripple table
    trials_info = mngs.io.load(eval(CONFIG["PATH_TRIALS_INFO"]))
    df_r = transfer_metadata(df_r, trials_info)
    df_r["subject"] = sub
    df_r["session"] = session

    # Saving
    # Ripple
    spath_ripple = eval(CONFIG["PATH_RIPPLE"])
    mngs.io.save(df_r, spath_ripple, from_cwd=True, dry_run=False)
    # Ripple-band iEEG data
    spath_ripple_band_iEEG = eval(CONFIG["PATH_iEEG"]).replace(
        "iEEG", "iEEG_ripple_preprocessed"
    )
    mngs.io.save(
        (xx_r, fs_r),
        spath_ripple_band_iEEG,
        from_cwd=True,
        dry_run=False,
    )


def main():
    LPATHS_iEEG = mngs.gen.natglob(CONFIG["PATH_iEEG"])
    for lpath_iEEG in LPATHS_iEEG:
        main_lpath(lpath_iEEG)


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
