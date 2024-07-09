#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-10 00:14:49 (ywatanabe)"
# ./scripts/ripple/detect_SWR_p.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys
from bisect import bisect_left
from functools import partial

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


# from scripts.load import load_iEEG
from scripts.utils import parse_lpath

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def transfer_metadata(df, trials_info):
    df.index.name = "trial_number"

    # Transfer information to the dataframe
    transfer_keys = [
        "set_size",
        "match",
        "correct",
        "response_time",
    ]
    # Initialization
    for k in transfer_keys:
        df[k] = str(np.nan)

    assert trials_info.index.name == "trial_number"
    for ii in trials_info.index:
        for k in transfer_keys:
            df.loc[ii, k] = trials_info.loc[ii, k]

    # Remove NaN rows
    df = df[~df.isna().any(axis=1)].copy()

    return df


def add_firing_patterns(df):

    def _add_firing_patterns(df_row, spike_times):

        trial_number = df_row.name
        i_trial = trial_number - 1

        firing_pattern = spike_times[i_trial][
            (float(df_row.start_s) < spike_times[i_trial])
            * (spike_times[i_trial] < float(df_row.end_s))
        ]
        firing_pattern = (~firing_pattern.isna()).sum()
        return [firing_pattern]

    # Fetch metadata to load spike_times
    sub = df.subject.iloc[0]
    session = df.session.iloc[0]
    roi = df.roi.iloc[0]

    # Loading
    spike_times = mngs.io.load(eval(CONFIG.PATH.SPIKE_TIMES))
    spike_times = [
        st.replace("", np.nan).astype(float) + 6 for st in spike_times
    ]
    df["firing_pattern"] = df.apply(
        partial(_add_firing_patterns, spike_times=spike_times), axis=1
    )

    return df


def main_lpath(lpath_iEEG):

    # LPATHS_iEEG = mngs.gen.natglob(CONFIG["PATH_iEEG"])
    # lpath = LPATHS_iEEG[0]

    # Parsing variables from lpath
    parsed = parse_lpath(lpath_iEEG)
    sub = parsed["sub"]
    session = parsed["session"]
    roi = parsed["roi"]

    # Loading
    iEEG = xx = mngs.io.load(lpath_iEEG)

    # Skipping ripple data for no channels data
    if xx.shape[1] == 0:
        fake_df = pd.DataFrame(
            columns=["start_s", "end_s"],
            data=np.array([[np.nan, np.nan]]),
        )
        return

    # Main
    df, xx_r, fs_r = mngs.dsp.detect_ripples(
        xx,
        CONFIG.FS.iEEG,
        CONFIG.RIPPLE.LOW_HZ,
        CONFIG.RIPPLE.HIGH_HZ,
        sd=CONFIG.RIPPLE.THRES_SD,
        min_duration_ms=CONFIG.RIPPLE.MIN_DUR_MS,
        return_preprocessed_signal=True,
    )
    df.index += 1
    df.index.name = "trial_number"

    # Adds metadata to the ripple table
    trials_info = mngs.io.load(eval(CONFIG.PATH.TRIALS_INFO))
    trials_info.set_index("trial_number", inplace=True)
    df = transfer_metadata(df, trials_info)
    df["subject"] = sub
    df["session"] = session
    df["roi"] = roi

    # Remove NaN rows here
    df = df[~df.isna().any(axis=1)]

    # Firing pattern
    df = add_firing_patterns(df)

    # Saving
    # Ripple
    spath_ripple = eval(CONFIG.PATH.RIPPLE)
    mngs.io.save(df, spath_ripple, from_cwd=True, dry_run=False)
    # Ripple-band iEEG data
    spath_ripple_band_iEEG = eval(CONFIG.PATH.iEEG).replace(
        "iEEG", "iEEG_ripple_preprocessed"
    )
    mngs.io.save(
        (xx_r, fs_r),
        spath_ripple_band_iEEG,
        from_cwd=True,
        dry_run=False,
    )


def main():
    LPATHS_iEEG = mngs.gen.natglob(CONFIG.PATH.iEEG)
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
