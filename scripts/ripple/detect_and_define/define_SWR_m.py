#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-01 08:38:26 (ywatanabe)"
# ./scripts/ripple/detect_and_define/define_SWR_m.py


"""This script defines SWR- as control events for SWR+."""


"""Imports"""
import random
import sys
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts.ripple.detect_and_define.detect_SWR_p import (add_firing_patterns,
                                                           transfer_metadata)
from scripts.utils import parse_lpath

"""Config"""
CONFIG = mngs.gen.load_configs()


"""Functions & Classes"""


def add_peak_s(row, xxr, fs_r):
    trial_number = row.name
    i_trial = trial_number - 1
    _xxr = xxr[i_trial][int(row.start_s * fs_r) : int(row.end_s * fs_r)]
    peak_pos = _xxr.argmax()
    peak_s = row.start_s + peak_pos / fs_r
    return peak_s


def add_rel_peak_pos(row, xxr, fs_r):
    trial_number = row.name
    i_trial = trial_number - 1
    _xxr = xxr[i_trial][int(row.start_s * fs_r) : int(row.end_s * fs_r)]
    return _xxr.argmax() / len(_xxr)


def add_peak_amp_sd(row, xxr, fs_r):
    trial_number = row.name
    i_trial = trial_number - 1
    _xxr = xxr[i_trial][int(row.start_s * fs_r) : int(row.end_s * fs_r)]
    return _xxr.max()


def main():
    LPATHS_RIPPLE = mngs.gen.natglob(CONFIG.PATH.RIPPLE)
    LPATHS_iEEG_RIPPLE_BAND = mngs.gen.natglob(CONFIG.PATH.iEEG_RIPPLE_BAND)

    for lpath_ripple, lpath_iEEG in zip(
        LPATHS_RIPPLE, LPATHS_iEEG_RIPPLE_BAND
    ):
        main_lpath(lpath_ripple, lpath_iEEG)


def main_lpath(lpath_ripple, lpath_iEEG):
    # Loading
    # SWR+
    df_p = mngs.io.load(lpath_ripple)
    (iEEG_ripple_band, fs_r) = (xxr, fs_r) = mngs.io.load(lpath_iEEG)

    # Parsing lpath
    parsed = parse_lpath(lpath_ripple)
    sub = parsed["sub"]
    session = parsed["session"]
    roi = parsed["roi"]

    # Trials info
    trials_info = mngs.io.load(
        mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, parsed)
    )
    trials_info.set_index("trial_number", inplace=True)

    # Starts defining SWR- using SWR+, iEEG signal, and trials_info
    df_m = df_p[["start_s", "end_s", "duration_s"]].copy()

    # Shuffle ripple period (row) within a session as controls
    df_m = df_m.iloc[np.random.permutation(np.arange(len(df_m)))]

    # Override trial_number
    new_trial_numbers = pd.Series(
        np.random.choice(trials_info.index, len(df_m)).astype(int),
        name="trial_number",
    )

    assert trials_info.index.min() <= new_trial_numbers.min()
    assert new_trial_numbers.max() <= trials_info.index.max()

    df_m.index = new_trial_numbers
    df_m = df_m.sort_index()

    # Adds metadata for the control data
    df_m = transfer_metadata(df_m, trials_info)

    # peak_s
    df_m["peak_s"] = df_m.apply(
        partial(add_peak_s, xxr=xxr, fs_r=fs_r), axis=1
    )

    # rel_peak_pos
    df_m["rel_peak_pos"] = df_m.apply(
        partial(add_rel_peak_pos, xxr=xxr, fs_r=fs_r), axis=1
    )

    # peak_amp_sd
    df_m["peak_amp_sd"] = df_m.apply(
        partial(add_peak_amp_sd, xxr=xxr, fs_r=fs_r), axis=1
    )

    # subject
    df_m.loc[:, "subject"] = sub

    # session
    df_m.loc[:, "session"] = session

    # session
    df_m.loc[:, "roi"] = roi

    # Firing patterns
    df_m = add_firing_patterns(df_m)

    assert len(df_p) == len(df_m)

    # Saving
    spath = lpath_ripple.replace("SWR_p", "SWR_m")
    mngs.io.save(df_m, spath, from_cwd=True)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, random=random, np=np
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
