#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-09 08:22:37 (ywatanabe)"
# calc_NT_with_GPFA.py


"""This script does XYZ."""


"""Imports"""
import logging
import re
import sys
from glob import glob

import matplotlib.pyplot as plt
import mngs
import neo
import numpy as np
import quantities as pq
from elephant.gpfa import GPFA
from natsort import natsorted
from utils import parse_lpath

"""Config"""
CONFIG = mngs.gen.load_configs()


"""Functions & Classes"""


def spiketimes_to_spiketrains(
    spike_times_all_trials, without_retrieval_phase=False
):
    spike_trains_all_trials = []
    for st_trial in spike_times_all_trials:
        spike_trains_trial = []
        for col, col_df in st_trial.items():
            spike_times = col_df[col_df != ""]
            if without_retrieval_phase:
                spike_times = spike_times[spike_times < 0]
                train = neo.SpikeTrain(
                    list(spike_times) * pq.s, t_start=-6.0, t_stop=0
                )
            else:
                train = neo.SpikeTrain(
                    list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0
                )
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)

    return spike_trains_all_trials


def switch_regarding_match(spike_trains, subject, session, match):
    if match != "all":
        trials_info = mngs.io.load(
            f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
        )
        indi = trials_info.match == match
        spike_trains = [
            spike_trains[ii] for ii, bl in enumerate(indi) if bl == True
        ]
    return spike_trains


def determine_spath(lpath_spike_times, match, without_retrieval_phase):
    spath_NTs = lpath_spike_times.replace("spike_times", "NT").replace(
        ".pkl", f"_match_{match}.npy"
    )
    if without_retrieval_phase:
        spath_NTs = spath_NTs.replace(".npy", "_wo_R.npy")
    return spath_NTs


def main(match="all", without_retrieval_phase=False):
    # Parameters
    BIN_SIZE = CONFIG.GPFA.BIN_SIZE_MS * pq.ms

    # Loads spike timings
    LPATHS_SPIKE_TIMES = mngs.gen.glob(CONFIG.PATH.SPIKE_TIMES)

    for lpath in LPATHS_SPIKE_TIMES:
        sub, session, roi = parse_lpath(lpath)

        # Spike trains of all trials;
        # some of spike trains data are unavailable in the original datset.
        spike_times_all_trials = mngs.io.load(lpath)
        spike_trains = spiketimes_to_spiketrains(
            spike_times_all_trials,
            without_retrieval_phase=without_retrieval_phase,
        )
        spike_trains = switch_regarding_match(
            spike_trains, sub, session, match
        )

        # GPFA calculation
        gpfa = GPFA(bin_size=BIN_SIZE, x_dim=8)
        try:
            NTs = np.stack(gpfa.fit_transform(spike_trains), axis=0)

            # Saving
            spath_NTs = determine_spath(lpath, match, without_retrieval_phase)
            mngs.io.save(NTs, spath_NTs, from_cwd=True)

        except Exception as e:
            logging.warn(
                f"\nError raised during GPFA calculation. Spike_trains might be unavailable. "
                f"Skipping {spath_NTs}.:\n",
                e,
            )


if __name__ == "__main__":
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main(match="all", without_retrieval_phase=False)
    mngs.gen.close(CONFIG, verbose=False, notify=True)


# for f in `find /mnt/ssd/ripple-wm-code/scripts/NT/calc_NT_with_GPFA/data -type f -name "*.npy"`; do
#     echo $f
#     tgt=$(echo $f | sed 's|/mnt/ssd/ripple-wm-code/scripts/NT/calc_NT_with_GPFA/||')
#     ln -sfr $f $tgt
# done

# EOF
