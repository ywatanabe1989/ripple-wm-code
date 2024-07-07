#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-06 14:54:09 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/demographic/electrode_positions.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import seaborn as sns

mngs.gen.reload(mngs)
import warnings
from collections import OrderedDict
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

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
import os
import re
from collections import defaultdict


def find_subjects(subject_dirs):
    subjects = []

    for subject_dir in subject_dirs:
        match = re.match(r"./data/Sub_(\d+)/", subject_dir)
        if match:
            subject = match.groups()[0]  # Because match.groups() returns tuple
            subjects.append(subject)

    return subjects


def find_sessions(session_paths):
    subject_dict = defaultdict(set)
    session_dict = defaultdict(set)

    for session_path in session_paths:
        match = re.match(r"./data/Sub_(\d+)/Session_(\d+)/", session_path)
        if match:
            subject, session = match.groups()
            subject_dict[subject].add(session)
            session_dict[session].add(subject)

    # Convert sets to sorted lists and ensure all numbers are two-digit strings
    subject_dict = {
        k: sorted(v, key=lambda x: int(x)) for k, v in subject_dict.items()
    }

    return subject_dict


def find_rois(file_paths):
    roi_dict = defaultdict(set)

    for file_path in file_paths:
        match = re.match(
            r"./data/Sub_(\d+)/Session_(\d+)/iEEG/([A-Z]+)\.pkl", file_path
        )
        if match:
            subject, session, roi = match.groups()
            key = f"{subject:0>2}"
            roi_dict[key].add(roi)

    # Convert sets to sorted lists
    roi_dict = {k: sorted(v) for k, v in roi_dict.items()}

    return roi_dict


def to_csv(out):
    df_sessions = (
        mngs.pd.force_df(out["SESSIONS"])
        .melt()
        .rename(columns={"variable": "subject", "value": "session"})
    ).set_index("subject")

    df_rois = (
        mngs.pd.force_df(out["ROIS"])
        .melt()
        .rename(columns={"variable": "subject", "value": "roi"})
    ).set_index("subject")

    result = OrderedDict()
    for subject in df_sessions.index.unique():
        sessions = df_sessions.loc[subject, "session"].tolist()
        rois = df_rois.loc[subject, "roi"].tolist()
        result[subject] = {"session": sessions, "roi": rois}
    df = pd.DataFrame(result)
    df.index.name = "subject"
    return df.T


def main():
    out = OrderedDict()

    # Subjects
    out["SUBJECTS"] = find_subjects(mngs.gen.natglob(CONFIG["DIR_SUBJECT"]))

    # Sessions
    out["SESSIONS"] = find_sessions(mngs.gen.natglob(CONFIG["DIR_SESSION"]))

    # ROIs
    out["ROIS"] = find_rois(mngs.gen.natglob(CONFIG["PATH_iEEG"]))

    # Prints
    pprint(out)

    # to CSV
    df = to_csv(out)

    # Saving
    mngs.io.save(out, "./config/demographic_data.yaml", from_cwd=True)
    mngs.io.save(df, "./data/demographic/demographic.csv", from_cwd=True)


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
