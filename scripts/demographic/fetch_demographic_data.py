#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-23 23:25:18 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/demographic/electrode_positions.py


"""This script processes demographic data, including subject information, sessions, and ROIs."""


"""Imports"""
import os
import re
import sys
import matplotlib.pyplot as plt
import mngs
from collections import OrderedDict
from pprint import pprint
import pandas as pd
from collections import defaultdict

"""Functions & Classes"""

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
    out["SUBJECTS"] = find_subjects(mngs.io.glob(CONFIG.DIR.SUBJECT))

    # Sessions
    out["SESSIONS"] = find_sessions(mngs.io.glob(CONFIG.DIR.SESSION))

    # ROIs
    out["ROIS"] = find_rois(mngs.io.glob(CONFIG.PATH.iEEG))

    # Prints
    pprint(out)

    # to CSV
    df = to_csv(out)

    # Saving
    mngs.io.save(out, "./config/demographic_data.yaml", from_cwd=True)
    mngs.io.save(df, "./data/demographic/demographic.csv", from_cwd=True)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
