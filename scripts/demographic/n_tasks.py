#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 20:14:26 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/demographic/n_tasks.py

"""This script does XYZ."""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import pandas as pd
from scripts import utils

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def load_TI():
    TI_all = []
    for lpath in mngs.gen.glob(CONFIG.PATH.TRIALS_INFO):
        TI = mngs.io.load(lpath)
        parsed = utils.parse_lpath(lpath)
        for k, v in parsed.items():
            TI[k] = v
        TI_all.append(TI)
    return pd.concat(TI_all)


def load_TI_CA1():
    TI_all = []
    for ca1 in CONFIG.ROI.CA1:
        TI = mngs.io.load(mngs.gen.replace(CONFIG.PATH.TRIALS_INFO, ca1))
        for k, v in ca1.items():
            TI[k] = v
        TI_all.append(TI)
    return pd.concat(TI_all)


def main():
    # All
    TI = load_TI()
    TI["n_trial"] = 1
    n_tasks = TI.groupby(["sub", "session", "roi", "match", "set_size"]).agg(
        {"n_trial": "sum"}
    )
    print(n_tasks)
    mngs.io.save(n_tasks, "n_tasks.csv")

    # CA1
    TI_CA1 = load_TI_CA1()
    TI_CA1["n_trial"] = 1
    n_tasks = TI_CA1.groupby(
        ["sub", "session", "roi", "match", "set_size"]
    ).agg({"n_trial": "sum"})
    print(n_tasks)
    mngs.io.save(n_tasks, "n_tasks_CA1.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
