#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 17:37:46 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/distance_from_O.py

"""This script calculates distance from O during pre-, mid-, and post-SWR+/- events"""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts import utils
from scripts.ripple.NT.distance.from_O import calc_dist_by_condi

"""Functions & Classes"""


# def run_statistical_tests(data):
#     # Implement your statistical tests here
#     # For example, comparing SWR+ vs SWR- for each condition
#     results = {}
#     for match in CONFIG.MATCHES:
#         for phase in PHASES:
#             swr_p = data[("SWR+", match, phase)]
#             swrm = data[("SWR-", match, phase)]
#             # Perform your statistical test here
#             # results[(match, phase)] = your_statistical_test(swr_p, swrm)
#     return results


def main():
    swr_p_all, swr_m_all = utils.load_ripples(with_NT=True)
    data = calc_dist_by_condi(swr_p_all, swr_m_all)

    # # Line Plot
    # fig = plot_line(data)
    # mngs.io.save(fig, "./SWR-triggered_distance_from_O.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        fig_scale=2,
        font_size_legend=3,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
