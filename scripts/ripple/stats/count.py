#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 19:39:11 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/stats/time_course.py

"""This script does XYZ."""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
from scripts.utils import load_ripples

"""Functions & Classes"""


def main():
    pp, mm = load_ripples()

    xx = pp

    xx = xx.set_index(["subject", "session", "roi"])
    xx["n_ripple"] = 1
    ns = xx.groupby(["subject", "session", "roi"]).agg({"n_ripple": ["sum"]})
    ns = xx.groupby(["subject", "roi"]).agg({"n_ripple": ["sum"]})

    print(ns)

    mngs.io.save(ns, "ns.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
