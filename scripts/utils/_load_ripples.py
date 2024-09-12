#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-12 21:52:39 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/_load_ripples.py

import mngs
import pandas as pd

"""Config"""
CONFIG = mngs.gen.load_configs()


def load_ripples():
    pp = []
    mm = []
    for ca1 in CONFIG.ROI.CA1:
        lpath_p = mngs.gen.replace(CONFIG.PATH.RIPPLE, ca1)
        lpath_m = mngs.gen.replace(CONFIG.PATH.RIPPLE_MINUS, ca1)

        pp.append(mngs.io.load(lpath_p))
        mm.append(mngs.io.load(lpath_m))

    pp = pd.concat(pp)
    mm = pd.concat(mm)

    return pp, mm
