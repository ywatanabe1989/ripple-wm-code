#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 09:03:38 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/_load_ripples.py

import mngs
import pandas as pd

"""Config"""
CONFIG = mngs.gen.load_configs()


def load_ripples(with_NT=False):
    pp = []
    mm = []
    lpath_p_exp = (
        CONFIG.PATH.RIPPLE if not with_NT else CONFIG.PATH.RIPPLE_WITH_NT
    )
    lpath_m_exp = (
        CONFIG.PATH.RIPPLE_MINUS
        if not with_NT
        else CONFIG.PATH.RIPPLE_MINUS_WITH_NT
    )
    for ca1 in CONFIG.ROI.CA1:
        lpath_p = mngs.gen.replace(lpath_p_exp, ca1)
        lpath_m = mngs.gen.replace(lpath_m_exp, ca1)

        pp.append(mngs.io.load(lpath_p))
        mm.append(mngs.io.load(lpath_m))

    pp = pd.concat(pp)
    mm = pd.concat(mm)

    return pp, mm
