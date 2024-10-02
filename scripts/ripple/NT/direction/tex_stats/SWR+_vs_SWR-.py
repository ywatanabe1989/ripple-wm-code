#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-02 23:25:15 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/direction/tex_stats/SWR+_vs_SWR-.py


"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""

"""Imports"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import mngs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from glob import glob
from pprint import pprint
import warnings
import logging
from tqdm import tqdm
import xarray as xr

try:
    from scripts import utils
except:
    pass

"""Warnings"""
mngs.pd.ignore_SettingWithCopyWarning()

"""Aliases"""
pt = print

"""Configs"""
# CONFIG = mngs.gen.load_configs()

"""Parameters"""
SORTED_COLS = [
            "v1",
            "v2",
            "p_value",
            "stars",
            "dof",
            "w_statistic",
            "effsize",
            "match",
            "set_size",
            "measure",
            "vSWR_def",
            "is_abs",
        ]

"""Functions & Classes"""

# Cosine similarity 1,Cosine similarity 2,P value,stars,W statistic,dof,Effect size,match
# "|cos_{sim}(\overrightarrow{\mathrm{rSWR_{time}^{+}}}, \overrightarrow{\mathrm{g_{E}g_{R}}})|","|cos_{sim}(\overrightarrow{\mathrm{rSWR_{time}^{+}}}, \overrightarrow{\mathrm{g_{R}}})|",0.002,**,3.071,369.538,0.588,Match IN


def preview_tex(tex_str_list):
    fig, axes = plt.subplots(nrows=len(tex_str_list), ncols=1, figsize=(10, 3*len(tex_str_list)))
    if len(tex_str_list) == 1:
        axes = [axes]
    for ax, tex_string in zip(axes, tex_str_list):
        ax.text(0.5, 0.7, tex_string, size=20, ha='center', va='center')
        ax.text(0.5, 0.3, f"${tex_string}$", size=20, ha='center', va='center')
        ax.axis('off')
    plt.tight_layout()
    return fig


def print_tex(tex_str_list):
    for tex_string in tex_str_list:
        print(f"Raw: {tex_string}")
        print(f"LaTeX: ${tex_string}$")
        print("-" * 40)


def rename_columns(df):
    col_map = {
        "v1": "Cosine similarity 1",
        "v2": "Cosine similarity 2",
        "p_value": "P value",
        "stars": "",
        "w_statistic": "W statistic",
        "effsize": "Effect size",
        "match": "Match",
        "set_size": "Set size",
        }
    # \n(P(x1 < x2) + 0.5 P(x1 = x2))
    return df.rename(columns=col_map)

def to_vec(v_str):
    return f"\\overrightarrow{{\\mathrm{{{v_str}}}}}"

def main():
    lpath = "./scripts/ripple/NT/direction/stats/SWR+_vs_SWR-/stats_all.csv"
    df = mngs.io.load(
        lpath
    )
    # Filtering
    df = df[df.measure == "cosine"]
    df = df[df.match.isin(['1','2'])]
    # df = df[df.set_size == "all"]

    # Sorting
    df = df[SORTED_COLS]

    # Reset Index
    df = df.reset_index().drop(columns=["index"])

    print(df)


    # Mapping
    mapper = {
        f"{er}SWR{sign}": to_vec(f"{er}SWR_{{direction}}^{{{sign}}}")
        for er in ["e", "r"] for sign in ["+", "-"]
    }
    mapper.update({"vER": to_vec("g_{E}g_{R}")})
    mapper.update({" - ": ", "})

    # Values
    df = mngs.pd.replace(df, mapper)
    for ii, row in df.iterrows():
        swr_direction = CONFIG.RIPPLE.DIRECTIONS[row.vSWR_def-1].split("_")[-1].replace("NT", "TIME").capitalize()
        v1 = df.iloc[ii].v1.replace("direction", swr_direction)
        v2 = df.iloc[ii].v2.replace("direction", swr_direction)
        df.at[ii, 'v1'] = v1
        df.at[ii, 'v2'] = v2

        if row.is_abs:
            df.iloc[ii].v1 = "| " + df.iloc[ii].v1 + " |"
    df = mngs.pd.replace(df, CONFIG.MATCHES_STR, cols=["match"])

    # Columns
    df = rename_columns(df)

    # Drops
    df = df.drop(columns=["measure", "is_abs"])

    # Render text as a figure
    fig = preview_tex(mapper.values())

    # Saving
    mngs.io.save(df, "tex_formatted.csv")
    mngs.io.save(fig, "tex_preview.jpg")



if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
