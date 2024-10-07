#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-11 20:41:30 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/jointplot_set_size_8.py


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
import seaborn as sns
import importlib

import mngs

importlib.reload(mngs)

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


def main():
    for meta in CONFIG.ROI.CA1:
        cache = mngs.gen.listed_dict()
        for sample_type in ["all", "SWR+", "SWR-"]:
            meta.update({"sample_type": sample_type})
            lpath = mngs.gen.replace(
                CONFIG.PATH.SCATTER,
                meta,
            )
            for match in [1, 2]:
                match_str = ["IN", "OUT"][match - 1]

                fname = mngs.path.split(lpath)[1]

                df = mngs.io.load(lpath)
                keys = [f"match: {match}", "set_size: 8"]
                for k in keys:
                    df = df[mngs.gen.search(k, df.columns)[1]]

                df.columns = pd.MultiIndex.from_tuples(
                    [
                        (
                            col.split(", ")[0].split(": ")[1],
                            col.split(", ")[-1].split(": ")[1],
                        )
                        for col in df.columns
                    ]
                )

                scatter_df = (
                    df.stack(level=0, future_stack=True)
                    .reset_index()
                    .rename(
                        columns={
                            "level_1": "phase",
                            0: "scatter_x",
                            1: "scatter_y",
                        }
                    )
                )

                title = f"{match_str}/{sample_type}/{fname}"

                hue_order = list(CONFIG.PHASES.keys())
                hue_color = [
                    CC[CONFIG.PHASES[phase].color] for phase in hue_order
                ]

                g = sns.jointplot(
                    data=scatter_df,
                    x=f"{sample_type}_scatter_x",
                    y=f"{sample_type}_scatter_y",
                    hue="phase",
                    hue_order=hue_order,
                    palette=hue_color,
                    kind="kde",
                    title=title,
                )
                # xlim = g.x.min(), g.x.max()
                # ylim = g.y.min(), g.y.max()

                xlim = (
                    scatter_df[f"{sample_type}_scatter_x"].min(),
                    scatter_df[f"{sample_type}_scatter_x"].max(),
                )
                ylim = (
                    scatter_df[f"{sample_type}_scatter_y"].min(),
                    scatter_df[f"{sample_type}_scatter_y"].max(),
                )

                ext = 1
                xlim = (xlim[0] - ext), (xlim[1] + ext)
                ylim = (ylim[0] - ext), (ylim[1] + ext)

                spath = f"./data/CA1/jointplot_set_size_8/{match_str}/{sample_type}/{fname}.jpg"

                cache["g"].append(g)
                cache["xlim"].append(xlim)
                cache["ylim"].append(ylim)
                cache["spath"].append(spath)

                # mngs.io.save(
                #     g,
                #     f"./data/CA1/jointplot_set_size_8/{match_str}/{sample_type}/{fname}.jpg",
                #     from_cwd=True,
                # )

        xlim_global = calculate_global_limit(cache["xlim"])
        ylim_global = calculate_global_limit(cache["ylim"])

        for g in cache["g"]:
            g.ax_joint.set_xlim(xlim_global)
            g.ax_joint.set_ylim(ylim_global)

        for g, spath in zip(cache["g"], cache["spath"]):
            mngs.io.save(g, spath, from_cwd=True)


def calculate_global_limit(cache_list):
    min_val = min([x[0] for x in cache_list])
    max_val = max([x[1] for x in cache_list])
    return (min_val, max_val)


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
