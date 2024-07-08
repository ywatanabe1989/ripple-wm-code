#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-09 04:39:03 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/custom_jointplot.py


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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import numpy as np


def custom_joint_plot(data, nrows, ncols, figsize=(15, 10)):
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
    )

    for i, ax in enumerate(axes.flat):
        if i >= len(data):
            ax.axis("off")  # Hide unused axes
            continue

        if not isinstance(data[i], pd.DataFrame):
            print(f"Error: Data at index {i} is not a pandas DataFrame.")
            continue

        # Create a divider for the existing axes
        divider = make_axes_locatable(ax)

        # Create axes for the marginal distributions
        ax_marg_x = divider.append_axes("top", size="20%", pad=0.02)
        ax_marg_x.set_xticks([])
        ax_marg_x.set_yticks([])

        ax_marg_y = divider.append_axes("right", size="20%", pad=0.02)
        ax_marg_y.set_xticks([])
        ax_marg_y.set_yticks([])

        # # Invert the y-axis marginal to align it properly with the main plot
        # ax_marg_y.invert_xaxis()

        data[i]["factor_2"] = np.log(data[i]["factor_2"] + 1e3)

        # Plot using seaborn
        sns.scatterplot(
            data=data[i],
            x="factor_1",
            y="factor_2",
            ax=ax,
            s=15,
            color="blue",
            alpha=0.6,
        )
        sns.kdeplot(
            data=data[i], x="factor_1", fill=True, ax=ax_marg_x, color="blue"
        )
        sns.kdeplot(
            data=data[i],
            x="factor_2",
            fill=True,
            ax=ax_marg_y,
            color="blue",
            vertical=True,
        )

        mngs.plt.ax.hide_spines(ax_marg_x)
        mngs.plt.ax.hide_spines(ax_marg_y)

        ax.set_xlim(
            data[i]["factor_1"].min(), data[i]["factor_1"].max()
        )  # Adjust x-limits to data range
        ax.set_ylim(
            data[i]["factor_2"].min(), data[i]["factor_2"].max()
        )  # Adjust y-limits to data range

    plt.tight_layout()
    plt.show()


# def custom_joint_plot(data, nrows, ncols, figsize=(15, 10)):
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

#     # Process each axis
#     for i, ax in enumerate(axes.flat):
#         if i >= len(data):  # If there's no more data, don't try to plot
#             ax.axis("off")
#             continue

#         if not isinstance(
#             data[i], pd.DataFrame
#         ):  # Check if the data is a DataFrame
#             print(f"Error: Data at index {i} is not a pandas DataFrame.")
#             continue

#         # Create a divider for the current axes
#         divider = make_axes_locatable(ax)

#         # Create the main plot axis to the rcenter
#         ax_main = divider.append_axes("bottom left", size="80%", pad=0.02)

#         # Create the marginal axes
#         ax_marg_x = divider.append_axes("top", size="20%", pad=0.02)
#         ax_marg_x.set_xticks([])  # Remove x-ticks as it's a marginal
#         ax_marg_x.set_yticks([])  # Remove y-ticks as it's a marginal

#         ax_marg_y = divider.append_axes("right", size="20%", pad=0.02)
#         ax_marg_y.set_xticks([])  # Remove x-ticks as it's a marginal
#         ax_marg_y.set_yticks([])  # Remove y-ticks as it's a marginal

#         # Plot using seaborn
#         sns.scatterplot(
#             data=data[i],
#             x="factor_1",
#             y="factor_2",
#             ax=ax_main,
#             s=15,
#             color="blue",
#             alpha=0.6,
#         )
#         sns.kdeplot(
#             data=data[i], x="factor_1", fill=True, ax=ax_marg_x, color="blue"
#         )
#         sns.kdeplot(
#             data=data[i],
#             x="factor_2",
#             fill=True,
#             ax=ax_marg_y,
#             color="blue",
#             vertical=True,
#         )

#         ax.axis("off")  # Hide the original axis

#     plt.tight_layout()
#     plt.show()


def main():
    # Generate example data for demonstration
    data = [
        pd.DataFrame(np.random.randn(300, 2), columns=["factor_1", "factor_2"])
        for _ in range(8)
    ]

    # Create the custom plot
    custom_joint_plot(data, nrows=2, ncols=4)

    pass


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
