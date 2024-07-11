#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-09 06:56:26 (ywatanabe)"
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
from scipy.stats import gaussian_kde


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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import skewnorm, lognorm


def generate_log_norm_skewed_biased_data(size, skewness, bias, mean, sigma):
    # Generate log-normal data
    data = lognorm.rvs(s=sigma, scale=np.exp(mean), size=size)
    # Apply skewness using skewnorm (adjusting the data distribution)
    skewed_data = skewnorm.rvs(a=skewness, loc=0, scale=1, size=size)
    skewed_scaled_data = skewed_data * (
        data.std() / skewed_data.std()
    )  # Scale to the log-normal std deviation
    final_data = (
        data + skewed_scaled_data
    )  # Combine the log-normal and skewed data
    return final_data + bias  # Add bias


def custom_joint_plot(data, nrows, ncols, figsize=(15, 10)):
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
    )

    for ax in axes.flat:
        ax.set_box_aspect(1)

    # # Drawing the canvas to get actual sizes
    # fig.canvas.draw()

    # Calculate global KDE maxima for consistent scale in density plots
    max_density_x = 0
    max_density_y = 0
    for df in data:
        kde_x = gaussian_kde(df["factor_1"])
        kde_y = gaussian_kde(df["factor_2"])
        x_values = np.linspace(
            df["factor_1"].min(), df["factor_1"].max(), 1000
        )
        y_values = np.linspace(
            df["factor_2"].min(), df["factor_2"].max(), 1000
        )
        max_density_x = max(max_density_x, max(kde_x(x_values)))
        max_density_y = max(max_density_y, max(kde_y(y_values)))

    # Global max density and override them for compatibility
    max_density = max(max_density_x, max_density_y)
    max_density_x = max_density_y = max_density

    for i, ax in enumerate(axes.flat):
        if i >= len(data):
            ax.axis("off")
            continue

        divider = make_axes_locatable(ax)

        ax_marg_x = divider.append_axes("top", size="20%", pad=0.1)
        ax_marg_x.set_box_aspect(0.2)  # Aspect ratio = 20% of the main axes

        ax_marg_y = divider.append_axes("right", size="20%", pad=0.1)
        ax_marg_y.set_box_aspect(
            0.2 ** (-1)
        )  # Inverse of 20% aspect to match the vertical stretch

        # dy = -7.5
        # mngs.plt.ax.shift(ax_marg_x, dy=-100)
        # mngs.plt.ax.shift(ax_marg_y, dy=dy)
        # mngs.plt.ax.shift(ax, dy=dy)

        sns.scatterplot(
            data=data[i],
            x="factor_1",
            y="factor_2",
            ax=ax,
            s=15,
            color="blue",
            alpha=0.6,
        )
        kdeplot_x = sns.kdeplot(
            data=data[i],
            x="factor_1",
            fill=True,
            ax=ax_marg_x,
            color="blue",
            # bw_adjust=0.5,
            common_norm=True,
        )
        kdeplot_y = sns.kdeplot(
            data=data[i],
            x="factor_2",
            fill=True,
            ax=ax_marg_y,
            color="blue",
            # bw_adjust=0.5,
            vertical=True,
            common_norm=True,
        )

        ax_marg_x.set_xlim(ax.get_xlim())
        ax_marg_y.set_ylim(ax.get_ylim())

        # Set the same density limits for all marginal plots
        ax_marg_x.set_ylim(0, max_density_x * 1.25)
        ax_marg_y.set_xlim(0, max_density_y * 1.25)

        # Hide spines
        mngs.plt.ax.hide_spines(ax_marg_x, bottom=False)
        mngs.plt.ax.hide_spines(ax_marg_y, left=False)

        # Hide ticks
        for ax_marg in [ax_marg_x, ax_marg_y]:
            ax_marg.set_xticks([])
            ax_marg.set_yticks([])
            ax_marg.set_xlabel(None)
            ax_marg.set_ylabel(None)

    plt.tight_layout()
    # plt.show()
    mngs.io.save(fig, "fig.jpg")


def main():
    # Generate skewed, biased, log-normally distributed example data for visualization
    data = [
        pd.DataFrame(
            {
                "factor_1": generate_log_norm_skewed_biased_data(
                    300,
                    skewness=np.random.randn(),
                    bias=np.random.randn(),
                    mean=0,  # np.random.randn(),
                    sigma=2,  # abs(np.random.randn()),
                ),
                "factor_2": generate_log_norm_skewed_biased_data(
                    300,
                    skewness=np.random.randn(),
                    bias=np.random.randn(),
                    mean=0,  # np.random.randn(),
                    sigma=2,  # abs(np.random.randn()),
                ),
            }
        )
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
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
