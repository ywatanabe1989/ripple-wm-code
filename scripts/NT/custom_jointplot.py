#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-09 05:01:47 (ywatanabe)"
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    ax_marg_x_shared = (
        None  # Variable to hold the shared x-axis for top marginal plots
    )
    ax_marg_y_shared = (
        None  # Variable to hold the shared y-axis for right marginal plots
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

        if ax_marg_x_shared is None:
            # This will set the first ax_marg_x as the shared axis
            ax_marg_x = divider.append_axes("top", size="20%", pad=0.02)
            ax_marg_x_shared = ax_marg_x  # Set the shared x-axis
        else:
            # Use the previously created ax_marg_x for sharing the x-axis
            ax_marg_x = divider.append_axes(
                "top", size="20%", pad=0.02, sharex=ax_marg_x_shared
            )

        ax_marg_x.set_xticks([])
        ax_marg_x.set_yticks([])

        if ax_marg_y_shared is None:
            # This will set the first ax_marg_y as the shared axis
            ax_marg_y = divider.append_axes("right", size="20%", pad=0.02)
            ax_marg_y_shared = ax_marg_y  # Set the shared y-axis
        else:
            # Use the previously created ax_marg_y for sharing the y-axis
            ax_marg_y = divider.append_axes(
                "right", size="20%", pad=0.02, sharey=ax_marg_y_shared
            )

        ax_marg_y.set_xticks([])

        # # Invert the y-axis marginal to align it properly with the main plot
        # ax_marg_y.invert_xaxis()

        mngs.plt.ax.hide_spines(ax_marg_x)
        mngs.plt.ax.hide_spines(ax_marg_y)

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
            data=data[i],
            x="factor_1",
            fill=True,
            ax=ax_marg_x,
            color="blue",
            common_norm=True,
        )
        sns.kdeplot(
            data=data[i],
            x="factor_2",
            fill=True,
            ax=ax_marg_y,
            color="blue",
            common_norm=True,
            vertical=True,
        )

    plt.tight_layout()
    plt.show()


def main():
    # Generate skewed, biased, log-normally distributed example data for visualization
    data = [
        pd.DataFrame(
            {
                "factor_1": generate_log_norm_skewed_biased_data(
                    300,
                    skewness=np.random.randn(),
                    bias=np.random.randn(),
                    mean=np.random.randn(),
                    sigma=abs(np.random.randn()),
                ),
                "factor_2": generate_log_norm_skewed_biased_data(
                    300,
                    skewness=np.random.randn(),
                    bias=np.random.randn(),
                    mean=np.random.randn(),
                    sigma=abs(np.random.randn()),
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
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
