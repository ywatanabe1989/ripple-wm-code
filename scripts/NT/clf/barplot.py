#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-02 10:22:07 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/clf/barplot.py

"""
1. Functionality:
   - Generates bar plots for classifier performance metrics
2. Input:
   - CSV file containing classifier performance data
3. Output:
   - Bar plot image file (bar.jpg)
4. Prerequisites:
   - Python 3.x, numpy, pandas, matplotlib, mngs package
"""

"""Imports"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import mngs

"""Configs"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def format_df(df):
    """
    Formats DataFrame for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with classifier performance metrics

    Returns
    -------
    tuple
        labels, mean values, and error values for plotting
    """
    cols = [
        f"match-{match}.0_set_size-{set_size}.0"
        for match in CONFIG.MATCHES
        for set_size in CONFIG.SET_SIZES
        for classifier in df.classifier.unique()
    ]
    df = df[df["condition"].isin(cols)].sort_values(["condition"])
    mngs.pd.merge_cols(df, "classifier", "condition", name="label")
    labels = np.array(df.label)
    mean_values = np.array(df["bACC_mean-mean"])
    ci = np.array([eval(_) for _ in df["bACC_mean-agg_ci"]])
    ci_low, ci_high = ci[:, 0], ci[:, 1]
    error_values = np.array([mean_values - ci_low, ci_high - mean_values])[
        0, :
    ]
    return labels, mean_values, error_values


def main():
    df = mngs.io.load(
        "scripts/NT/clf/linearSVC/Encoding_Retrieval/metrics_all.csv",
        index_col=[0, 1],
    ).reset_index()

    labels, mean_values, error_values = format_df(df)

    fig, axes = mngs.plt.subplots(ncols=len(CONFIG.MATCHES))
    for label, mean_value, error_value in zip(
        labels, mean_values, error_values
    ):
        ax = axes[0] if f"match-1.0" in label else axes[1]

        ax.bar(
            label,
            mean_value,
            yerr=error_value,
            capsize=5,
            label=label,
            id=label,
        )

    mngs.io.save(fig, "bar.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-02 08:00:41 (ywatanabe)"
# # /mnt/ssd/ripple-wm-code/scripts/NT/clf/barplot.py

# """
# 1. Functionality:
#    - (e.g., Executes XYZ operation)
# 2. Input:
#    - (e.g., Required data for XYZ)
# 3. Output:
#    - (e.g., Results of XYZ operation)
# 4. Prerequisites:
#    - (e.g., Necessary dependencies for XYZ)

# (Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
# """

# """Imports"""
# import os
# import re
# import sys

# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import importlib

# import mngs

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from icecream import ic
# from natsort import natsorted
# from glob import glob
# from pprint import pprint
# import warnings
# import logging
# from tqdm import tqdm
# import xarray as xr

# try:
#     from scripts import utils
# except:
#     pass

# """Config"""
# CONFIG = mngs.gen.load_configs()

# """Parameters"""

# """Functions & Classes"""

# def format_df(df):

#     cols = [
#         f"match-{match}.0_set_size-{set_size}.0"
#         for match in CONFIG.MATCHES
#         for set_size in CONFIG.SET_SIZES
#         for classifier in df.classifier.unique()
#     ]
#     df = df[df["condition"].isin(cols)]
#     df = df.sort_values(["condition"])
#     mngs.pd.merge_cols(df, "classifier", "condition", name="label")
#     xx = np.array(df.label)
#     yy = np.array(df['bACC_mean-mean'])
#     ci = np.array(df['bACC_mean-agg_ci'])
#     ci = np.array([eval(_) for _ in ci])
#     ci_low, ci_high = ci[:,0], ci[:,1]
#     yerr = np.array([yy - ci_low, ci_high - yy])
#     assert np.allclose(yerr[0,:], yerr[1,:], rtol=1e-5, atol=1e-8)
#     yerr = yerr[0,:]

#     return xx, yy, yerr

# def main():
#     df = mngs.io.load(
#         "scripts/NT/clf/linearSVC/Encoding_Retrieval/metrics_all.csv",
#         index_col=[0, 1],
#     ).reset_index()

#     # Extract necessary information
#     xx, yy, yerr = format_df(df)

#     # Plotting
#     fig, axes = mngs.plt.subplots(ncols=len(CONFIG.MATCHES))
#     for _xx, _yy, _yerr, match in zip(xx, yy, yerr, CONFIG.MATCHES):
#         ax = axes[0] if match == 1 else axes[1]
#         ax.bar(_xx, _yy, yerr=_yerr, capsize=5, label=_xx, id=_xx)

#     df = fig.to_sigma()
#     mngs.io.save(fig, "bar.jpg")


# if __name__ == "__main__":
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False, agg=True
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)

# # EOF
