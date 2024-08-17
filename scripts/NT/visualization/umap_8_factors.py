#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-14 17:38:46 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/kde.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import mngs
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import utils
from itertools import product
import logging

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


def clustering(
    dfs,
    target,
    supervised,
    # shuffle_target=False,
    axes=None,
):
    # Formatting Data
    data_all = []
    labels_all = []
    hues_all = []
    hues_colors_all = []
    for sample_type, df in dfs.items():
        # Copy df
        df = df.copy()

        # Slicing according to conditions
        indi = np.ones(len(df)).astype(bool)
        # indi = df.match == cc["match"]
        if "SWR" in sample_type:
            indi *= df.within_ripple
        df = df[indi]

        # Data
        cols_factors = mngs.gen.search("factor_", df.columns)[1]
        data = df[cols_factors]

        # # Target
        # if shuffle_target:
        #     rand_indi = np.random.permutation(np.arange(len(df)))
        #     df[target] = np.array(df[target].iloc[rand_indi])
        #     df.color = np.array(df.color.iloc[rand_indi])
        targets = df[target]

        # Hues
        hues = targets.to_list()
        colors = df["color"].tolist()

        if target == "set_size":
            __import__("ipdb").set_trace()

        # Caching
        data_all.append(data)
        labels_all.append(targets)
        hues_all.append(hues)
        hues_colors_all.append(colors)

    # Clustering and Plot
    fig, figs_legend, _umap = mngs.ml.clustering.umap(
        data=data_all,
        labels=labels_all,
        hues=hues_all,
        hues_colors=hues_colors_all,
        supervised=supervised,
        alpha=0.5,
        axes=axes,
        use_independent_legend=True,
        s=1,
        axes_titles=list(dfs.keys()),
    )

    fig.supxyt(t=f"{target}")

    return fig, figs_legend, _umap


def main():
    znorm = symlog = unbias = shuffle_target = False

    # condi = {
    #     # "match": CONFIG.MATCHES,
    #     "target": ["phase", "phase_set_size", "set_size"],
    #     "supervised": [True, False],
    #     "sample_type": ["all", "SWR+", "SWR-"],
    # }

    sample_types = ["all", "SWR+", "SWR-"]
    targets = ["phase", "phase_set_size", "set_size"]
    supervised_list = [True, False]

    # n_panels = mngs.gen.count_grids(condi)
    # sample_types = condi.pop("sample_type")

    for ca1_region in CONFIG.ROI.CA1:
        lpath_NT = mngs.gen.replace(CONFIG.PATH.NT, ca1_region)
        dfs = {
            sample_type: utils.load_NTdf(
                lpath_NT,
                sample_type,
                znorm=znorm,
                symlog=symlog,
                unbias=unbias,
            )
            for sample_type in sample_types
        }

        dfs = {
            k: mngs.pd.merge_columns(v, "phase", "set_size")
            for k, v in dfs.items()
        }

        for _, target in enumerate(targets):
            i_ax = 0
            fig, axes = mngs.plt.subplots(
                ncols=len(sample_types), nrows=len(supervised_list)
            )

            for _, supervised in enumerate(supervised_list):

                axes_sample_types = axes.flat[i_ax : i_ax + len(sample_types)]

                # Supervised on the all samples with the specified target
                # SWR+ and SWR- samples are plotted in the same embedding space
                fig, figs_legend, _umap = clustering(
                    dfs,
                    target,
                    supervised,
                    axes=axes_sample_types,
                )

                # Title
                axes_sample_types[0].set_title(f"Supervised: {supervised}")
                plt.tight_layout()
                mngs.plt.ax.sharex(axes_sample_types)
                mngs.plt.ax.sharey(axes_sample_types)

                i_ax += len(sample_types)

            for ax in axes.flat:
                ax._legend = None

            # Saving
            spath_fig = construct_spath(
                lpath_NT,
                symlog,
                znorm,
                unbias,
                target,
                shuffle_target,
                supervised,
            )
            mngs.io.save(fig, spath_fig, from_cwd=True)

            plt.close("all")


def construct_spath(
    lpath_NT,
    symlog,
    znorm,
    unbias,
    target,
    shuffle_target,
    supervised,
):
    parsed = utils.parse_lpath(lpath_NT)

    scale = "linear" if not symlog else "symlog"
    znorm_str = "NT" if not znorm else "NT_z"
    unbias_str = "unbiased" if unbias else "orig"
    is_shuffle_tgt = "target_shuffled" if shuffle_target else "target_orig"
    supervised_str = "supervised" if supervised else "unsupervised"
    spath_fig = (
        f"./data/CA1/umap/{znorm_str}-{scale}-{unbias_str}/{target}/{is_shuffle_tgt}/{supervised_str}/"
        + "_".join("-".join(item) for item in parsed.items())
        + ".jpg"
    )
    return spath_fig


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        # agg=True,
        alpha=1.0,
        fig_scale=10,
        font_size_legend=6,
        font_size_title=6,
    )

    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF

# rm ./data/CA1/umap -rf
# /mnt/ssd/ripple-wm-code/data/CA1/umap/NT-linear-orig/phase/target_orig/unsupervised/sub-01_session-01_roi-AHL.jpg
