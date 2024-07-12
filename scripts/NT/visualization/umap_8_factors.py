#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-12 02:04:40 (ywatanabe)"
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
    lpath_NT,
    sample_type,
    znorm=False,
    symlog=False,
    unbias=False,
    shuffle_label=False,
    supervised=False,
):
    df = utils.load_NTdf(
        lpath_NT, sample_type, znorm=znorm, symlog=symlog, unbias=unbias
    )

    labels = (
        df["phase"]
        if not shuffle_label
        else np.random.permutation(df["phase"])
    )

    # df.color = df.color.astype(str)
    # palette = df[["phase", "color"]].drop_duplicates()

    fig, legend_fig, _umap = mngs.ml.clustering.umap(
        data_all=[df[mngs.gen.search("factor_", df.columns)[1]]],
        labels_all=[labels],
        supervised=supervised,
        palette=[
            CC[CONFIG.PHASES[phase_str].color] for phase_str in CONFIG.PHASES
        ],
        alpha=1.0,
    )
    return fig


def main():
    for znorm, symlog, unbias, shuffle_label, supervised in product(
        [False],
        [False],
        [False],
        [True, False],
        [True, False],
    ):

        for ca1_region in CONFIG.ROI.CA1:
            lpath_NT = mngs.gen.replace(CONFIG.PATH.NT, ca1_region)
            parsed = utils.parse_lpath(lpath_NT)

            if parsed not in CONFIG.ROI.CA1:
                continue

            cache = mngs.gen.listed_dict()
            for sample_type in ["SWR+", "SWR-", "all"]:

                fig = clustering(
                    lpath_NT,
                    znorm=znorm,
                    symlog=symlog,
                    unbias=unbias,
                    sample_type=sample_type,
                    shuffle_label=shuffle_label,
                    supervised=supervised,
                )

                # Saving
                scale = "linear" if not symlog else "symlog"
                znorm_str = "NT" if not znorm else "NT_z"
                unbias_str = "unbiased" if unbias else "orig"
                label_str = "label_shuffled" if shuffle_label else "label_orig"
                supervised_str = "supervised" if supervised else "unsupervised"

                spath_fig = (
                    f"./data/CA1/umap/{znorm_str}-{scale}-{unbias_str}-{sample_type}-{label_str}-{supervised_str}/"
                    + "_".join("-".join(item) for item in parsed.items())
                    + ".jpg"
                )
                # fig.supxyt("Factor value", "KDE density", spath_fig)

                # mngs.io.save(
                #     axes.to_sigma(),
                #     spath_fig.replace(".jpg", ".csv"),
                #     from_cwd=True,
                # )
                mngs.io.save(fig, spath_fig, from_cwd=True)


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
        agg=True,
        alpha=1.0,
        fig_scale=2,
        font_size_legend=6,
    )

    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
