#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-05 16:16:35 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/define_putative_CA1_using_UMAP.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from scripts.utils import parse_lpath
from sklearn.metrics import silhouette_score
from tqdm import tqdm

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def calc_silhouette_score(lpath_ripple_p):
    # SWR+
    pp = mngs.io.load(lpath_ripple_p)
    X_pp = np.vstack(pp.firing_pattern)
    T_pp = np.ones(len(pp))  # ["SWR+" for _ in range(len(pp))]

    # SWR-
    lpath_ripple_m = lpath_ripple_p.replace("SWR_p", "SWR_m")
    mm = mngs.io.load(lpath_ripple_m)
    X_mm = np.vstack(mm.firing_pattern)
    T_mm = np.zeros(len(mm))

    # When firing patterns are not available
    if X_pp.size == X_mm.size == 0:
        return np.nan

    assert len(pp) == len(mm)

    # UMAP clustering
    fig, legend_figs, _umap = mngs.ml.clustering.umap(
        [np.vstack([X_pp, X_mm])],
        [np.hstack([T_pp, T_mm])],
        supervised=True,
    )
    plt.close()
    U_pp = _umap.transform(X_pp)
    U_mm = _umap.transform(X_mm)

    # Silhouette score
    sil_score = silhouette_score(
        np.vstack([U_pp, U_mm]), np.hstack([T_pp, T_mm])
    )
    sil_score = round(sil_score, 3)

    return sil_score


def sort_df(df):
    df = df.drop(columns=["lpath_ripple"])

    # Calculate count, mean, and standard deviation for 'session' and 'silhouette_score'
    df = (
        df.groupby(["sub", "roi"])
        .agg(
            silhouette_score_mean=("silhouette_score", "mean"),
            silhouette_score_std=("silhouette_score", "std"),
            session_count=("session", "count"),
        )
        .round(3)
        .reset_index()
        .set_index("sub")
    )

    # Format the string as "mean ± std (n = count)"
    df["formatted"] = df.apply(
        lambda row: (
            f"{row['silhouette_score_mean']:.3f} ± {row['silhouette_score_std']:.3f} "
            f"(n = {row['session_count']})"
            if pd.notnull(row["silhouette_score_mean"])
            else "NaN"
        ),
        axis=1,
    )

    # Sort the DataFrame for the sorted_table output
    df_sorted = (
        df.reset_index()
        .sort_values(["silhouette_score_mean", "sub"], ascending=False)
        .set_index("sub")[["roi", "formatted"]]
    )

    # Pivot the DataFrame to get the desired format
    df_pivot = (
        df.reset_index()
        .pivot(index="sub", columns="roi", values="formatted")
        .fillna("NaN")
    )

    return df_pivot, df_sorted


def main():
    for roi in tqdm(CONFIG["ROIS"].keys()):
        out = mngs.gen.listed_dict()
        for lpath_ripple_p in tqdm(mngs.gen.natglob(CONFIG["PATH_RIPPLE"])):
            parsed = parse_lpath(lpath_ripple_p)

            if not parsed["roi"] in CONFIG["ROIS"][roi]:
                continue

            sil_score = calc_silhouette_score(lpath_ripple_p)

            out["sub"].append(parsed["sub"])
            out["session"].append(parsed["session"])
            out["roi"].append(parsed["roi"])
            out["silhouette_score"].append(sil_score)
            out["lpath_ripple"].append(lpath_ripple_p)

        df = pd.DataFrame(out)
        df_pivot, df_sorted = sort_df(df)

        # Printing
        print(df)
        print(df_pivot)
        print(df_sorted)

        # Saving
        for var, sname in (
            (df, "raw"),
            (df_pivot, "table"),
            (df_sorted, "sorted_table"),
        ):
            mngs.io.save(
                var,
                f"./data/silhouette_scores/{roi}/{sname}.csv",
                from_cwd=True,
            )

        # mngs.io.save(
        #     df, f"./data/silhouette_scores/{roi}/raw.csv", from_cwd=True
        # )
        # mngs.io.save(
        #     df_pivot,
        #     f"./data/silhouette_scores/{roi}/table.csv",
        #     from_cwd=True,
        # )
        # mngs.io.save(
        #     df_sorted,
        #     f"./data/silhouette_scores/{roi}/sorted_table.csv",
        #     from_cwd=True,
        # )


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
