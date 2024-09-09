#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-03 18:38:12 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_stats.py


"""
This script does XYZ.
"""

"""Imports"""
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import gaussian_kde

"""Warnings"""
mngs.pd.ignore_SettingWithCopyWarning()
# warnings.simplefilter("ignore", UserWarning)


"""Functions & Classes"""
PHASES_TO_PLOT = ["Encoding", "Retrieval"]


def run_pairwise_stats_test(df):
    _df = mngs.pd.merge_cols(df, "group", "match")

    df_stats = []
    for g1, g2 in combinations(_df.merged.unique(), 2):
        group1 = _df.loc[_df.merged == g1, "dist"]
        group2 = _df.loc[_df.merged == g2, "dist"]

        # Undersample to the size of the smaller group
        min_size = min(len(group1), len(group2))

        sampled_group1 = np.random.choice(group1, size=min_size, replace=False)
        sampled_group2 = np.random.choice(group2, size=min_size, replace=False)

        statistic, p_value = stats.wilcoxon(sampled_group1, sampled_group2)

        result = {
            "group_1": g1,
            "group_2": g2,
            "statistic": statistic,
            "p_val_unc": p_value,
        }

        df_stats.append(pd.Series(result))

    df_stats = pd.DataFrame(df_stats)
    df_stats["p_val"] = (df_stats["p_val_unc"] * len(df_stats)).clip(upper=1.0)
    df_stats["statistic"] = df_stats["statistic"].astype(int)
    df_stats["p_val_unc"] = df_stats["p_val_unc"].round(4)
    df_stats["p_val"] = df_stats["p_val"].round(4)

    # Organizing the df_stats
    pivot = df_stats.pivot(index="group_1", columns="group_2", values="p_val")
    pivot = pivot.combine_first(pivot.T)
    pivot = pivot.fillna(1.0)
    np.fill_diagonal(pivot.values, 1.0)  # Set diagonal to 1.0

    # Sort the pivot table
    columns = [
        "group-$g_E-NT_E$_match-1.0",
        "group-$g_R-NT_E$_match-1.0",
        "group-$g_E-NT_R$_match-1.0",
        "group-$g_R-NT_R$_match-1.0",
        "group-$g_E-NT_E$_match-2.0",
        "group-$g_R-NT_E$_match-2.0",
        "group-$g_E-NT_R$_match-2.0",
        "group-$g_R-NT_R$_match-2.0",
    ]
    pivot = pivot.reindex(index=columns, columns=columns)

    return pivot


def plot_p_values(pivot):

    # Create heatmap
    fig, ax = mngs.plt.subplots()
    ax.imshow2d(pivot, vmin=0, vmax=1.0, cmap="viridis_r")
    ax.set_xyt(None, None, "P-values for Group Comparisons")
    ax.set_ticks(
        xvals="auto",
        xticks=pivot.columns,
        yvals="auto",
        yticks=pivot.index,
    )
    return fig


def rename_phases(df):
    df["phase_combi"] = df["phase_combi"].replace(
        {
            "phase_g-Encoding": "$g_{E}$",
            "phase_g-Retrieval": "$g_{R}$",
            "_phase_nt-Encoding": "-$NT_{E}$",
            "_phase_nt-Retrieval": "-$NT_{R}$",
        },
        regex=True,
    )
    return df


def dist2rank(df):
    """Converts distance to rank data in each session."""

    def rank_session(group):
        group["dist"] = mngs.gen.to_rank(group["dist"])
        return group

    return df.groupby("global_session", group_keys=False).apply(rank_session)


def calc_kde(df, n_points=100):

    xmin = df.dist.min()
    xmax = df.dist.max()

    kde_values = []

    for ses in df["global_session"].unique():
        df_ses = df[df["global_session"] == ses]
        for match in CONFIG.MATCHES:
            df_ses_m = df_ses[df_ses.match == match]
            for phase in df_ses["phase_combi"].unique():
                data = df_ses_m[df_ses_m["phase_combi"] == phase]["dist"]
                kde = gaussian_kde(data)
                xx = np.linspace(xmin, xmax, n_points)
                yy = kde(xx)

                kde_values.append(
                    {
                        "global_session": ses,
                        "match": match,
                        "phase_combi": phase,
                        "xx": xx,
                        "yy": yy,
                    }
                )
    return pd.DataFrame(kde_values)


def calc_kde_mean_std(df):
    mngs.pd.merge_cols(df, "phase_combi", "match", name="phase_combi_match")
    df = (
        df.groupby(["phase_combi_match"])
        .agg(
            {
                "xx": [
                    lambda x: x.iloc[0]
                ],  # Take the first list as it's identical
                "yy": [
                    lambda x: np.mean(np.array(x.tolist()), axis=0),
                    lambda x: np.std(np.array(x.tolist()), axis=0),
                    lambda x: len(x),
                ],
            }
        )
        .reset_index()
    )

    # Rename columns
    df.columns = ["phase_combi_match", "xx", "yy_mean", "yy_std", "yy_count"]

    # ci
    df["yy_ci"] = 1.96 * (df["yy_std"] / np.sqrt(df["yy_count"]))

    return df


def parse_match(df):
    df["match"] = np.nan
    df.loc[mngs.gen.search("match-1", df["phase_combi_match"])[0], "match"] = 1
    df.loc[mngs.gen.search("match-2", df["phase_combi_match"])[0], "match"] = 2
    return df


def plot_kde(df):
    # Plotting
    fig, axes = mngs.plt.subplots(
        ncols=len(df.match.unique()), sharex=True, sharey=True
    )
    for _, row in df.iterrows():
        label = row.phase_combi_match
        ax = axes[0] if row.match == 1 else axes[1]
        ax.plot_with_ci(
            row.xx,
            row.yy_mean,
            row.yy_std,
            alpha=0.1,
            label=label,
            id=label,
        )
        ax.legend()
    return fig


def main():

    df = mngs.io.load("./data/CA1/dist.csv")

    mngs.pd.merge_cols(df, "sub", "session", name="global_session")
    mngs.pd.merge_cols(df, "phase_g", "phase_nt", name="phase_combi")

    df = rename_phases(df)
    df = dist2rank(df)

    df_stats = run_pairwise_stats_test(df)
    fig_pvals = plot_p_values(df_stats)

    df = calc_kde(df)

    df = calc_kde_mean_std(df)

    df = parse_match(df)

    df = mngs.pd.replace(
        df,
        {"phase_combi-": "", "_match-1": "", "_match-2": ""},
        cols=["phase_combi_match"],
    )

    fig_kde = plot_kde(df)

    # Saving
    SDIR = "./data/CA1/dist_rank_summary/"
    mngs.io.save(fig_pvals, SDIR + "pvals.jpg", from_cwd=True)
    mngs.io.save(
        mngs.pd.to_xyz(fig_pvals.to_sigma()), SDIR + "pvals.csv", from_cwd=True
    )
    # mngs.io.save(df_stats, SDIR + "pvals.csv", from_cwd=True)
    mngs.io.save(
        fig_kde,
        SDIR + "kde.jpg",
        from_cwd=True,
    )
    mngs.io.save(
        fig_kde.to_sigma(),
        SDIR + "kde.csv",
        from_cwd=True,
    )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        line_width=1.0,
        agg=True,
        np=np,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
