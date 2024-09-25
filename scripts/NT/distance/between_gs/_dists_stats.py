#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-26 07:41:49 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_stats.py


"""This script does XYZ."""

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


def run_pairwise_stats_test(df):
    _df = mngs.pd.merge_cols(df, "group", "match")

    df_stats = []
    for g1, g2 in combinations(_df.merged.unique(), 2):
        group1 = _df.loc[_df.merged == g1, "dist_rank"]
        group2 = _df.loc[_df.merged == g2, "dist_rank"]

        group1 = group1[~group1.isna()]
        group2 = group2[~group2.isna()]

        # Undersample to the size of the smaller group for the requirements of Wilcoxon test
        min_size = min(len(group1), len(group2))

        sampled_group1 = np.random.choice(group1, size=min_size, replace=False)
        sampled_group2 = np.random.choice(group2, size=min_size, replace=False)

        statistic, p_value = stats.wilcoxon(sampled_group1, sampled_group2)

        result = {
            "group_1": g1,
            "group_2": g2,
            "statistic": statistic,
            "p_val_unc": p_value,
            "n": min_size,
        }

        df_stats.append(pd.Series(result))

    df_stats = pd.DataFrame(df_stats)

    # Bonferroni correction
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

    # Renaming
    pivot.columns = [
        col.replace("group-$", "").replace("$_", "_").replace(".0", "")
        for col in pivot.columns
    ]
    pivot.index = [
        ii.replace("group-$", "").replace("$_", "_").replace(".0", "")
        for ii in pivot.index
    ]

    return pivot


def plot_p_values(df_pvals, match):
    if match != "all":
        indi_match = mngs.gen.search(f"match-{match}", df_pvals.index)[1]
        cols_match = mngs.gen.search(f"match-{match}", df_pvals.columns)[1]
        df_pvals_match = df_pvals.loc[indi_match, cols_match]
    else:
        df_pvals_match = df_pvals

    # Create heatmap
    fig, ax = mngs.plt.subplots()
    ax.imshow2d(df_pvals_match, vmin=0, vmax=1.0, cmap="viridis_r")
    ax = mngs.plt.ax.rotate_labels(ax, x=30, y=30)
    ax.set_xyt(None, None, "P-values for Group Comparisons")
    ax.set_ticks(
        xvals="auto",
        xticks=df_pvals_match.columns,
        yvals="auto",
        yticks=df_pvals_match.index,
    )
    fig.tight_layout()
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


def dist2rank_by_session(df):
    """Converts distance to rank data in each session."""
    from scipy.stats import rankdata

    def rank_session(group):
        group["dist_rank"] = rankdata(group["dist"], method='average')
        return group

    rank_data = df.groupby("global_session", group_keys=False).apply(
        rank_session
    )

    return rank_data


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
        ax.plot_(
            xx=row.xx,
            mean=row.yy_mean,
            ci=row.yy_ci,
            alpha=0.1,
            label=label,
            id=label,
        )
        ax.legend(loc="lower center")
    return fig


def main(phases_to_plot):

    # Loading
    df = mngs.io.load(
        f"./scripts/NT/distance/between_gs/calc_dists/{'_'.join(phases_to_plot)}/dist_ca1.csv"
    )

    # Preparation
    mngs.pd.merge_cols(df, "sub", "session", name="global_session")
    mngs.pd.merge_cols(df, "phase_g", "phase_nt", name="phase_combi")
    # indi = (mngs.gen.search(phases_to_plot, df.phase_g, as_bool=True))[0] * (
    #     mngs.gen.search(phases_to_plot, df.phase_nt, as_bool=True)
    # )[0]
    df = rename_phases(df)

    # To rank
    df = dist2rank_by_session(df)

    # Wilcoxon test
    df_stats = run_pairwise_stats_test(df)
    fig_pvals_all = plot_p_values(df_stats, match="all")
    fig_pvals_in = plot_p_values(df_stats, match=1)
    fig_pvals_out = plot_p_values(df_stats, match=2)

    # KDE
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
    sdir = f"./{'_'.join(phases_to_plot)}/"
    for obj, spath in [
        (fig_pvals_all, sdir + "pvals_match_all.jpg"),
        (fig_pvals_in, sdir + "pvals_match_in.jpg"),
        (fig_pvals_out, sdir + "pvals_match_out.jpg"),
    ]:
        mngs.io.save(obj, spath)
        mngs.io.save(
            mngs.pd.to_xyz(obj.to_sigma()), spath.replace(".jpg", ".csv")
        )

    mngs.io.save(
        fig_kde,
        sdir + "kde.jpg",
        from_cwd=False,
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
    for phases_to_plot in [
        ["Encoding", "Retrieval"],
        ["Fixation", "Encoding", "Maintenance", "Retrieval"],
    ]:
        main(phases_to_plot)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
