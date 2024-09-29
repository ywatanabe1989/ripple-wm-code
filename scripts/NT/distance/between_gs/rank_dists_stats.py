#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-29 15:15:27 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/TDA/n_samples_stats.py


"""This script does XYZ."""


"""Imports"""
import sys
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from itertools import combinations, product

import scipy.stats as stats
from scipy.stats import rankdata
import joypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests
import mngs.stats

mngs.pd.ignore_SettingWithCopyWarning()

"""Functions & Classes"""

ORDER = [
    "NT_E-g_E",
    "NT_E-g_R",
    "NT_R-g_E",
    "NT_R-g_R",
]

def run_stats_test(df):
    """
    Performs pairwise statistical tests on groups in the dataframe.

    Example
    -------
    df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'dist': [1, 2, 3, 4]})
    results = perform_pairwise_statistical_test(df)
    print(results)

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing 'group' and 'dist' columns

    Returns
    -------
    pandas.DataFrame
        Results of pairwise statistical tests
    """
    results = []
    for col1, col2 in product(df.group.unique(), df.group.unique()):
        x1 = df.loc[df.group == col1, "dist"]
        x2 = df.loc[df.group == col2, "dist"]

        if np.all(np.array(x1) == np.array(x2)):
            statistic_wc, p_value_wc = np.nan, 1.0
            statistic_bm, p_value_bm = np.nan, 1.0
            statistic_ks, p_value_ks = np.nan, 1.0
        else:
            statistic_wc, p_value_wc = stats.wilcoxon(x1, x2)
            bm_results = mngs.stats.brunner_munzel_test(x1, x2)
            statistic_bm, p_value_bm = bm_results["w_statistic"], bm_results["p_value"]
            statistic_ks, p_value_ks = stats.ks_2samp(x1, x2)

        result = {
            "col1": col1,
            "col2": col2,
            "n1": len(x1),
            "n2": len(x2),
            "statistic_wc": statistic_wc,
            "p_val_unc_wc": p_value_wc,
            "statistic_bm": statistic_bm,
            "p_val_unc_bm": p_value_bm,
            "statistic_ks": statistic_ks,
            "p_val_unc_ks": p_value_ks,
        }
        results.append(pd.Series(result))

    results = pd.DataFrame(results)

    for i_col, col in enumerate(results.columns):
        if "p_val_unc" in col:
            # Bonferroni correction
            bonf_corrected = (results[col] * len(results)).clip(upper=1.0)
            col_corrected = f"{col}".replace("_unc", "_bonf")
            results[col_corrected] = bonf_corrected
            mngs.pd.mv(results, col_corrected, i_col+1)

            # Benjamini-Hochberg FDR
            _, fdr_corrected = fdrcorrection(results[col])
            col_corrected = f"{col}".replace("_unc", "_fdr")
            results[col_corrected] = fdr_corrected
            mngs.pd.mv(results, col_corrected, i_col+2)

            # Holm-Bonferroni
            _, holm_corrected, _, _ = multipletests(results[col], method='holm')
            col_corrected = f"{col}".replace("_unc", "_holm")
            results[col_corrected] = holm_corrected
            mngs.pd.mv(results, col_corrected, i_col+3)

    # Round all float columns
    for col in results.columns:
        if results[col].dtype == float:
            results[col] = results[col].round(3)

    return results


def plot_kde(df):
    fig, ax = mngs.plt.subplots()
    # hue_order = [
    #     "NT_E-g_E",
    #     "NT_E-g_R",
    #     "NT_R-g_E",
    #     "NT_R-g_R",
    # ]
    # hue_colors = {
    #     "NT_E-g_E": CC["blue"],
    #     "NT_E-g_R": CC["light_blue"],
    #     "NT_R-g_E": CC["pink"],
    #     "NT_R-g_R": CC["red"],
    # }

    ax.sns_kdeplot(
        data=df,
        x="dist",
        hue="group",
        hue_order=ORDER,
        hue_colors={k: CC[CONFIG.COLORS[k]] for k in ORDER},
        # hue_order=hue_order,
        # hue_colors=hue_colors,
        xlim=(df.dist.min(), df.dist.max()),
        cumulative=False,
        id="_".join(phases_to_plot),
    )
    ax.legend()
    return fig

def plot_box(df):
    fig, ax = mngs.plt.subplots()
    # hue_order = [
    #     "NT_E-g_E",
    #     "NT_E-g_R",
    #     "NT_R-g_E",
    #     "NT_R-g_R",
    # ]
    # hue_colors = {
    #     "NT_E-g_E": CC["blue"],
    #     "NT_E-g_R": CC["light_blue"],
    #     "NT_R-g_E": CC["pink"],
    #     "NT_R-g_R": CC["red"],
    # }

    ax.sns_boxplot(
        data=df,
        y="dist",
        # hue="group",
        # hue_order=hue_order,
        # hue_colors=hue_colors,
        x="group",
        hue_order=ORDER,
        hue_colors={k: CC[CONFIG.COLORS[k]] for k in ORDER},
        # order=hue_order,
        # palette=hue_colors,
        # strip=True,
        id="_".join(phases_to_plot),
    )
    ax.legend()
    return fig

# def plot_hist(df):
#     fig, ax = mngs.plt.subplots()
#     hue_order = [
#         "NT_E-g_E",
#         "NT_E-g_R",
#         "NT_R-g_E",
#         "NT_R-g_R",
#     ]
#     hue_colors = {
#         "NT_E-g_E": CC["blue"],
#         "NT_E-g_R": CC["light_blue"],
#         "NT_R-g_E": CC["pink"],
#         "NT_R-g_R": CC["red"],
#     }

#     ax.sns_histplot(
#         data=df,
#         y="dist",
#         hue="group",
#         hue_order=hue_order,
#         hue_colors=hue_colors,
#         id="_".join(phases_to_plot),
#     )
#     ax.legend()
#     return fig

# def plot_violin(df):
#     fig, ax = mngs.plt.subplots()
#     hue_order = [
#         "NT_E-g_E",
#         "NT_E-g_E_fake",
#         "NT_E-g_R",
#         "NT_E-g_R_fake",
#         "NT_R-g_E",
#         "NT_R-g_E_fake",
#         "NT_R-g_R",
#         "NT_R-g_R_fake",
#     ]
#     transparent_color = (1.,1.,1.,0.)
#     hue_colors = {
#         "NT_E-g_E": CC["blue"],
#         "NT_E-g_E_fake": transparent_color,
#         "NT_E-g_R": CC["light_blue"],
#         "NT_E-g_R_fake": transparent_color,
#         "NT_R-g_E": CC["pink"],
#         "NT_R-g_E_fake": transparent_color,
#         "NT_R-g_R": CC["red"],
#         "NT_R-g_R_fake": transparent_color,
#     }

#     df_fake = df.copy()
#     df_fake["dist"] = np.nan
#     df_fake['group'] = df_fake['group'] + "_fake"
#     combined_df = pd.concat([df, df_fake])

#     ax.sns_violinplot(
#         data=combined_df,
#         y="dist",
#         hue="group",
#         hue_order=hue_order,
#         hue_colors=hue_colors,
#         palette=hue_colors,
#         split=True,
#         # inner="quart",
#         id="_".join(phases_to_plot),
#     )
#     ax.legend()
#     return fig

def plot_violin(df):
    fig, ax = mngs.plt.subplots()
    # hue_order = [
    #     "NT_E-g_E",
    #     "NT_E-g_R",
    #     "NT_R-g_E",
    #     "NT_R-g_R",
    # ]
    # hue_colors = {
    #     "NT_E-g_E": CC["blue"],
    #     "NT_E-g_R": CC["light_blue"],
    #     "NT_R-g_E": CC["pink"],
    #     "NT_R-g_R": CC["red"],
    # }

    ax.sns_violinplot(
        data=df,
        y="dist",
        hue="group",
        hue_order=ORDER,
        hue_colors={k: CC[CONFIG.COLORS[k]] for k in ORDER},
        palette={k: CC[CONFIG.COLORS[k]] for k in ORDER},
        # hue_order=hue_order,
        # hue_colors=hue_colors,
        # palette=hue_colors,
        split=True,
        inner="quart",
        id="_".join(phases_to_plot),
    )
    ax.legend()
    return fig


def plot_joy(df):
    # group_order = ["NT_E-g_E", "NT_E-g_R", "NT_R-g_E", "NT_R-g_R"]
    # color_map = {
    #     "NT_E-g_E": CC["blue"],
    #     "NT_E-g_R": CC["light_blue"],
    #     "NT_R-g_E": CC["pink"],
    #     "NT_R-g_R": CC["red"]
    # }
    group_order = ORDER
    color_map = {k: CC[CONFIG.COLORS[k]] for k in ORDER}
    colors = [color_map[group] for group in group_order]

    fig, axes = joypy.joyplot(
        data=df,
        by="group",
        column="dist",
        # color=colors,
        title="_".join(phases_to_plot),
        # labels_color="black",
        overlap=0.1,
        # order=group_order
        # x_range=(0, 2000),
        hist=True,
    )

    plt.xlabel("Distance")
    plt.ylabel("Group")
    return fig

def plot_heatmap(stats, z):
    vmin = 0 if "p_val" in z else np.nanmin(stats[z])
    vmax = 1 if "p_val" in z else np.nanmax(stats[z])
    cmap = "viridis_r" if "p_val" in z else "viridis"

    # Heatmap data
    hm = mngs.pd.from_xyz(stats, x="col1", y="col2", z=z)

    # Sorting
    # order = [
    #     "NT_E-g_E",
    #     "NT_E-g_R",
    #     "NT_R-g_E",
    #     "NT_R-g_R",
    # ]
    hm = hm.reindex(columns=ORDER, index=ORDER)

    # Main
    fig, ax = mngs.plt.subplots()
    ax.imshow2d(hm, vmin=vmin, vmax=vmax, cmap=cmap, xyz=True)
    ax.rotate_labels()
    ax.set_xyt(None, None, z)
    ax.set_ticks(
        xvals="auto",
        xticks=hm.columns,
        yvals="auto",
        yticks=hm.index,
    )
    fig.tight_layout()
    return fig


def main(phases_to_plot):
    """
    phases_to_plot = ["Encoding", "Retrieval"]
    """

    # Loading
    lpath = mngs.io.glob(
        f"./scripts/NT/distance/between_gs/to_rank_dists/{'_'.join(phases_to_plot)}/dist_ca1.csv",
        ensure_one=True,
    )[0]
    df = mngs.io.load(lpath)

    # Save directory
    sdir = f"./{'_'.join(phases_to_plot)}/"

    # Verify balanced data
    df["n"] = 1
    df.groupby("group").agg({"n": "sum", "dist": ["mean", "min", "max"]})
    df.group = df.group.replace(
        {
            "$g_E-NT_E$": "NT_E-g_E",
            "$g_E-NT_R$": "NT_R-g_E",
            "$g_R-NT_E$": "NT_E-g_R",
            "$g_R-NT_R$": "NT_R-g_R",
        }
    )

    # Conditioning
    df_all = df.copy()
    df_in = df[df.match == 1]
    df_out = df[df.match == 2]

    # Main
    for match in ["all"] + CONFIG.MATCHES:

        df_match = {
            "1": df_in,
            "2": df_out,
            "all": df_all,
        }[str(match)]

        match_str = CONFIG.MATCHES_STR[str(match)]

        # KDE plot
        fig_kde = plot_kde(df_match)
        mngs.io.save(fig_kde, sdir + f"kde_{match_str}.jpg")

        # Box plot
        fig_box = plot_box(df_match)
        mngs.io.save(fig_box, sdir + f"box_{match_str}.jpg")

        # # Hist plot
        # fig_hist = plot_hist(df_match)
        # mngs.io.save(fig_hist, sdir + f"hist_{match_str}.jpg")

        # Violin plot
        fig_violin = plot_violin(df_match)
        mngs.io.save(fig_violin, sdir + f"violin_{match_str}.jpg")

        # Joy plot
        fig_joy = plot_joy(df_match)
        mngs.io.save(fig_joy, sdir + f"joy_{match_str}.jpg")

        # Statistical test (Wilcoxon, Brunner-Munzel, and KS)
        stats = run_stats_test(df_match)
        mngs.io.save(stats, sdir + f"stats_{match_str}.csv")

        for test_type in ["wc", "bm", "ks"]:

            for correction_method in ["bonf", "fdr", "holm"]:
                # P-values
                fig_hm_pval = plot_heatmap(stats, f"p_val_{correction_method}_{test_type}")
                mngs.io.save(fig_hm_pval, sdir + f"heatmap_{test_type}_pval_{correction_method}_{match_str}.jpg")

            # P-values (Uncorrected)
            fig_hm_pval = plot_heatmap(stats, f"p_val_unc_{test_type}")
            mngs.io.save(fig_hm_pval, sdir + f"heatmap_{test_type}_pval_unc_{match_str}.jpg")


            # Statistics
            fig_hm_stat = plot_heatmap(stats, f"statistic_{test_type}")
            mngs.io.save(fig_hm_stat, sdir + f"heatmap_{test_type}_stat_{match_str}.jpg")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, line_width=1.0, np=np, agg=True
    )
    for phases_to_plot in [
        ["Encoding", "Retrieval"],
        ["Fixation", "Encoding", "Maintenance", "Retrieval"],
    ]:
        main(phases_to_plot)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
