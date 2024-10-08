#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-08 09:14:48 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/check_CA1.py


"""
This script does XYZ.
"""


"""Imports"""
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs

"""Functions & Classes"""


def main():
    n_ca1 = len(CONFIG.ROI.CA1)
    fig, axes = mngs.plt.subplots(nrows=n_ca1, sharex=True)
    for ca1, ax in zip(CONFIG.ROI.CA1, axes):
        df = mngs.io.load(mngs.gen.replace(CONFIG.PATH.RIPPLE, ca1))

        trial_numbers = df.index.unique()
        peak_posis = []
        for trial_number in trial_numbers:
            peak_posis.append(df.loc[trial_number].peak_s.tolist())
            time = mngs.dsp.time(
                0,
                CONFIG.TRIAL.DUR_SEC,
                CONFIG.TRIAL.DUR_SEC * CONFIG.FS.iEEG,
            )
            ax.raster(peak_posis, time=time)
            ax.set_xyt(None, str(ca1).replace(", ", "\n"), None)

    # axes[-1].set_ticks(xvals=time - 6, xticks=[-6, -5, -3, 0, 2])

    fig.supxyt("Time [s]", "Putative CA1 region", "Ripple time raster plot")

    # fig.tight_layout()

    mngs.io.save(fig, "raster.jpg")
    mngs.io.save(ax.to_sigma(), "raster.csv")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        font_size_title=8,
        font_size_axis_label=4,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
