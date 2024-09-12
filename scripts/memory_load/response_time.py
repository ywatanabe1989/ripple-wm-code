#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-08 17:27:50 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/figures/01.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
from correct_rate import _load_data

"""
Config
"""
# CONFIG = mngs.gen.load_configs()
CONFIG = mngs.gen.load_configs()

"""
Functions & Classes
"""


def main():
    df = _load_data("response_time")
    fig, ax = mngs.plt.subplots()
    ax.bar(
        df["set_size"],
        df["mean"],
        yerr=df["std"],
    )
    ax.set_xyt("Set size", "Response time [s]", None)

    mngs.io.save(fig, "./data/memory_load/response_time.jpg", from_cwd=True)
    mngs.io.save(
        fig.to_sigma(), "./data/memory_load/response_time.csv", from_cwd=True
    )
    return fig


if __name__ == "__main__":
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        font_size_axis_label=6,
        font_size_title=6,
        alpha=0.5,
        dpi_display=100,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
