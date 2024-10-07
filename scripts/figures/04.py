#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-09 08:45:47 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/figures/01.py


"""This script does XYZ."""


"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs

"""Config"""
CONFIG = mngs.gen.load_configs()


"""Functions & Classes"""


def A():
    pass


def B():
    pass


def C():
    pass


def D():
    pass


def E():
    pass


def main():
    """Executes main analysis and plotting routines."""
    fig_A = A()
    fig_B = B()
    fig_C = C()
    fig_D = D()
    fig_E = E()
    plt.show()


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        # agg=True,
        font_size_axis_label=6,
        font_size_title=6,
        alpha=0.5,
        dpi_display=100,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
