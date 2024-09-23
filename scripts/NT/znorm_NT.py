#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-24 01:27:57 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/znorm_NT.py


"""This script does XYZ."""


"""Imports"""
import sys
import matplotlib.pyplot as plt
import mngs

"""CONFIG"""
CONFIG = mngs.io.load_configs()

"""Functions & Classes"""
def main():

    LPATHS_NT = mngs.io.glob(CONFIG.PATH.NT)

    # bys = ["by_trial", "by_session"]
    bys = ["by_session"]

    for lpath_nt in LPATHS_NT:
        if "_z_by" not in lpath_nt:
            for by in bys:
                NT = mngs.io.load(lpath_nt)
                dim = 0 if by == "by_session" else -1
                NT_z = mngs.gen.to_z(NT, dim=dim)
                lpath_z = lpath_nt.replace(".npy", f"_z_{by}.npy")
                mngs.io.save(NT_z, lpath_z, from_cwd=True)


if __name__ == "__main__":
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        alpha=0.75,
        fig_scale=2,
        font_size_legend=2,
    )

    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

    # find ./data -name "*by_trial*" | xargs -I {} rm {}
