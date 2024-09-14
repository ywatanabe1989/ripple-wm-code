#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 22:42:06 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/adds_NT.py

"""This script associates SWR data with neural trajectory."""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np

"""Functions & Classes"""


def find_peak_i(swr):
    NT_TIME = eval(CONFIG.NT.TIME_AXIS)
    swr["peak_i"] = swr["peak_s"].apply(
        lambda x: mngs.gen.find_closest(NT_TIME, x)[1]
    )
    return swr


def add_NT(swr):
    """Add Neural Trajectory (NT) data to the DataFrame."""

    def _load_nt(row):
        return mngs.io.load(
            mngs.gen.replace(
                CONFIG.PATH.NT_Z,
                dict(sub=row.subject, session=row.session, roi=row.roi),
            )
        )

    def _slice_and_pad(nt, row, i_trial):
        # Slicing
        lim = (0, eval(CONFIG.NT.N_BINS))
        start = row.peak_i + CONFIG.RIPPLE.BINS.pre[0]
        end = row.peak_i + CONFIG.RIPPLE.BINS.post[1]
        width_ideal = end - start
        start_clipped, end_clipped = np.clip([start, end], *lim).astype(int)
        nt_slice = nt[i_trial, :, start_clipped:end_clipped]

        # Padding
        if nt_slice.shape[1] == width_ideal:
            return nt_slice

        padded = np.full((nt_slice.shape[0], width_ideal), np.nan)
        n_left_pad = abs(start - start_clipped)
        padded[:, n_left_pad : n_left_pad + nt_slice.shape[1]] = nt_slice
        return padded

    def _add_NT_single(row):
        i_trial = row.name - 1
        nt = _load_nt(row)
        nt_padded = _slice_and_pad(nt, row, i_trial)
        return nt_padded

    swr["NT"] = swr.apply(_add_NT_single, axis=1)
    np.vstack(swr.NT)
    return swr


def add_phase(swr):
    swr["phase"] = str(np.nan)
    for phase, phase_data in CONFIG.PHASES.items():
        indi_phase = (phase_data.start <= swr.peak_i) * (
            swr.peak_i < phase_data.end
        )
        swr.loc[indi_phase, "phase"] = phase
    return swr


def add_cosine(swr, ca1):
    # Loading
    GS = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1))
    # SWR = mngs.io.load(mngs.gen.replace(CONFIG.PATH.RIPPLE_WITH_NT, ca1))
    SWR = swr

    # Base
    vER = GS["Retrieval"] - GS["Encoding"]

    # SWR direction
    nt_swr = np.stack(SWR.NT, axis=0)
    start, end = nt_swr.shape[-1] // 2 + np.array(CONFIG.RIPPLE.BINS.mid)
    vSWR = nt_swr[..., end] - nt_swr[..., start]

    cosine = np.array(
        [mngs.linalg.cosine(vER, vSWR[ii]) for ii in range(len(vSWR))]
    )

    SWR["cosine_with_vER"] = cosine
    return SWR


def main():
    for ripple_type in ["RIPPLE", "RIPPLE_MINUS"]:
        for ca1 in CONFIG.ROI.CA1:
            # PATHs
            lpath = mngs.gen.replace(CONFIG.PATH[ripple_type], ca1)
            spath = mngs.gen.replace(
                CONFIG.PATH[f"{ripple_type}_WITH_NT"], ca1
            )

            # Main
            swr = mngs.io.load(lpath)
            swr = find_peak_i(swr)
            swr = add_NT(swr)
            swr = add_phase(swr)
            swr = add_cosine(swr, ca1)

            # Saving
            mngs.io.save(
                swr,
                spath,
                dry_run=False,
                from_cwd=True,
                verbose=True,
            )


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
