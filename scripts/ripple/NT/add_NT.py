#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-21 19:44:10 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/NT/adds_NT.py

"""
Associates SWR data with neural trajectory (NT) and adds relevant information.

Key operations:
1. Adds NT during SWR to the SWR dataframe
2. Adds phase of SWR
3. Associates vector of geometric medians from encoding to retrieval
4. Defines two types of SWR directions:
   a. NT direction from start to end of mid-SWR
   b. NT direction of the jump (to peak direction from the average of +/- 50 ms NT coordinates) during mid-SWR
5. Calculates radians between SWR vectors and encoding-retrieval vector

Processes both RIPPLE and RIPPLE_MINUS data for specified CA1 regions.
"""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np

"""Functions & Classes"""


def find_peak_i(swr):
    """Finds the peak index for each SWR."""
    NT_TIME = eval(CONFIG.NT.TIME_AXIS)
    swr["peak_i"] = swr["peak_s"].apply(
        lambda x: mngs.gen.find_closest(NT_TIME, x)[1]
    )
    return swr


def add_NT(swr):
    """Adds Neural Trajectory (NT) data to the DataFrame."""

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
    """Adds SWR phase information to the DataFrame."""
    swr["phase"] = str(np.nan)
    for phase, phase_data in CONFIG.PHASES.items():
        indi_phase = (phase_data.start <= swr.peak_i) * (
            swr.peak_i < phase_data.end
        )
        swr.loc[indi_phase, "phase"] = phase
    return swr


def add_vER(SWR, ca1):
    """Adds encoding-retrieval vector to the DataFrame."""
    GS = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1))
    SWR["vER"] = [
        np.array(GS["Retrieval"] - GS["Encoding"]) for _ in range(len(SWR))
    ]
    return SWR

def add_vOR(SWR, ca1):
    """Adds origin-retrieval vector to the DataFrame."""
    GS = mngs.io.load(mngs.gen.replace(CONFIG.PATH.NT_GS_SESSION, ca1))
    SWR["vOR"] = [
        np.array(GS["Retrieval"] - 0) for _ in range(len(SWR))
    ]
    return SWR


def add_vSWR_NT(SWR):
    """Adds SWR direction definition 1 (NT) to the DataFrame."""
    nt_swr = np.stack(SWR.NT, axis=0)
    start, end = nt_swr.shape[-1] // 2 + np.array(CONFIG.RIPPLE.BINS.mid)
    vSWR = nt_swr[..., end] - nt_swr[..., start]
    SWR["vSWR_NT"] = [vSWR[ii] for ii in range(len(vSWR))]
    return SWR


def add_vSWR_JUMP(SWR):
    """Adds SWR direction definition 2 (base to peak) to the DataFrame."""
    nt_swr = np.stack(SWR.NT, axis=0)

    # Indices
    mid = nt_swr.shape[-1] // 2
    mid_start, mid_end = mid + np.array(CONFIG.RIPPLE.BINS.mid)

    # Direction
    coord_base = (nt_swr[..., mid_end] + nt_swr[..., mid_start]) / 2
    coord_peak = nt_swr[..., mid]
    vSWR = coord_peak - coord_base

    SWR["vSWR_JUMP"] = [vSWR[ii] for ii in range(len(vSWR))]

    return SWR

def add_radian_NT(SWR):
    SWR["radian_NT"] = [
        np.arccos(
            mngs.linalg.cosine(SWR["vSWR_NT"].iloc[ii], SWR["vER"].iloc[ii])
        )
        for ii in range(len(SWR))
    ]
    return SWR


def add_radian_peak(SWR):
    SWR["radian_peak"] = [
        np.arccos(
            mngs.linalg.cosine(SWR["vSWR_JUMP"].iloc[ii], SWR["vER"].iloc[ii])
        )
        for ii in range(len(SWR))
    ]
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
            swr = add_vER(swr, ca1)
            swr = add_vOR(swr, ca1)
            swr = add_vSWR_NT(swr)
            swr = add_vSWR_JUMP(swr)
            swr = add_radian_NT(swr)
            swr = add_radian_peak(swr)

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
