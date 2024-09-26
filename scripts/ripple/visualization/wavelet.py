#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-26 17:13:30 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/ripple/visualization/wavelet.py

"""This script calculates and visualize wavelet-transformed iEEG signals to validate the ripple (80-140 Hz) in our dataset."""

import sys
import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch


def process_batch(data, fs, device="cuda", batch_size=2):
    """
    Process a single batch of data using wavelet transform.

    Parameters
    ----------
    data : numpy.ndarray
        Input data for wavelet transform
    fs : float
        Sampling frequency
    device : str, optional
        Computation device (default is "cuda")
    batch_size : int, optional
        Size of each batch (default is 2)

    Returns
    -------
    tuple
        Processed phase, amplitude, and frequencies
    """
    return mngs.dsp.wavelet(
        data,
        fs,
        freq_scale="linear",
        out_scale="linear",
        device=device,
        batch_size=batch_size,
    )


def main(batch_size=10):
    """
    Main function to process iEEG data using wavelet transform.

    Parameters
    ----------
    batch_size : int, optional
        Size of each batch for processing (default is 10)
    """
    for ca1 in CONFIG.ROI.CA1:
        lpath = mngs.gen.replace(CONFIG.PATH.iEEG, ca1)
        spath = mngs.gen.replace(CONFIG.PATH.iEEG_WAVELET, ca1)

        data = mngs.io.load(lpath)
        data = torch.tensor(np.array(data), dtype=torch.float32).cuda()

        n_batches = len(data) // batch_size
        pha, amp, freqs = [], [], []
        for i_batch in range(n_batches):
            start_idx = i_batch * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]

            _pha, _amp, _freqs = process_batch(batch_data, CONFIG.FS.iEEG)
            pha.append(_pha.cpu().numpy())
            amp.append(_amp.cpu().numpy())
            freqs.append(_freqs.cpu().numpy())

            torch.cuda.empty_cache()

        pha = np.vstack(pha)
        amp = np.vstack(amp)
        freqs = np.vstack(freqs)

        mngs.io.save(pha, spath.replace(".pkl", "_pha.pkl"), dry_run=True)
        mngs.io.save(amp, spath.replace(".pkl", "_amp.pkl"), dry_run=True)
        mngs.io.save(freqs, spath.replace(".pkl", "_freqs.pkl"), dry_run=True)


def main_representative(batch_size=10):
    # Parameters
    repr = CONFIG.REPRESENTATIVE
    trial_number = repr.pop("trial")
    ca1 = repr
    lpath = mngs.gen.replace(CONFIG.PATH.iEEG, ca1)
    FS_DOWN_SAMPLED = CONFIG.FS.iEEG//10
    # spath = mngs.gen.replace(CONFIG.PATH.iEEG_WAVELET, ca1)

    # Loading
    xx = np.array(mngs.io.load(lpath))
    i_trial = int(trial_number) - 1
    xx = xx[i_trial][np.newaxis, ...]
    xx = mngs.dsp.resample(xx, src_fs=CONFIG.FS.iEEG, tgt_fs=FS_DOWN_SAMPLED)

    # Wavelet transformation
    pha, amp, freqs = mngs.dsp.wavelet(
        xx,
        FS_DOWN_SAMPLED,
        freq_scale="log",
        out_scale="log",
    )
    pha = pha.squeeze(0)
    amp = amp.squeeze(0)
    freqs = freqs.squeeze(0)


    # Plotting
    fig, axes = mngs.plt.subplots(nrows=len(amp))
    for i_ax, ax in enumerate(axes):
        xx = amp[i_ax]
        ax.imshow2d(xx.T)
    mngs.io.save(fig, "wavelet.jpg")

    #     mngs.io.save(pha, spath.replace(".pkl", "_pha.pkl"), dry_run=True)
    #     mngs.io.save(amp, spath.replace(".pkl", "_amp.pkl"), dry_run=True)
    #     mngs.io.save(freqs, spath.replace(".pkl", "_freqs.pkl"), dry_run=True)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main_representative()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
