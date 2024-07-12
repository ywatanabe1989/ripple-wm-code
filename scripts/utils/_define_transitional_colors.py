#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-12 01:02:35 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/utils/_define_transition_colors.py

import numpy as np
import mngs

"""
Config
"""


"""
Functions & Classes
"""


def define_transitional_colors():
    CONFIG = mngs.gen.load_configs()
    CC = mngs.plt.PARAMS["RGBA_NORM"]

    phases = list(CONFIG.PHASES.keys())
    colors = []
    for i_phase in range(len(phases) - 1):
        phase_curr = phases[i_phase]
        phase_next = phases[i_phase + 1]

        mid_curr = np.mean(
            [
                CONFIG.PHASES[phase_curr].start,
                CONFIG.PHASES[phase_curr].end,
            ]
        ).astype(int)

        mid_next = np.mean(
            [
                CONFIG.PHASES[phase_next].start,
                CONFIG.PHASES[phase_next].end,
            ]
        ).astype(int)

        color_curr = CONFIG.PHASES[phase_curr].color
        color_next = CONFIG.PHASES[phase_next].color

        _colors = mngs.plt.interp_colors(
            CC[color_curr], CC[color_next], mid_next - mid_curr
        )
        colors.extend(_colors)

    # Extending the head and tail
    head = [(colors[0])] * (
        np.mean(
            [
                CONFIG.PHASES.Fixation.start,
                CONFIG.PHASES.Fixation.end,
            ]
        ).astype(int)
        - 0
    )

    tail = [(colors[-1])] * (
        CONFIG.PHASES.Retrieval.end
        - np.mean(
            [
                CONFIG.PHASES.Retrieval.start,
                CONFIG.PHASES.Retrieval.end,
            ]
        ).astype(int)
    )

    return head + colors + tail


if __name__ == "__main__":
    define_transitional_colors()
