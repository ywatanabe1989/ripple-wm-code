#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-06 04:28:21 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/utils/_sort_phases.py

import mngs

"""
Config
"""
CONFIG = mngs.gen.load_configs()


def sort_phases(unsorted_phases):
    PHASES = CONFIG.PHASES.keys()
    return mngs.gen.search(unsorted_phases, PHASES)[1]


if __name__ == "__main__":
    print(sort_phases(["Retrieval", "Encoding"]))
