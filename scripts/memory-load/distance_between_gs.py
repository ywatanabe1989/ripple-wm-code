#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-09 09:00:39 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/memory-load/distance_between_gs.py

"""This script does XYZ."""

"""Imports"""
import importlib
import logging
import os
import re
import sys
import warnings
from glob import glob
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

sys.path = ["."] + sys.path
try:
    from scripts import load, utils
except Exception as e:
    pass

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""


def main():
    mngs.gen.glob(CONFIG.PATH.NT_GS_SESSION)
    mngs.gen.glob(CONFIG.PATH.TRIALS_INFO)
    pass


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
