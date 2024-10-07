#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 19:49:08 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/calc_dist_between_gs_trial.py

"""
Functionality:
    - Calculates distances between geometric medians (GS) for different conditions (match x set_size)
Input:
    - GS data for different conditions
Output:
    - Distances between GS for different conditions
Prerequisites:
    - mngs package, numpy, pandas, xarray
"""

import sys
from itertools import combinations
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr
from scripts import utils

def process_gs_file(lpath_gs: str, n_factors: int) -> pd.DataFrame:
    """
    Process a single GS file and calculate distances.

    Parameters
    ----------
    lpath_gs : str
        Path to the GS data file
    n_factors : int
        Number of factors to consider

    Returns
    -------
    pd.DataFrame
        Processed distances
    """
    gs = mngs.io.load(lpath_gs).iloc[:n_factors, :]

    dists = []
    for p1, p2 in combinations(CONFIG.PHASES.keys(), 2):
        for match in ["all"] + CONFIG.MATCHES:
            for set_size in ["all"] + CONFIG.SET_SIZES:
                g1 = gs[f"{p1}-match_{match}_set_size_{set_size}"]
                g2 = gs[f"{p2}-match_{match}_set_size_{set_size}"]
                dist = mngs.linalg.euclidean_distance(g1, g2)
                _df = pd.DataFrame(pd.Series({
                    "match": match,
                    "set_size": set_size,
                    "phase_combination": f"{p1[0]}{p2[0]}",
                    "distance": dist,
                })).T
                dists.append(_df)
    dists = pd.concat(dists)

    return dists

def main(n_factors: int = 3) -> None:
    """
    Main function to process all GS files and save results.

    Parameters
    ----------
    n_factors : int, optional
        Number of factors to consider (default is 3)
    """
    lpaths_gs = mngs.io.glob(CONFIG.PATH.NT_GS_MATCH_SET_SIZE)
    for lpath_gs in lpaths_gs:
        dists = process_gs_file(lpath_gs, n_factors)
        spath = mngs.gen.replace(
            CONFIG.PATH.NT_DIST_BETWEEN_GS_MATCH_SET_SIZE,
            utils.parse_lpath(lpath_gs),
        )
        mngs.io.save(dists, spath, from_cwd=True)

if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True, np=np)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-06 20:58:38 (ywatanabe)"
# # /mnt/ssd/ripple-wm-code/scripts/NT/distance/between_gs/calc_dist_between_gs_trial.py

# """
# 1. Functionality:
#    - Calculates distances between geometric medians (GS) for different trial phases
# 2. Input:
#    - geometric median (GS) data and corresponding trial information
# 3. Output:
#    - Distances between GS trials for different phase combinations
# 4. Prerequisites:
#    - mngs package, numpy, pandas, xarray
# """

# import sys
# from itertools import combinations
# from typing import Dict, Tuple

# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# import pandas as pd
# import xarray as xr

# try:
#     from scripts import utils
# except ImportError:
#     pass



# def main(n_factors: int = 3) -> None:
#     """
#     Main function to process all GS files and save results.

#     Parameters
#     ----------
#     n_factors : int, optional
#         Number of factors to consider (default is 3)
#     """
#     lpaths_gs = mngs.io.glob(CONFIG.PATH.NT_GS_MATCH_SET_SIZE)
#     for lpath_gs in lpaths_gs:
#         dists = process_gs_file(lpath_gs, n_factors)
#         spath = mngs.gen.replace(
#             CONFIG.PATH.NT_DIST_BETWEEN_GS_MATCH_SET_SIZE,
#             utils.parse_lpath(lpath_gs),
#             )
#         mngs.io.save(dists, spath, from_cwd=True)

# if __name__ == '__main__':
#     np.random.seed(42)
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)

# # EOF
