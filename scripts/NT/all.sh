#!/usr/bin/env bash
# Script created on: 2024-07-06 04:39:15
# Script path: /mnt/ssd/ripple-wm-code/scripts/load/all.sh


# Global Parameters
LOG_PATH="$0".log

################################################################################
# Main
################################################################################

# Functions
main() {
    # Opening
    echo -e "$0 starts."

    # Main
    # find data -name '*NT*' | xargs rm -rf
    ./scripts/NT/znorm_NT.py

    ./scripts/NT/visualization/kde_8_factors.py
    ./scripts/NT/visualization/scatter_kde.py # good for set-size separations
    ./scripts/NT/visualization/umap_8_factors.py # fxime; colors, supervised

    # Classification
    ./scripts/NT/clf/linearSVC.py

    # Distance
    ./scripts/NT/distance/from_O/MTL_regions.py
    ./scripts/NT/distance/between_gs/geometrics_medians.py
    ./scripts/NT/distance/between_gs/calc_dists.py
    ./scripts/NT/distance/between_gs/dists_stats.py
    ./scripts/NT/distance/between_gs/MTL_regions.py

    # Closing
    echo -e "$0 ends"
}

################################################################################

touch $LOG_PATH
main | tee $LOG_PATH
echo -e "
Logged to: $LOG_PATH"

# EOF
