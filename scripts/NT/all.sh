#!/usr/bin/env bash
# Script created on: 2024-07-06 04:40:09
# Script path: /mnt/ssd/ripple-wm-code/scripts/NT/all.sh


################################################################################

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
    find data -name '*NT*' | xargs rm -rf
    ./scripts/NT/calc_NT_with_GPFA.py
    ./scripts/NT/znorm_NT.py

    # Closing
    echo -e "$0 ends"
}

################################################################################

touch $LOG_PATH
main | tee $LOG_PATH
echo -e "
Logged to: $LOG_PATH"

# EOF
