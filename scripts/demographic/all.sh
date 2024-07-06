#!/usr/bin/env bash
# Script created on: 2024-07-06 04:38:38
# Script path: /mnt/ssd/ripple-wm-code/scripts/demographic/all.sh


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
    ./scripts/demographic/fetch_demographic_data.py

    # Closing
    echo -e "$0 ends"
}

################################################################################

touch $LOG_PATH
main | tee $LOG_PATH
echo -e "
Logged to: $LOG_PATH"

# EOF
