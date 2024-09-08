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
    ./scripts/memory-load/correct_rate.py
    ./scripts/memory-load/response_time.py    

    # Closing
    echo -e "$0 ends"
}

################################################################################

touch $LOG_PATH
main | tee $LOG_PATH
echo -e "
Logged to: $LOG_PATH"

# EOF
