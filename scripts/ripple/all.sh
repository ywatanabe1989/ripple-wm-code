#!/usr/bin/env bash
# Script created on: 2024-07-06 04:37:42
# Script path: /mnt/ssd/ripple-wm-code/scripts/ripple/all.sh


# Functions
main() {
    # Opening
    echo -e "$0 starts."

    # Main
    ./scripts/ripple/detect_SWR_p.py
    ./scripts/ripple/plot_SWR_p.py
    ./scripts/ripple/define_SWR_m.py
    ./scripts/ripple/define_putative_CA1_using_UMAP.py


    # Closing
    echo -e "$0 ends"
}

################################################################################

touch $LOG_PATH
main | tee $LOG_PATH
echo -e "
Logged to: $LOG_PATH"

# EOF
