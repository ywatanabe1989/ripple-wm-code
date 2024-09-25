#!/usr/bin/env bash
# Script created on: 2024-07-06 04:39:15
# Script path: /mnt/ssd/ripple-wm-code/scripts/NT/all.sh

LOG_PATH="${0%.sh}.log"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

usage() {
    echo "Usage: $0 [-h|--help]"
    echo
    echo "Options:"
    echo " -h, --help Display this help message"
    echo
    echo "Example:"
    echo " $0"
    exit 1
}

# Functions
main() {
    # Opening
    echo -e "$0 starts."

    local scripts=(
        "./znorm_NT.py"

        "./visualization/kde_8_factors.py"
        "./visualization/scatter_kde.py" # good for set-size separations
        "./visualization/umap_8_factors.py" # fxime; colors, supervised

        # Distance
        "./distance/from_O/MTL_regions.py"
        "./distance/between_gs/geometrics_medians.py"
        "./distance/between_gs/calc_dists.py"
        "./distance/between_gs/to_rank_dists.py"
        "./distance/between_gs/dists_stats.py"
        "./distance/between_gs/MTL_regions.py"

        # Classification
        "./clf/linearSVC.py"
    )

    for script in "${scripts[@]}"; do
        echo -e "\n--------------------------------------------------------------------------------"
        echo "$script starts"
        echo -e "--------------------------------------------------------------------------------\n"
        python "$SCRIPT_DIR"/"$script" 2>&1 | tee -a "$LOG_PATH"
    done

    echo "$0 ends"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
    shift
done

rm -f "$LOG_PATH"
touch "$LOG_PATH"
main 2>&1 | tee -a "$LOG_PATH"
echo -e "\nLogged to: $LOG_PATH"

# Notification
if command -v notify &> /dev/null
then
    notify \
        -s "$0 ends" \
        -m "$0 ends" \
        -t "ywata1989@gmail.com"
else
    echo "notify command not found"
fi

# EOF

# #!/usr/bin/env bash
# # Script created on: 2024-07-06 04:39:15
# # Script path: /mnt/ssd/ripple-wm-code/scripts/load/all.sh


# # Global Parameters
# LOG_PATH="$0".log

# ################################################################################
# # Main
# ################################################################################

# # Functions
# main() {
#     # Opening
#     echo -e "$0 starts."

#     # Main
#     # find data -name '*NT*' | xargs rm -rf
#     ./scripts/NT/znorm_NT.py

#     ./scripts/NT/visualization/kde_8_factors.py
#     ./scripts/NT/visualization/scatter_kde.py # good for set-size separations
#     ./scripts/NT/visualization/umap_8_factors.py # fxime; colors, supervised

#     # Classification
#     ./scripts/NT/clf/linearSVC.py

#     # Distance
#     ./scripts/NT/distance/from_O/MTL_regions.py
#     ./scripts/NT/distance/between_gs/geometrics_medians.py
#     ./scripts/NT/distance/between_gs/calc_dists.py # fixme; pval changed from Sep 3
#     ./scripts/NT/distance/between_gs/dists_stats.py # fixme; pval changed from Sep 3
#     ./scripts/NT/distance/between_gs/MTL_regions.py


#     # Closing
#     echo -e "$0 ends"
# }

# ################################################################################

# touch $LOG_PATH
# main | tee $LOG_PATH
# echo -e "
# Logged to: $LOG_PATH"

# # EOF
