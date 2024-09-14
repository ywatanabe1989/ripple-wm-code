#!/bin/bash
# Script created on: 2024-07-06 04:37:42
# Script path: ./scripts/ripple/all.sh

LOG_PATH="${0%.sh}.log"

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

main() {
    echo "$0 starts."

    local scripts=(
        "./scripts/ripple/detect_and_define/detect_SWR_p.py"
        "./scripts/ripple/detect_and_define/define_SWR_m.py"
        "./scripts/ripple/detect_and_define/define_putative_CA1_using_UMAP.py"
        "./scripts/ripple/plot_SWR_p.py"
        "./scripts/ripple/stats/duration_amplitude.py"
        "./scripts/ripple/stats/time_course.py"
        "./scripts/ripple/NT/add_NT.py"
        "./scripts/ripple/NT/distance/from_O_lineplot.py"        
        "./scripts/ripple/NT/distance/from_O_boxplot.py"
        # "./scripts/ripple/check_SWR.py"
    )

    for script in "${scripts[@]}"; do
        echo -e "\n--------------------------------------------------------------------------------"
        echo "$script starts"
        echo -e "--------------------------------------------------------------------------------\n"
        python "$script" 2>&1 | tee -a "$LOG_PATH"
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

# EOF
