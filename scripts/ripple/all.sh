#!/bin/bash
# Script created on: 2024-07-06 04:37:42
# Script path: ./scripts/ripple/all.sh

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

main() {
    echo "$0 starts."

    local scripts=(
        "./detect_and_define/detect_SWR_p.py"
        "./detect_and_define/define_SWR_m.py"
        "./detect_and_define/define_putative_CA1_using_UMAP.py"
        "./plot_SWR_p.py"
        "./stats/count.py"        
        "./stats/duration_amplitude.py"
        "./stats/time_course.py"
        "./NT/add_NT.py"
        "./NT/distance/from_O_lineplot.py"        
        "./NT/distance/from_O_boxplot.py"
        "./NT/distance/stats.py"
        "./NT/direction/kde_plot.py"
        # "./NT/direction/cosine_kde_plot.py"        
        # "./NT/direction/cosine_kde_plot.py"
        # "./NT/direction/radian_kde_plot.py"        
        # "./check_SWR.py"
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
