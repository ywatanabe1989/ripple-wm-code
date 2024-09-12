#!/bin/bash
# Script created on: 2024-07-06 04:37:42
# Script path: ./scripts/ripple/all.sh

LOG_PATH_STDOUT="${0%.sh}.log"
LOG_PATH_STDERR="${0%.sh}.err"

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
        # "./scripts/ripple/check_SWR.py"
    )

    for script in "${scripts[@]}"; do
        echo -e "\n--------------------------------------------------------------------------------"
        echo "$script starts"
        echo -e "--------------------------------------------------------------------------------\n"
        if python "$script" 2>> "$LOG_PATH_STDERR" | tee -a "$LOG_PATH_STDOUT"; then
            echo "$script completed successfully" | tee -a "$LOG_PATH_STDOUT"
        else
            echo "Error: $script failed with exit code $?" | tee -a "$LOG_PATH_STDOUT" "$LOG_PATH_STDERR"
        fi
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

rm -f "$LOG_PATH_STDOUT" "$LOG_PATH_STDERR"
touch "$LOG_PATH_STDOUT" "$LOG_PATH_STDERR" || { echo "Error: Unable to create log files" >&2; exit 1; }
main 2>&1 | tee -a "$LOG_PATH_STDOUT"
echo -e "\nLogged to: $LOG_PATH_STDOUT (stdout and stderr) and $LOG_PATH_STDERR (errors only)"

# EOF
