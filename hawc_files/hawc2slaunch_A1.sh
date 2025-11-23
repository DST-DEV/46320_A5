#!/bin/bash

# Script to submit multiple HAWC2S simulations as jobs
# Usage: ./submit_hawc2s_jobs.sh <base_path>
# Example: ./submit_hawc2s_jobs.sh ./htc/hawc2s/DTU_10MW

if [ $# -eq 0 ]; then
    echo "Error: No base path provided"
    echo "Usage: $0 <base_path>"
    echo "Example: $0 ./htc/hawc2s/DTU_10MW"
    exit 1
fi

BASE_PATH="$1"

# Array of suffixes
SUFFIXES=("_1wsp" "_multitsr" "_rigid" "_flex")

# Submit jobs for each suffix
for suffix in "${SUFFIXES[@]}"; do
    HTCFILE="${BASE_PATH}_hawc2s${suffix}.htc"
    
    # Check if the .htc file exists
    if [ ! -f "$HTCFILE" ]; then
        echo "Warning: HTC file not found: $HTCFILE"
        continue
    fi
    
    echo "Submitting job for: $HTCFILE"
    
    # Submit the job
    bsub -env "HTCFILE=$HTCFILE" < hawc2slaunch.sh
    
    # Optional: small delay between submissions
    sleep 1
done

echo "All jobs submitted!"