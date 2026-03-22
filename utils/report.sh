#!/bin/bash

# Default values (can be overridden by command-line arguments or environment)
FOLDER="${1:-results}"
OUTPUT_DIR="${2:-conference_results_go}"
TIME_INTERVAL="${3:-2.0}"
MAX_TIME="${4:-600.0}"

pwd 

echo "=== Running Go data processor ==="
go run generate-report-csv.go \
    -folder="$FOLDER" \
    -output="$OUTPUT_DIR" \
    -interval="$TIME_INTERVAL" \
    -maxtime="$MAX_TIME"

if [ $? -ne 0 ]; then
    echo "Go processing failed."
    exit 1
fi

echo "=== Generating plots with Python ==="
python3 plot.py --output_dir="$OUTPUT_DIR" --max_time="$MAX_TIME"

echo "=== Done ==="