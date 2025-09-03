#!/bin/bash

# Run Gastronet baseline evaluation
# This script evaluates the pretrained Gastronet model without fine-tuning

echo "Running Gastronet baseline evaluation..."

# Check if data path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <data_path> [output_dir]"
    echo "Example: $0 data/train"
    echo "Example: $0 data/train output/gastronet_results"
    exit 1
fi

DATA_PATH=$1
OUTPUT_DIR=${2:-""}

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path '$DATA_PATH' does not exist"
    exit 1
fi

echo "Data path: $DATA_PATH"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
    python3 evaluate_gastronet_baseline.py --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR"
else
    echo "Output directory: auto-generated"
    python3 evaluate_gastronet_baseline.py --data_path "$DATA_PATH"
fi

echo "Gastronet baseline evaluation completed!"

