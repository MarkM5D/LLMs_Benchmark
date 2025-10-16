#!/bin/bash

# Dataset download script for ShareGPT English 10K vLLM serving benchmark
# Usage: bash download_dataset.sh

set -euo pipefail

DATASET_NAME="heka-ai/sharegpt-english-10k-vllm-serving-benchmark"
DATASET_DIR="./datasets"
OUTPUT_FILE="$DATASET_DIR/sharegpt_prompts.jsonl"

echo "=========================================="
echo "Downloading ShareGPT Dataset for Benchmarking"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Create datasets directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# Check if Python and required packages are available
echo "Checking dependencies..."
python3 -c "import datasets, json" || {
    echo "Error: Required Python packages not found. Installing..."
    pip install datasets
}

# Download and convert dataset
echo "Downloading dataset from Hugging Face..."
python3 << 'EOF'
from datasets import load_dataset
import json
import os

# Load the dataset
print("Loading dataset: heka-ai/sharegpt-english-10k-vllm-serving-benchmark")
dataset = load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark")

# Convert to JSONL format for benchmarking
output_file = "./datasets/sharegpt_prompts.jsonl"
print(f"Converting to JSONL format: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    for item in dataset['train']:  # Assuming 'train' split exists
        # Format the conversations according to our benchmark needs
        if 'conversations' in item:
            # Extract the first human prompt for benchmarking
            for conv in item['conversations']:
                if conv.get('from') == 'human' and conv.get('value'):
                    benchmark_item = {
                        "prompt": conv['value'],
                        "length": len(conv['value'].split()),
                        "conversations": item['conversations']
                    }
                    f.write(json.dumps(benchmark_item, ensure_ascii=False) + '\n')
                    break
        elif 'prompt' in item:
            # Direct prompt format
            benchmark_item = {
                "prompt": item['prompt'],
                "length": len(item['prompt'].split()) if 'prompt' in item else 0
            }
            f.write(json.dumps(benchmark_item, ensure_ascii=False) + '\n')

print(f"Dataset successfully converted to {output_file}")
print("Dataset download and conversion completed!")
EOF

# Verify the downloaded file
if [ -f "$OUTPUT_FILE" ]; then
    echo "✅ Dataset downloaded successfully!"
    echo "📊 Dataset statistics:"
    echo "   - File: $OUTPUT_FILE"
    echo "   - Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "   - Lines: $(wc -l < "$OUTPUT_FILE")"
    echo "   - Sample entry:"
    head -n 1 "$OUTPUT_FILE" | python3 -m json.tool
else
    echo "❌ Dataset download failed!"
    exit 1
fi

echo "Dataset is ready for benchmarking!"