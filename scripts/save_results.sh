#!/bin/bash
"""
Results Archive and Summary Generator
Organizes benchmark results, creates timestamped archives, and generates summary reports
"""

set -euo pipefail

# Parse arguments
ENGINE=${1:-""}
TEST_TYPE=${2:-""}
ARCHIVE_NAME=${3:-""}

if [ -z "$ENGINE" ]; then
    echo "Usage: $0 <engine> [test_type] [archive_name]"
    echo "       $0 vllm s1_throughput my_benchmark"
    echo "       $0 all"
    echo ""
    echo "Available engines: vllm, sglang, tensorrt, all"
    echo "Available test types: s1_throughput, s2_json_struct, s3_low_latency, all"
    exit 1
fi

echo "=========================================="
echo "Results Archive and Summary Generator"
echo "=========================================="

# Create archives directory
ARCHIVES_DIR="./archives"
mkdir -p "$ARCHIVES_DIR"

# Generate timestamp
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

# Determine archive name
if [ -n "$ARCHIVE_NAME" ]; then
    FINAL_ARCHIVE_NAME="${ARCHIVE_NAME}_${TIMESTAMP}"
else
    if [ "$ENGINE" = "all" ]; then
        FINAL_ARCHIVE_NAME="all_engines_${TIMESTAMP}"
    else
        if [ -n "$TEST_TYPE" ] && [ "$TEST_TYPE" != "all" ]; then
            FINAL_ARCHIVE_NAME="${ENGINE}_${TEST_TYPE}_${TIMESTAMP}"
        else
            FINAL_ARCHIVE_NAME="${ENGINE}_all_tests_${TIMESTAMP}"
        fi
    fi
fi

ARCHIVE_PATH="$ARCHIVES_DIR/$FINAL_ARCHIVE_NAME"
mkdir -p "$ARCHIVE_PATH"

echo "Archive: $FINAL_ARCHIVE_NAME"
echo "Path: $ARCHIVE_PATH"
echo ""

# Function to copy results for a specific engine and test
copy_results() {
    local engine=$1
    local test_type=$2
    
    echo "📁 Copying results for $engine / $test_type..."
    
    # Source directory
    if [ "$test_type" = "all" ]; then
        SOURCE_DIR="./results/$engine"
        DEST_DIR="$ARCHIVE_PATH/results/$engine"
    else
        # Handle both formats: s1_throughput and s1/throughput
        if [ -d "./results/$engine/${test_type}" ]; then
            SOURCE_DIR="./results/$engine/${test_type}"
            DEST_DIR="$ARCHIVE_PATH/results/$engine/${test_type}"
        elif [ -d "./results/$engine/${test_type//_//}" ]; then
            SOURCE_DIR="./results/$engine/${test_type//_//}"
            DEST_DIR="$ARCHIVE_PATH/results/$engine/${test_type//_//}"
        else
            echo "  ⚠️  Results directory not found for $engine / $test_type"
            return 1
        fi
    fi
    
    if [ -d "$SOURCE_DIR" ]; then
        mkdir -p "$(dirname "$DEST_DIR")"
        cp -r "$SOURCE_DIR" "$DEST_DIR"
        
        # Count files
        FILE_COUNT=$(find "$DEST_DIR" -name "*.json" | wc -l)
        echo "  ✅ Copied $FILE_COUNT result files"
    else
        echo "  ❌ Source directory not found: $SOURCE_DIR"
        return 1
    fi
}

# Function to copy logs
copy_logs() {
    echo "📋 Copying logs..."
    
    if [ -d "./logs" ]; then
        cp -r "./logs" "$ARCHIVE_PATH/"
        LOG_COUNT=$(find "$ARCHIVE_PATH/logs" -type f | wc -l)
        echo "  ✅ Copied $LOG_COUNT log files"
    else
        echo "  ⚠️  No logs directory found"
    fi
}

# Function to generate summary report
generate_summary() {
    echo "📊 Generating summary report..."
    
    SUMMARY_FILE="$ARCHIVE_PATH/BENCHMARK_SUMMARY.md"
    
    {
        echo "# Benchmark Results Summary"
        echo ""
        echo "**Generated:** $(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)"
        echo "**Archive:** $FINAL_ARCHIVE_NAME"
        echo ""
        
        echo "## Test Configuration"
        echo ""
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Engine(s) | $ENGINE |"
        echo "| Test Type(s) | $TEST_TYPE |"
        echo "| Archive Date | $(date -u +%Y-%m-%d) |"
        echo ""
        
        echo "## Results Overview"
        echo ""
        
        # Analyze results in archive
        if [ -d "$ARCHIVE_PATH/results" ]; then
            echo "### Available Results"
            echo ""
            
            for engine_dir in "$ARCHIVE_PATH/results"/*; do
                if [ -d "$engine_dir" ]; then
                    engine_name=$(basename "$engine_dir")
                    echo "#### $engine_name"
                    echo ""
                    
                    # Find test directories or files
                    for test_dir in "$engine_dir"/*; do
                        if [ -d "$test_dir" ]; then
                            test_name=$(basename "$test_dir")
                            json_files=$(find "$test_dir" -name "*.json" | wc -l)
                            echo "- **$test_name**: $json_files result files"
                            
                            # Try to extract key metrics from the latest result
                            latest_result=$(find "$test_dir" -name "*.json" -type f -exec ls -t {} + | head -1)
                            if [ -n "$latest_result" ] && [ -f "$latest_result" ]; then
                                if command -v python3 &> /dev/null; then
                                    python3 -c "
import json, sys
try:
    with open('$latest_result', 'r') as f:
        data = json.load(f)
    
    if 'metrics' in data:
        metrics = data['metrics']
        if 'average_tokens_per_second' in metrics:
            print(f'  - Avg Throughput: {metrics[\"average_tokens_per_second\"]:.1f} tokens/s')
        if 'success_rate_percent' in metrics:
            print(f'  - Success Rate: {metrics[\"success_rate_percent\"]:.1f}%')
        if 'latency_statistics' in metrics and 'total_latency' in metrics['latency_statistics']:
            lat = metrics['latency_statistics']['total_latency']
            if 'mean_ms' in lat:
                print(f'  - Avg Latency: {lat[\"mean_ms\"]:.1f}ms')
            if 'p95_ms' in lat:
                print(f'  - P95 Latency: {lat[\"p95_ms\"]:.1f}ms')
except Exception as e:
    pass  # Ignore parsing errors
" 2>/dev/null || true
                                fi
                            fi
                        fi
                    done
                    echo ""
                fi
            done
        else
            echo "No results found in archive."
        fi
        
        echo "## System Information"
        echo ""
        
        # Include system info if available
        ENV_LOG=$(find "$ARCHIVE_PATH/logs" -name "env_info_*.log" -type f 2>/dev/null | head -1)
        if [ -n "$ENV_LOG" ] && [ -f "$ENV_LOG" ]; then
            echo "### Hardware Configuration"
            echo ""
            echo "\`\`\`"
            
            # Extract key system info
            grep -A 5 "## System Information" "$ENV_LOG" 2>/dev/null || echo "System info not available"
            echo ""
            grep -A 10 "## GPU Information" "$ENV_LOG" 2>/dev/null | head -15 || echo "GPU info not available"
            echo ""
            grep -A 5 "## PyTorch Information" "$ENV_LOG" 2>/dev/null || echo "PyTorch info not available"
            
            echo "\`\`\`"
        else
            echo "System information not available."
        fi
        
        echo ""
        echo "## Files in Archive"
        echo ""
        echo "\`\`\`"
        find "$ARCHIVE_PATH" -type f | sort
        echo "\`\`\`"
        
        echo ""
        echo "---"
        echo "*Generated by LLM Benchmark Suite*"
        
    } > "$SUMMARY_FILE"
    
    echo "  ✅ Summary report generated: $SUMMARY_FILE"
}

# Function to create compressed archive
create_compressed_archive() {
    echo "🗜️  Creating compressed archive..."
    
    cd "$ARCHIVES_DIR"
    
    if command -v tar &> /dev/null; then
        COMPRESSED_FILE="${FINAL_ARCHIVE_NAME}.tar.gz"
        tar -czf "$COMPRESSED_FILE" "$FINAL_ARCHIVE_NAME"
        
        # Get sizes
        ORIGINAL_SIZE=$(du -sh "$FINAL_ARCHIVE_NAME" | cut -f1)
        COMPRESSED_SIZE=$(du -sh "$COMPRESSED_FILE" | cut -f1)
        
        echo "  ✅ Compressed archive created: $COMPRESSED_FILE"
        echo "     Original: $ORIGINAL_SIZE, Compressed: $COMPRESSED_SIZE"
        
        cd - > /dev/null
        return 0
    else
        echo "  ⚠️  tar not available, skipping compression"
        cd - > /dev/null
        return 1
    fi
}

# Main execution
echo "🚀 Starting results archival..."

# Copy results based on parameters
if [ "$ENGINE" = "all" ]; then
    # Archive all engines
    for engine in vllm sglang tensorrt; do
        if [ "$TEST_TYPE" = "" ] || [ "$TEST_TYPE" = "all" ]; then
            copy_results "$engine" "all" || true
        else
            copy_results "$engine" "$TEST_TYPE" || true
        fi
    done
else
    # Archive specific engine
    if [ "$TEST_TYPE" = "" ] || [ "$TEST_TYPE" = "all" ]; then
        copy_results "$ENGINE" "all" || true
    else
        copy_results "$ENGINE" "$TEST_TYPE" || true
    fi
fi

# Copy logs
copy_logs

# Copy benchmark configuration files
echo "📋 Copying benchmark configuration..."
if [ -d "./benchmarks" ]; then
    cp -r "./benchmarks" "$ARCHIVE_PATH/"
    echo "  ✅ Copied benchmark scripts"
fi

if [ -f "./TODO.md" ]; then
    cp "./TODO.md" "$ARCHIVE_PATH/"
fi

# Generate summary
generate_summary

# Create compressed version
create_compressed_archive

# Final summary
echo ""
echo "🎉 Archive creation completed!"
echo ""
echo "📁 Archive Location: $ARCHIVE_PATH"
echo "📋 Summary Report: $ARCHIVE_PATH/BENCHMARK_SUMMARY.md"

if [ -f "$ARCHIVES_DIR/${FINAL_ARCHIVE_NAME}.tar.gz" ]; then
    echo "🗜️  Compressed Archive: $ARCHIVES_DIR/${FINAL_ARCHIVE_NAME}.tar.gz"
fi

echo ""
echo "Archive Contents:"
find "$ARCHIVE_PATH" -type f | wc -l | xargs echo "  Total Files:"
find "$ARCHIVE_PATH" -name "*.json" | wc -l | xargs echo "  Result Files:"
du -sh "$ARCHIVE_PATH" | cut -f1 | xargs echo "  Total Size:"

echo ""
echo "✅ Results successfully archived and summarized!"