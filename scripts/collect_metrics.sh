#!/bin/bash
"""
GPU and System Metrics Collection
Collects real-time GPU utilization, memory usage, temperature, and system metrics during benchmarking
"""

set -euo pipefail

# Default values
DURATION=${1:-60}      # Duration in seconds
INTERVAL=${2:-1}       # Sampling interval in seconds
OUTPUT_DIR=${3:-"./logs"}  # Output directory

echo "=========================================="
echo "GPU & System Metrics Collection"
echo "Duration: ${DURATION}s, Interval: ${INTERVAL}s"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamped filenames
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
GPU_LOG="$OUTPUT_DIR/gpu_metrics_${TIMESTAMP}.csv"
SYSTEM_LOG="$OUTPUT_DIR/system_metrics_${TIMESTAMP}.csv"
PROCESS_LOG="$OUTPUT_DIR/process_metrics_${TIMESTAMP}.log"

echo "GPU metrics: $GPU_LOG"
echo "System metrics: $SYSTEM_LOG" 
echo "Process metrics: $PROCESS_LOG"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping metrics collection..."
    
    # Kill all background jobs
    for job in $(jobs -p); do
        kill $job 2>/dev/null || true
    done
    
    echo "✅ Metrics collection stopped"
    echo "Results saved:"
    echo "  - GPU: $GPU_LOG"
    echo "  - System: $SYSTEM_LOG"
    echo "  - Processes: $PROCESS_LOG"
    
    # Show brief summary
    if [ -f "$GPU_LOG" ]; then
        echo ""
        echo "📊 GPU Utilization Summary:"
        if command -v python3 &> /dev/null; then
            python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$GPU_LOG')
    if 'gpu_util' in df.columns:
        print(f'  Average GPU Util: {df[\"gpu_util\"].mean():.1f}%')
        print(f'  Peak GPU Util: {df[\"gpu_util\"].max():.1f}%')
    if 'gpu_memory_used_mb' in df.columns:
        print(f'  Peak GPU Memory: {df[\"gpu_memory_used_mb\"].max():.0f} MB')
except Exception as e:
    print(f'  Error analyzing GPU data: {e}')
" 2>/dev/null || echo "  (Analysis requires pandas)"
        fi
    fi
}

trap cleanup EXIT INT TERM

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU monitoring disabled."
    GPU_MONITORING=false
else
    echo "✅ nvidia-smi found. GPU monitoring enabled."
    GPU_MONITORING=true
fi

# Initialize CSV files with headers
if [ "$GPU_MONITORING" = true ]; then
    echo "timestamp,gpu_id,gpu_util,gpu_memory_used_mb,gpu_memory_total_mb,gpu_temp_c,gpu_power_w,gpu_clock_mhz,gpu_mem_clock_mhz" > "$GPU_LOG"
fi

echo "timestamp,cpu_util_percent,memory_used_mb,memory_total_mb,memory_percent,load_1min,load_5min,load_15min" > "$SYSTEM_LOG"

# Function to collect GPU metrics
collect_gpu_metrics() {
    while true; do
        if [ "$GPU_MONITORING" = true ]; then
            TIMESTAMP=$(date -u +%Y-%m-%d\ %H:%M:%S)
            
            # Query GPU metrics - handle multiple GPUs
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r gpu_id util mem_used mem_total temp power gpu_clock mem_clock; do
                # Clean up values (remove spaces, handle 'N/A')
                gpu_id=$(echo "$gpu_id" | tr -d ' ')
                util=$(echo "$util" | tr -d ' ' | sed 's/N\/A/0/g')
                mem_used=$(echo "$mem_used" | tr -d ' ' | sed 's/N\/A/0/g')
                mem_total=$(echo "$mem_total" | tr -d ' ' | sed 's/N\/A/0/g')
                temp=$(echo "$temp" | tr -d ' ' | sed 's/N\/A/0/g')
                power=$(echo "$power" | tr -d ' ' | sed 's/N\/A/0/g')
                gpu_clock=$(echo "$gpu_clock" | tr -d ' ' | sed 's/N\/A/0/g')
                mem_clock=$(echo "$mem_clock" | tr -d ' ' | sed 's/N\/A/0/g')
                
                echo "$TIMESTAMP,$gpu_id,$util,$mem_used,$mem_total,$temp,$power,$gpu_clock,$mem_clock" >> "$GPU_LOG"
            done
        fi
        sleep "$INTERVAL"
    done
}

# Function to collect system metrics
collect_system_metrics() {
    while true; do
        TIMESTAMP=$(date -u +%Y-%m-%d\ %H:%M:%S)
        
        # Get CPU utilization (1-minute average)
        if command -v top &> /dev/null; then
            # Get CPU usage from top (works on most systems)
            CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' 2>/dev/null || echo "0")
        else
            CPU_UTIL="0"
        fi
        
        # Get memory info
        if [ -f /proc/meminfo ]; then
            MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')  # in KB
            MEM_AVAILABLE=$(grep MemAvailable /proc/meminfo | awk '{print $2}' 2>/dev/null || grep MemFree /proc/meminfo | awk '{print $2}')
            MEM_USED=$((MEM_TOTAL - MEM_AVAILABLE))
            MEM_PERCENT=$((MEM_USED * 100 / MEM_TOTAL))
            
            # Convert to MB
            MEM_TOTAL_MB=$((MEM_TOTAL / 1024))
            MEM_USED_MB=$((MEM_USED / 1024))
        else
            MEM_TOTAL_MB="0"
            MEM_USED_MB="0" 
            MEM_PERCENT="0"
        fi
        
        # Get load averages
        if [ -f /proc/loadavg ]; then
            read LOAD_1MIN LOAD_5MIN LOAD_15MIN _ _ < /proc/loadavg
        else
            LOAD_1MIN="0"
            LOAD_5MIN="0" 
            LOAD_15MIN="0"
        fi
        
        echo "$TIMESTAMP,$CPU_UTIL,$MEM_USED_MB,$MEM_TOTAL_MB,$MEM_PERCENT,$LOAD_1MIN,$LOAD_5MIN,$LOAD_15MIN" >> "$SYSTEM_LOG"
        
        sleep "$INTERVAL"
    done
}

# Function to collect process metrics
collect_process_metrics() {
    while true; do
        TIMESTAMP=$(date -u +%Y-%m-%d\ %H:%M:%S)
        
        {
            echo "=== $TIMESTAMP ==="
            
            # Top processes by CPU
            echo "Top CPU processes:"
            ps aux --sort=-%cpu | head -6
            echo ""
            
            # Top processes by memory
            echo "Top Memory processes:"
            ps aux --sort=-%mem | head -6
            echo ""
            
            # Python processes (likely our benchmarks)
            echo "Python processes:"
            ps aux | grep python | grep -v grep | head -5 || echo "No Python processes found"
            echo ""
            
        } >> "$PROCESS_LOG"
        
        sleep $((INTERVAL * 5))  # Less frequent for process logs
    done
}

# Start background metric collection
echo "🚀 Starting metrics collection..."

collect_gpu_metrics &
GPU_PID=$!

collect_system_metrics &
SYSTEM_PID=$!

collect_process_metrics &
PROCESS_PID=$!

echo "✅ Metrics collection started (PIDs: GPU=$GPU_PID, System=$SYSTEM_PID, Process=$PROCESS_PID)"
echo "📊 Collecting metrics for ${DURATION} seconds..."
echo "   (Press Ctrl+C to stop early)"

# Wait for specified duration or until interrupted
sleep "$DURATION"

# Cleanup will be called automatically by the trap