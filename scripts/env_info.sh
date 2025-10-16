#!/bin/bash
"""
System Environment Information Collector
Collects detailed information about the system environment, GPU, CUDA, Python packages
"""

set -euo pipefail

echo "=========================================="
echo "System Environment Information Collection"
echo "Timestamp: $(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Output file with timestamp
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
OUTPUT_FILE="./logs/env_info_${TIMESTAMP}.log"

echo "Saving environment info to: $OUTPUT_FILE"

{
    echo "# System Environment Information"
    echo "Generated at: $(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)"
    echo ""
    
    echo "## System Information"
    echo "OS: $(uname -a)"
    echo "Hostname: $(hostname)"
    echo "Uptime: $(uptime)"
    echo ""
    
    echo "## CPU Information"
    if command -v lscpu &> /dev/null; then
        echo "CPU Details:"
        lscpu | head -20
    else
        echo "CPU Info: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2)"
        echo "CPU Cores: $(nproc)"
    fi
    echo ""
    
    echo "## Memory Information"
    if command -v free &> /dev/null; then
        echo "Memory Usage:"
        free -h
        echo ""
        echo "Memory Details:"
        cat /proc/meminfo | head -10
    fi
    echo ""
    
    echo "## Storage Information"
    echo "Disk Usage:"
    df -h
    echo ""
    
    echo "## GPU Information"
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU Info:"
        nvidia-smi
        echo ""
        echo "NVIDIA GPU Details:"
        nvidia-smi -q | head -50
    else
        echo "nvidia-smi not found - no NVIDIA GPU detected"
    fi
    echo ""
    
    echo "## CUDA Information"
    if command -v nvcc &> /dev/null; then
        echo "NVCC Version:"
        nvcc --version
    else
        echo "NVCC not found"
    fi
    
    if [ -f /usr/local/cuda/version.txt ]; then
        echo "CUDA Version File:"
        cat /usr/local/cuda/version.txt
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA Driver Version:"
        nvidia-smi --query-gpu=driver_version,cuda_version --format=csv
    fi
    echo ""
    
    echo "## Python Environment"
    echo "Python Version: $(python3 --version 2>&1)"
    echo "Python Path: $(which python3)"
    echo "Pip Version: $(pip --version 2>&1)"
    echo ""
    
    echo "## Python Packages (Relevant to LLM Inference)"
    echo "Checking for key packages..."
    
    # Check for LLM-related packages
    packages=(
        "torch" "torchvision" "transformers" "datasets" 
        "vllm" "sglang" "tensorrt" "tensorrt-llm"
        "numpy" "pandas" "json" "requests"
        "accelerate" "bitsandbytes" "flash-attn"
        "triton" "xformers" "optimum"
    )
    
    for package in "${packages[@]}"; do
        if python3 -c "import $package; print(f'✅ $package: {$package.__version__}')" 2>/dev/null; then
            python3 -c "import $package; print(f'✅ $package: {$package.__version__}')" 2>/dev/null
        else
            echo "❌ $package: not installed"
        fi
    done
    echo ""
    
    echo "## PyTorch Information"
    python3 -c "
try:
    import torch
    print(f'PyTorch Version: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Version (PyTorch): {torch.version.cuda}')
        print(f'CUDA Device Count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB')
    print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
except ImportError:
    print('PyTorch not installed')
except Exception as e:
    print(f'Error getting PyTorch info: {e}')
"
    echo ""
    
    echo "## Network Information"
    echo "Network Interfaces:"
    ip addr show 2>/dev/null || ifconfig 2>/dev/null || echo "No network info available"
    echo ""
    
    echo "## Environment Variables"
    echo "CUDA-related environment variables:"
    env | grep -i cuda || echo "No CUDA environment variables found"
    echo ""
    echo "Python-related environment variables:"
    env | grep -i python || echo "No Python environment variables found"
    echo ""
    
    echo "## Process Information"
    echo "Running processes (GPU-related):"
    ps aux | grep -E "(python|nvidia|cuda)" | head -10 || echo "No relevant processes found"
    echo ""
    
    echo "## Container/Docker Information"
    if [ -f /.dockerenv ]; then
        echo "Running in Docker container"
        if command -v docker &> /dev/null; then
            echo "Docker version: $(docker --version)"
        fi
    else
        echo "Not running in Docker container"
    fi
    echo ""
    
    echo "## Additional System Libraries"
    echo "Checking system libraries..."
    
    # Check for CUDA libraries
    if [ -d /usr/local/cuda/lib64 ]; then
        echo "CUDA Libraries found:"
        ls -la /usr/local/cuda/lib64/libcuda* 2>/dev/null || echo "No CUDA libraries found"
        ls -la /usr/local/cuda/lib64/libcublas* 2>/dev/null || echo "No cuBLAS libraries found"
        ls -la /usr/local/cuda/lib64/libcudnn* 2>/dev/null || echo "No cuDNN libraries found"
    fi
    echo ""
    
    # Check for TensorRT
    if [ -d /usr/local/tensorrt ]; then
        echo "TensorRT Installation found:"
        ls -la /usr/local/tensorrt/ 2>/dev/null || echo "Cannot access TensorRT directory"
    fi
    echo ""
    
    echo "## Benchmark Project Information"
    echo "Current Directory: $(pwd)"
    echo "Project Structure:"
    find . -maxdepth 3 -type d | head -20
    echo ""
    
    echo "Dataset Information:"
    if [ -f "./datasets/sharegpt_prompts.jsonl" ]; then
        echo "✅ Dataset file found: $(wc -l < ./datasets/sharegpt_prompts.jsonl) lines"
        echo "Sample entries:"
        head -2 ./datasets/sharegpt_prompts.jsonl
    else
        echo "❌ Dataset file not found"
    fi
    echo ""
    
    echo "Benchmark Scripts:"
    echo "Available engines:"
    for engine in vllm sglang tensorrt; do
        if [ -d "./benchmarks/$engine" ]; then
            echo "✅ $engine: $(ls ./benchmarks/$engine/*.py | wc -l) test files"
        else
            echo "❌ $engine: directory not found"
        fi
    done
    echo ""
    
    echo "=========================================="
    echo "Environment info collection completed"
    echo "=========================================="
    
} > "$OUTPUT_FILE" 2>&1

# Also display to console
echo ""
echo "📊 Environment Summary:"
echo "- OS: $(uname -s)"
echo "- Python: $(python3 --version 2>&1)"
echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'Not detected')"
echo "- CUDA: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1 || echo 'Not detected')"

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    echo "- PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)"
else
    echo "- PyTorch: Not installed"
fi

echo ""
echo "✅ Full environment information saved to: $OUTPUT_FILE"
echo ""

# Return the output file path for other scripts to use
echo "$OUTPUT_FILE"