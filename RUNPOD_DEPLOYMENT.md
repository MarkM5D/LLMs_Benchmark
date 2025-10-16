# RunPod H100 LLM Benchmark Deployment Guide

## Quick Setup Commands for RunPod

### 1. Clone Repository
```bash
git clone https://github.com/MarkM5D/LLMs_Benchmark.git
cd LLMs_Benchmark
```

### 2. Environment Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Make scripts executable
chmod +x scripts/*.sh
```

### 3. System Verification
```bash
# Check GPU and CUDA
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check system environment (if bash available)
bash scripts/env_info.sh || echo "Bash scripts not available, continuing..."
```

### 4. Create Dataset (Quick Start)
```bash
# Create sample dataset for immediate testing
python scripts/create_sample_dataset.py

# OR download real ShareGPT dataset (optional)
python scripts/download_dataset.py
```

### 5. Install LLM Engines (Modular Installation)

⚠️ **Important**: Install engines one by one to avoid version conflicts

#### Option A: Install One Engine at a Time (Recommended)
```bash
# Install and test vLLM
python scripts/install_engines.py --engine vllm
python scripts/run_benchmark.py --engine vllm --test s1_throughput

# Uninstall and install next engine
python scripts/uninstall_engines.py --engine vllm
python scripts/install_engines.py --engine sglang
python scripts/run_benchmark.py --engine sglang --test s1_throughput

# Continue with TensorRT-LLM
python scripts/uninstall_engines.py --engine sglang  
python scripts/install_engines.py --engine tensorrt
python scripts/run_benchmark.py --engine tensorrt --test s1_throughput
```

#### Option B: Install All (May Have Version Conflicts)
```bash
python scripts/install_engines.py --engine all
# If conflicts occur, use uninstall_engines.py and install individually
```

#### Option C: Check What's Already Installed
```bash
python scripts/uninstall_engines.py --list
```

### 6. Run Benchmarks

#### Complete Benchmark Suite
```bash
# All engines, all tests
python scripts/run_benchmark.py --engine all --test all

# With monitoring
./scripts/collect_metrics.sh 1800 10 &  # 30min monitoring
python scripts/run_benchmark.py --engine all --test all
```

#### Quick Test (Single Engine)
```bash
# Test vLLM throughput
python scripts/run_benchmark.py --engine vllm --test s1_throughput

# Test SGLang structured generation
python scripts/run_benchmark.py --engine sglang --test s2_json_struct

# Test TensorRT-LLM latency
python scripts/run_benchmark.py --engine tensorrt --test s3_low_latency
```

### 7. Analysis and Results
```bash
# Generate analysis report
python scripts/analyze_results.py --visualizations --format both

# Archive results
./scripts/save_results.sh all

# Check fairness
python scripts/analyze_test_fairness.py
```

## Expected Performance on RunPod H100

### Performance Targets
- **Throughput Test**: 2500-3500+ tokens/s
- **JSON Structure**: 1200-2000+ tokens/s  
- **Low Latency**: 30-60ms P95 response time
- **Success Rate**: 95%+ across all tests

### Resource Usage
- **GPU Memory**: 40-70GB (depending on model size)
- **System RAM**: 16-32GB recommended
- **Storage**: 100GB+ for models and results

## Troubleshooting on RunPod

### Common Issues
```bash
# CUDA memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Permission issues
sudo chmod +x scripts/*.sh

# Package conflicts
pip install --upgrade --force-reinstall torch

# Check logs
tail -f logs/benchmark_*.log
```

### Performance Optimization
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Monitor system resources
./scripts/collect_metrics.sh 300 5 &

# Validate test fairness
python scripts/analyze_test_fairness.py --verbose
```

## RunPod-Specific Notes

1. **Persistent Storage**: Save results to `/workspace` for persistence
2. **Jupyter Access**: Access via RunPod's Jupyter interface if needed
3. **SSH Access**: Use RunPod's SSH for command-line operations
4. **GPU Memory**: H100 80GB should handle all tests comfortably
5. **Network**: Fast download speeds for model downloading

## Quick Start Commands for RunPod

### One-Command Setup (Engine by Engine - Recommended)
```bash
# Setup dataset and test one engine
python scripts/create_sample_dataset.py
python scripts/install_engines.py --engine vllm
python scripts/run_benchmark.py --engine vllm --test s1_throughput
python scripts/analyze_results.py --format markdown
```

### Complete Benchmark (All Engines Sequentially)
```bash
# Create benchmark script for sequential engine testing
cat > run_full_benchmark.py << 'EOF'
import subprocess
import time

engines = ['vllm', 'sglang', 'tensorrt']
tests = ['s1_throughput', 's2_json_struct', 's3_low_latency']

print("🚀 Starting complete LLM benchmark suite...")

# Create dataset first
subprocess.run(['python', 'scripts/create_sample_dataset.py'])

for engine in engines:
    print(f"\n{'='*50}")
    print(f"Testing {engine.upper()}")
    print(f"{'='*50}")
    
    # Install engine
    result = subprocess.run(['python', 'scripts/install_engines.py', '--engine', engine])
    if result.returncode != 0:
        print(f"❌ Failed to install {engine}, skipping...")
        continue
    
    # Run all tests for this engine
    for test in tests:
        print(f"\n🔄 Running {engine} - {test}")
        subprocess.run(['python', 'scripts/run_benchmark.py', '--engine', engine, '--test', test])
        time.sleep(5)  # Brief pause between tests
    
    # Uninstall to prevent conflicts (except for the last engine)
    if engine != engines[-1]:
        subprocess.run(['python', 'scripts/uninstall_engines.py', '--engine', engine])
        time.sleep(10)  # Wait for cleanup

# Generate analysis
print("\n📊 Generating analysis...")
subprocess.run(['python', 'scripts/analyze_results.py', '--visualizations', '--format', 'both'])

print("\n🎉 Complete benchmark finished!")
print("Check analysis_output/ for results.")
EOF

# Run the complete benchmark
python run_full_benchmark.py
```

### Monitoring Commands (if bash available)
```bash
# Background monitoring (if shell scripts work)
bash scripts/collect_metrics.sh 1800 10 &  # 30min monitoring

# Python-based monitoring alternative
python -c "
import psutil, time, json
from pathlib import Path

Path('logs').mkdir(exist_ok=True)
with open('logs/system_monitor.jsonl', 'w') as f:
    for i in range(180):  # 30min with 10s intervals
        stats = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        f.write(json.dumps(stats) + '\n')
        f.flush()
        time.sleep(10)
"
```

This provides a more robust approach for RunPod deployment with better error handling.