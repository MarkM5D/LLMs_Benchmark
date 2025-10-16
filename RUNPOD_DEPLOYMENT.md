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

# Check system environment
./scripts/env_info.sh
```

### 4. Install LLM Engines (Choose Option)

#### Option A: Install All Engines
```bash
python scripts/install_engines.py --engine all
```

#### Option B: Install Specific Engine
```bash
# vLLM only
python scripts/install_engines.py --engine vllm

# SGLang only  
python scripts/install_engines.py --engine sglang

# TensorRT-LLM only
python scripts/install_engines.py --engine tensorrt
```

### 5. Download Dataset
```bash
./scripts/download_dataset.sh
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

## One-Command Full Benchmark
```bash
# Complete benchmark with monitoring (recommended)
nohup bash -c '
./scripts/env_info.sh > logs/runpod_environment.log 2>&1;
./scripts/collect_metrics.sh 3600 15 &
python scripts/run_benchmark.py --engine all --test all;
python scripts/analyze_results.py --visualizations --format both;
./scripts/save_results.sh all all runpod_h100_$(date +%Y%m%d);
echo "Benchmark completed! Check analysis_output/ for results."
' > benchmark_run.log 2>&1 &
```

This will run the complete benchmark suite in background and save all results.