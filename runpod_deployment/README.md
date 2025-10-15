# 🚀 LLM Inference Engine Benchmark Framework

A comprehensive benchmarking suite for comparing next-generation Large Language Model inference engines: **vLLM**, **SGLang**, and **TensorRT-LLM**.

## 📊 Overview

This project implements a scientific benchmark to evaluate the performance, efficiency, and architectural trade-offs of three leading LLM inference engines on RunPod H100 80GB instances.

### 🎯 Benchmark Objectives

- **Throughput**: Measure tokens/second performance
- **Latency**: Analyze P50/P95 response times  
- **GPU Efficiency**: Monitor memory usage and utilization
- **Fair Comparison**: Consistent parameters across all engines

## 🏗️ Architecture

```
Benchmarking_Next-Generation_LLM/
├── 📄 benchmark_plan.md          # Detailed benchmark specification
├── 🚀 run_all.py                 # Main orchestrator script
├── 🔧 setup_env.sh              # Environment setup script
├── 📊 aggregate_results.py       # Results analysis and reporting
├── benchmarks/
│   ├── 📈 metrics.py            # Shared performance metrics utility
│   ├── ⚡ vllm_benchmark.py     # vLLM benchmark implementation
│   ├── 🔥 sglang_benchmark.py   # SGLang benchmark implementation
│   ├── 🏎️ tensorrtllm_benchmark.py # TensorRT-LLM benchmark
│   ├── data/                    # Dataset storage
│   │   └── sharegpt-10k.jsonl   # ShareGPT benchmark dataset
│   └── results/                 # Benchmark results and reports
└── README.md                    # This documentation
```

## ⚙️ Configuration

### Benchmark Parameters (Consistent Across All Engines)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | `gpt-oss-20b` | Target LLM model |
| **Max Tokens** | `128` | Maximum output length |
| **Batch Size** | `32` | Requests per batch |
| **Concurrency** | `16` | Parallel request limit |
| **Temperature** | `0.8` | Sampling randomness |
| **Top-p** | `0.95` | Nucleus sampling threshold |

### Hardware Requirements

- **GPU**: H100 80GB (RunPod instance)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ available space
- **CUDA**: 12.1+ with compatible drivers

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or set up the benchmark workspace
cd Benchmarking_Next-Generation_LLM

# Run complete environment setup (installs dependencies, downloads dataset)
bash setup_env.sh
```

### 2. Run Full Benchmark Suite

```bash
# Execute complete benchmarking pipeline
python run_all.py

# Or with custom options
python run_all.py --results-dir ./my_results --skip-setup
```

### 3. Run Individual Engines

```bash
# Test single engine
python run_all.py --engine vllm
python run_all.py --engine sglang  
python run_all.py --engine tensorrtllm
```

## 📊 Results and Analysis

### Automatic Report Generation

The benchmark generates comprehensive results:

- **📄 `benchmark_report.txt`**: Executive summary and recommendations
- **📊 `benchmark_results.csv`**: Detailed performance metrics
- **🏆 `performance_rankings.csv`**: Comparative rankings
- **📈 `performance_comparison.png`**: Visual performance charts
- **🔍 `benchmark_summary.json`**: Machine-readable summary

### Key Metrics

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Throughput** | Processing speed | tokens/second |
| **Latency P50** | Median response time | milliseconds |
| **Latency P95** | 95th percentile latency | milliseconds |
| **GPU Utilization** | Average GPU usage | percentage |
| **Memory Peak** | Maximum GPU memory | percentage |

## 🔧 Advanced Usage

### Custom Configuration

```bash
# Skip environment setup
python run_all.py --skip-setup

# Custom results directory
python run_all.py --results-dir /workspace/custom_results

# Skip aggregation (run benchmarks only)
python run_all.py --no-aggregation

# Generate only charts from existing results
python aggregate_results.py --create-charts
```

### Individual Components

```bash
# Run specific benchmark scripts directly
python benchmarks/vllm_benchmark.py --output-dir ./results
python benchmarks/sglang_benchmark.py --output-dir ./results
python benchmarks/tensorrtllm_benchmark.py --output-dir ./results

# Aggregate existing results
python aggregate_results.py --results-dir ./results
```

## 🏗️ Engine-Specific Details

### ⚡ vLLM
- **Technology**: PagedAttention for memory efficiency
- **Benchmark Method**: Built-in `vllm.bench` + Python API direct sampling
- **Strengths**: Mature ecosystem, extensive model support

### 🔥 SGLang  
- **Technology**: RadixAttention for optimized inference
- **Benchmark Method**: Python API direct sampling
- **Strengths**: Advanced attention mechanisms, competitive performance

### 🏎️ TensorRT-LLM
- **Technology**: NVIDIA GPU-optimized compilation
- **Benchmark Method**: Pre-compiled engine + Python API
- **Strengths**: Hardware-specific optimizations, peak performance potential

## 📋 Dependencies

### Core Requirements

```bash
# Python packages (auto-installed by setup_env.sh)
pip install torch torchvision torchaudio
pip install vllm sglang tensorrt-llm
pip install transformers accelerate datasets
pip install numpy pandas matplotlib seaborn
pip install psutil nvidia-ml-py3 tqdm
```

### System Requirements

```bash
# System packages
apt-get install nvidia-driver cuda-toolkit
apt-get install wget curl git htop tmux
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Package Import Errors**
```bash
# Reinstall problematic packages
pip install --force-reinstall vllm sglang tensorrt-llm
```

**3. Dataset Download Failures**
```bash
# Manual dataset preparation
cd benchmarks/data
python -c "
from datasets import load_dataset
ds = load_dataset('heka-ai/sharegpt-english-10k-vllm-serving-benchmark')
# Process and save as JSONL
"
```

**4. Memory Issues**
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## 📈 Performance Optimization Tips

### For RunPod Deployment

1. **Use NVIDIA PyTorch Container**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
2. **Pre-warm GPU**: Run warm-up iterations before benchmarking
3. **Monitor Resources**: Use `nvidia-smi` and `htop` during execution
4. **Clean Between Runs**: Clear GPU cache between engine tests

### Benchmark Tuning

```bash
# Adjust batch size based on GPU memory
export BENCHMARK_BATCH_SIZE=16  # For smaller GPUs
export BENCHMARK_MAX_TOKENS=64  # For faster testing

# Custom dataset size
export BENCHMARK_PROMPT_LIMIT=500  # Limit prompts for testing
```

## 🤝 Contributing

### Adding New Engines

1. Create `benchmarks/newengine_benchmark.py` following the template
2. Implement the `BenchmarkMetrics` integration
3. Add engine configuration to `run_all.py`
4. Update documentation and test

### Extending Metrics

1. Add new metrics to `benchmarks/metrics.py`
2. Update result aggregation in `aggregate_results.py`
3. Modify visualization charts as needed

## 📄 License & Citation

This benchmark framework is designed for research and development purposes. When using this framework for publications or research, please cite:

```bibtex
@software{llm_inference_benchmark,
  title={LLM Inference Engine Benchmark Framework},
  author={AI Multiple Research Team},
  year={2024},
  url={https://github.com/aimultiple/llm-inference-benchmark}
}
```

## 🆘 Support

For issues, questions, or contributions:

1. **Check Logs**: Review `benchmarks/results/` for detailed error logs
2. **Environment**: Ensure all dependencies are properly installed
3. **Documentation**: Review this README and `benchmark_plan.md`
4. **Community**: Reach out for technical support

---

## 🎯 Expected Outcomes

After running the complete benchmark, you'll receive:

- **Performance Rankings**: Which engine performs best overall
- **Use Case Recommendations**: Best engine for throughput vs. latency
- **Resource Efficiency Analysis**: GPU utilization and memory usage patterns
- **Scalability Insights**: How each engine handles different load patterns

**🚀 Ready to benchmark? Run `python run_all.py` and let the competition begin!**