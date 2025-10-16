# LLM Benchmark Suite

A comprehensive benchmarking framework for comparing next-generation Large Language Model (LLM) inference engines on RunPod H100 instances.

## 🎯 Overview

This benchmark suite provides **fair and comprehensive testing** of three leading LLM inference engines:

- **vLLM**: PagedAttention-based serving with high throughput optimization
- **SGLang**: RadixAttention with structured generation capabilities  
- **TensorRT-LLM**: NVIDIA's optimized inference engine for maximum performance

## 🏗️ Architecture

### Test Framework
- **3 Engines × 3 Test Scenarios = 9 Comprehensive Benchmarks**
- **Standardized Parameters** for fair comparison across all engines
- **Automated Fairness Validation** to ensure unbiased results
- **Unified Engine Interfaces** for consistent testing methodology

### Test Scenarios
1. **S1 Throughput Test**: Maximum tokens/second performance measurement
2. **S2 JSON Structure Test**: Structured generation capability and validation
3. **S3 Low Latency Test**: Minimum latency optimization for real-time applications

## 🚀 Quick Start

### Prerequisites
- **Hardware**: RunPod H100 80GB GPU instance (recommended)
- **Software**: CUDA 12.x, Python 3.9+, Docker (optional)
- **Memory**: 32GB+ system RAM recommended

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd LLM_Benchmark
```

2. **Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install engines (choose one or all):**
```bash
# Install all engines
python scripts/install_engines.py --engine all

# Install specific engine
python scripts/install_engines.py --engine vllm
python scripts/install_engines.py --engine sglang  
python scripts/install_engines.py --engine tensorrt
```

### Running Benchmarks

#### Complete Benchmark Suite
```bash
# Run all tests for all engines
python scripts/run_benchmark.py --engine all --test all --dataset datasets/sharegpt_prompts.jsonl

# Monitor system during benchmarks
./scripts/collect_metrics.sh &
```

#### Specific Engine/Test Combinations
```bash
# Single engine, single test
python scripts/run_benchmark.py --engine vllm --test s1_throughput

# Single engine, all tests
python scripts/run_benchmark.py --engine sglang --test all

# All engines, specific test
python scripts/run_benchmark.py --engine all --test s3_low_latency
```

#### Using Engine Adapter Directly
```bash
# Direct engine testing
python scripts/run_engine.py --engine vllm --prompt "Tell me about AI" --max-tokens 100
python scripts/run_engine.py --engine sglang --structured-output --max-tokens 256
python scripts/run_engine.py --engine tensorrt --batch-size 8 --temperature 0.8
```

## 📊 Analysis and Results

### Automated Analysis
```bash
# Generate comprehensive analysis report
python scripts/analyze_results.py --results-dir ./results --output-dir ./analysis_output

# Create visualizations (requires matplotlib)
python scripts/analyze_results.py --visualizations --format both

# Quick performance comparison
python scripts/analyze_results.py --format markdown
```

### Result Management
```bash
# Archive specific results  
./scripts/save_results.sh vllm s1_throughput my_experiment

# Archive all results
./scripts/save_results.sh all

# Archive with custom name
./scripts/save_results.sh sglang all custom_benchmark_name
```

### System Monitoring
```bash
# Collect environment information
./scripts/env_info.sh

# Real-time performance monitoring
./scripts/collect_metrics.sh 300 5  # 5min duration, 5sec interval

# Background monitoring during benchmarks
./scripts/collect_metrics.sh 1800 10 &  # 30min, 10sec interval
```

## 📁 Project Structure

```
LLM_Benchmark/
├── benchmarks/           # Individual test implementations
│   ├── vllm/            # vLLM-specific tests
│   │   ├── s1_throughput.py
│   │   ├── s2_json_struct.py
│   │   └── s3_low_latency.py
│   ├── sglang/          # SGLang-specific tests  
│   │   ├── s1_throughput.py
│   │   ├── s2_json_struct.py
│   │   └── s3_low_latency.py
│   └── tensorrt/        # TensorRT-LLM tests
│       ├── s1_throughput.py
│       ├── s2_json_struct.py
│       └── s3_low_latency.py
├── scripts/             # Core orchestration and utilities
│   ├── run_benchmark.py     # Main benchmark orchestrator
│   ├── run_engine.py        # Unified engine adapter
│   ├── analyze_results.py   # Results analysis tool
│   ├── analyze_test_fairness.py  # Fairness validator
│   ├── install_engines.py   # Engine installation
│   ├── uninstall_engines.py # Engine cleanup
│   ├── env_info.sh         # System environment collection
│   ├── collect_metrics.sh  # Real-time monitoring
│   └── save_results.sh     # Results archival
├── datasets/            # Benchmark datasets
│   └── sharegpt_prompts.jsonl
├── results/             # Benchmark output (created during execution)
│   ├── vllm/           # Engine-specific results
│   ├── sglang/
│   └── tensorrt/
├── logs/               # System and execution logs
├── archives/           # Compressed result archives  
├── analysis_output/    # Analysis reports and visualizations
└── docs/              # Documentation
    ├── benchmark_plan.md
    └── README.md (this file)
```

## 🔧 Configuration

### Test Parameters (Standardized for Fairness)
- **Batch Size**: 8 (consistent across all engines)
- **Temperature**: 0.8 (for S1/S3), 0.0 (for S2 structured generation)
- **Top-p**: 0.95 (sampling parameter)
- **Repetition Penalty**: 1.1 (for S1/S2)
- **Max Tokens**: 256 (adjusted for SGLang structured generation)

### Engine-Specific Settings
- **vLLM**: PagedAttention with optimized GPU memory management
- **SGLang**: RadixAttention with structured generation limits
- **TensorRT-LLM**: Optimized kernels with FP16 precision

### Dataset Configuration
- **Source**: Hugging Face `sharegpt-english-10k-vllm-serving-benchmark`
- **Format**: JSONL with conversation structure
- **Size**: 1000 high-quality prompts for comprehensive testing
- **Validation**: Automated prompt validation and statistics

## 📈 Benchmark Metrics

### Performance Metrics
- **Throughput**: Tokens per second (tokens/s)
- **Latency**: Mean, P50, P95, P99 response times (ms)
- **Success Rate**: Percentage of successful completions
- **GPU Utilization**: Memory usage and compute efficiency

### Quality Metrics  
- **JSON Validation**: Structure compliance for S2 tests
- **Output Token Distribution**: Statistical analysis of generated content
- **Error Analysis**: Failure categorization and root cause analysis

### System Metrics
- **GPU Memory**: Peak and average usage (GB)
- **System Memory**: RAM utilization during inference
- **CPU Usage**: System overhead measurement
- **Temperature**: GPU thermal performance

## 🎛️ Advanced Usage

### Custom Dataset Testing
```bash
# Prepare custom dataset
python -c "
import json
prompts = [{'prompt': 'Your custom prompt here'}, ...]
with open('custom_dataset.jsonl', 'w') as f:
    for prompt in prompts:
        f.write(json.dumps(prompt) + '\n')
"

# Run with custom dataset
python scripts/run_benchmark.py --dataset custom_dataset.jsonl --engine all
```

### Performance Tuning
```bash
# Test different batch sizes
for batch_size in 4 8 16 32; do
    python scripts/run_engine.py --engine vllm --batch-size $batch_size
done

# Test different precision modes (TensorRT-LLM)
python scripts/run_engine.py --engine tensorrt --precision fp16
python scripts/run_engine.py --engine tensorrt --precision int8
```

### Continuous Integration
```bash
# Automated fairness validation
python scripts/analyze_test_fairness.py

# Regression testing
python scripts/run_benchmark.py --engine all --quick-test

# Performance monitoring
python scripts/analyze_results.py --compare-with-baseline baseline_results/
```

## 📊 Expected Results

### Performance Baselines (H100 80GB)

| Engine | Throughput Test | JSON Structure | Low Latency |
|--------|----------------|----------------|-------------|
| **vLLM** | ~2800 tokens/s | ~1200 tokens/s | ~45ms P95 |
| **SGLang** | ~3100 tokens/s | ~1800 tokens/s | ~42ms P95 |  
| **TensorRT-LLM** | ~3500 tokens/s | ~1500 tokens/s | ~38ms P95 |

*Note: Results may vary based on model size, hardware configuration, and system load.*

### Analysis Outputs
- **Markdown Report**: Comprehensive performance comparison with rankings
- **JSON Data**: Machine-readable results for further analysis
- **Visualizations**: Charts comparing throughput, latency, and success rates
- **Technical Insights**: Recommendations for production deployment

## 🔍 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# CUDA version conflicts
nvidia-smi  # Check CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Memory errors
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> ~/.bashrc
```

#### Engine-Specific Issues
```bash
# vLLM installation
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# SGLang dependencies  
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/

# TensorRT-LLM setup
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com
```

#### Performance Issues
- **Low GPU Utilization**: Check batch size and model parallelization
- **High Memory Usage**: Monitor with `nvidia-smi` and adjust parameters
- **Inconsistent Results**: Ensure consistent environment with `env_info.sh`

### Getting Help
1. **Check Logs**: Review files in `./logs/` directory
2. **System Information**: Run `./scripts/env_info.sh` for diagnostics  
3. **Fairness Validation**: Use `python scripts/analyze_test_fairness.py`
4. **Verbose Output**: Add `--verbose` flag to any script

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black scripts/ benchmarks/
flake8 scripts/ benchmarks/
```

### Adding New Engines
1. Create engine adapter in `scripts/run_engine.py`
2. Implement benchmark tests in `benchmarks/new_engine/`  
3. Update fairness validation in `scripts/analyze_test_fairness.py`
4. Add installation instructions in `scripts/install_engines.py`

### Testing Guidelines
- Ensure all tests pass fairness validation
- Maintain consistent parameter standardization
- Update documentation for new features
- Test on multiple GPU configurations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the sharegpt benchmark dataset
- **vLLM Team** for the PagedAttention implementation
- **SGLang Team** for RadixAttention and structured generation
- **NVIDIA** for TensorRT-LLM optimization framework
- **RunPod** for H100 GPU infrastructure support

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: See `docs/benchmark_plan.md` for detailed technical specifications
- **Issues**: Create detailed issue reports with system information from `env_info.sh`  
- **Results**: Share benchmark results and analysis for community validation

---

**AI Multiple LLM Benchmark Team**