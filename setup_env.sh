#!/bin/bash

# =============================================================================
# LLM Benchmark Environment Setup Script - H100 Optimized
# =============================================================================
# Description: Sets up the complete environment for benchmarking vLLM, SGLang,
#              and TensorRT-LLM inference engines on RunPod H100 instances
#              with H100 Hopper architecture optimizations
# =============================================================================

set -e  # Exit on any error

echo "🚀 Starting LLM Benchmark Environment Setup for H100..."
echo "======================================================="

# H100 CRITICAL: Set CUDA architecture flags for Hopper (Compute Capability 9.0)
export CUDA_ARCHITECTURES="90"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_COMPUTE_CAPABILITIES="9.0"

echo "🏗️ H100 Hopper Architecture Configuration:"
echo "   CUDA_ARCHITECTURES: $CUDA_ARCHITECTURES"
echo "   TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p /workspace/benchmarks/data
mkdir -p /workspace/benchmarks/results
mkdir -p /workspace/benchmarks/scripts
mkdir -p /workspace/benchmarks/logs

# Update system packages
echo "🔄 Updating system packages..."
apt-get update -qq
apt-get install -y wget curl git htop tmux

# Install Python dependencies
echo "🐍 Installing Python packages..."

# Core ML packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Benchmark and utility packages
pip install numpy pandas tqdm matplotlib seaborn
pip install transformers accelerate
pip install datasets huggingface_hub
pip install psutil gpustat nvidia-ml-py3

# vLLM installation - GPT-oss compatible version for H100
echo "⚡ Installing vLLM with GPT-oss support and H100 optimizations..."
# Install GPT-oss compatible vLLM version
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Alternative: Build from source for maximum H100 performance
# git clone https://github.com/vllm-project/vllm.git
# cd vllm && CUDA_ARCHITECTURES=90 pip install -e .

# SGLang installation - H100 Optimized
echo "🔥 Installing SGLang with H100 optimizations..."
pip install --upgrade pip
# Set compilation flags for H100
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"
export SGLANG_CUDA_ARCH_LIST="9.0"
pip install "sglang[all]" --no-build-isolation

# TensorRT-LLM installation - H100 Native Support
echo "🏎️ Installing TensorRT-LLM for H100..."
# TensorRT-LLM has native H100 support
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Verify H100 optimizations are active
echo "🔧 Verifying H100 optimizations..."
python3 -c "
import os
print('🏗️ H100 Build Configuration:')
print(f'   CUDA_ARCHITECTURES: {os.environ.get(\"CUDA_ARCHITECTURES\", \"Not set\")}')
print(f'   TORCH_CUDA_ARCH_LIST: {os.environ.get(\"TORCH_CUDA_ARCH_LIST\", \"Not set\")}')

try:
    import torch
    print(f'📊 PyTorch CUDA Architectures: {torch.cuda.get_arch_list()}')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        print(f'🎮 GPU: {gpu_name}')
        print(f'🔢 Compute Capability: {compute_cap[0]}.{compute_cap[1]}')
        if compute_cap >= (9, 0):
            print('✅ H100 Hopper architecture detected and supported!')
        else:
            print('⚠️ Non-H100 GPU detected')
except Exception as e:
    print(f'❌ Error checking GPU: {e}')
"

echo "📊 Downloading ShareGPT benchmark dataset..."
cd /workspace/benchmarks/data

# Download the ShareGPT dataset from Hugging Face
python3 << 'EOF'
import os
from datasets import load_dataset
import json

print("Downloading heka-ai/sharegpt-english-10k-vllm-serving-benchmark...")

try:
    # Load the dataset from Hugging Face
    dataset = load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark")
    
    # Convert to list of conversations for benchmarking
    conversations = []
    for item in dataset["train"]:
        # Extract conversation data
        conversation = {
            "conversation": item.get("conversation", []),
            "turns": item.get("turns", 1),
            "prompt": "",
            "completion": ""
        }
        
        # Extract first user message as prompt
        if item.get("conversation") and len(item["conversation"]) > 0:
            for turn in item["conversation"]:
                if turn.get("from") == "human":
                    conversation["prompt"] = turn.get("value", "")
                    break
                    
        conversations.append(conversation)
    
    # Save as JSONL for vLLM compatibility
    with open("sharegpt-10k.jsonl", "w", encoding="utf-8") as f:
        for conv in conversations:
            json.dump(conv, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✅ Downloaded {len(conversations)} conversations")
    print("✅ Saved as sharegpt-10k.jsonl")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("📝 Creating a small sample dataset as fallback...")
    
    # Fallback: Create a small sample dataset
    sample_data = []
    for i in range(100):
        sample_data.append({
            "prompt": f"Explain the concept of artificial intelligence in simple terms. This is sample prompt {i+1}.",
            "conversation": [
                {"from": "human", "value": f"Explain AI concept {i+1}"},
                {"from": "gpt", "value": "AI is a technology that enables machines to simulate human intelligence..."}
            ]
        })
    
    with open("sharegpt-10k.jsonl", "w") as f:
        for item in sample_data:
            json.dump(item, f)
            f.write("\n")
    
    print("✅ Created sample dataset with 100 entries")

EOF

echo "🔧 Setting up GPU monitoring utilities..."

# Create a simple GPU monitoring script
cat > /workspace/benchmarks/scripts/monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import json
import time
import sys

def get_gpu_stats():
    """Get current GPU utilization and memory usage"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        gpu_stats = []
        
        for i, line in enumerate(lines):
            parts = [x.strip() for x in line.split(',')]
            gpu_stats.append({
                'gpu_id': i,
                'utilization_percent': int(parts[0]),
                'memory_used_mb': int(parts[1]),
                'memory_total_mb': int(parts[2]),
                'temperature_c': int(parts[3]) if parts[3] != 'N/A' else 0,
                'power_draw_w': float(parts[4]) if parts[4] != 'N/A' else 0
            })
        
        return gpu_stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(get_gpu_stats(), indent=2))
    else:
        stats = get_gpu_stats()
        for gpu in stats:
            print(f"GPU {gpu['gpu_id']}: {gpu['utilization_percent']}% util, "
                  f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB memory, "
                  f"{gpu['temperature_c']}°C, {gpu['power_draw_w']}W")
EOF

chmod +x /workspace/benchmarks/scripts/monitor_gpu.py

echo "🧪 Verifying installations..."

# Check CUDA
echo "🔍 CUDA Version:"
nvcc --version || echo "⚠️  CUDA not found in PATH"

# Check Python packages
echo "🔍 Checking Python installations:"
python3 -c "
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except ImportError:
    print('❌ PyTorch not installed')

try:
    import vllm
    print(f'✅ vLLM: {vllm.__version__}')
except ImportError:
    print('❌ vLLM not installed')

try:
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')
except ImportError:
    print('❌ SGLang not installed')

try:
    import tensorrt_llm
    print(f'✅ TensorRT-LLM: {tensorrt_llm.__version__}')
except ImportError:
    print('⚠️ TensorRT-LLM not installed (may require NVIDIA container)')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not installed')
"

echo "📊 GPU Information:"
/workspace/benchmarks/scripts/monitor_gpu.py

echo ""
echo "✅ Environment setup completed successfully!"
echo "================================================="
echo "📁 Data directory: /workspace/benchmarks/data"
echo "📊 Results directory: /workspace/benchmarks/results"
echo "🔧 Scripts directory: /workspace/benchmarks/scripts"
echo "📋 Dataset: sharegpt-10k.jsonl"
echo ""
echo "🚀 Ready to run benchmarks!"
echo "Next steps:"
echo "  1. Run: python benchmarks/vllm_benchmark.py"
echo "  2. Run: python benchmarks/sglang_benchmark.py"
echo "  3. Run: python benchmarks/tensorrtllm_benchmark.py"
echo "  4. Run: python aggregate_results.py"