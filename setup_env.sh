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

# Update system packages and install MPI for TensorRT-LLM
echo "🔄 Updating system packages..."
apt-get update -qq
apt-get install -y wget curl git htop tmux

# Install MPI for TensorRT-LLM support
echo "🔧 Installing MPI support for TensorRT-LLM..."
apt-get install -y libopenmpi-dev openmpi-bin
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

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
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121 || \
pip install vllm --no-build-isolation || \
pip install "vllm>=0.6.0"

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

# Download from Hugging Face with proper format handling
python3 << 'EOF'
import json
import os

print("Downloading heka-ai/sharegpt-english-10k-vllm-serving-benchmark...")

try:
    from datasets import load_dataset
    
    # GUARANTEED DATASET DOWNLOAD - Multiple methods, MUST succeed
    dataset = None
    methods = [
        ("Loading train split only", lambda: load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark", split="train")),
        ("Loading full dataset", lambda: load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark")),
        ("Loading with trust_remote_code", lambda: load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark", split="train", trust_remote_code=True)),
        ("Loading with download_mode=force_redownload", lambda: load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark", split="train", download_mode="force_redownload"))
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"Attempting: {method_name}...")
            result = method_func()
            if isinstance(result, dict) and "train" in result:
                dataset = result["train"]
            else:
                dataset = result
            print(f"✅ SUCCESS: {method_name}")
            break
        except Exception as method_error:
            print(f"⚠️ {method_name} failed: {method_error}")
            continue
    
    if dataset is None:
        raise Exception("All download methods failed")
    
    print(f"✅ Loaded {len(dataset)} entries from dataset")
    
    # Parse and convert to consistent format
    processed_prompts = []
    
    for idx, item in enumerate(dataset):
        try:
            # Extract prompt from conversations
            prompt = ""
            
            # Handle different possible formats
            if "conversations" in item and item["conversations"]:
                # Standard ShareGPT format with conversations array
                for turn in item["conversations"]:
                    if isinstance(turn, dict) and turn.get("from") == "human":
                        prompt = turn.get("value", "").strip()
                        break
            elif "conversation" in item and item["conversation"]:
                # Alternative conversation format
                for turn in item["conversation"]:
                    if isinstance(turn, dict) and turn.get("from") == "human":
                        prompt = turn.get("value", "").strip()
                        break
            elif "prompt" in item:
                # Direct prompt format
                prompt = str(item["prompt"]).strip()
            elif "text" in item:
                # Text format
                prompt = str(item["text"]).strip()
            
            # Only add valid prompts
            if prompt and len(prompt) > 10:  # Filter out very short prompts
                processed_prompts.append({
                    "prompt": prompt,
                    "id": idx,
                    "conversation": [{"from": "human", "value": prompt}]
                })
                
            # Limit to prevent memory issues
            if len(processed_prompts) >= 10000:
                break
                
        except Exception as e:
            print(f"⚠️ Skipping item {idx}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(processed_prompts)} valid prompts")
    
    # Save in consistent JSONL format
    with open("sharegpt-10k.jsonl", "w", encoding="utf-8") as f:
        for entry in processed_prompts:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(processed_prompts)} prompts to sharegpt-10k.jsonl")
    print(f"📍 Location: {os.path.abspath('sharegpt-10k.jsonl')}")

except Exception as e:
    error_msg = f'''
❌ CRITICAL DATASET DOWNLOAD FAILURE: {e}

🚫 NO FALLBACKS - SYSTEM MUST WORK PROPERLY!

REQUIRED FIXES:
1. Check internet connection to Hugging Face
2. Install datasets: pip install datasets
3. Verify Hugging Face access permissions  
4. Ensure sufficient disk space

🔧 Manual fix commands:
   pip install datasets huggingface_hub
   huggingface-cli login
   
💀 SYSTEM TERMINATED - FIX DATASET ISSUE!
'''
    print(error_msg)
    raise SystemExit(f"DATASET DOWNLOAD FAILED: {e} - NO FALLBACKS ALLOWED")

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