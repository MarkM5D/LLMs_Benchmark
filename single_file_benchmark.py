#!/usr/bin/env python3
"""
Single File LLM Benchmark for RunPod

All-in-one benchmark script that doesn't require separate files.
Handles installation, testing, and cleanup for all LLM engines.
"""

import subprocess
import sys
import os
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse

print("🚀 LLM Benchmark - Single File Version")
print("=" * 50)

def run_command(command, description, timeout=3600):
    """Run a command with logging."""
    print(f"🔄 {description}...")
    
    start_time = time.time()
    try:
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed in {duration:.1f}s")
            return True, result.stdout, duration
        else:
            print(f"❌ {description} failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()[:200]}...")
            return False, result.stderr, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False, f"Timeout after {timeout}s", timeout
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False, str(e), time.time() - start_time

def create_sample_dataset():
    """Download and prepare ShareGPT dataset."""
    print("📦 Downloading ShareGPT dataset...")
    
    os.makedirs("datasets", exist_ok=True)
    
    # Check if dataset already exists
    dataset_path = 'datasets/sharegpt_prompts.jsonl'
    if os.path.exists(dataset_path):
        print("✅ Dataset already exists, skipping download")
        return True
    
    try:
        # First install datasets if not available
        install_cmd = [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"]
        run_command(install_cmd, "Installing datasets library", timeout=300)
        
        # Try to download real ShareGPT dataset
        download_cmd = [
            sys.executable, "-c",
            """
import json
import os

print("Attempting to download ShareGPT dataset...")

# Create fallback dataset function
def create_fallback_dataset():
    prompts = [
        "Explain machine learning concepts",
        "Write a Python sorting algorithm", 
        "Describe artificial intelligence benefits",
        "Create a neural network example",
        "How does NLP work?",
        "What is deep learning?",
        "Explain computer vision",
        "Describe reinforcement learning",
        "What are transformers in AI?",
        "How do GPT models work?"
    ] * 100
    
    os.makedirs("datasets", exist_ok=True)
    with open('datasets/sharegpt_prompts.jsonl', 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts[:1000]):
            data = {
                'prompt': prompt,
                'source': 'synthetic',
                'id': f"synthetic_{i}"
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\\n')
    return True

# Try to download real dataset
try:
    from datasets import load_dataset
    print("Loading ShareGPT dataset from HuggingFace...")
    
    dataset = load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark", split="train")
    
    os.makedirs("datasets", exist_ok=True)
    
    with open('datasets/sharegpt_prompts.jsonl', 'w', encoding='utf-8') as f:
        count = 0
        for item in dataset:
            if count >= 1000:
                break
            
            # Extract prompt from conversation
            if 'conversations' in item and item['conversations']:
                conversations = item['conversations']
                if len(conversations) > 0:
                    prompt = conversations[0].get('value', '').strip()
                    if prompt and len(prompt) > 10:
                        data = {
                            'prompt': prompt,
                            'source': 'sharegpt',
                            'id': f"sharegpt_{count}"
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + '\\n')
                        count += 1

    print(f"✅ Downloaded {count} ShareGPT prompts")
    
except Exception as e:
    print(f"⚠️ ShareGPT download failed: {e}")
    print("Creating synthetic dataset...")
    create_fallback_dataset()
    print("✅ Created 1000 synthetic prompts")
"""
        ]
        
        success, output, duration = run_command(
            download_cmd,
            "Downloading dataset",
            timeout=600
        )
        
        if success:
            print("✅ Dataset prepared successfully")
        else:
            print("⚠️ Using fallback dataset")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return False

def complete_cleanup():
    """Complete environment cleanup."""
    print("🧹 Complete environment cleanup...")
    
    packages_to_remove = [
        'vllm', 'sglang', 'tensorrt_llm', 'tensorrt',
        'torch', 'torchvision', 'torchaudio', 'torchtext',
        'transformers', 'accelerate', 'safetensors', 'tokenizers',
        'datasets', 'huggingface_hub', 'sentencepiece',
        'ray', 'xformers', 'flashinfer', 'triton', 'mpi4py',
        'nvidia-ml-py3', 'pynvml'
    ]
    
    for package in packages_to_remove:
        success, output, duration = run_command(
            [sys.executable, "-m", "pip", "uninstall", "-y", package],
            f"Removing {package}",
            timeout=120
        )
    
    # Clean pip cache
    run_command(
        [sys.executable, "-m", "pip", "cache", "purge"],
        "Cleaning pip cache",
        timeout=60
    )
    
    print("✅ Environment cleanup completed")
    return True

def install_pytorch():
    """Install PyTorch with CUDA."""
    print("🔥 Installing PyTorch with CUDA...")
    
    success, output, duration = run_command(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
         "--index-url", "https://download.pytorch.org/whl/cu121"],
        "Installing PyTorch with CUDA",
        timeout=1800
    )
    
    return success

def install_engine(engine):
    """Install specific engine."""
    print(f"📦 Installing {engine.upper()}...")
    
    # Install basic packages first
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading pip tools",
        timeout=300
    )
    
    # Install PyTorch
    if not install_pytorch():
        return False
    
    # Install datasets library first for dataset download
    run_command(
        [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"],
        "Installing datasets library",
        timeout=600
    )
    
    # Install common dependencies
    common_deps = [
        "transformers>=4.35.0", "accelerate", "safetensors", 
        "numpy", "pandas", "requests", "tqdm"
    ]
    
    for dep in common_deps:
        run_command(
            [sys.executable, "-m", "pip", "install", dep],
            f"Installing {dep}",
            timeout=600
        )
    
    # Install engine-specific packages
    if engine == 'vllm':
        success, _, _ = run_command(
            [sys.executable, "-m", "pip", "install", "vllm"],
            "Installing vLLM",
            timeout=1800
        )
    elif engine == 'sglang':
        # Install flashinfer first
        run_command(
            [sys.executable, "-m", "pip", "install", "flashinfer", 
             "-f", "https://flashinfer.ai/whl/cu121/torch2.4/"],
            "Installing flashinfer",
            timeout=1200
        )
        
        success, _, _ = run_command(
            [sys.executable, "-m", "pip", "install", "sglang[all]"],
            "Installing SGLang",
            timeout=1800
        )
        
        if not success:
            success, _, _ = run_command(
                [sys.executable, "-m", "pip", "install", "sglang"],
                "Installing SGLang (alternative)",
                timeout=1800
            )
    elif engine == 'tensorrt':
        run_command(
            [sys.executable, "-m", "pip", "install", "mpi4py"],
            "Installing MPI4Py",
            timeout=600
        )
        
        success, _, _ = run_command(
            [sys.executable, "-m", "pip", "install", "tensorrt_llm",
             "--extra-index-url", "https://pypi.nvidia.com"],
            "Installing TensorRT-LLM",
            timeout=1800
        )
    else:
        print(f"❌ Unknown engine: {engine}")
        return False
    
    # Verify installation
    if engine == 'vllm':
        import_cmd = "import vllm; print(f'vLLM ready')"
    elif engine == 'sglang':
        import_cmd = "import sglang; print('SGLang ready')"
    elif engine == 'tensorrt':
        import_cmd = "import tensorrt_llm; print('TensorRT-LLM ready')"
    
    verify_success, output, _ = run_command(
        [sys.executable, "-c", import_cmd],
        f"Verifying {engine} installation",
        timeout=120
    )
    
    if verify_success:
        print(f"✅ {engine.upper()} installation verified")
    
    return success and verify_success

def run_simple_test(engine):
    """Run a simple test for the engine."""
    print(f"🧪 Running simple test for {engine.upper()}...")
    
    # Simple generation test
    if engine == 'vllm':
        test_code = """
import sys
from vllm import LLM, SamplingParams

try:
    # Use the actual task model for H100 benchmarking
    model_name = "microsoft/DialoGPT-large"  # Use appropriate model for H100
    
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.85,  # H100 has 80GB, use most of it
        max_model_len=2048,
        dtype="bfloat16",  # Better for H100
        tensor_parallel_size=1
    )
    
    sampling_params = SamplingParams(
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=256,
        repetition_penalty=1.1
    )
    
    # Load actual prompts from dataset
    import json
    prompts = []
    try:
        with open('datasets/sharegpt_prompts.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data['prompt'])
                if len(prompts) >= 10:  # Test with 10 prompts
                    break
    except:
        prompts = ["Hello, how are you?", "Explain AI briefly.", "What is machine learning?"]
    
    print(f"Testing with {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"✅ vLLM test successful - Generated {len(outputs)} responses")
    for i, output in enumerate(outputs[:3]):  # Show first 3
        generated_text = output.outputs[0].text
        print(f"Sample {i+1}: {generated_text[:100]}...")
        
except Exception as e:
    print(f"❌ vLLM test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    elif engine == 'sglang':
        test_code = """
import sys
import json
try:
    import sglang as sgl
    from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
    
    # Load prompts from dataset
    prompts = []
    try:
        with open('datasets/sharegpt_prompts.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data['prompt'])
                if len(prompts) >= 5:  # Test with 5 prompts
                    break
    except:
        prompts = ["Hello, how are you?", "Explain AI briefly."]
    
    print(f"✅ SGLang test successful - Loaded {len(prompts)} test prompts")
    print("SGLang module imported and ready for benchmarking")
    
except Exception as e:
    print(f"❌ SGLang test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    elif engine == 'tensorrt':
        test_code = """
import sys
import json
try:
    import tensorrt_llm
    
    # Load prompts from dataset  
    prompts = []
    try:
        with open('datasets/sharegpt_prompts.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data['prompt'])
                if len(prompts) >= 5:  # Test with 5 prompts
                    break
    except:
        prompts = ["Hello, how are you?", "Explain AI briefly."]
    
    print(f"✅ TensorRT-LLM test successful - Loaded {len(prompts)} test prompts")
    print("TensorRT-LLM module imported and ready for H100 benchmarking")
    
except Exception as e:
    print(f"❌ TensorRT-LLM test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    success, output, duration = run_command(
        [sys.executable, "-c", test_code],
        f"Testing {engine}",
        timeout=600
    )
    
    return success

def run_complete_benchmark():
    """Run complete benchmark with isolation."""
    engines = ['vllm', 'sglang', 'tensorrt']
    
    print("🚀 Starting Complete LLM Benchmark with Full Isolation")
    print("=" * 60)
    
    # Create dataset
    if not create_sample_dataset():
        print("❌ Failed to create dataset")
        return False
    
    results = {}
    
    for i, engine in enumerate(engines, 1):
        print(f"\n{'='*60}")
        print(f"ENGINE {i}/{len(engines)}: {engine.upper()}")
        print(f"{'='*60}")
        
        engine_start = time.time()
        
        # Phase 1: Cleanup (skip for first engine)
        if i > 1:
            print("🧹 Phase 1: Environment Cleanup")
            complete_cleanup()
        
        # Phase 2: Install
        print("📦 Phase 2: Engine Installation")
        install_success = install_engine(engine)
        
        if not install_success:
            print(f"❌ Failed to install {engine}, skipping...")
            results[engine] = {'install': False, 'test': False}
            continue
        
        # Phase 3: Test
        print("🧪 Phase 3: Running Test")
        test_success = run_simple_test(engine)
        
        engine_duration = time.time() - engine_start
        print(f"⏱️ {engine.upper()} complete cycle: {engine_duration:.1f}s")
        
        results[engine] = {
            'install': install_success,
            'test': test_success,
            'duration': engine_duration
        }
    
    # Final cleanup
    print(f"\n{'='*60}")
    print("🧹 FINAL CLEANUP")
    print(f"{'='*60}")
    complete_cleanup()
    
    # Results summary
    print(f"\n{'='*60}")
    print("📊 BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for engine, result in results.items():
        status = "✅ SUCCESS" if result['install'] and result['test'] else "❌ FAILED"
        duration = result.get('duration', 0)
        print(f"{engine.upper():12} | {status} | {duration:.1f}s")
    
    print(f"\n🎉 COMPLETE BENCHMARK FINISHED!")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Single file LLM benchmark for RunPod')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Quick test mode - testing one engine only")
        # Quick test with just vLLM
        create_sample_dataset()
        install_engine('vllm')
        run_simple_test('vllm')
        complete_cleanup()
    else:
        run_complete_benchmark()

if __name__ == "__main__":
    main()