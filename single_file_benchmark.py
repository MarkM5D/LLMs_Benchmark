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
    """Create sample dataset."""
    print("📦 Creating sample dataset...")
    
    os.makedirs("datasets", exist_ok=True)
    
    prompts = []
    base_prompts = [
        "Explain machine learning in simple terms.",
        "Write a Python function for sorting numbers.",
        "What are the benefits of artificial intelligence?",
        "Create a simple neural network example.",
        "How does natural language processing work?"
    ]
    
    # Generate 1000 prompts with variations
    for i in range(200):
        base = base_prompts[i % len(base_prompts)]
        variations = [
            base,
            f"Please explain: {base.lower()}",
            f"Can you help with: {base.lower()}",
            f"I need information about: {base.lower()}",
            f"Provide details on: {base.lower()}"
        ]
        
        for j, variation in enumerate(variations):
            if len(prompts) >= 1000:
                break
            prompts.append({
                'prompt': variation,
                'source': 'sample',
                'id': f"sample_{i}_{j}"
            })
    
    # Save dataset
    with open('datasets/sharegpt_prompts.jsonl', 'w', encoding='utf-8') as f:
        for prompt in prompts[:1000]:
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(prompts)} sample prompts")
    return True

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
    
    # Install common dependencies
    common_deps = [
        "transformers>=4.35.0", "accelerate", "safetensors", 
        "datasets", "numpy", "pandas", "requests", "tqdm"
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
    llm = LLM(model="microsoft/DialoGPT-medium", gpu_memory_utilization=0.3)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    
    prompts = ["Hello, how are you?", "Explain AI briefly."]
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"✅ vLLM test successful - Generated {len(outputs)} responses")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Sample: {generated_text[:50]}...")
        
except Exception as e:
    print(f"❌ vLLM test failed: {e}")
    sys.exit(1)
"""
    
    elif engine == 'sglang':
        test_code = """
import sys
try:
    import sglang as sgl
    from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
    
    print("✅ SGLang test successful - Module imported and basic functions available")
    
except Exception as e:
    print(f"❌ SGLang test failed: {e}")
    sys.exit(1)
"""
    
    elif engine == 'tensorrt':
        test_code = """
import sys
try:
    import tensorrt_llm
    print("✅ TensorRT-LLM test successful - Module imported")
    
except Exception as e:
    print(f"❌ TensorRT-LLM test failed: {e}")
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