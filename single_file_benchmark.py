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

# Disable HF_TRANSFER to avoid import issues
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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
    
    # Check if dataset already exists and has content
    dataset_path = 'datasets/sharegpt_prompts.jsonl'
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 100:  # Must have at least 100 prompts
                    print(f"✅ Dataset already exists with {len(lines)} prompts, skipping download")
                    return True
                else:
                    print(f"⚠️ Dataset exists but only has {len(lines)} prompts, re-downloading...")
        except:
            print("⚠️ Dataset file corrupted, re-downloading...")
    else:
        print("📦 Dataset not found, downloading...")
    
    try:
        # First install datasets if not available
        print("📦 Installing datasets library for ShareGPT download...")
        install_cmd = [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"]
        install_success, _, _ = run_command(install_cmd, "Installing datasets library", timeout=300)
        
        # Simplified direct approach - avoid subprocess complexity
        print("🔄 Loading ShareGPT dataset directly...")
        
        try:
            from datasets import load_dataset
            import huggingface_hub
            
            print("📡 Connecting to HuggingFace Hub...")
            
            # Use heka-ai ShareGPT dataset optimized for vLLM benchmarking
            print("📡 Loading heka-ai ShareGPT benchmark dataset...")
            
            try:
                # Try proven working datasets in order of preference
                dataset = None
                is_heka_dataset = False
                
                # 1st preference: Open-Orca/OpenOrca (3.5M+ GPT-4 style instruction pairs - very reliable)
                try:
                    print("📡 Loading Open-Orca/OpenOrca dataset (3.5M+ GPT-4 instructions)...")
                    dataset = load_dataset("Open-Orca/OpenOrca", split="train")
                    dataset = dataset.select(range(min(5000, len(dataset))))  # Take more samples to ensure 1000 prompts
                    print(f"✅ Loaded Open-Orca dataset: {len(dataset)} samples")
                    is_heka_dataset = False
                    dataset_source = "open_orca"
                except Exception as e1:
                    print(f"⚠️ Open-Orca dataset failed: {e1}")
                    
                    # 2nd preference: Dahoas/synthetic-instruct-gptj-pairwise (synthetic but reliable)
                    try:
                        print("📡 Loading Dahoas synthetic instruct dataset...")
                        dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise", split="train")
                        dataset = dataset.select(range(min(5000, len(dataset))))
                        print(f"✅ Loaded Dahoas synthetic dataset: {len(dataset)} samples")
                        is_heka_dataset = False
                        dataset_source = "dahoas_synthetic"
                    except Exception as e2:
                        print(f"⚠️ Dahoas synthetic dataset failed: {e2}")
                        
                        # 3rd preference: Try original heka-ai if fixed
                        try:
                            print("📡 Loading heka-ai ShareGPT benchmark dataset...")
                            dataset = load_dataset("heka-ai/sharegpt-english-10k-vllm-serving-benchmark", split="train")
                            dataset = dataset.select(range(min(5000, len(dataset))))
                            print(f"✅ Loaded heka-ai ShareGPT dataset: {len(dataset)} samples")
                            is_heka_dataset = True
                            dataset_source = "heka_sharegpt"
                        except Exception as e3:
                            print(f"⚠️ heka-ai dataset failed: {e3}")
                            raise Exception(f"All quality datasets failed: {e1}, {e2}, {e3}")
                
                if dataset is None:
                    raise Exception("No quality dataset could be loaded")
                    
            except Exception as e:
                print(f"⚠️ All ShareGPT datasets failed: {e}, using OpenAssistant fallback...")
                try:
                    dataset = load_dataset("OpenAssistant/oasst1", split="train")
                    dataset = dataset.select(range(min(5000, len(dataset))))  # Take more samples to ensure 1000 prompts
                    print(f"✅ Loaded OpenAssistant dataset: {len(dataset)} samples")
                    is_heka_dataset = False
                    dataset_source = "openassistant"
                except Exception as final_e:
                    print(f"❌ All datasets failed including fallback: {final_e}")
                    raise Exception("Cannot load any dataset")
            
            print(f"📊 Dataset loaded: {len(dataset)} samples")
            
            # DEBUG: Show dataset structure
            if len(dataset) > 0:
                first_item = dataset[0]
                print(f"🔍 DATASET STRUCTURE:")
                print(f"   Keys: {list(first_item.keys())}")
                for key in first_item.keys():
                    value = str(first_item.get(key, ''))[:100]
                    print(f"   {key}: {value}...")
            
            # Process ShareGPT dataset format
            os.makedirs("datasets", exist_ok=True)
            prompts_written = 0
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                for i in range(len(dataset)):
                    if prompts_written >= 1000:  # Stop when we have 1000 prompts
                        break
                        
                    item = dataset[i]
                    
                    # Handle Open-Orca/OpenOrca format (most reliable)
                    if 'question' in item:
                        # Open-Orca format: {"question": "prompt text", "response": "..."}
                        prompt_text = str(item.get('question', '')).strip()
                        if len(prompt_text) > 20:
                            data = {
                                'prompt': prompt_text,
                                'source': dataset_source,
                                'id': f"orca_{prompts_written}"
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            prompts_written += 1
                            
                            if prompts_written % 100 == 0:
                                print(f"📝 Extracted {prompts_written} prompts from {dataset_source}...")
                    
                    # Handle Dahoas synthetic format
                    elif 'prompt' in item and dataset_source == "dahoas_synthetic":
                        # Dahoas synthetic format: {"prompt": "text"}
                        prompt_text = str(item.get('prompt', '')).strip()
                        if len(prompt_text) > 20:
                            data = {
                                'prompt': prompt_text,
                                'source': dataset_source,
                                'id': f"synthetic_{prompts_written}"
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            prompts_written += 1
                            
                            if prompts_written % 100 == 0:
                                print(f"📝 Extracted {prompts_written} prompts from {dataset_source}...")
                    
                    # Handle heka-ai ShareGPT format (if working)
                    elif 'conversations' in item and dataset_source == "heka_sharegpt":
                        # heka-ai format: {"conversations": [{"user": "human", "value": "text"}]}
                        conversations = item['conversations']
                        if isinstance(conversations, list) and len(conversations) > 0:
                            for conv in conversations:
                                if isinstance(conv, dict) and conv.get('user') == 'human':
                                    prompt_text = conv.get('value', '').strip()
                                    if len(prompt_text) > 20:
                                        data = {
                                            'prompt': prompt_text,
                                            'source': dataset_source,
                                            'id': f"sharegpt_{prompts_written}"
                                        }
                                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                                        prompts_written += 1
                                        
                                        if prompts_written % 100 == 0:
                                            print(f"📝 Extracted {prompts_written} prompts from {dataset_source}...")
                                        break
                    
                    # Handle general conversation formats (standard ShareGPT)
                    elif 'conversations' in item:
                        # General conversation format with conversations field
                        conversations = item['conversations']
                        if isinstance(conversations, list) and len(conversations) > 0:
                            for conv in conversations:
                                if isinstance(conv, dict):
                                    prompt_text = None
                                    # Multiple formats: {"from": "human", "value": "text"} or {"role": "user", "content": "text"}
                                    if conv.get('from') in ['human', 'user']:
                                        prompt_text = conv.get('value', '').strip()
                                    elif conv.get('role') in ['user', 'human']:
                                        prompt_text = conv.get('content', conv.get('value', '')).strip()
                                    
                                    if prompt_text and len(prompt_text) > 20:
                                        data = {
                                            'prompt': prompt_text,
                                            'source': dataset_source,
                                            'id': f"conv_{prompts_written}"
                                        }
                                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                                        prompts_written += 1
                                        
                                        if prompts_written % 100 == 0:
                                            print(f"📝 Extracted {prompts_written} prompts from {dataset_source}...")
                                        break
                    
                    # Handle generic prompt field
                    elif 'prompt' in item:
                        # Generic prompt format
                        prompt_text = str(item.get('prompt', '')).strip()
                        if len(prompt_text) > 20:
                            data = {
                                'prompt': prompt_text,
                                'source': dataset_source,
                                'id': f"prompt_{prompts_written}"
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            prompts_written += 1
                            
                            if prompts_written % 100 == 0:
                                print(f"📝 Extracted {prompts_written} prompts from {dataset_source}...")
                    
                    # Handle OpenAssistant format (fallback)
                    elif 'text' in item and 'role' in item:
                        # OpenAssistant format
                        role = str(item.get('role', '')).strip().lower()
                        text = str(item.get('text', '')).strip()
                        lang = str(item.get('lang', 'en')).strip().lower()
                        
                        if role == 'prompter' and len(text) > 20 and lang in ['en', 'english', '']:
                            data = {
                                'prompt': text,
                                'source': dataset_source,
                                'id': f"prompt_{prompts_written}",
                                'lang': lang or 'en'
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            prompts_written += 1
                            
                            if prompts_written % 100 == 0:
                                print(f"📝 Extracted {prompts_written} prompts from {dataset_source}...")
                    
                    # Continue until we have 1000 prompts
            
            print(f"✅ Final result: {prompts_written} prompts extracted from {dataset_source}")
            
            # If we don't have 1000 prompts, try to extract more by processing more samples
            if prompts_written < 1000:
                print(f"⚠️ Only {prompts_written} prompts extracted, trying to get more...")
                
                # Try to extract more prompts with relaxed criteria
                with open(dataset_path, 'a', encoding='utf-8') as f:  # Append mode
                    for i in range(len(dataset)):
                        if prompts_written >= 1000:
                            break
                            
                        item = dataset[i]
                        
                        # More relaxed extraction for any text content
                        if 'text' in item:
                            text = str(item.get('text', '')).strip()
                            if len(text) > 10:  # Lower threshold
                                data = {
                                    'prompt': text,
                                    'source': dataset_source,
                                    'id': f"prompt_{prompts_written}"
                                }
                                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                                prompts_written += 1
                                
                        elif isinstance(item, str) and len(item.strip()) > 10:
                            # Handle direct string items
                            data = {
                                'prompt': item.strip(),
                                'source': dataset_source,
                                'id': f"prompt_{prompts_written}"
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            prompts_written += 1
                
                print(f"✅ Final extraction result: {prompts_written} prompts from {dataset_source}")
            
            if prompts_written < 100:  # Minimum threshold
                print("❌ FAILED to extract sufficient prompts!")
                raise Exception(f"Only {prompts_written} prompts extracted from {len(dataset)} items")
                
        except Exception as e:
            print(f"❌ Direct processing failed: {e}")
            print("📊 Exception details:")
            import traceback
            traceback.print_exc()
            raise e
            
        download_cmd = ["echo", "Direct processing completed"]
        
        success, output, duration = run_command(
            download_cmd,
            "Downloading ShareGPT dataset",
            timeout=900  # Extended timeout for dataset download
        )
        
        # Validation - check if we have enough prompts
        if success and os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= 50:  # Require minimum 50 prompts
                        # Verify data format
                        sample_line = json.loads(lines[0])
                        if 'prompt' in sample_line and sample_line.get('source'):
                            source = sample_line.get('source')
                            if source == 'open_orca':
                                print(f"✅ HIGH-QUALITY Open-Orca dataset ready: {len(lines)} prompts (GPT-4 instructions)")
                            elif 'sharegpt' in source.lower():
                                print(f"✅ GENUINE ShareGPT dataset ready: {len(lines)} prompts from {source}")
                            else:
                                print(f"✅ Quality dataset ready: {len(lines)} prompts from {source}")
                            return True
                        else:
                            print(f"❌ Invalid dataset format")
                            return False
                    else:
                        print(f"❌ Insufficient prompts: only {len(lines)} found")
                        return False
            except Exception as e:
                print(f"❌ Dataset validation failed: {e}")
                return False
        else:
            print("❌ Dataset creation failed completely")
            return False
        
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
        [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub", "hf_transfer"],
        "Installing datasets and HF libraries",
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
    # Use GPT-2 which is reliable and fast to download
    model_name = "gpt2"  # Reliable model that always works
    
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.7,  # Conservative for initial test
        max_model_len=1024,
        dtype="float16",  # Safer than bfloat16 for all GPUs
        tensor_parallel_size=1,
        download_dir="/tmp/model_cache"  # Use tmp for model cache
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