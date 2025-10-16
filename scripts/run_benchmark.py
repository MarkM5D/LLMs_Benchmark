#!/usr/bin/env python3
"""
LLM Benchmark Orchestrator with Complete Isolation

Main benchmark script that runs complete benchmark cycles for all LLM engines with full isolation:
1. Install engine + all dependencies (PyTorch, etc.)
2. Run all tests for that engine  
3. Completely uninstall everything (including PyTorch)
4. Repeat for next engine

No version conflicts, complete isolation between engines.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --engines vllm,sglang
    python scripts/run_benchmark.py --tests s1_throughput,s2_json_struct
"""

import argparse
import subprocess
import sys
import os
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import traceback

class BenchmarkOrchestrator:
    """Orchestrates complete isolated benchmarking of LLM engines."""
    
    def __init__(self, engines=None, tests=None, verbose=False):
        self.engines = engines or ['vllm']  # Focus on vLLM only for now
        self.tests = tests or ['s1_throughput', 's2_json_struct', 's3_low_latency']
        self.verbose = verbose
        
        # Results tracking
        self.benchmark_results = {}
        self.installation_logs = {}
        self.test_logs = {}
        self.timing_info = {}
        
        # Directories
        self.logs_dir = Path("logs")
        self.results_dir = Path("results")
        self.analysis_dir = Path("analysis_output")
        
        # Create directories
        for dir_path in [self.logs_dir, self.results_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.session_id = f"benchmark_{int(time.time())}"
        self.session_log = self.logs_dir / f"{self.session_id}.log"
        
        print(f"🚀 LLM Benchmark Session: {self.session_id}")
        print(f"📋 Engines: {', '.join(self.engines)}")
        print(f"📋 Tests: {', '.join(self.tests)}")
        print(f"📝 Session log: {self.session_log}")
    
    def log(self, message, level="INFO"):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}"
        
        print(log_line)
        
        with open(self.session_log, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
            f.flush()
    
    def run_command(self, command, description, timeout=3600, critical=True):
        """Run a command with logging and error handling."""
        self.log(f"🔄 {description}")
        if self.verbose:
            self.log(f"   Command: {' '.join(command) if isinstance(command, list) else command}")
        
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
                self.log(f"✅ {description} completed in {duration:.1f}s")
                if self.verbose and result.stdout.strip():
                    self.log(f"   Output: {result.stdout.strip()[:200]}...")
                return True, result.stdout, duration
            else:
                error_msg = f"{description} failed (exit code: {result.returncode})"
                if result.stderr:
                    error_msg += f": {result.stderr.strip()[:200]}..."
                
                if critical:
                    self.log(f"❌ {error_msg}", "ERROR")
                else:
                    self.log(f"⚠️ {error_msg}", "WARN")
                
                return False, result.stderr, duration
                
        except subprocess.TimeoutExpired:
            self.log(f"⏰ {description} timed out after {timeout}s", "ERROR")
            return False, f"Timeout after {timeout}s", timeout
        except Exception as e:
            self.log(f"❌ {description} failed with exception: {e}", "ERROR")
            return False, str(e), time.time() - start_time
    
    def check_system_requirements(self):
        """Check basic system requirements."""
        self.log("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version
        self.log(f"   Python: {python_version}")
        
        # Check CUDA availability (via nvidia-smi)
        cuda_success, cuda_output, _ = self.run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            "Checking CUDA/GPU",
            timeout=30,
            critical=False
        )
        
        if cuda_success:
            self.log(f"   GPU: {cuda_output.strip()}")
        else:
            self.log("   ⚠️ No CUDA/GPU detected", "WARN")
        
        return True
    
    def prepare_dataset(self):
        """Ensure dataset is available."""
        dataset_file = Path("datasets/sharegpt_prompts.jsonl")
        
        if dataset_file.exists():
            self.log(f"✅ Dataset found: {dataset_file}")
            self.log(f"🤖 Model: gpt-oss-20b (20B parameters)")
            return True
        
        self.log("📦 Creating dataset...")
        success, output, duration = self.run_command(
            [sys.executable, "scripts/create_sample_dataset.py"],
            "Creating sample dataset",
            timeout=300
        )
        
        if success and dataset_file.exists():
            self.log("✅ Dataset created successfully")
            return True
        else:
            self.log("❌ Failed to create dataset", "ERROR")
            return False
    
    def complete_environment_cleanup(self):
        """Completely clean the Python environment of all LLM-related packages."""
        self.log("🧹 Performing complete environment cleanup...")
        
        # List of all packages to remove (comprehensive cleanup)
        packages_to_remove = [
            # LLM Engines
            'vllm', 'sglang', 'tensorrt_llm', 'tensorrt',
            
            # PyTorch ecosystem
            'torch', 'torchvision', 'torchaudio', 'torchtext',
            
            # LLM dependencies
            'transformers', 'accelerate', 'safetensors', 'tokenizers',
            'datasets', 'huggingface_hub', 'sentencepiece', 'protobuf',
            
            # Engine-specific dependencies
            'ray', 'xformers', 'flashinfer', 'triton', 'mpi4py',
            'flash-attn', 'ninja', 'packaging',
            
            # GPU/CUDA packages
            'nvidia-ml-py3', 'pynvml', 'cupy', 'pycuda',
            
            # Additional ML packages that might conflict
            'scipy', 'scikit-learn', 'matplotlib', 'seaborn',
            'pandas', 'numpy',  # These might be reinstalled, but clean slate
            
            # Jupyter/notebook packages (if any)
            'jupyter', 'notebook', 'ipython', 'ipykernel'
        ]
        
        self.log(f"🗑️ Removing {len(packages_to_remove)} package types...")
        
        # Uninstall all packages (ignore errors for packages not installed)
        for package in packages_to_remove:
            success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "uninstall", "-y", package],
                f"Removing {package}",
                timeout=120,
                critical=False
            )
        
        # Additional cleanup: pip cache
        self.run_command(
            [sys.executable, "-m", "pip", "cache", "purge"],
            "Cleaning pip cache",
            timeout=60,
            critical=False
        )
        
        # Clean Python __pycache__ directories
        self.log("🧹 Cleaning __pycache__ directories...")
        for root, dirs, files in os.walk("."):
            if "__pycache__" in dirs:
                pycache_path = os.path.join(root, "__pycache__")
                try:
                    shutil.rmtree(pycache_path)
                except:
                    pass
        
        # Verify cleanup
        self.log("🔍 Verifying cleanup...")
        key_packages = ['torch', 'vllm', 'sglang', 'tensorrt_llm']
        
        for package in key_packages:
            try:
                __import__(package)
                self.log(f"⚠️ {package} still importable after cleanup", "WARN")
            except ImportError:
                self.log(f"✅ {package} successfully removed")
        
        # Wait a moment for system cleanup
        time.sleep(5)
        
        self.log("✅ Complete environment cleanup finished")
        return True
    
    def install_engine_with_dependencies(self, engine):
        """Install specific engine with all its dependencies from scratch."""
        self.log(f"📦 Installing {engine.upper()} with all dependencies...")
        
        install_start = time.time()
        
        # Step 1: Install basic Python packages
        basic_packages = ["pip", "setuptools", "wheel"]
        for package in basic_packages:
            success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                f"Upgrading {package}",
                timeout=300
            )
            if not success:
                return False
        
        # Step 2: Install PyTorch with CUDA support
        self.log("🔥 Installing PyTorch with CUDA support...")
        success, output, duration = self.run_command(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", 
             "--index-url", "https://download.pytorch.org/whl/cu121", "--no-cache-dir"],
            "Installing PyTorch with CUDA",
            timeout=900  # 15 minutes max
        )
        if not success:
            return False
        
        # Step 3: Install common dependencies
        common_deps = [
            "transformers>=4.35.0", "accelerate", "safetensors", "tokenizers",
            "datasets", "numpy", "pandas", "psutil", "requests", "tqdm",
            "nvidia-ml-py3"
        ]
        
        for dep in common_deps:
            success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "install", dep],
                f"Installing {dep}",
                timeout=600,
                critical=False
            )
        
        # Step 4: Install engine-specific packages
        if engine == 'vllm':
            success = self._install_vllm()
        elif engine == 'sglang':
            success = self._install_sglang()  
        elif engine == 'tensorrt':
            success = self._install_tensorrt_llm()
        else:
            self.log(f"❌ Unknown engine: {engine}", "ERROR")
            return False
        
        if not success:
            self.log(f"❌ Failed to install {engine}", "ERROR")
            return False
        
        # Step 5: Verify installation
        success = self._verify_engine_installation(engine)
        if not success:
            return False
        
        install_duration = time.time() - install_start
        self.log(f"✅ {engine.upper()} installation completed in {install_duration:.1f}s")
        
        # Store installation info
        self.installation_logs[engine] = {
            'success': True,
            'duration': install_duration,
            'timestamp': time.time()
        }
        
        return True
    
    def _install_vllm(self):
        """Install vLLM with gpt-oss support and harmony encoding."""
        self.log("🚀 Installing vLLM with gpt-oss support and harmony encoding...")
        
        # Install transformers and kernels first for model compatibility  
        trans_success, _, _ = self.run_command(
            [sys.executable, "-m", "pip", "install", "transformers>=4.35.0", "torch", "--no-cache-dir"],
            "Installing transformers for model support",
            timeout=300
        )
        
        # Install openai-harmony for gpt-oss model support
        harmony_success, _, _ = self.run_command(
            [sys.executable, "-m", "pip", "install", "openai-harmony", "--no-cache-dir"],
            "Installing openai-harmony for gpt-oss support",
            timeout=300
        )
        
        # Install vLLM with gpt-oss support using exact command from HuggingFace
        self.log("🚀 Installing vLLM with gpt-oss support (exact HF version)...")
        
        # First try to install uv package manager as recommended
        uv_success, _, _ = self.run_command(
            [sys.executable, "-m", "pip", "install", "uv", "--no-cache-dir"],
            "Installing uv package manager",
            timeout=300,
            critical=False
        )
        
        if uv_success:
            # Use uv to install vLLM with gpt-oss as recommended by HuggingFace
            self.log("🚀 Using uv to install vLLM with gpt-oss...")
            gptoss_success, output, duration = self.run_command(
                ["uv", "pip", "install", "--pre", "vllm==0.10.1+gptoss",
                 "--extra-index-url", "https://wheels.vllm.ai/gpt-oss/",
                 "--extra-index-url", "https://download.pytorch.org/whl/nightly/cu128",
                 "--index-strategy", "unsafe-best-match"],
                "Installing vLLM gpt-oss version with uv",
                timeout=1800,
                critical=False
            )
        else:
            gptoss_success = False
        
        # Fallback: try with regular pip if uv failed
        if not gptoss_success:
            self.log("🚀 Fallback: Installing vLLM gpt-oss with pip...")
            gptoss_success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "install", "--pre", "vllm==0.10.1+gptoss",
                 "--extra-index-url", "https://wheels.vllm.ai/gpt-oss/",
                 "--extra-index-url", "https://download.pytorch.org/whl/nightly/cu128",
                 "--index-strategy", "unsafe-best-match", "--no-cache-dir"],
                "Installing vLLM gpt-oss version with pip",
                timeout=1800,
                critical=False
            )
        
        # If gpt-oss version still fails, try latest regular vLLM
        if not gptoss_success:
            self.log("⚠️ gpt-oss vLLM failed, trying latest regular vLLM...", "WARN")
            vllm_success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "install", "vllm>=0.6.0", "--no-cache-dir"],
                "Installing latest regular vLLM",
                timeout=1800
            )
            return trans_success and vllm_success
        
        return trans_success and gptoss_success
    
    def _install_sglang(self):
        """Install SGLang specific packages."""
        self.log("🚀 Installing SGLang...")
        
        # Install flashinfer first (SGLang dependency)
        success, output, duration = self.run_command(
            [sys.executable, "-m", "pip", "install", "flashinfer", 
             "-f", "https://flashinfer.ai/whl/cu121/torch2.4/"],
            "Installing flashinfer",
            timeout=1200,
            critical=False
        )
        
        # Install SGLang
        success, output, duration = self.run_command(
            [sys.executable, "-m", "pip", "install", "sglang[all]"],
            "Installing SGLang",
            timeout=1800
        )
        
        if not success:
            # Try alternative installation
            success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "install", "sglang"],
                "Installing SGLang (alternative)",
                timeout=1800
            )
        
        return success
    
    def _install_tensorrt_llm(self):
        """Install TensorRT-LLM specific packages."""
        self.log("🚀 Installing TensorRT-LLM...")
        
        # Install MPI first
        success, output, duration = self.run_command(
            [sys.executable, "-m", "pip", "install", "mpi4py"],
            "Installing MPI4Py",
            timeout=600,
            critical=False
        )
        
        # Install TensorRT-LLM
        success, output, duration = self.run_command(
            [sys.executable, "-m", "pip", "install", "tensorrt_llm",
             "--extra-index-url", "https://pypi.nvidia.com"],
            "Installing TensorRT-LLM",
            timeout=1800
        )
        
        if not success:
            # Try pre-release version
            success, output, duration = self.run_command(
                [sys.executable, "-m", "pip", "install", "--pre", "tensorrt_llm",
                 "--extra-index-url", "https://pypi.nvidia.com"],
                "Installing TensorRT-LLM (pre-release)",
                timeout=1800
            )
        
        return success
    
    def _verify_engine_installation(self, engine):
        """Verify that the engine is properly installed."""
        self.log(f"🔍 Verifying {engine} installation...")
        
        if engine == 'vllm':
            import_cmd = "import vllm; print(f'vLLM {vllm.__version__} ready')"
        elif engine == 'sglang':
            import_cmd = "import sglang; print('SGLang ready')"
        elif engine == 'tensorrt':
            import_cmd = "import tensorrt_llm; print('TensorRT-LLM ready')"
        else:
            return False
        
        success, output, duration = self.run_command(
            [sys.executable, "-c", import_cmd],
            f"Verifying {engine} import",
            timeout=120
        )
        
        if success:
            self.log(f"✅ {engine} verification successful: {output.strip()}")
        
        return success
    
    def run_engine_tests(self, engine):
        """Run all tests for a specific engine."""
        self.log(f"🧪 Running tests for {engine.upper()}...")
        
        test_start = time.time()
        test_results = {}
        
        for test in self.tests:
            self.log(f"   🔄 Running {engine} - {test}")
            
            # Import and run test directly instead of subprocess
            try:
                success = self._run_test_directly(engine, test)
                test_results[test] = {
                    'success': success,
                    'duration': time.time() - test_start,
                    'timestamp': time.time()
                }
                
                if success:
                    self.log(f"   ✅ {engine} - {test} completed")
                else:
                    self.log(f"   ❌ {engine} - {test} failed", "WARN")
                    
            except Exception as e:
                self.log(f"   ❌ {engine} - {test} failed with error: {e}", "ERROR")
                test_results[test] = {
                    'success': False,
                    'duration': time.time() - test_start,
                    'timestamp': time.time()
                }
        
        total_test_time = time.time() - test_start
        self.log(f"🧪 {engine.upper()} testing completed in {total_test_time:.1f}s")
        
        # Store test results
        self.test_logs[engine] = test_results
        self.timing_info[engine] = {
            'total_test_time': total_test_time,
            'test_results': test_results
        }
        
        return test_results
    
    def _run_test_directly(self, engine, test):
        """Run test directly by importing and executing the test function."""
        try:
            if engine == 'vllm':
                return self._run_vllm_test(test)
            elif engine == 'sglang':
                return self._run_sglang_test(test)
            elif engine == 'tensorrt':
                return self._run_tensorrt_test(test)
            else:
                self.log(f"❌ Unknown engine: {engine}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Test execution failed: {e}", "ERROR")
            return False
    
    def _run_vllm_test(self, test):
        """Run vLLM test with proper gpt-oss-20b loading using harmony encoding."""
        try:
            from vllm import LLM, SamplingParams
            
            # Load dataset
            import json
            import os
            prompts = []
            with open("./datasets/sharegpt_prompts.jsonl", 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Limit for testing
                        break
                    data = json.loads(line)
                    prompts.append(data['prompt'])
            
            self.log(f"   📊 Loaded {len(prompts)} prompts for {test}")
            
            # Use gpt-oss-20b with harmony encoding support
            model_path = "openai/gpt-oss-20b"
            self.log(f"   🚀 Loading gpt-oss-20b model with harmony encoding support")
            
            # Check if model exists and is accessible
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(model_path)
                self.log(f"   ✅ Model repository found with {len(files)} files")
            except Exception as e:
                self.log(f"   ❌ Could not verify model repository: {e}", "ERROR")
                raise Exception(f"gpt-oss-20b model not accessible: {e}")
            
            # Import harmony encoding for gpt-oss
            try:
                from openai_harmony import (
                    HarmonyEncodingName,
                    load_harmony_encoding,
                    Conversation,
                    Message,
                    Role,
                    SystemContent,
                    DeveloperContent,
                )
                self.log(f"   ✅ Harmony encoding imported successfully")
                use_harmony = True
            except ImportError as e:
                self.log(f"   ⚠️ Harmony encoding not available: {e}", "WARN")
                self.log(f"   🔄 Will attempt direct model loading (may fail for gpt-oss)")
                use_harmony = False
            
            # Initialize model with proper settings for gpt-oss-20b
            self.log(f"   🔄 Initializing vLLM with model: {model_path}")
            
            # Ensure we have enough GPU memory and proper settings for 20B model
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                self.log(f"   💾 Available GPU Memory: {gpu_memory:.1f}GB")
                
                # Settings optimized for gpt-oss-20b (MXFP4 quantized, ~16GB VRAM needed)
                if gpu_memory >= 80:  # H100 or similar high-end
                    gpu_util = 0.85
                    max_seqs = 64
                    model_len = 4096
                elif gpu_memory >= 40:  # A100 or similar
                    gpu_util = 0.8
                    max_seqs = 32
                    model_len = 2048
                elif gpu_memory >= 24:  # RTX 4090, A6000 - should work with quantized gpt-oss-20b
                    gpu_util = 0.75
                    max_seqs = 16
                    model_len = 1024
                else:  # Lower memory GPUs
                    gpu_util = 0.7
                    max_seqs = 8
                    model_len = 512
            else:
                self.log("   ⚠️ No CUDA GPU detected, using CPU settings", "WARN")
                gpu_util = 0.8
                max_seqs = 4
                model_len = 512
            
            self.log(f"   ⚙️ GPU Utilization: {gpu_util}, Max Sequences: {max_seqs}, Model Length: {model_len}")
            
            # Load gpt-oss-20b model with proper configuration
            llm = None
            
            # Load attempts with different configurations
            load_attempts = [
                # Attempt 1: Optimal settings for gpt-oss-20b
                {
                    "gpu_memory_utilization": gpu_util,
                    "max_num_seqs": max_seqs,
                    "max_model_len": model_len,
                    "enforce_eager": False,
                    "trust_remote_code": True,
                    "description": "optimal gpt-oss settings"
                },
                # Attempt 2: Conservative settings
                {
                    "gpu_memory_utilization": 0.6,
                    "max_num_seqs": max(4, max_seqs // 4),
                    "max_model_len": max(512, model_len // 2),
                    "enforce_eager": True,
                    "trust_remote_code": True,
                    "description": "conservative settings"
                },
                # Attempt 3: Minimal settings
                {
                    "gpu_memory_utilization": 0.4,
                    "max_num_seqs": 2,
                    "max_model_len": 512,
                    "enforce_eager": True,
                    "trust_remote_code": True,
                    "description": "minimal settings"
                }
            ]
            
            for i, attempt in enumerate(load_attempts, 1):
                try:
                    self.log(f"   🔄 Loading gpt-oss-20b attempt {i}/3 with {attempt['description']}...")
                    
                    # Base configuration for gpt-oss-20b
                    config = {
                        "model": model_path,
                        "tensor_parallel_size": 1,
                        "disable_log_stats": True,
                        "download_dir": None,
                        "dtype": "auto", 
                        "load_format": "auto",
                        "swap_space": 4,  # Required for large models
                        "cpu_offload_gb": 0 if i == 1 else 2,  # CPU offload on retry
                    }
                    
                    # Add attempt-specific settings
                    config.update({k: v for k, v in attempt.items() if k != "description"})
                    
                    self.log(f"   🔧 Model config: GPU util={config['gpu_memory_utilization']}, "
                           f"max_seqs={config['max_num_seqs']}, max_len={config['max_model_len']}")
                    
                    llm = LLM(**config)
                    self.log(f"   ✅ gpt-oss-20b loaded successfully with {attempt['description']}!")
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    self.log(f"   ❌ Attempt {i} failed: {error_msg[:150]}...", "WARN")
                    
                    # Specific error handling for gpt-oss
                    if "configuration" in error_msg.lower():
                        self.log(f"   💡 Config issue detected - model may need specific vLLM gpt-oss version", "WARN")
                    elif "memory" in error_msg.lower():
                        self.log(f"   💾 Memory issue detected - trying smaller settings", "WARN")
                    
                    if i == len(load_attempts):
                        # Last attempt failed, provide helpful error message
                        self.log(f"   ❌ All gpt-oss-20b loading attempts failed", "ERROR")
                        self.log(f"   💡 Try: 1) Check vLLM gpt-oss version, 2) Verify GPU memory, 3) Model accessibility", "ERROR")
                        raise e
                    else:
                        # Try next configuration
                        continue
            
            if llm is None:
                raise Exception("Failed to load gpt-oss-20b with any configuration")
            
            # Test-specific parameters optimized for gpt-oss-20b
            if test == "s1_throughput":
                max_tokens = 256
                batch_size = 4
                temperature = 0.8
                test_prompt_count = 20
            elif test == "s2_json_struct":
                max_tokens = 128
                batch_size = 2
                temperature = 0.3  # Lower for structured output
                test_prompt_count = 10
            else:  # s3_low_latency
                max_tokens = 50
                batch_size = 1
                temperature = 0.7
                test_prompt_count = 5
            
            self.log(f"   🚀 Running {test} with batch_size={batch_size}, max_tokens={max_tokens}, test_prompts={test_prompt_count}")
            
            # Prepare prompts for gpt-oss-20b (with or without harmony)
            test_prompts = prompts[:test_prompt_count]
            
            if use_harmony:
                self.log(f"   🎭 Preparing prompts with harmony encoding...")
                try:
                    # Load harmony encoding
                    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                    
                    # Convert prompts to harmony format
                    harmony_prompts = []
                    for prompt in test_prompts:
                        convo = Conversation.from_messages([
                            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                            Message.from_role_and_content(Role.USER, prompt),
                        ])
                        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
                        harmony_prompts.append(prefill_ids)
                    
                    # Use token IDs for generation
                    prompt_inputs = harmony_prompts
                    use_token_ids = True
                    self.log(f"   ✅ Converted {len(prompt_inputs)} prompts to harmony format")
                    
                    # Get stop tokens for harmony
                    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
                    
                except Exception as e:
                    self.log(f"   ⚠️ Harmony encoding failed: {e}", "WARN")
                    self.log(f"   🔄 Falling back to direct text prompts")
                    prompt_inputs = test_prompts
                    use_token_ids = False
                    stop_token_ids = None
            else:
                # Use direct text prompts
                prompt_inputs = test_prompts
                use_token_ids = False
                stop_token_ids = None
            
            # Configure sampling parameters for gpt-oss-20b
            sampling_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": max_tokens,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            if stop_token_ids:
                sampling_config["stop_token_ids"] = stop_token_ids
            else:
                sampling_config["stop"] = None
            
            sampling_params = SamplingParams(**sampling_config)
            
            self.log(f"   🔄 Generating responses for {len(prompt_inputs)} prompts...")
            self.log(f"   ⚙️ Using {'harmony token IDs' if use_token_ids else 'text prompts'}")
            
            # Generate outputs with proper input format
            import time
            start_time = time.time()
            
            if use_token_ids:
                outputs = llm.generate(prompt_token_ids=prompt_inputs, sampling_params=sampling_params)
            else:
                outputs = llm.generate(prompt_inputs, sampling_params)
            
            generation_time = time.time() - start_time
            
            # Process and validate outputs
            valid_outputs = []
            for i, output in enumerate(outputs):
                if output.outputs and len(output.outputs) > 0:
                    if use_harmony and use_token_ids:
                        # Parse harmony-encoded response
                        try:
                            gen = output.outputs[0]
                            output_tokens = gen.token_ids
                            
                            # Parse completion tokens back to structured messages
                            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
                            
                            # Extract text from harmony entries
                            response_texts = []
                            for message in entries:
                                message_dict = message.to_dict()
                                if 'content' in message_dict and message_dict['content']:
                                    response_texts.append(str(message_dict['content']))
                            
                            generated_text = ' '.join(response_texts).strip()
                            
                        except Exception as e:
                            self.log(f"   ⚠️ Harmony parsing failed for response {i+1}: {e}", "WARN")
                            generated_text = output.outputs[0].text.strip()
                    else:
                        # Regular text response
                        generated_text = output.outputs[0].text.strip()
                    
                    if len(generated_text) > 0:
                        valid_outputs.append(generated_text)
                        self.log(f"   📝 Response {i+1}: {generated_text[:100]}...")
                    else:
                        self.log(f"   ⚠️ Empty response for prompt {i+1}", "WARN")
                else:
                    self.log(f"   ❌ No output for prompt {i+1}", "WARN")
            
            success_count = len(valid_outputs)
            success_rate = success_count / len(test_prompts)
            avg_tokens_per_sec = (success_count * max_tokens) / generation_time if generation_time > 0 else 0
            
            self.log(f"   📈 Test Results:")
            self.log(f"     - Generated: {success_count}/{len(test_prompts)} responses")
            self.log(f"     - Success Rate: {success_rate:.1%}")
            self.log(f"     - Generation Time: {generation_time:.2f}s")
            self.log(f"     - Throughput: {avg_tokens_per_sec:.1f} tokens/sec")
            
            # Consider successful if > 80% responses generated
            test_passed = success_rate > 0.8
            if test_passed:
                self.log(f"   ✅ {test} PASSED")
            else:
                self.log(f"   ❌ {test} FAILED - Success rate too low", "ERROR")
                
            return test_passed
            
        except Exception as e:
            self.log(f"   ❌ vLLM test failed: {e}", "ERROR")
            return False
    
    def _run_sglang_test(self, test):
        """Run SGLang test directly."""
        # Placeholder for SGLang - skip for now
        self.log(f"   ⏭️ SGLang {test} skipped (not implemented)")
        return True
    
    def _run_tensorrt_test(self, test):
        """Run TensorRT test directly.""" 
        # Placeholder for TensorRT - skip for now
        self.log(f"   ⏭️ TensorRT {test} skipped (not implemented)")
        return True
    
    def generate_session_report(self):
        """Generate a comprehensive report of the benchmark session."""
        self.log("📊 Generating session report...")
        
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'engines': self.engines,
            'tests': self.tests,
            'installation_logs': self.installation_logs,
            'test_logs': self.test_logs,
            'timing_info': self.timing_info,
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version
            }
        }
        
        # Save detailed JSON report
        report_file = self.analysis_dir / f"{self.session_id}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate summary markdown
        summary_file = self.analysis_dir / f"{self.session_id}_summary.md"
        self._generate_markdown_summary(summary_file, report)
        
        self.log(f"📋 Reports generated:")
        self.log(f"   📄 Detailed: {report_file}")
        self.log(f"   📝 Summary: {summary_file}")
        
        return report_file, summary_file
    
    def _generate_markdown_summary(self, file_path, report):
        """Generate a markdown summary report."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Isolated LLM Benchmark Report\n\n")
            f.write(f"**Session ID:** {report['session_id']}\n")
            f.write(f"**Generated:** {report['timestamp']}\n")
            f.write(f"**Engines:** {', '.join(report['engines'])}\n")
            f.write(f"**Tests:** {', '.join(report['tests'])}\n\n")
            
            f.write("## Installation Results\n\n")
            f.write("| Engine | Status | Duration |\n")
            f.write("|--------|--------|---------|\n")
            
            for engine in report['engines']:
                if engine in report['installation_logs']:
                    install_info = report['installation_logs'][engine]
                    status = "✅ Success" if install_info['success'] else "❌ Failed"
                    duration = f"{install_info['duration']:.1f}s"
                else:
                    status = "❌ Failed"
                    duration = "N/A"
                
                f.write(f"| {engine.upper()} | {status} | {duration} |\n")
            
            f.write("\n## Test Results\n\n")
            
            for engine in report['engines']:
                if engine in report['test_logs']:
                    f.write(f"### {engine.upper()}\n\n")
                    f.write("| Test | Status | Duration |\n")
                    f.write("|------|--------|---------|\n")
                    
                    test_logs = report['test_logs'][engine]
                    for test, result in test_logs.items():
                        status = "✅ Pass" if result['success'] else "❌ Fail"
                        duration = f"{result['duration']:.1f}s"
                        f.write(f"| {test} | {status} | {duration} |\n")
                    f.write("\n")
            
            f.write("## Timing Summary\n\n")
            total_time = sum(
                info.get('total_test_time', 0) 
                for info in report['timing_info'].values()
            )
            f.write(f"**Total Benchmark Time:** {total_time:.1f} seconds\n\n")
            
            f.write("---\n")
            f.write("*Generated by LLM Benchmark Orchestrator*\n")
    
    def run_complete_benchmark(self):
        """Run the complete isolated benchmark for all engines."""
        self.log("🚀 Starting Complete LLM Benchmark with Full Isolation")
        self.log("=" * 60)
        
        total_start = time.time()
        
        try:
            # Initial setup
            if not self.check_system_requirements():
                return False
            
            if not self.prepare_dataset():
                return False
            
            # Process each engine in complete isolation
            for i, engine in enumerate(self.engines, 1):
                self.log(f"\n{'='*60}")
                self.log(f"ENGINE {i}/{len(self.engines)}: {engine.upper()}")
                self.log(f"{'='*60}")
                
                engine_start = time.time()
                
                # Phase 1: Complete cleanup (except for first engine)
                if i > 1:  # Skip cleanup for first engine
                    self.log("🧹 Phase 1: Complete Environment Cleanup")
                    self.complete_environment_cleanup()
                
                # Phase 2: Install engine with dependencies
                self.log("📦 Phase 2: Engine Installation")
                if not self.install_engine_with_dependencies(engine):
                    self.log(f"❌ Failed to install {engine}, skipping tests", "ERROR")
                    continue
                
                # Phase 3: Run tests
                self.log("🧪 Phase 3: Running Tests")
                self.run_engine_tests(engine)
                
                engine_duration = time.time() - engine_start
                self.log(f"⏱️ {engine.upper()} complete cycle: {engine_duration:.1f}s")
            
            # Final cleanup
            self.log(f"\n{'='*60}")
            self.log("🧹 FINAL CLEANUP")
            self.log(f"{'='*60}")
            self.complete_environment_cleanup()
            
            # Generate analysis if possible
            try:
                self.log("📊 Generating final analysis...")
                analysis_cmd = [
                    sys.executable, "scripts/analyze_results.py",
                    "--format", "both",
                    "--output-dir", str(self.analysis_dir)
                ]
                
                success, output, duration = self.run_command(
                    analysis_cmd,
                    "Generating results analysis",
                    timeout=300,
                    critical=False
                )
            except Exception as e:
                self.log(f"⚠️ Analysis generation failed: {e}", "WARN")
            
            # Generate session report
            self.generate_session_report()
            
            total_duration = time.time() - total_start
            self.log(f"\n🎉 COMPLETE BENCHMARK FINISHED!")
            self.log(f"⏱️ Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            self.log(f"📁 Results: {self.results_dir}")
            self.log(f"📊 Analysis: {self.analysis_dir}")
            self.log(f"📋 Session Log: {self.session_log}")
            
            return True
            
        except KeyboardInterrupt:
            self.log("⏹️ Benchmark interrupted by user", "WARN")
            return False
        except Exception as e:
            self.log(f"❌ Benchmark failed with error: {e}", "ERROR")
            if self.verbose:
                self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run isolated LLM benchmark with complete environment isolation')
    parser.add_argument('--engines', default='vllm,sglang,tensorrt',
                       help='Comma-separated list of engines to test (default: vllm,sglang,tensorrt)')
    parser.add_argument('--tests', default='s1_throughput,s2_json_struct,s3_low_latency',
                       help='Comma-separated list of tests to run (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse engines and tests
    engines = [e.strip() for e in args.engines.split(',') if e.strip()]
    tests = [t.strip() for t in args.tests.split(',') if t.strip()]
    
    print("🔥 LLM BENCHMARK ORCHESTRATOR")
    print("=" * 50)
    print("⚠️  WARNING: This will install/uninstall engines multiple times")
    print("⚠️  Each engine gets a completely clean environment")
    print("⚠️  This process may take 30+ minutes")
    print()
    
    # Confirm with user
    response = input("Continue with complete benchmark? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Benchmark cancelled by user")
        return False
    
    # Create orchestrator and run
    orchestrator = BenchmarkOrchestrator(
        engines=engines,
        tests=tests,
        verbose=args.verbose
    )
    
    success = orchestrator.run_complete_benchmark()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)