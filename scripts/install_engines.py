#!/usr/bin/env python3
"""
LLM Engine Installation Manager

Modular installation system for LLM inference engines.
Each engine can be installed/uninstalled independently to avoid version conflicts.

Supported engines:
- vLLM: PagedAttention-based serving
- SGLang: RadixAttention with structured generation  
- TensorRT-LLM: NVIDIA optimized inference

Usage:
    python scripts/install_engines.py --engine vllm
    python scripts/install_engines.py --engine sglang
    python scripts/install_engines.py --engine tensorrt
    python scripts/install_engines.py --engine all
"""

import argparse
import subprocess
import sys
import os
import platform
from pathlib import Path
import json
import time

class EngineInstaller:
    """Manages installation of LLM inference engines."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.install_log = []
        
    def _get_system_info(self):
        """Get system information for installation decisions."""
        try:
            # Check CUDA availability
            cuda_available = False
            cuda_version = "unknown"
            
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    cuda_available = True
                    # Extract CUDA version from nvidia-smi output
                    for line in result.stdout.split('\n'):
                        if 'CUDA Version:' in line:
                            cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                            break
            except FileNotFoundError:
                pass
            
            return {
                'platform': platform.system(),
                'python_version': sys.version,
                'cuda_available': cuda_available,
                'cuda_version': cuda_version,
                'arch': platform.machine()
            }
        except Exception as e:
            print(f"⚠️  Warning: Could not detect system info: {e}")
            return {}
    
    def _run_command(self, command, description="Running command", timeout=1800):
        """Run a shell command with logging."""
        print(f"🔄 {description}...")
        print(f"   Command: {' '.join(command) if isinstance(command, list) else command}")
        
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
                self.install_log.append({
                    'command': ' '.join(command),
                    'status': 'success',
                    'duration': duration,
                    'description': description
                })
                return True, result.stdout
            else:
                print(f"❌ {description} failed (exit code: {result.returncode})")
                print(f"   Error: {result.stderr}")
                self.install_log.append({
                    'command': ' '.join(command),
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr,
                    'description': description
                })
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out after {timeout}s")
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            print(f"❌ {description} failed with exception: {e}")
            return False, str(e)
    
    def _check_pytorch(self):
        """Check if PyTorch is installed with CUDA support."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            torch_version = torch.__version__
            print(f"✅ PyTorch {torch_version} found (CUDA: {'Yes' if cuda_available else 'No'})")
            return True, cuda_available
        except ImportError:
            print("❌ PyTorch not found")
            return False, False
    
    def install_pytorch(self):
        """Install PyTorch with CUDA support."""
        print("🔧 Installing PyTorch with CUDA support...")
        
        if self.system_info.get('cuda_available'):
            # Install PyTorch with CUDA 12.1 support (common for H100)
            commands = [
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                 "--index-url", "https://download.pytorch.org/whl/cu121"]
            ]
        else:
            # Install CPU-only version
            print("⚠️  No CUDA detected, installing CPU-only PyTorch")
            commands = [
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                 "--index-url", "https://download.pytorch.org/whl/cpu"]
            ]
        
        for cmd in commands:
            success, output = self._run_command(cmd, f"Installing {cmd[3] if len(cmd) > 3 else 'packages'}")
            if not success:
                return False
        
        return True
    
    def install_vllm(self):
        """Install vLLM engine."""
        print("\n🚀 Installing vLLM (PagedAttention)...")
        
        # Check PyTorch first
        torch_ok, cuda_ok = self._check_pytorch()
        if not torch_ok:
            print("📦 Installing PyTorch first...")
            if not self.install_pytorch():
                return False
        
        # Install vLLM
        if self.system_info.get('cuda_available'):
            cmd = [sys.executable, "-m", "pip", "install", "vllm"]
        else:
            print("⚠️  Installing vLLM CPU version (limited performance)")
            cmd = [sys.executable, "-m", "pip", "install", "vllm", "--extra-index-url", 
                   "https://download.pytorch.org/whl/cpu"]
        
        success, output = self._run_command(cmd, "Installing vLLM", timeout=3600)
        
        if success:
            # Verify installation
            verify_cmd = [sys.executable, "-c", "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"]
            success, output = self._run_command(verify_cmd, "Verifying vLLM installation")
            if success:
                print("✅ vLLM installation verified")
        
        return success
    
    def install_sglang(self):
        """Install SGLang engine."""
        print("\n🚀 Installing SGLang (RadixAttention + Structured Generation)...")
        
        # Check PyTorch first
        torch_ok, cuda_ok = self._check_pytorch()
        if not torch_ok:
            print("📦 Installing PyTorch first...")
            if not self.install_pytorch():
                return False
        
        # Install SGLang with dependencies
        commands = [
            # Install flashinfer first (SGLang dependency)
            [sys.executable, "-m", "pip", "install", "flashinfer", "-f", 
             "https://flashinfer.ai/whl/cu121/torch2.4/"],
            # Install SGLang
            [sys.executable, "-m", "pip", "install", "sglang[all]"]
        ]
        
        for cmd in commands:
            success, output = self._run_command(cmd, f"Installing {cmd[3]}", timeout=3600)
            if not success:
                # Try alternative installation for SGLang
                if "sglang" in cmd[3]:
                    print("🔄 Trying alternative SGLang installation...")
                    alt_cmd = [sys.executable, "-m", "pip", "install", "sglang"]
                    success, output = self._run_command(alt_cmd, "Installing SGLang (alternative)", timeout=3600)
                    if not success:
                        return False
        
        # Verify installation
        verify_cmd = [sys.executable, "-c", "import sglang; print('SGLang installed successfully')"]
        success, output = self._run_command(verify_cmd, "Verifying SGLang installation")
        if success:
            print("✅ SGLang installation verified")
        
        return success
    
    def install_tensorrt_llm(self):
        """Install TensorRT-LLM engine."""
        print("\n🚀 Installing TensorRT-LLM (NVIDIA Optimized)...")
        
        if not self.system_info.get('cuda_available'):
            print("❌ TensorRT-LLM requires CUDA. No CUDA detected on this system.")
            return False
        
        # Check PyTorch first
        torch_ok, cuda_ok = self._check_pytorch()
        if not torch_ok:
            print("📦 Installing PyTorch first...")
            if not self.install_pytorch():
                return False
        
        # Install TensorRT-LLM
        commands = [
            # Install TensorRT-LLM from NVIDIA index
            [sys.executable, "-m", "pip", "install", "tensorrt_llm", 
             "--extra-index-url", "https://pypi.nvidia.com"],
            # Install additional dependencies
            [sys.executable, "-m", "pip", "install", "mpi4py"]
        ]
        
        for cmd in commands:
            success, output = self._run_command(cmd, f"Installing {cmd[3]}", timeout=3600)
            if not success:
                # TensorRT-LLM can be tricky, try alternatives
                if "tensorrt_llm" in cmd[3]:
                    print("🔄 Trying alternative TensorRT-LLM installation...")
                    alt_cmd = [sys.executable, "-m", "pip", "install", "--pre", "tensorrt_llm", 
                              "--extra-index-url", "https://pypi.nvidia.com"]
                    success, output = self._run_command(alt_cmd, "Installing TensorRT-LLM (pre-release)", timeout=3600)
                    if not success:
                        print("⚠️  TensorRT-LLM installation failed. This is common due to version compatibility.")
                        print("    You can continue with vLLM and SGLang benchmarks.")
                        return False
        
        # Verify installation
        verify_cmd = [sys.executable, "-c", "import tensorrt_llm; print('TensorRT-LLM installed successfully')"]
        success, output = self._run_command(verify_cmd, "Verifying TensorRT-LLM installation")
        if success:
            print("✅ TensorRT-LLM installation verified")
        
        return success
    
    def install_common_dependencies(self):
        """Install common dependencies needed by all engines."""
        print("📦 Installing common dependencies...")
        
        dependencies = [
            "transformers>=4.35.0",
            "datasets",
            "accelerate", 
            "safetensors",
            "numpy",
            "pandas",
            "psutil",
            "nvidia-ml-py3",
            "requests",
            "tqdm"
        ]
        
        for dep in dependencies:
            cmd = [sys.executable, "-m", "pip", "install", dep]
            success, output = self._run_command(cmd, f"Installing {dep}")
            if not success:
                print(f"⚠️  Failed to install {dep}, continuing...")
        
        # Optional visualization dependencies
        print("📊 Installing optional visualization dependencies...")
        viz_deps = ["matplotlib", "seaborn"]
        for dep in viz_deps:
            cmd = [sys.executable, "-m", "pip", "install", dep]
            success, output = self._run_command(cmd, f"Installing {dep}")
            if not success:
                print(f"⚠️  Visualization library {dep} not installed (optional)")
    
    def save_install_log(self):
        """Save installation log for debugging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"installation_{int(time.time())}.json"
        
        install_info = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'install_log': self.install_log
        }
        
        with open(log_file, 'w') as f:
            json.dump(install_info, f, indent=2)
        
        print(f"📋 Installation log saved: {log_file}")


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description='Install LLM inference engines')
    parser.add_argument('--engine', choices=['vllm', 'sglang', 'tensorrt', 'all'], 
                       required=True, help='Engine to install')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip common dependency installation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔧 LLM Engine Installation Manager")
    print("=" * 50)
    
    installer = EngineInstaller()
    
    # Show system info
    if args.verbose:
        print(f"🖥️  System Info:")
        for key, value in installer.system_info.items():
            print(f"   {key}: {value}")
        print()
    
    success_count = 0
    total_count = 0
    
    try:
        # Install common dependencies first
        if not args.skip_deps:
            installer.install_common_dependencies()
        
        # Install requested engines
        if args.engine == 'all':
            engines = ['vllm', 'sglang', 'tensorrt']
        else:
            engines = [args.engine]
        
        for engine in engines:
            total_count += 1
            print(f"\n{'='*50}")
            print(f"Installing {engine.upper()}")
            print(f"{'='*50}")
            
            if engine == 'vllm':
                success = installer.install_vllm()
            elif engine == 'sglang':
                success = installer.install_sglang() 
            elif engine == 'tensorrt':
                success = installer.install_tensorrt_llm()
            else:
                print(f"❌ Unknown engine: {engine}")
                continue
            
            if success:
                success_count += 1
                print(f"✅ {engine.upper()} installed successfully!")
            else:
                print(f"❌ {engine.upper()} installation failed!")
        
        # Save installation log
        installer.save_install_log()
        
        # Summary
        print(f"\n{'='*50}")
        print(f"INSTALLATION SUMMARY")
        print(f"{'='*50}")
        print(f"✅ Successful: {success_count}/{total_count}")
        print(f"❌ Failed: {total_count - success_count}/{total_count}")
        
        if success_count > 0:
            print(f"\n🎉 You can now run benchmarks with the installed engines:")
            for engine in engines:
                print(f"   python scripts/run_benchmark.py --engine {engine} --test s1_throughput")
        
        if success_count == total_count:
            print("\n🚀 All engines installed successfully! Ready for benchmarking!")
            return True
        else:
            print(f"\n⚠️  {total_count - success_count} engine(s) failed to install.")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Installation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)