#!/usr/bin/env python3
"""
LLM Engine Uninstallation Manager

Clean uninstallation system for LLM inference engines.
Removes engines and their dependencies to prevent version conflicts.

Usage:
    python scripts/uninstall_engines.py --engine vllm
    python scripts/uninstall_engines.py --engine sglang  
    python scripts/uninstall_engines.py --engine tensorrt
    python scripts/uninstall_engines.py --engine all
    python scripts/uninstall_engines.py --clean-all  # Complete cleanup
"""

import argparse
import subprocess
import sys
import os
import time
import json
from pathlib import Path

class EngineUninstaller:
    """Manages uninstallation of LLM inference engines."""
    
    def __init__(self):
        self.uninstall_log = []
    
    def _run_command(self, command, description="Running command"):
        """Run a shell command with logging."""
        print(f"🔄 {description}...")
        
        start_time = time.time()
        try:
            if isinstance(command, str):
                command = command.split()
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {description} completed in {duration:.1f}s")
                self.uninstall_log.append({
                    'command': ' '.join(command),
                    'status': 'success',
                    'duration': duration,
                    'description': description
                })
                return True, result.stdout
            else:
                print(f"⚠️  {description} returned exit code {result.returncode}")
                # For pip uninstall, non-zero exit might just mean package not found
                if 'pip uninstall' in ' '.join(command) and 'not installed' in result.stderr:
                    print(f"   Package not installed, continuing...")
                    return True, result.stderr
                else:
                    print(f"   Error: {result.stderr}")
                
                self.uninstall_log.append({
                    'command': ' '.join(command),
                    'status': 'warning',
                    'duration': duration,
                    'error': result.stderr,
                    'description': description
                })
                return True, result.stderr  # Continue anyway for uninstall
                
        except Exception as e:
            print(f"❌ {description} failed with exception: {e}")
            return False, str(e)
    
    def _check_package_installed(self, package_name):
        """Check if a package is installed."""
        try:
            cmd = [sys.executable, "-c", f"import {package_name}; print('installed')"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return result.returncode == 0
        except:
            return False
    
    def uninstall_vllm(self):
        """Uninstall vLLM and its specific dependencies."""
        print("\n🗑️  Uninstalling vLLM...")
        
        if not self._check_package_installed('vllm'):
            print("✅ vLLM not installed, nothing to remove")
            return True
        
        # vLLM specific packages to remove
        vllm_packages = [
            'vllm',
            'ray',  # vLLM dependency
            'xformers',  # vLLM optimization
        ]
        
        success = True
        for package in vllm_packages:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package]
            pkg_success, output = self._run_command(cmd, f"Uninstalling {package}")
            if not pkg_success:
                success = False
        
        # Verify uninstallation
        if not self._check_package_installed('vllm'):
            print("✅ vLLM successfully uninstalled")
        else:
            print("⚠️  vLLM might still be partially installed")
        
        return success
    
    def uninstall_sglang(self):
        """Uninstall SGLang and its specific dependencies."""
        print("\n🗑️  Uninstalling SGLang...")
        
        if not self._check_package_installed('sglang'):
            print("✅ SGLang not installed, nothing to remove")
            return True
        
        # SGLang specific packages to remove
        sglang_packages = [
            'sglang',
            'flashinfer',  # SGLang dependency
            'triton',  # SGLang optimization
        ]
        
        success = True
        for package in sglang_packages:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package]
            pkg_success, output = self._run_command(cmd, f"Uninstalling {package}")
            if not pkg_success:
                success = False
        
        # Verify uninstallation
        if not self._check_package_installed('sglang'):
            print("✅ SGLang successfully uninstalled")
        else:
            print("⚠️  SGLang might still be partially installed")
        
        return success
    
    def uninstall_tensorrt_llm(self):
        """Uninstall TensorRT-LLM and its specific dependencies."""
        print("\n🗑️  Uninstalling TensorRT-LLM...")
        
        if not self._check_package_installed('tensorrt_llm'):
            print("✅ TensorRT-LLM not installed, nothing to remove")
            return True
        
        # TensorRT-LLM specific packages to remove
        tensorrt_packages = [
            'tensorrt_llm',
            'tensorrt',
            'mpi4py',  # TensorRT-LLM dependency
        ]
        
        success = True
        for package in tensorrt_packages:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package]
            pkg_success, output = self._run_command(cmd, f"Uninstalling {package}")
            if not pkg_success:
                success = False
        
        # Verify uninstallation
        if not self._check_package_installed('tensorrt_llm'):
            print("✅ TensorRT-LLM successfully uninstalled")
        else:
            print("⚠️  TensorRT-LLM might still be partially installed")
        
        return success
    
    def clean_pytorch(self):
        """Remove PyTorch (use with caution)."""
        print("\n🗑️  Uninstalling PyTorch...")
        
        pytorch_packages = [
            'torch',
            'torchvision', 
            'torchaudio',
            'torchtext',
        ]
        
        success = True
        for package in pytorch_packages:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package]
            pkg_success, output = self._run_command(cmd, f"Uninstalling {package}")
            if not pkg_success:
                success = False
        
        return success
    
    def clean_common_dependencies(self):
        """Remove common dependencies (use with caution)."""
        print("\n🗑️  Removing common dependencies...")
        
        common_deps = [
            'transformers',
            'datasets',
            'accelerate',
            'safetensors',
            'nvidia-ml-py3',
        ]
        
        success = True
        for package in common_deps:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package]
            pkg_success, output = self._run_command(cmd, f"Uninstalling {package}")
            if not pkg_success:
                success = False
        
        return success
    
    def clean_pip_cache(self):
        """Clean pip cache to free up space."""
        print("\n🧹 Cleaning pip cache...")
        
        cmd = [sys.executable, "-m", "pip", "cache", "purge"]
        success, output = self._run_command(cmd, "Cleaning pip cache")
        
        return success
    
    def list_installed_engines(self):
        """List which engines are currently installed."""
        print("\n📋 Checking installed engines...")
        
        engines = {
            'vLLM': 'vllm',
            'SGLang': 'sglang', 
            'TensorRT-LLM': 'tensorrt_llm'
        }
        
        installed = []
        for name, module in engines.items():
            if self._check_package_installed(module):
                installed.append(name)
                print(f"   ✅ {name} is installed")
            else:
                print(f"   ❌ {name} is not installed")
        
        return installed
    
    def save_uninstall_log(self):
        """Save uninstallation log for debugging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"uninstallation_{int(time.time())}.json"
        
        uninstall_info = {
            'timestamp': time.time(),
            'uninstall_log': self.uninstall_log
        }
        
        with open(log_file, 'w') as f:
            json.dump(uninstall_info, f, indent=2)
        
        print(f"📋 Uninstallation log saved: {log_file}")


def main():
    """Main uninstallation function."""
    parser = argparse.ArgumentParser(description='Uninstall LLM inference engines')
    parser.add_argument('--engine', choices=['vllm', 'sglang', 'tensorrt', 'all'],
                       help='Engine to uninstall')
    parser.add_argument('--clean-all', action='store_true',
                       help='Remove all engines and dependencies (complete cleanup)')
    parser.add_argument('--clean-pytorch', action='store_true',
                       help='Also remove PyTorch (use with caution)')
    parser.add_argument('--clean-deps', action='store_true',
                       help='Also remove common dependencies (use with caution)')
    parser.add_argument('--list', action='store_true',
                       help='List currently installed engines')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.engine or args.clean_all or args.list):
        parser.error("Must specify --engine, --clean-all, or --list")
    
    print("🗑️  LLM Engine Uninstallation Manager")
    print("=" * 50)
    
    uninstaller = EngineUninstaller()
    
    try:
        # List installed engines if requested
        if args.list:
            uninstaller.list_installed_engines()
            return True
        
        success_count = 0
        total_count = 0
        
        # Determine which engines to uninstall
        if args.clean_all:
            engines = ['vllm', 'sglang', 'tensorrt']
            print("🚨 Complete cleanup requested - removing all engines")
        elif args.engine == 'all':
            engines = ['vllm', 'sglang', 'tensorrt']
        else:
            engines = [args.engine] if args.engine else []
        
        # Show current status
        installed_engines = uninstaller.list_installed_engines()
        
        # Uninstall requested engines
        for engine in engines:
            total_count += 1
            print(f"\n{'='*50}")
            print(f"Uninstalling {engine.upper()}")
            print(f"{'='*50}")
            
            if engine == 'vllm':
                success = uninstaller.uninstall_vllm()
            elif engine == 'sglang':
                success = uninstaller.uninstall_sglang()
            elif engine == 'tensorrt':
                success = uninstaller.uninstall_tensorrt_llm()
            else:
                print(f"❌ Unknown engine: {engine}")
                continue
            
            if success:
                success_count += 1
        
        # Clean additional components if requested
        if args.clean_deps or args.clean_all:
            print(f"\n{'='*50}")
            print("Cleaning common dependencies")
            print(f"{'='*50}")
            uninstaller.clean_common_dependencies()
        
        if args.clean_pytorch or args.clean_all:
            print(f"\n{'='*50}")
            print("Cleaning PyTorch")
            print(f"{'='*50}")
            uninstaller.clean_pytorch()
        
        # Clean pip cache
        uninstaller.clean_pip_cache()
        
        # Save uninstallation log
        uninstaller.save_uninstall_log()
        
        # Final check
        print(f"\n{'='*50}")
        print("FINAL STATUS CHECK")
        print(f"{'='*50}")
        remaining_engines = uninstaller.list_installed_engines()
        
        if not remaining_engines:
            print("\n🎉 All LLM engines successfully removed!")
        else:
            print(f"\n📋 Remaining engines: {', '.join(remaining_engines)}")
        
        print(f"\n✅ Uninstallation completed!")
        print(f"💡 To reinstall engines: python scripts/install_engines.py --engine <engine_name>")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Uninstallation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Uninstallation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)