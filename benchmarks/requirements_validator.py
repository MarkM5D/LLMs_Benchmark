#!/usr/bin/env python3
"""
Requirements Validator for LLM Benchmark System

This module validates that all required packages are properly installed
and configured before running benchmarks. NO FALLBACKS - system must be
properly configured or benchmarks fail immediately.
"""

import sys
import subprocess
import importlib.util
from typing import Dict, List, Tuple


class RequirementsValidator:
    """Validate all benchmark requirements - ZERO TOLERANCE for missing packages"""
    
    REQUIRED_PACKAGES = {
        "core": [
            "torch",
            "transformers", 
            "numpy",
            "pandas",
            "tqdm",
            "psutil"
        ],
        "vllm": [
            "vllm"
        ],
        "sglang": [
            "sglang"
        ],
        "tensorrt_llm": [
            "tensorrt_llm"
        ]
    }
    
    CUDA_REQUIREMENTS = [
        "nvidia-ml-py3",
        "pynvml"
    ]
    
    @classmethod
    def validate_all_requirements(cls, engine: str = None) -> bool:
        """
        Validate all requirements for specified engine or all engines
        
        Args:
            engine: Specific engine to validate ("vllm", "sglang", "tensorrt_llm") 
                   or None for all
        
        Returns:
            True if all requirements satisfied, raises SystemExit otherwise
        """
        validator = cls()
        
        print("🔍 VALIDATING BENCHMARK REQUIREMENTS - ZERO TOLERANCE MODE")
        print("=" * 60)
        
        # Validate core requirements
        if not validator._validate_core_requirements():
            raise SystemExit("❌ Core requirements validation failed")
        
        # Validate CUDA requirements
        if not validator._validate_cuda_requirements():
            raise SystemExit("❌ CUDA requirements validation failed")
        
        # Validate engine-specific requirements
        if engine:
            if not validator._validate_engine_requirements(engine):
                raise SystemExit(f"❌ {engine.upper()} requirements validation failed")
        else:
            # Validate all engines
            for eng in ["vllm", "sglang", "tensorrt_llm"]:
                if not validator._validate_engine_requirements(eng):
                    raise SystemExit(f"❌ {eng.upper()} requirements validation failed")
        
        print("✅ ALL REQUIREMENTS VALIDATED SUCCESSFULLY")
        print("🚀 System ready for benchmark execution")
        return True
    
    def _validate_core_requirements(self) -> bool:
        """Validate core Python packages"""
        print("📦 Validating core packages...")
        
        for package in self.REQUIRED_PACKAGES["core"]:
            if not self._check_package_import(package):
                error_msg = (
                    f"❌ CRITICAL: Core package '{package}' not available!\n"
                    f"   Install: pip install {package}\n"
                    f"   Core packages are mandatory for all benchmarks"
                )
                print(error_msg)
                return False
            print(f"  ✅ {package}")
        
        return True
    
    def _validate_cuda_requirements(self) -> bool:
        """Validate CUDA and GPU requirements"""
        print("🎮 Validating CUDA/GPU requirements...")
        
        # Check CUDA packages
        for package in self.CUDA_REQUIREMENTS:
            if not self._check_package_import(package):
                error_msg = (
                    f"❌ CRITICAL: CUDA package '{package}' not available!\n"
                    f"   Install: pip install {package}\n"
                    f"   CUDA packages are mandatory for GPU benchmarks"
                )
                print(error_msg)
                return False
            print(f"  ✅ {package}")
        
        # Validate nvidia-smi availability
        if not self._check_nvidia_smi():
            error_msg = (
                f"❌ CRITICAL: nvidia-smi not available!\n"
                f"   NVIDIA drivers not properly installed\n"
                f"   GPU monitoring requires working nvidia-smi"
            )
            print(error_msg)
            return False
        print(f"  ✅ nvidia-smi")
        
        return True
    
    def _validate_engine_requirements(self, engine: str) -> bool:
        """Validate specific engine requirements"""
        print(f"🚀 Validating {engine.upper()} requirements...")
        
        if engine not in self.REQUIRED_PACKAGES:
            print(f"❌ Unknown engine: {engine}")
            return False
        
        for package in self.REQUIRED_PACKAGES[engine]:
            if not self._check_package_import(package):
                error_msg = (
                    f"❌ CRITICAL: {engine.upper()} package '{package}' not available!\n"
                    f"   Install {engine.upper()}: {self._get_install_command(engine)}\n"
                    f"   {engine.upper()} is mandatory for its benchmark"
                )
                print(error_msg)
                return False
            print(f"  ✅ {package}")
        
        # Additional engine-specific validations
        if engine == "vllm":
            return self._validate_vllm_specific()
        elif engine == "sglang":
            return self._validate_sglang_specific()
        elif engine == "tensorrt_llm":
            return self._validate_tensorrt_specific()
        
        return True
    
    def _validate_vllm_specific(self) -> bool:
        """Validate vLLM-specific requirements"""
        try:
            from vllm import LLM, SamplingParams
            print("  ✅ vLLM API classes available")
            return True
        except ImportError as e:
            print(f"  ❌ vLLM API import failed: {e}")
            return False
    
    def _validate_sglang_specific(self) -> bool:
        """Validate SGLang-specific requirements"""
        try:
            from sglang import LLM, SamplingParams
            print("  ✅ SGLang API classes available")
            return True
        except ImportError as e:
            print(f"  ❌ SGLang API import failed: {e}")
            return False
    
    def _validate_tensorrt_specific(self) -> bool:
        """Validate TensorRT-LLM-specific requirements"""
        try:
            from tensorrt_llm import LLM, SamplingParams
            print("  ✅ TensorRT-LLM API classes available")
            return True
        except ImportError as e:
            print(f"  ❌ TensorRT-LLM API import failed: {e}")
            return False
    
    def _check_package_import(self, package: str) -> bool:
        """Check if package can be imported"""
        try:
            spec = importlib.util.find_spec(package)
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available and working"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _get_install_command(self, engine: str) -> str:
        """Get installation command for engine"""
        commands = {
            "vllm": "pip install vllm",
            "sglang": "pip install sglang[all]", 
            "tensorrt_llm": "pip install tensorrt-llm"
        }
        return commands.get(engine, f"pip install {engine}")


# Standalone validation function for easy import
def validate_benchmark_requirements(engine: str = None) -> bool:
    """
    Validate benchmark requirements for specific engine or all engines
    
    Args:
        engine: Engine name ("vllm", "sglang", "tensorrt_llm") or None for all
        
    Returns:
        True if validation passes, raises SystemExit otherwise
    """
    return RequirementsValidator.validate_all_requirements(engine)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate LLM Benchmark Requirements")
    parser.add_argument("--engine", choices=["vllm", "sglang", "tensorrt_llm"],
                       help="Validate specific engine only")
    
    args = parser.parse_args()
    
    try:
        validate_benchmark_requirements(args.engine)
        print("\n🎉 ALL VALIDATIONS PASSED - SYSTEM READY!")
        sys.exit(0)
    except SystemExit as e:
        print(f"\n💥 VALIDATION FAILED - SYSTEM NOT READY!")
        print(f"Fix the above issues before running benchmarks")
        sys.exit(1)