#!/usr/bin/env python3
"""
H100 GPU Optimization Checker and Setup

This script verifies H100 Hopper architecture optimizations and provides
specific configuration recommendations for maximum performance.
"""

import os
import sys
import subprocess
import json
import torch


def check_h100_environment():
    """Check if environment is properly configured for H100"""
    print("🔍 H100 Environment Analysis")
    print("=" * 50)
    
    # Detect environment
    is_runpod = os.path.exists("/workspace") or "runpod" in os.environ.get("HOSTNAME", "").lower()
    is_local_dev = not is_runpod
    
    if is_local_dev:
        print("🏠 Local development environment detected")
        print("   H100 optimizations will be tested on RunPod /workspace")
        print()
    
    results = {
        "environment": "runpod" if is_runpod else "local_dev",
        "gpu_detected": False,
        "h100_detected": False,
        "compute_capability": None,
        "cuda_architectures": None,
        "optimizations": {}
    }
    
    # Check GPU
    try:
        if torch.cuda.is_available():
            results["gpu_detected"] = True
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            results["compute_capability"] = f"{compute_cap[0]}.{compute_cap[1]}"
            
            print(f"🎮 GPU: {gpu_name}")
            print(f"🔢 Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            # Check if H100
            if "H100" in gpu_name.upper():
                results["h100_detected"] = True
                print("✅ H100 GPU detected!")
                
                if compute_cap >= (9, 0):
                    print("✅ Hopper architecture (9.0+) confirmed")
                else:
                    print("⚠️ Unexpected compute capability for H100")
            else:
                print(f"⚠️ Non-H100 GPU detected: {gpu_name}")
        else:
            if is_local_dev:
                print("ℹ️ No CUDA GPU detected (expected in local dev)")
            else:
                print("❌ No CUDA GPU detected in RunPod!")
    except Exception as e:
        if is_local_dev:
            print(f"ℹ️ GPU check failed (expected in local dev): {e}")
        else:
            print(f"❌ GPU check failed: {e}")
    
    # Check CUDA architecture settings
    cuda_arch = os.environ.get("CUDA_ARCHITECTURES")
    torch_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    
    print(f"\n🏗️ Architecture Configuration:")
    print(f"   CUDA_ARCHITECTURES: {cuda_arch or 'Not set'}")
    print(f"   TORCH_CUDA_ARCH_LIST: {torch_arch or 'Not set'}")
    
    results["cuda_architectures"] = {
        "CUDA_ARCHITECTURES": cuda_arch,
        "TORCH_CUDA_ARCH_LIST": torch_arch
    }
    
    # PyTorch CUDA architectures
    try:
        arch_list = torch.cuda.get_arch_list()
        print(f"   PyTorch CUDA Arch List: {arch_list}")
        results["optimizations"]["pytorch_arch"] = arch_list
        
        # Check if 9.0 is in the list (H100 support)
        if any("sm_90" in arch for arch in arch_list):
            print("✅ PyTorch compiled with H100 support (sm_90)")
            results["optimizations"]["h100_support"] = True
        else:
            print("⚠️ PyTorch may not be optimized for H100")
            results["optimizations"]["h100_support"] = False
    except Exception as e:
        print(f"❌ Error checking PyTorch architectures: {e}")
    
    return results


def check_inference_engines():
    """Check if inference engines are H100 optimized"""
    print("\n🚀 Inference Engine H100 Optimization Check")
    print("=" * 50)
    
    engines = {
        "vllm": False,
        "sglang": False,
        "tensorrt_llm": False
    }
    
    # Check vLLM
    try:
        import vllm
        engines["vllm"] = True
        print(f"✅ vLLM {vllm.__version__} installed")
        
        # Check if vLLM was compiled with proper CUDA arch
        try:
            # This is a heuristic check - vLLM should work optimally on H100
            from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel
            print("   ✅ vLLM parallel utilities available (good for H100)")
        except ImportError:
            print("   ⚠️ vLLM parallel utilities not available")
            
    except ImportError:
        print("❌ vLLM not installed")
    
    # Check SGLang
    try:
        import sglang
        engines["sglang"] = True
        print(f"✅ SGLang {sglang.__version__} installed")
    except ImportError:
        print("❌ SGLang not installed")
    
    # Check TensorRT-LLM
    try:
        import tensorrt_llm
        engines["tensorrt_llm"] = True
        print(f"✅ TensorRT-LLM {tensorrt_llm.__version__} installed")
        print("   ✅ TensorRT-LLM has native H100 optimizations")
    except ImportError:
        print("❌ TensorRT-LLM not installed")
    
    return engines


def generate_h100_recommendations(env_results, engine_results):
    """Generate specific H100 optimization recommendations"""
    print("\n💡 H100 Optimization Recommendations")
    print("=" * 50)
    
    recommendations = []
    
    if not env_results["h100_detected"]:
        recommendations.append({
            "priority": "HIGH",
            "issue": "H100 GPU not detected",
            "solution": "Ensure you're running on an H100 instance. Check RunPod instance type."
        })
    
    # Check CUDA architecture settings
    if not env_results["cuda_architectures"]["CUDA_ARCHITECTURES"] == "90":
        recommendations.append({
            "priority": "HIGH", 
            "issue": "CUDA_ARCHITECTURES not set to 90 for H100",
            "solution": "export CUDA_ARCHITECTURES=90 before installing inference engines"
        })
    
    if not env_results["optimizations"].get("h100_support", False):
        recommendations.append({
            "priority": "HIGH",
            "issue": "PyTorch not compiled with H100 support",
            "solution": "Install PyTorch with CUDA 12.1+ and Hopper support"
        })
    
    # Engine-specific recommendations
    if not engine_results["vllm"]:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "vLLM not installed",
            "solution": "pip install vllm --no-build-isolation with CUDA_ARCHITECTURES=90"
        })
    
    if not engine_results["tensorrt_llm"]:
        recommendations.append({
            "priority": "MEDIUM", 
            "issue": "TensorRT-LLM not installed",
            "solution": "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
        })
    
    # Print recommendations
    for rec in recommendations:
        priority_icon = "🔥" if rec["priority"] == "HIGH" else "⚠️"
        print(f"{priority_icon} {rec['priority']}: {rec['issue']}")
        print(f"   💡 Solution: {rec['solution']}")
        print()
    
    if not recommendations:
        print("✅ Environment looks properly configured for H100!")
    
    return recommendations


def generate_optimized_build_commands():
    """Generate H100-optimized build commands"""
    print("\n🔧 H100-Optimized Installation Commands")
    print("=" * 50)
    
    commands = [
        "# Set H100 environment variables",
        "export CUDA_ARCHITECTURES=90",
        "export TORCH_CUDA_ARCH_LIST=9.0", 
        "export CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=90'",
        "",
        "# Install PyTorch with H100 support",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "",
        "# Install vLLM with H100 optimization",
        "pip install vllm --no-build-isolation",
        "",
        "# Install SGLang with H100 optimization", 
        "pip install 'sglang[all]' --no-build-isolation",
        "",
        "# Install TensorRT-LLM (has native H100 support)",
        "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com",
        "",
        "# Verify installation",
        "python3 h100_optimize.py"
    ]
    
    for cmd in commands:
        print(cmd)


def main():
    """Main optimization checker"""
    print("🏎️ H100 GPU Optimization Checker")
    print("=" * 60)
    
    # Check environment
    env_results = check_h100_environment()
    
    # Check inference engines
    engine_results = check_inference_engines()
    
    # Generate recommendations
    recommendations = generate_h100_recommendations(env_results, engine_results)
    
    # Show optimized commands if needed
    if recommendations:
        generate_optimized_build_commands()
    
    # Save results
    results = {
        "timestamp": __import__("time").time(),
        "environment": env_results,
        "engines": engine_results,
        "recommendations": recommendations
    }
    
    with open("h100_optimization_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Full report saved to: h100_optimization_report.json")


if __name__ == "__main__":
    main()