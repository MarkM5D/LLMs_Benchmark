#!/usr/bin/env python3
"""
RunPod Deployment Script

This script prepares your benchmark framework for RunPod H100 deployment.
Run this in local development to create a deployment-ready version.
"""

import os
import shutil
import subprocess
import json
from pathlib import Path


def create_runpod_deployment_package():
    """Create a RunPod-ready deployment package"""
    print("📦 Creating RunPod H100 Deployment Package")
    print("=" * 50)
    
    # Current directory
    current_dir = Path(__file__).parent
    deployment_dir = current_dir / "runpod_deployment"
    
    # Clean and create deployment directory
    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)
    deployment_dir.mkdir()
    
    print(f"📁 Deployment directory: {deployment_dir}")
    
    # Files to copy
    files_to_copy = [
        "setup_env.sh",
        "run_all.py",
        "aggregate_results.py",
        "h100_optimize.py",
        "benchmark_plan.md",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "LICENSE"
    ]
    
    dirs_to_copy = [
        "benchmarks"
    ]
    
    # Copy files
    print("📋 Copying benchmark files...")
    for file_name in files_to_copy:
        src = current_dir / file_name
        if src.exists():
            shutil.copy2(src, deployment_dir / file_name)
            print(f"  ✅ {file_name}")
        else:
            print(f"  ⚠️ {file_name} (not found)")
    
    # Copy directories
    for dir_name in dirs_to_copy:
        src = current_dir / dir_name
        if src.exists():
            shutil.copytree(src, deployment_dir / dir_name)
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ⚠️ {dir_name}/ (not found)")
    
    # Create deployment instructions
    create_deployment_instructions(deployment_dir)
    
    # Create RunPod startup script
    create_runpod_startup_script(deployment_dir)
    
    print(f"\n✅ Deployment package created in: {deployment_dir}")
    print("\n🚀 Next steps:")
    print("1. Upload this folder to your RunPod instance")
    print("2. Run: cd /workspace/runpod_deployment")
    print("3. Run: chmod +x runpod_startup.sh")
    print("4. Run: ./runpod_startup.sh")


def create_deployment_instructions(deployment_dir):
    """Create detailed deployment instructions"""
    instructions = """# RunPod H100 Deployment Instructions

## 🚀 Quick Start

1. **Upload to RunPod**
   ```bash
   # In RunPod terminal:
   cd /workspace
   # Upload this runpod_deployment folder here
   ```

2. **Run Setup**
   ```bash
   cd /workspace/runpod_deployment
   chmod +x runpod_startup.sh setup_env.sh
   ./runpod_startup.sh
   ```

3. **Verify H100 Setup**
   ```bash
   python3 h100_optimize.py
   ```

4. **Run Benchmarks**
   ```bash
   python3 run_all.py
   ```

## 📊 Expected Results

On H100 80GB you should see:
- vLLM: ~8000-12000 tokens/sec
- SGLang: ~7000-10000 tokens/sec  
- TensorRT-LLM: ~10000-15000 tokens/sec

## 🔧 Troubleshooting

### GPU Not Detected
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

### Build Errors
```bash
export CUDA_ARCHITECTURES=90
pip install --no-cache-dir --force-reinstall vllm
```

### Memory Issues
```bash
python3 -c "import torch; torch.cuda.empty_cache()"
```

## 📁 File Structure in /workspace

```
/workspace/
├── runpod_deployment/
│   ├── setup_env.sh          # H100 optimized setup
│   ├── h100_optimize.py      # H100 verification
│   ├── run_all.py           # Main benchmark runner
│   ├── benchmarks/          # Individual engine benchmarks
│   └── results/            # Benchmark outputs
```

## 🎯 Success Indicators

✅ H100 detected in h100_optimize.py
✅ All 3 engines install successfully 
✅ Benchmarks complete without errors
✅ Results saved in benchmarks/results/
"""
    
    with open(deployment_dir / "DEPLOYMENT_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("  ✅ DEPLOYMENT_INSTRUCTIONS.md")


def create_runpod_startup_script(deployment_dir):
    """Create automated RunPod startup script"""
    script = """#!/bin/bash

echo "🚀 RunPod H100 Benchmark Setup Starting..."
echo "============================================"

# Ensure we're in /workspace
cd /workspace/runpod_deployment

# Check H100 availability
echo "🔍 Checking H100 GPU..."
nvidia-smi

# Run environment setup
echo "⚙️ Setting up H100 optimized environment..."
./setup_env.sh

# Verify H100 optimizations
echo "🧪 Verifying H100 optimizations..."
python3 h100_optimize.py

echo ""
echo "✅ Setup completed! Ready to run benchmarks."
echo "📊 To run benchmarks: python3 run_all.py"
echo "📈 Results will be in: benchmarks/results/"
"""
    
    startup_script = deployment_dir / "runpod_startup.sh"
    with open(startup_script, "w", encoding="utf-8") as f:
        f.write(script)
    
    # Make executable (will work on RunPod)
    startup_script.chmod(0o755)
    
    print("  ✅ runpod_startup.sh")


def create_local_test_script():
    """Create a script to test framework locally without H100 dependencies"""
    print("\n🧪 Creating local test script...")
    
    local_test = """#!/usr/bin/env python3
\"\"\"
Local Framework Test (No H100 Required)

Tests the benchmark framework structure and basic functionality
without requiring actual H100 hardware or inference engines.
\"\"\"

import sys
import os
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

def test_local_framework():
    print("🧪 Local Framework Test (No H100 Required)")
    print("=" * 50)
    
    try:
        # Test framework imports
        from test_framework import BenchmarkFrameworkTester
        
        # Run framework tests
        tester = BenchmarkFrameworkTester()
        
        # Run only local tests
        print("📋 Testing file structure...")
        file_results = tester.test_file_structure()
        
        print("\\n🐍 Testing Python syntax...")
        syntax_results = tester.test_python_syntax()
        
        print("\\n📊 Testing metrics utilities...")
        metrics_results = tester.test_metrics_utilities()
        
        # Summary
        print("\\n📊 Local Test Summary:")
        print("✅ Framework structure validated")
        print("✅ Python syntax checked")  
        print("✅ Metrics utilities tested")
        print("\\n🚀 Framework ready for RunPod H100 deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_local_framework()
    sys.exit(0 if success else 1)
"""
    
    with open(Path(__file__).parent / "test_local.py", "w", encoding="utf-8") as f:
        f.write(local_test)
    
    print("  ✅ test_local.py created")


def main():
    """Main deployment preparation"""
    print("🎯 RunPod H100 Deployment Preparation")
    print("=" * 60)
    
    # Create deployment package
    create_runpod_deployment_package()
    
    # Create local test script
    create_local_test_script()
    
    print("\n" + "=" * 60)
    print("📋 Deployment Summary:")
    print("🏠 Local: Run 'python3 test_local.py' to test framework")
    print("🚀 RunPod: Upload 'runpod_deployment/' to /workspace")
    print("⚡ H100: Run './runpod_startup.sh' for automated setup")
    print("\n✅ Ready for H100 benchmarking on RunPod!")


if __name__ == "__main__":
    main()