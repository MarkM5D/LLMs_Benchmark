#!/bin/bash

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
