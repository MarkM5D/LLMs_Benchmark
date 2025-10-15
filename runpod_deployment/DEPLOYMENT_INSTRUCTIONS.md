# RunPod H100 Deployment Instructions

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
