# 🧠 TASK: Implement Complete LLM Benchmark Automation (vLLM, SGLang, TensorRT-LLM)

You are to implement the **"Benchmarking Next-Generation LLM Inference Engines Test Plan"** exactly as described below.  
Your job: Write complete setup + benchmark automation scripts (in Python + Bash) to run **vLLM**, **SGLang**, and **TensorRT-LLM** on a single RunPod GPU instance (H100 80GB).  

---

## 🎯 Objective
Measure throughput (tokens/sec), latency (p50/p95), GPU memory peak, and GPU utilization for each engine on the **gpt-oss-20b** model.  
All three engines must be tested under **identical** parameters and environment conditions.

---

## ⚙️ Environment Setup
- Docker Image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- GPU: `H100 80GB`
- Dataset: `heka-ai/sharegpt-english-10k-vllm-serving-benchmark` (download from Hugging Face)
- Create directories:
  - `/workspace/benchmarks/data`
  - `/workspace/benchmarks/results`
  - `/workspace/benchmarks/scripts`
- Ensure `pip install` commands install required packages:
  - `vllm`, `sglang`, `tensorrt-llm`, `torch`, `transformers`, `tqdm`, `numpy`, `pandas`
  - and any additional dependencies automatically detected by imports

---

## 🧪 Benchmark Parameters
Use the same parameters for all engines:
- `max_tokens = 128`
- `batch_size = 32`
- `concurrency = 16`
- `temperature = 0.8`
- `top_p = 0.95`

---

## 🧰 What to Implement

### 1️⃣ Environment Setup Script
- Bash script: `setup_env.sh`
  - Installs dependencies
  - Downloads dataset to `/workspace/benchmarks/data`
  - Creates necessary folders

---

### 2️⃣ Engine Setup & Benchmark Scripts

For each engine, create a separate Python script:

#### (A) vLLM Benchmark (`vllm_benchmark.py`)
- Import and run the built-in vLLM benchmark tool:
  ```bash
  vllm.bench \
    --model gpt-oss-20b \
    --dataset /workspace/benchmarks/data/sharegpt-10k.jsonl \
    --max-numseqs 32 \
    --max-num-batched-tokens 8192 \
    --temperature 0.8 \
    --top-p 0.95
  ```
- Save output logs and results (throughput, latency, GPU memory) to `/workspace/benchmarks/results/vllm_results.json`.

#### (B) SGLang Benchmark (`sglang_benchmark.py`)
- Load model using:
  ```python
  from sg.lang import LLM, SamplingParams
  ```
- Generate text for all prompts from dataset using:
  ```python
  llm = LLM("gpt-oss-20b")
  params = SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95)
  llm.generate(prompts, sampling_params=params, batch_size=32, concurrency=16)
  ```
- Measure:
  - Tokens/sec (throughput)
  - Latency p50/p95 (use Python `time` module)
  - GPU memory peak + utilization (via `nvidia-smi`)
- Save metrics to `/workspace/benchmarks/results/sglang_results.json`.

#### (C) TensorRT-LLM Benchmark (`tensorrtllm_benchmark.py`)
- Load compiled model: `"gpt-oss-20b.trt"`
- Use TensorRT-LLM Python API to perform direct sampling with the same parameters.
- Measure identical metrics and store results to `/workspace/benchmarks/results/tensorrtllm_results.json`.

---

## 📊 Metrics Collection
- Implement a shared Python utility file: `metrics.py`
  - Functions to record:
    - `throughput (tokens/sec)`
    - `latency (p50, p95)`
    - `gpu_peak_mem`, `gpu_utilization`
  - Use:
    - `time.time()` for latency
    - `subprocess` to call `nvidia-smi --query-gpu=utilization.gpu,memory.used`
- Run each test **3 times**, average the metrics, and ignore outliers.

---

## 🧾 Execution Sequence
1. Run `bash setup_env.sh`
2. Run `python vllm_benchmark.py`
3. Run `python sglang_benchmark.py`
4. Run `python tensorrtllm_benchmark.py`
5. Aggregate results with `aggregate_results.py`
   - Combine JSON results into one CSV/table.
   - Compute mean, p50/p95, and rank engines by throughput and efficiency.

---

## 🧠 Output
- Save summary file: `/workspace/benchmarks/results/final_report.csv`
- Include fields:  
  `engine, throughput, latency_p50, latency_p95, gpu_util, gpu_mem_peak`
- Print ranking table in console.

---

## 🧮 Constraints
- Each engine tested on **fresh RunPod instance**
- Perform **warm-up phase** before each benchmark with same parameters
- No network/web API latency — only direct sampling
- All file paths and parameters must be consistent

---

## ✅ Deliverables
- `setup_env.sh`
- `vllm_benchmark.py`
- `sglang_benchmark.py`
- `tensorrtllm_benchmark.py`
- `metrics.py`
- `aggregate_results.py`
- Result folder with all metrics and logs

---

## 💡 Goal for Copilot
Generate all the scripts above with:
- Realistic metric measurement code
- Clean, modular structure
- Comments explaining each step
- Automatic CSV + JSON result saving
- GPU-safe execution (handle CUDA errors gracefully)

This is an automation pipeline for scientific benchmarking.
Follow the plan strictly and ensure reproducibility.