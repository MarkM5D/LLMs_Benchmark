#!/usr/bin/env python3
"""
vLLM Benchmark Script

This script benchmarks vLLM inference engine using the built-in benchmark tool
and direct Python API for consistent measurements with other engines.

Parameters (matching benchmark plan):
- Model: gpt-oss-20b
- Max tokens: 128
- Batch size: 32
- Concurrency: 16
- Temperature: 0.8
- Top-p: 0.95
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to sys.path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import BenchmarkMetrics, save_results, warm_up_gpu, clear_gpu_memory
from benchmarks.requirements_validator import validate_benchmark_requirements


class VLLMBenchmark:
    """vLLM benchmark implementation"""
    
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        self.max_tokens = 128
        self.batch_size = 32
        self.concurrency = 16
        self.temperature = 0.8
        self.top_p = 0.95
        self.dataset_path = "/workspace/benchmarks/data/sharegpt-10k.jsonl"
        
        # H100 optimization settings
        self.h100_optimizations = {
            "enable_cuda_graphs": True,
            "max_num_batched_tokens": self.batch_size * self.max_tokens * 2,  # H100 has more memory
            "gpu_memory_utilization": 0.95,  # H100 80GB allows higher utilization
            "trust_remote_code": True,
            "enforce_eager": False,  # Allow CUDA graphs for H100
            "quantization": None  # Can be set to "awq" or "gptq" for H100 if needed
        }
        
        # Alternative local paths for development
        if not os.path.exists(self.dataset_path):
            local_path = os.path.join(os.path.dirname(__file__), "..", "data", "sharegpt-10k.jsonl")
            if os.path.exists(local_path):
                self.dataset_path = local_path
    
    def run_builtin_benchmark(self) -> dict:
        """Run vLLM's guaranteed built-in benchmark - NO FALLBACKS"""
        print("🚀 Running vLLM built-in benchmark - guaranteed mode...")
        
        # vLLM standard benchmark command - MUST work
        cmd = [
            "python", "-m", "vllm.entrypoints.llm_benchmark",
            "--model", self.model_name,
            "--dataset", self.dataset_path,
            "--max-num-seqs", str(self.batch_size),
            "--max-num-batched-tokens", str(self.batch_size * self.max_tokens),
            "--num-prompts", "1000"
        ]
        
        print(f"📝 Command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            error_msg = (
                f"❌ CRITICAL vLLM BUILTIN BENCHMARK ERROR!\n"
                f"   Command failed: {' '.join(cmd)}\n"
                f"   Error: {result.stderr}\n"
                f"   No fallback available - fix vLLM installation"
            )
            print(error_msg)
            raise SystemExit(f"vLLM builtin benchmark failed: {result.stderr}")
        
        return self._parse_builtin_output(result.stdout, result.stderr)
    
    def _parse_builtin_output(self, stdout: str, stderr: str) -> dict:
        """Parse vLLM benchmark output"""
        # vLLM benchmark typically outputs metrics in the stdout
        lines = stdout.split('\n') + stderr.split('\n')
        
        metrics = {
            "throughput_tokens_per_second": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "total_requests": 0,
            "raw_output": stdout
        }
        
        # Parse common vLLM benchmark output patterns
        for line in lines:
            line = line.strip().lower()
            
            if "throughput" in line and "token" in line:
                # Look for patterns like "Throughput: 123.45 tokens/s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if "token" in part and i > 0:
                        try:
                            metrics["throughput_tokens_per_second"] = float(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            
            elif "p50" in line or "median" in line:
                # Look for P50 latency
                parts = line.split()
                for i, part in enumerate(parts):
                    if ("p50" in part or "median" in part) and i < len(parts) - 1:
                        try:
                            metrics["latency_p50"] = float(parts[i+1])
                        except (ValueError, IndexError):
                            pass
            
            elif "p95" in line:
                # Look for P95 latency
                parts = line.split()
                for i, part in enumerate(parts):
                    if "p95" in part and i < len(parts) - 1:
                        try:
                            metrics["latency_p95"] = float(parts[i+1])
                        except (ValueError, IndexError):
                            pass
            
            elif "total" in line and ("request" in line or "prompt" in line):
                # Look for total requests
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        metrics["total_requests"] = int(part)
                        break
        
        return metrics
    
    def run_direct_benchmark(self) -> dict:
        """Run direct vLLM benchmark - GUARANTEED SUCCESS ONLY"""
        print("🔬 Running vLLM direct Python API benchmark - guaranteed mode...")
        
        # HARD REQUIREMENT: vLLM must be properly installed
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            error_msg = (
                f"❌ CRITICAL ERROR: vLLM not properly installed!\n"
                f"   Import error: {e}\n"
                f"   Install vLLM first: pip install vllm\n"
                f"   Or: pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121"
            )
            print(error_msg)
            raise SystemExit("vLLM installation required - no fallbacks available")
        
        try:
            
            # Initialize metrics tracking
            benchmark_metrics = BenchmarkMetrics("vLLM")
            
            # Load dataset
            prompts = self._load_dataset()
            if not prompts:
                raise ValueError("No prompts loaded from dataset")
            
            print(f"📚 Loaded {len(prompts)} prompts")
            
            print(f"🤖 Loading model: {self.model_name}")
            
            # vLLM initialization with H100 optimizations
            llm = LLM(
                model=self.model_name,
                max_num_seqs=self.batch_size,
                max_num_batched_tokens=self.h100_optimizations["max_num_batched_tokens"],
                trust_remote_code=self.h100_optimizations["trust_remote_code"],
                gpu_memory_utilization=self.h100_optimizations["gpu_memory_utilization"],
                enforce_eager=self.h100_optimizations["enforce_eager"],  # Enable CUDA graphs for H100
                quantization=self.h100_optimizations["quantization"],
                # H100-specific optimizations
                enable_chunked_prefill=True,  # Better for large batches on H100
                max_model_len=None,  # Let vLLM auto-detect based on H100 memory
                dtype="auto"  # Use optimal dtype for H100
            )
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            
            # Start benchmark
            benchmark_metrics.start_benchmark()
            
            # Warm-up phase
            print("🔥 Performing warm-up...")
            warmup_prompts = prompts[:min(10, len(prompts))]
            for _ in range(3):
                llm.generate(warmup_prompts, sampling_params)
            
            benchmark_metrics.complete_warmup()
            
            # Main benchmark
            print("📊 Running main benchmark...")
            
            # Process prompts in batches
            batch_size = self.batch_size
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            
            for i in range(0, min(len(prompts), 1000), batch_size):  # Limit to 1000 prompts
                batch_prompts = prompts[i:i + batch_size]
                batch_start = time.time()
                
                # Generate responses
                outputs = llm.generate(batch_prompts, sampling_params)
                
                batch_end = time.time()
                batch_latency = batch_end - batch_start
                
                # Count tokens in this batch
                batch_tokens = 0
                for output in outputs:
                    for completion in output.outputs:
                        batch_tokens += len(completion.token_ids)
                
                # Record metrics for each request in the batch
                avg_latency_per_request = batch_latency / len(batch_prompts)
                avg_tokens_per_request = batch_tokens / len(batch_prompts)
                
                for _ in batch_prompts:
                    benchmark_metrics.add_request_result(
                        avg_latency_per_request, 
                        avg_tokens_per_request
                    )
                
                # Progress update
                batch_num = (i // batch_size) + 1
                print(f"📈 Batch {batch_num}/{min(total_batches, 1000//batch_size)}: "
                      f"{len(batch_prompts)} prompts, {batch_tokens} tokens, "
                      f"{batch_latency:.2f}s")
            
            # End benchmark and get results
            results = benchmark_metrics.end_benchmark()
            return results
            
        except Exception as e:
            error_msg = (
                f"❌ CRITICAL vLLM BENCHMARK ERROR: {e}\n"
                f"   vLLM benchmark failed - system cannot continue\n"
                f"   Check vLLM installation and model availability\n"
                f"   No fallback mode - fix the issue and retry"
            )
            print(error_msg)
            raise SystemExit(f"vLLM benchmark failed: {e}")
    
    def _load_dataset(self) -> list:
        """Load prompts from ShareGPT dataset"""
        prompts = []
        
        try:
            if os.path.exists(self.dataset_path):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line_num >= 2000:  # Limit for testing
                            break
                        try:
                            data = json.loads(line.strip())
                            
                            # Extract prompt from different possible formats
                            prompt = ""
                            if "prompt" in data:
                                prompt = data["prompt"]
                            elif "conversation" in data:
                                for turn in data["conversation"]:
                                    if turn.get("from") == "human":
                                        prompt = turn.get("value", "")
                                        break
                            
                            if prompt.strip():
                                prompts.append(prompt.strip())
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception:
                            continue
            
            # NO FALLBACKS - dataset MUST exist
            if not prompts:
                error_msg = (
                    f"❌ CRITICAL ERROR: No prompts loaded from dataset!\n"
                    f"   Dataset path: {self.dataset_path}\n"
                    f"   Download required dataset: heka-ai/sharegpt-english-10k-vllm-serving-benchmark\n"
                    f"   No fallback prompts available - fix dataset issue"
                )
                print(error_msg)
                raise SystemExit("Dataset loading failed - no fallbacks available")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            
        return prompts


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description="vLLM Benchmark")
    parser.add_argument("--mode", choices=["builtin", "direct", "both"], 
                       default="both", help="Benchmark mode")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 vLLM Inference Engine Benchmark")
    print("=" * 60)
    
    # MANDATORY REQUIREMENTS VALIDATION - NO TOLERANCE FOR MISSING PACKAGES
    print("🔍 Validating vLLM requirements...")
    validate_benchmark_requirements("vllm")
    print("✅ vLLM requirements validated successfully!")
    
    # Prepare environment
    clear_gpu_memory()
    warm_up_gpu()
    
    # Initialize benchmark
    benchmark = VLLMBenchmark()
    results = {"engine": "vLLM", "timestamp": time.time()}
    
    # Run benchmark(s)
    if args.mode in ["builtin", "both"]:
        builtin_results = benchmark.run_builtin_benchmark()
        if builtin_results:
            results["builtin_benchmark"] = builtin_results
    
    if args.mode in ["direct", "both"]:
        direct_results = benchmark.run_direct_benchmark()
        if direct_results:
            results["direct_benchmark"] = direct_results
    
    # Save results
    if results.get("builtin_benchmark") or results.get("direct_benchmark"):
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "vllm_results.json")
        save_results(results, output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 vLLM Benchmark Summary")
        print("=" * 60)
        
        if "direct_benchmark" in results:
            metrics = results["direct_benchmark"]
            print(f"🎯 Throughput: {metrics.get('throughput_tokens_per_second', 0):.2f} tokens/sec")
            if "latency_statistics" in metrics:
                lat_stats = metrics["latency_statistics"]
                print(f"⏱️ Latency P50: {lat_stats.get('p50', 0):.3f}s")
                print(f"⏱️ Latency P95: {lat_stats.get('p95', 0):.3f}s")
            
            if "gpu_statistics" in metrics:
                gpu_stats = metrics["gpu_statistics"]
                for gpu_id, stats in gpu_stats.items():
                    if isinstance(stats, dict):
                        print(f"🎮 {gpu_id.upper()}: {stats.get('utilization_mean_percent', 0):.1f}% avg util, "
                              f"{stats.get('memory_peak_percent', 0):.1f}% peak memory")
        
        print(f"💾 Results saved to: {output_path}")
    else:
        print("❌ No benchmark results obtained")
        sys.exit(1)
    
    print("✅ vLLM benchmark completed!")


if __name__ == "__main__":
    main()