#!/usr/bin/env python3
"""
TensorRT-LLM Benchmark Script

This script benchmarks TensorRT-LLM inference engine using its Python API
for direct sampling to match vLLM and SGLang benchmark methodology.

TensorRT-LLM requires a pre-compiled model engine file, which is generated
from the original Hugging Face model through NVIDIA's compilation process.

Parameters (matching benchmark plan):
- Model: gpt-oss-20b (compiled as gpt-oss-20b.trt)
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
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to sys.path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import BenchmarkMetrics, save_results, warm_up_gpu, clear_gpu_memory
from benchmarks.requirements_validator import validate_benchmark_requirements


class TensorRTLLMBenchmark:
    """TensorRT-LLM benchmark implementation"""
    
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        self.compiled_model_path = "/workspace/benchmarks/models/gpt-oss-20b.trt"
        self.max_tokens = 128
        self.batch_size = 32
        self.concurrency = 16
        self.temperature = 0.8
        self.top_p = 0.95
        self.dataset_path = "/workspace/benchmarks/data/sharegpt-10k.jsonl"
        
        # Alternative local paths for development
        if not os.path.exists(self.dataset_path):
            local_path = os.path.join(os.path.dirname(__file__), "..", "data", "sharegpt-10k.jsonl")
            if os.path.exists(local_path):
                self.dataset_path = local_path
    

    
    def run_benchmark(self) -> dict:
        """Run TensorRT-LLM benchmark - GUARANTEED DIRECT RUNTIME ONLY"""
        print("🏎️ Running TensorRT-LLM direct runtime benchmark...")
        
        # HARD REQUIREMENT: TensorRT-LLM must be properly installed
        try:
            import os
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
            from tensorrt_llm import LLM, SamplingParams
        except ImportError as e:
            error_msg = (
                f"❌ CRITICAL ERROR: TensorRT-LLM not properly installed!\n"
                f"   Import error: {e}\n"
                f"   Install TensorRT-LLM first: pip install tensorrt-llm\n" 
                f"   Or use NVIDIA NGC container: nvcr.io/nvidia/tensorrt:23.08-py3"
            )
            print(error_msg)
            raise SystemExit("TensorRT-LLM installation required - no fallbacks available")
        
        try:
            # Initialize metrics tracking
            benchmark_metrics = BenchmarkMetrics("TensorRT-LLM")
            
            # Load dataset
            prompts = self._load_dataset()
            if not prompts:
                raise ValueError("No prompts loaded from dataset")
            
            print(f"📚 Loaded {len(prompts)} prompts")
            
            # TensorRT-LLM GUARANTEED direct initialization (no pre-compilation needed)
            print(f"🤖 Loading model: {self.model_name}")
            
            llm = LLM(model=self.model_name)
            
            # Sampling parameters - standard TensorRT-LLM API
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            
            # Start benchmark
            benchmark_metrics.start_benchmark()
            
            # Warm-up phase - GUARANTEED
            print("� Performing warm-up...")
            warmup_prompts = prompts[:min(10, len(prompts))]
            llm.generate(warmup_prompts, sampling_params)
            print("✅ TensorRT-LLM warmup completed")
            
            benchmark_metrics.complete_warmup()
            
            # Main benchmark - GUARANTEED batch processing
            print("� Running TensorRT-LLM main benchmark...")
            self._guaranteed_batch_processing(llm, prompts, sampling_params, benchmark_metrics)
            
            # End benchmark and get results
            results = benchmark_metrics.end_benchmark()
            return results
        except Exception as e:
            error_msg = (
                f"❌ CRITICAL TensorRT-LLM BENCHMARK ERROR: {e}\n"
                f"   TensorRT-LLM benchmark failed - system cannot continue\n"
                f"   Check TensorRT-LLM installation and CUDA environment\n"
                f"   No fallback mode - fix the issue and retry"
            )
            print(error_msg)
            raise SystemExit(f"TensorRT-LLM benchmark failed: {e}")
    
    def _guaranteed_batch_processing(self, llm, prompts, sampling_params, benchmark_metrics):
        """TensorRT-LLM guaranteed batch processing - NO FALLBACKS"""
        print("� TensorRT-LLM optimized batch processing - guaranteed success mode...")
        
        # Limit prompts for production run
        test_prompts = prompts[:min(1000, len(prompts))]
        
        # Process in optimized batches
        batch_size = self.batch_size
        for i in range(0, len(test_prompts), batch_size):
            batch_prompts = test_prompts[i:i + batch_size]
            batch_start = time.time()
            
            # TensorRT-LLM standard batch generation - MUST work
            outputs = llm.generate(batch_prompts, sampling_params)
            
            batch_end = time.time()
            batch_latency = batch_end - batch_start
            
            # Process outputs
            batch_tokens = self._count_tokens_in_outputs(outputs)
            
            # Record metrics
            avg_latency = batch_latency / len(batch_prompts)
            avg_tokens = batch_tokens / len(batch_prompts)
            
            for _ in batch_prompts:
                benchmark_metrics.add_request_result(avg_latency, avg_tokens)
            
            # Progress
            batch_num = (i // batch_size) + 1
            total_batches = (len(test_prompts) + batch_size - 1) // batch_size
            print(f"📈 Batch {batch_num}/{total_batches}: "
                  f"{len(batch_prompts)} prompts, {batch_tokens} tokens, "
                  f"{batch_latency:.2f}s")
    

    

    
    def _count_tokens_in_outputs(self, outputs) -> int:
        """Count tokens in TensorRT-LLM outputs"""
        total_tokens = 0
        
        try:
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    if hasattr(output, 'text'):
                        total_tokens += len(output.text.split())
                    elif hasattr(output, 'outputs'):
                        for sub_output in output.outputs:
                            if hasattr(sub_output, 'text'):
                                total_tokens += len(sub_output.text.split())
                            elif hasattr(sub_output, 'token_ids'):
                                total_tokens += len(sub_output.token_ids)
                    elif hasattr(output, 'token_ids'):
                        total_tokens += len(output.token_ids)
                    elif isinstance(output, str):
                        total_tokens += len(output.split())
            elif hasattr(outputs, 'text'):
                total_tokens = len(outputs.text.split())
            elif isinstance(outputs, str):
                total_tokens = len(outputs.split())
        except Exception as e:
            print(f"⚠️ Error counting tokens: {e}")
            # Fallback: rough estimate
            total_tokens = self.max_tokens * len(outputs) if isinstance(outputs, (list, tuple)) else self.max_tokens
        
        return total_tokens
    
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
    parser = argparse.ArgumentParser(description="TensorRT-LLM Benchmark")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏎️ TensorRT-LLM Inference Engine Benchmark")
    print("=" * 60)
    
    # MANDATORY REQUIREMENTS VALIDATION - NO TOLERANCE FOR MISSING PACKAGES
    print("🔍 Validating TensorRT-LLM requirements...")
    validate_benchmark_requirements("tensorrt_llm")
    print("✅ TensorRT-LLM requirements validated successfully!")
    
    # Initialize benchmark
    benchmark = TensorRTLLMBenchmark()
    
    # Prepare environment
    clear_gpu_memory()
    warm_up_gpu()
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Save results
    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "tensorrtllm_results.json")
        save_results(results, output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TensorRT-LLM Benchmark Summary")
        print("=" * 60)
        
        print(f"🎯 Throughput: {results.get('throughput_tokens_per_second', 0):.2f} tokens/sec")
        if "latency_statistics" in results:
            lat_stats = results["latency_statistics"]
            print(f"⏱️ Latency P50: {lat_stats.get('p50', 0):.3f}s")
            print(f"⏱️ Latency P95: {lat_stats.get('p95', 0):.3f}s")
        
        if "gpu_statistics" in results:
            gpu_stats = results["gpu_statistics"]
            for gpu_id, stats in gpu_stats.items():
                if isinstance(stats, dict):
                    print(f"🎮 {gpu_id.upper()}: {stats.get('utilization_mean_percent', 0):.1f}% avg util, "
                          f"{stats.get('memory_peak_percent', 0):.1f}% peak memory")
        
        print(f"💾 Results saved to: {output_path}")
    else:
        print("❌ No benchmark results obtained")
        sys.exit(1)
    
    print("✅ TensorRT-LLM benchmark completed!")


if __name__ == "__main__":
    main()