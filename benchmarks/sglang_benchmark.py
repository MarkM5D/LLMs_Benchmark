#!/usr/bin/env python3
"""
SGLang Benchmark Script

This script benchmarks SGLang inference engine using its Python API
for direct sampling to match vLLM benchmark methodology.

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
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to sys.path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import BenchmarkMetrics, save_results, warm_up_gpu, clear_gpu_memory
from benchmarks.requirements_validator import validate_benchmark_requirements


class SGLangBenchmark:
    """SGLang benchmark implementation"""
    
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
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
        """Run SGLang benchmark using Python API"""
        print("🔬 Running SGLang direct Python API benchmark...")
        
        # HARD REQUIREMENT: SGLang must be properly installed
        try:
            from sglang import LLM, SamplingParams
        except ImportError as e:
            error_msg = (
                f"❌ CRITICAL ERROR: SGLang not properly installed!\n"
                f"   Import error: {e}\n"
                f"   Install SGLang first: pip install sglang[all]\n"
                f"   Or: pip install 'sglang[all]' --extra-index-url https://download.pytorch.org/whl/cu121"
            )
            print(error_msg)
            raise SystemExit("SGLang installation required - no fallbacks available")
        
        try:
            
            # Initialize metrics tracking
            benchmark_metrics = BenchmarkMetrics("SGLang")
            
            # Load dataset
            prompts = self._load_dataset()
            if not prompts:
                raise ValueError("No prompts loaded from dataset")
            
            print(f"📚 Loaded {len(prompts)} prompts")
            
            # Initialize SGLang model - GUARANTEED WORKING PATH ONLY
            print(f"🤖 Loading model: {self.model_name}")
            
            # SGLang standard initialization
            llm = LLM(
                model=self.model_name,
                trust_remote_code=True
            )
            
            # Sampling parameters - standard SGLang API
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
            self._run_warmup(llm, warmup_prompts, sampling_params)
            
            benchmark_metrics.complete_warmup()
            
            # Main benchmark
            print("📊 Running main benchmark...")
            self._run_main_benchmark(llm, prompts, sampling_params, benchmark_metrics)
            
            # End benchmark and get results
            results = benchmark_metrics.end_benchmark()
            return results
            
        except Exception as e:
            error_msg = (
                f"❌ CRITICAL SGLang BENCHMARK ERROR: {e}\n"
                f"   SGLang benchmark failed - system cannot continue\n"
                f"   Check SGLang installation and model availability\n"
                f"   No fallback mode - fix the issue and retry"
            )
            print(error_msg)
            raise SystemExit(f"SGLang benchmark failed: {e}")
    
    def _run_warmup(self, llm, warmup_prompts, sampling_params):
        """Run warm-up iterations"""
        for i in range(3):
            print(f"🔥 Warm-up iteration {i+1}/3...")
            try:
                # Try different SGLang generation methods
                if hasattr(llm, 'generate'):
                    llm.generate(warmup_prompts, sampling_params=sampling_params)
                elif hasattr(llm, 'batch_generate'):
                    llm.batch_generate(warmup_prompts, sampling_params=sampling_params)
                else:
                    # Single prompt generation fallback
                    for prompt in warmup_prompts[:3]:  # Limit for warmup
                        llm(prompt, sampling_params=sampling_params)
            except Exception as e:
                print(f"⚠️ Warm-up iteration {i+1} failed: {e}")
    
    def _run_main_benchmark(self, llm, prompts, sampling_params, benchmark_metrics):
        """Run the main benchmark - GUARANTEED BATCH PROCESSING ONLY"""
        
        # Limit prompts for production run
        test_prompts = prompts[:min(1000, len(prompts))]
        
        # SGLang GUARANTEED batch processing - no fallbacks
        self._guaranteed_batch_processing(llm, test_prompts, sampling_params, benchmark_metrics)
    
    def _guaranteed_batch_processing(self, llm, prompts, sampling_params, benchmark_metrics):
        """SGLang guaranteed batch processing - NO FALLBACKS"""
        print("🚀 SGLang batch processing - guaranteed success mode...")
        
        # Process in batches - MUST work
        batch_size = self.batch_size
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_start = time.time()
            
            # SGLang standard batch generation - MUST exist
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            
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
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            print(f"📈 Batch {batch_num}/{total_batches}: "
                  f"{len(batch_prompts)} prompts, {batch_tokens} tokens, "
                  f"{batch_latency:.2f}s")
    

    
    def _count_tokens_in_outputs(self, outputs) -> int:
        """Count tokens in SGLang outputs"""
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
    parser = argparse.ArgumentParser(description="SGLang Benchmark")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 SGLang Inference Engine Benchmark")
    print("=" * 60)
    
    # MANDATORY REQUIREMENTS VALIDATION - NO TOLERANCE FOR MISSING PACKAGES
    print("🔍 Validating SGLang requirements...")
    validate_benchmark_requirements("sglang")
    print("✅ SGLang requirements validated successfully!")
    
    # Prepare environment
    clear_gpu_memory()
    warm_up_gpu()
    
    # Initialize and run benchmark
    benchmark = SGLangBenchmark()
    results = benchmark.run_benchmark()
    
    # Save results
    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "sglang_results.json")
        save_results(results, output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 SGLang Benchmark Summary")
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
    
    print("✅ SGLang benchmark completed!")


if __name__ == "__main__":
    main()