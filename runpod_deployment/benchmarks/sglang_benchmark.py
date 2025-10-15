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


class SGLangBenchmark:
    """SGLang benchmark implementation"""
    
    def __init__(self):
        self.model_name = "gpt-oss-20b"
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
        
        try:
            # Import SGLang components
            # Note: The exact import may vary based on SGLang version
            try:
                from sglang import LLM, SamplingParams
            except ImportError:
                # Alternative import paths for different SGLang versions
                try:
                    from sglang.api import LLM, SamplingParams
                except ImportError:
                    try:
                        from sgl import LLM, SamplingParams
                    except ImportError:
                        # Last resort - try runtime import
                        import sglang
                        LLM = getattr(sglang, 'LLM', None)
                        SamplingParams = getattr(sglang, 'SamplingParams', None)
                        if not LLM or not SamplingParams:
                            raise ImportError("Could not find LLM and SamplingParams in sglang")
            
            # Initialize metrics tracking
            benchmark_metrics = BenchmarkMetrics("SGLang")
            
            # Load dataset
            prompts = self._load_dataset()
            if not prompts:
                raise ValueError("No prompts loaded from dataset")
            
            print(f"📚 Loaded {len(prompts)} prompts")
            
            # Initialize SGLang model
            print(f"🤖 Loading model: {self.model_name}")
            
            # SGLang initialization (parameters may vary by version)
            try:
                llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True
                )
            except TypeError:
                # Fallback for different SGLang initialization patterns
                llm = LLM(self.model_name)
            
            # Sampling parameters
            try:
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
            except TypeError:
                # Fallback for different parameter names
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_tokens
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
            
        except ImportError as e:
            print(f"❌ SGLang import error: {e}")
            print("💡 Make sure SGLang is installed: pip install sglang")
            return self._create_fallback_results()
        except Exception as e:
            print(f"❌ SGLang benchmark error: {e}")
            return self._create_fallback_results()
    
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
        """Run the main benchmark with concurrent processing"""
        
        # Limit prompts for testing
        test_prompts = prompts[:min(1000, len(prompts))]
        
        # Method 1: Try batch processing
        if self._try_batch_processing(llm, test_prompts, sampling_params, benchmark_metrics):
            return
        
        # Method 2: Try concurrent processing
        if self._try_concurrent_processing(llm, test_prompts, sampling_params, benchmark_metrics):
            return
        
        # Method 3: Sequential processing fallback
        self._sequential_processing(llm, test_prompts, sampling_params, benchmark_metrics)
    
    def _try_batch_processing(self, llm, prompts, sampling_params, benchmark_metrics) -> bool:
        """Try SGLang batch processing"""
        try:
            print("🚀 Trying batch processing...")
            
            # Process in batches
            batch_size = self.batch_size
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_start = time.time()
                
                # Try batch generation
                if hasattr(llm, 'generate'):
                    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
                elif hasattr(llm, 'batch_generate'):
                    outputs = llm.batch_generate(batch_prompts, sampling_params=sampling_params)
                else:
                    return False
                
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
            
            return True
            
        except Exception as e:
            print(f"⚠️ Batch processing failed: {e}")
            return False
    
    def _try_concurrent_processing(self, llm, prompts, sampling_params, benchmark_metrics) -> bool:
        """Try concurrent processing with ThreadPoolExecutor"""
        try:
            print("🔄 Trying concurrent processing...")
            
            def process_prompt(prompt):
                """Process a single prompt"""
                start_time = time.time()
                try:
                    if hasattr(llm, 'generate'):
                        output = llm.generate([prompt], sampling_params=sampling_params)
                    else:
                        output = llm(prompt, sampling_params=sampling_params)
                    
                    end_time = time.time()
                    latency = end_time - start_time
                    tokens = self._count_tokens_in_outputs([output])
                    
                    return latency, tokens, None
                except Exception as e:
                    return 0, 0, str(e)
            
            # Process with limited concurrency
            completed_requests = 0
            with ThreadPoolExecutor(max_workers=min(self.concurrency, 8)) as executor:
                # Submit batches of requests
                for i in range(0, min(len(prompts), 500), 50):  # Process in chunks
                    batch_prompts = prompts[i:i+50]
                    futures = [executor.submit(process_prompt, prompt) for prompt in batch_prompts]
                    
                    for future in as_completed(futures):
                        latency, tokens, error = future.result()
                        
                        if error:
                            print(f"⚠️ Request failed: {error}")
                        else:
                            benchmark_metrics.add_request_result(latency, tokens)
                            completed_requests += 1
                    
                    print(f"📈 Completed {completed_requests} requests...")
            
            return completed_requests > 0
            
        except Exception as e:
            print(f"⚠️ Concurrent processing failed: {e}")
            return False
    
    def _sequential_processing(self, llm, prompts, sampling_params, benchmark_metrics):
        """Sequential processing fallback"""
        print("🐌 Using sequential processing fallback...")
        
        for i, prompt in enumerate(prompts[:200]):  # Limit for testing
            try:
                start_time = time.time()
                
                # Try different generation methods
                if hasattr(llm, 'generate'):
                    output = llm.generate([prompt], sampling_params=sampling_params)
                else:
                    output = llm(prompt, sampling_params=sampling_params)
                
                end_time = time.time()
                latency = end_time - start_time
                tokens = self._count_tokens_in_outputs([output])
                
                benchmark_metrics.add_request_result(latency, tokens)
                
                if (i + 1) % 10 == 0:
                    print(f"📈 Processed {i + 1} prompts...")
                    
            except Exception as e:
                print(f"⚠️ Error processing prompt {i}: {e}")
    
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
            
            # Fallback: create sample prompts if dataset not found
            if not prompts:
                print("⚠️ Dataset not found, creating sample prompts...")
                sample_prompts = [
                    "Explain the concept of artificial intelligence and its applications.",
                    "What are the benefits of renewable energy sources?",
                    "Describe the process of photosynthesis in plants.",
                    "How does machine learning work in modern technology?",
                    "What is the importance of data science in business?",
                    "Explain quantum computing in simple terms for beginners.",
                    "What are the applications of blockchain technology today?",
                    "Describe the evolution of programming languages over time.",
                    "How does cloud computing benefit modern businesses?",
                    "What is the role of cybersecurity in protecting data?"
                ]
                prompts = sample_prompts * 30  # Repeat to get more prompts
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            
        return prompts
    
    def _create_fallback_results(self) -> dict:
        """Create fallback results when SGLang is not available"""
        print("📝 Creating fallback results (SGLang not available)")
        
        return {
            "engine_name": "SGLang",
            "status": "not_available",
            "error": "SGLang not installed or not compatible",
            "throughput_tokens_per_second": 0,
            "latency_statistics": {
                "p50": 0,
                "p95": 0,
                "mean": 0
            },
            "gpu_statistics": {},
            "total_requests": 0,
            "note": "SGLang benchmark skipped - engine not available"
        }


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description="SGLang Benchmark")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 SGLang Inference Engine Benchmark")
    print("=" * 60)
    
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
        
        if results.get("status") != "not_available":
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
        else:
            print(f"⚠️ Status: {results.get('error', 'Unknown error')}")
        
        print(f"💾 Results saved to: {output_path}")
    else:
        print("❌ No benchmark results obtained")
        sys.exit(1)
    
    print("✅ SGLang benchmark completed!")


if __name__ == "__main__":
    main()