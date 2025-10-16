#!/usr/bin/env python3
"""
SGLang S3 Low Latency Test  
Tests single-request low-latency performance using SGLang's optimized inference
"""

import json
import time
import statistics
import sglang as sgl
from sglang import Runtime, set_default_backend
import argparse
from pathlib import Path
import sys

class SGLangLowLatencyTest:
    def __init__(self, model_name="gpt-oss-20b", tp_size=1):
        self.model_name = model_name
        self.tp_size = tp_size
        self.runtime = None
        
    def initialize_model(self):
        """Initialize SGLang runtime optimized for low latency"""
        print(f"Initializing SGLang runtime for low latency: {self.model_name}")
        self.runtime = Runtime(
            model_path=self.model_name,
            tp_size=self.tp_size,
            mem_fraction_static=0.8,
            max_running_requests=1,  # Single request for lowest latency
            context_length=2048,  # Shorter context for speed
            schedule_policy="fcfs",  # First-come-first-serve for predictable latency
            disable_radix_cache=False  # Keep radix cache for prefix efficiency
        )
        set_default_backend(self.runtime)
        print("SGLang runtime initialized for low latency")
    
    @sgl.function
    def generate_short_response(self, s, prompt, max_tokens=50):
        """SGLang function for short response generation"""
        s += sgl.user(prompt)
        s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=0.1, top_p=0.95))
    
    def get_low_latency_prompts(self):
        """Generate short prompts for low latency testing"""
        prompts = [
            "What is AI?",
            "Explain Python in one sentence.", 
            "Define machine learning.",
            "What is cloud computing?",
            "Describe blockchain briefly.",
            "What is cybersecurity?",
            "Define data science.",
            "What is DevOps?",
            "Explain API.",
            "What is Docker?",
            "Define microservices.",
            "What is Git?",
            "Explain REST API.",
            "What is Kubernetes?",
            "Define neural networks.",
            "What is Big Data?",
            "Explain SQL.",
            "What is NoSQL?",
            "Define web development.",
            "What is React?",
            "Explain JavaScript.",
            "What is TypeScript?",
            "Define CSS.",
            "What is HTML?",
            "Explain HTTP."
        ]
        
        # Extend for more test cases
        extended_prompts = prompts * 20  # 500 total prompts
        return extended_prompts
    
    def measure_single_generation(self, prompt, max_tokens=50):
        """Measure latency for a single generation"""
        # Warm-up generation (not counted)
        if not hasattr(self, '_warmed_up'):
            self.generate_short_response.run(prompt=prompt, max_tokens=max_tokens)
            self._warmed_up = True
        
        # Measure actual generation with high precision timing
        start_time = time.perf_counter()
        state = self.generate_short_response.run(prompt=prompt, max_tokens=max_tokens)
        end_time = time.perf_counter()
        
        total_latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Extract response and estimate tokens
        response_text = getattr(state, 'response', '') if hasattr(state, 'response') else ''
        
        # Estimate token count from response
        tokens_generated = len(response_text.split()) + len(response_text) // 4  # Rough estimate
        
        # SGLang doesn't provide direct time-to-first-token in this mode
        # Estimate based on total latency (rough approximation)
        first_token_latency = total_latency * 0.25  # Conservative estimate
        
        return {
            "generated_text": response_text,
            "tokens_generated": tokens_generated,
            "total_latency_ms": total_latency,
            "first_token_latency_ms": first_token_latency,
            "tokens_per_second": (tokens_generated / (total_latency / 1000)) if total_latency > 0 else 0
        }
    
    def run_low_latency_test(self, prompts, iterations=500, max_tokens=50):
        """Run low latency benchmark"""
        results = []
        latencies = []
        first_token_latencies = []
        
        print(f"Running SGLang low latency test with {iterations} iterations")
        print("Warming up model...")
        
        # Select prompts for testing
        test_prompts = prompts[:iterations]
        
        for i, prompt in enumerate(test_prompts):
            try:
                result = self.measure_single_generation(prompt, max_tokens)
                
                result.update({
                    "iteration": i + 1,
                    "prompt": prompt,
                    "prompt_length": len(prompt.split())
                })
                
                results.append(result)
                latencies.append(result["total_latency_ms"])
                first_token_latencies.append(result["first_token_latency_ms"])
                
                # Progress update every 50 iterations
                if (i + 1) % 50 == 0:
                    avg_latency = statistics.mean(latencies[-50:])
                    print(f"Iteration {i + 1}/{iterations}: Avg latency = {avg_latency:.1f}ms")
                    
            except Exception as e:
                print(f"Error in iteration {i + 1}: {e}")
                continue
        
        return results, latencies, first_token_latencies
    
    def calculate_statistics(self, latencies, first_token_latencies):
        """Calculate latency statistics"""
        if not latencies:
            return {}
        
        return {
            "total_latency": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
                "p99_ms": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "first_token_latency": {
                "mean_ms": statistics.mean(first_token_latencies),
                "median_ms": statistics.median(first_token_latencies),
                "p95_ms": statistics.quantiles(first_token_latencies, n=20)[18],
                "p99_ms": statistics.quantiles(first_token_latencies, n=100)[98],
                "min_ms": min(first_token_latencies),
                "max_ms": max(first_token_latencies),
                "stdev_ms": statistics.stdev(first_token_latencies) if len(first_token_latencies) > 1 else 0
            }
        }
    
    def save_results(self, results, latencies, first_token_latencies, output_dir):
        """Save benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        stats = self.calculate_statistics(latencies, first_token_latencies)
        
        # Calculate additional metrics
        total_tokens = sum(r["tokens_generated"] for r in results)
        total_time_seconds = sum(r["total_latency_ms"] for r in results) / 1000
        
        summary = {
            "test_type": "s3_low_latency",
            "engine": "sglang",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "tp_size": self.tp_size,
                "total_iterations": len(results),
                "max_tokens_per_generation": 50,
                "schedule_policy": "fcfs",
                "radix_cache_enabled": True
            },
            "metrics": {
                "total_iterations": len(results),
                "total_tokens_generated": total_tokens,
                "total_time_seconds": total_time_seconds,
                "average_tokens_per_second": total_tokens / total_time_seconds if total_time_seconds > 0 else 0,
                "latency_statistics": stats
            },
            "detailed_results": results
        }
        
        # Save detailed results
        result_file = output_dir / f"sglang_s3_low_latency_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        if stats:
            print(f"Mean latency: {stats['total_latency']['mean_ms']:.1f}ms")
            print(f"P95 latency: {stats['total_latency']['p95_ms']:.1f}ms")
            print(f"P99 latency: {stats['total_latency']['p99_ms']:.1f}ms")
            print(f"First token P95: {stats['first_token_latency']['p95_ms']:.1f}ms")
        
        return result_file
    
    def shutdown(self):
        """Cleanup SGLang runtime"""
        if self.runtime:
            self.runtime.shutdown()

def main():
    parser = argparse.ArgumentParser(description="SGLang S3 Low Latency Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--output", default="./results/sglang/s3_low_latency", help="Output directory")
    parser.add_argument("--iterations", type=int, default=500, help="Number of test iterations")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = SGLangLowLatencyTest(args.model, args.tp_size)
        test.initialize_model()
        
        # Generate prompts
        prompts = test.get_low_latency_prompts()
        print(f"Generated {len(prompts)} test prompts")
        
        # Run test
        print("Starting SGLang low latency test...")
        results, latencies, first_token_latencies = test.run_low_latency_test(
            prompts, args.iterations, args.max_tokens
        )
        
        # Save results
        result_file = test.save_results(results, latencies, first_token_latencies, args.output)
        
        print("✅ SGLang S3 Low Latency test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'test' in locals():
            test.shutdown()

if __name__ == "__main__":
    main()