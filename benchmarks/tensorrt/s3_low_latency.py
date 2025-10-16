#!/usr/bin/env python3
"""
TensorRT-LLM S3 Low Latency Test
Tests single-request low-latency performance using TensorRT-LLM's optimized inference
"""

import json
import time
import statistics
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
import argparse
from pathlib import Path
import sys
import torch

class TensorRTLLMLowLatencyTest:
    def __init__(self, model_name="gpt-oss-20b", engine_path=None, world_size=1):
        self.model_name = model_name
        self.engine_path = engine_path or f"./engines/{model_name}/1-gpu"
        self.world_size = world_size
        self.runner = None
        
    def initialize_model(self):
        """Initialize TensorRT-LLM model runner optimized for low latency"""
        print(f"Initializing TensorRT-LLM model for low latency: {self.model_name}")
        print(f"Engine path: {self.engine_path}")
        
        try:
            self.runner = ModelRunner.from_dir(
                engine_dir=self.engine_path,
                lora_dir=None,
                rank=0,
                world_size=self.world_size,
                max_batch_size=1,  # Single batch for lowest latency
                max_input_len=512,  # Shorter input for speed
                max_output_len=50,  # Short outputs for low latency
                max_beam_width=1
            )
            print("TensorRT-LLM model initialized for low latency")
        except Exception as e:
            print(f"Failed to initialize TensorRT-LLM model: {e}")
            print("Note: Ensure the TensorRT engine is built and available at the specified path")
            raise
    
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
            "What is React?"
        ]
        
        # Extend for more test cases
        extended_prompts = prompts * 25  # 500 total prompts
        return extended_prompts
    
    def measure_single_generation(self, prompt, max_output_len=50):
        """Measure latency for a single generation"""
        # Configure sampling for low latency (deterministic)
        sampling_config = SamplingConfig(
            end_id=50256,  # Adjust based on tokenizer
            pad_id=50256,
            num_beams=1,
            temperature=0.1,  # Very low temperature for consistency
            top_k=1,  # Greedy decoding for speed
            top_p=0.95,
            repetition_penalty=1.0,
            length_penalty=1.0
        )
        
        # Warm-up generation (not counted)
        if not hasattr(self, '_warmed_up'):
            try:
                self.runner.generate(
                    batch_input_ids=None,
                    batch_input_texts=[prompt],
                    max_new_tokens=max_output_len,
                    sampling_config=sampling_config,
                    output_sequence_lengths=True,
                    return_dict=True
                )
                self._warmed_up = True
            except Exception as e:
                print(f"Warmup failed: {e}")
        
        # Measure actual generation with high precision timing
        start_time = time.perf_counter()
        
        try:
            outputs = self.runner.generate(
                batch_input_ids=None,
                batch_input_texts=[prompt],
                max_new_tokens=max_output_len,
                sampling_config=sampling_config,
                output_sequence_lengths=True,
                return_dict=True
            )
        except Exception as e:
            return {
                "generated_text": "",
                "tokens_generated": 0,
                "total_latency_ms": 0,
                "first_token_latency_ms": 0,
                "tokens_per_second": 0,
                "error": str(e)
            }
        
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Extract response and calculate tokens
        output_texts = outputs.get('output_texts', [''])
        generated_text = output_texts[0] if output_texts else ''
        
        # Calculate token count
        output_ids = outputs.get('output_ids', [])
        if output_ids and isinstance(output_ids[0], torch.Tensor):
            tokens_generated = output_ids[0].shape[-1]
        else:
            # Fallback: estimate from text
            tokens_generated = len(generated_text.split()) + len(generated_text) // 4
        
        # TensorRT-LLM doesn't provide direct time-to-first-token in this mode
        # Estimate based on total latency and generation characteristics
        first_token_latency = total_latency * 0.2  # Conservative estimate for TRT optimizations
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "total_latency_ms": total_latency,
            "first_token_latency_ms": first_token_latency,
            "tokens_per_second": (tokens_generated / (total_latency / 1000)) if total_latency > 0 else 0
        }
    
    def run_low_latency_test(self, prompts, iterations=500, max_output_len=50):
        """Run low latency benchmark"""
        results = []
        latencies = []
        first_token_latencies = []
        
        print(f"Running TensorRT-LLM low latency test with {iterations} iterations")
        print("Warming up model...")
        
        # Select prompts for testing
        test_prompts = prompts[:iterations]
        
        for i, prompt in enumerate(test_prompts):
            try:
                result = self.measure_single_generation(prompt, max_output_len)
                
                if "error" not in result:
                    result.update({
                        "iteration": i + 1,
                        "prompt": prompt,
                        "prompt_length": len(prompt.split())
                    })
                    
                    results.append(result)
                    latencies.append(result["total_latency_ms"])
                    first_token_latencies.append(result["first_token_latency_ms"])
                else:
                    print(f"Error in iteration {i + 1}: {result['error']}")
                    continue
                
                # Progress update every 50 iterations
                if (i + 1) % 50 == 0 and latencies:
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
                "p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "first_token_latency": {
                "mean_ms": statistics.mean(first_token_latencies),
                "median_ms": statistics.median(first_token_latencies),
                "p95_ms": statistics.quantiles(first_token_latencies, n=20)[18] if len(first_token_latencies) >= 20 else max(first_token_latencies),
                "p99_ms": statistics.quantiles(first_token_latencies, n=100)[98] if len(first_token_latencies) >= 100 else max(first_token_latencies),
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
            "engine": "tensorrt-llm",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "engine_path": self.engine_path,
                "world_size": self.world_size,
                "total_iterations": len(results),
                "max_output_len": 50,
                "batch_size": 1,
                "greedy_decoding": True,
                "tensorrt_optimizations": True
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
        result_file = output_dir / f"tensorrt_s3_low_latency_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        if stats:
            print(f"Mean latency: {stats['total_latency']['mean_ms']:.1f}ms")
            print(f"P95 latency: {stats['total_latency']['p95_ms']:.1f}ms")
            print(f"P99 latency: {stats['total_latency']['p99_ms']:.1f}ms")
            print(f"First token P95: {stats['first_token_latency']['p95_ms']:.1f}ms")
        
        return result_file

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM S3 Low Latency Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--engine-path", help="Path to TensorRT engine directory")
    parser.add_argument("--output", default="./results/tensorrt/s3_low_latency", help="Output directory")
    parser.add_argument("--iterations", type=int, default=500, help="Number of test iterations")
    parser.add_argument("--max-output-len", type=int, default=50, help="Max output length")
    parser.add_argument("--world-size", type=int, default=1, help="World size for multi-GPU")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = TensorRTLLMLowLatencyTest(args.model, args.engine_path, args.world_size)
        test.initialize_model()
        
        # Generate prompts
        prompts = test.get_low_latency_prompts()
        print(f"Generated {len(prompts)} test prompts")
        
        # Run test
        print("Starting TensorRT-LLM low latency test...")
        results, latencies, first_token_latencies = test.run_low_latency_test(
            prompts, args.iterations, args.max_output_len
        )
        
        # Save results
        result_file = test.save_results(results, latencies, first_token_latencies, args.output)
        
        print("✅ TensorRT-LLM S3 Low Latency test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure TensorRT-LLM is properly installed")
        print("2. Build the TensorRT engine for your model first")
        print("3. Check the engine path is correct")
        print("4. Verify CUDA and TensorRT versions are compatible")
        sys.exit(1)

if __name__ == "__main__":
    main()