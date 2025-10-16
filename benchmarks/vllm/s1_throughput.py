#!/usr/bin/env python3
"""
vLLM S1 Throughput Test
Tests high-throughput scenario with concurrent requests
"""

import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
import argparse
from pathlib import Path
import sys

class VLLMThroughputTest:
    def __init__(self, model_name="gpt-oss-20b", tensor_parallel_size=1):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        
    def initialize_model(self):
        """Initialize vLLM model"""
        print(f"Initializing vLLM model: {self.model_name}")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.85,
            max_num_seqs=200,
            max_model_len=4096
        )
        print("Model initialized successfully")
    
    def load_prompts(self, dataset_path):
        """Load prompts from JSONL file"""
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])
        return prompts[:1000]  # Limit to 1000 for throughput test
    
    def run_throughput_test(self, prompts, batch_size=8, max_tokens=512):
        """Run throughput benchmark"""
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens,
            repetition_penalty=1.1
        )
        
        results = []
        total_tokens = 0
        
        print(f"Running throughput test with {len(prompts)} prompts, batch_size={batch_size}")
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            start_time = time.time()
            outputs = self.llm.generate(batch_prompts, sampling_params)
            batch_time = time.time() - start_time
            
            batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens += batch_tokens
            
            batch_result = {
                "batch_id": i // batch_size,
                "batch_size": len(batch_prompts),
                "batch_time_seconds": batch_time,
                "batch_tokens": batch_tokens,
                "tokens_per_second": batch_tokens / batch_time if batch_time > 0 else 0,
                "prompts_per_second": len(batch_prompts) / batch_time if batch_time > 0 else 0
            }
            results.append(batch_result)
            
            print(f"Batch {i//batch_size + 1}: {batch_tokens} tokens in {batch_time:.2f}s "
                  f"({batch_tokens/batch_time:.1f} tokens/s)")
        
        return results, total_tokens
    
    def save_results(self, results, total_tokens, output_dir):
        """Save benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics
        total_time = sum(r["batch_time_seconds"] for r in results)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        summary = {
            "test_type": "s1_throughput",
            "engine": "vllm",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "total_batches": len(results),
                "total_prompts": sum(r["batch_size"] for r in results)
            },
            "metrics": {
                "total_tokens_generated": total_tokens,
                "total_time_seconds": total_time,
                "average_tokens_per_second": avg_tokens_per_sec,
                "peak_tokens_per_second": max(r["tokens_per_second"] for r in results),
                "min_tokens_per_second": min(r["tokens_per_second"] for r in results)
            },
            "batch_results": results
        }
        
        # Save detailed results
        result_file = output_dir / f"vllm_s1_throughput_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        print(f"Average throughput: {avg_tokens_per_sec:.1f} tokens/second")
        
        return result_file

def main():
    parser = argparse.ArgumentParser(description="vLLM S1 Throughput Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--dataset", default="./datasets/sharegpt_prompts.jsonl", help="Dataset path")
    parser.add_argument("--output", default="./results/vllm/s1_throughput", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = VLLMThroughputTest(args.model, args.tensor_parallel)
        test.initialize_model()
        
        # Load prompts
        prompts = test.load_prompts(args.dataset)
        print(f"Loaded {len(prompts)} prompts")
        
        # Run test
        print("Starting throughput test...")
        results, total_tokens = test.run_throughput_test(
            prompts, args.batch_size, args.max_tokens
        )
        
        # Save results
        result_file = test.save_results(results, total_tokens, args.output)
        
        print("✅ vLLM S1 Throughput test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()