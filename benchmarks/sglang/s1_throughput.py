#!/usr/bin/env python3
"""
SGLang S1 Throughput Test
Tests high-throughput scenario with concurrent requests using SGLang engine
"""

import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sglang as sgl
from sglang import Runtime, set_default_backend
import argparse
from pathlib import Path
import sys

class SGLangThroughputTest:
    def __init__(self, model_name="gpt-oss-20b", tp_size=1):
        self.model_name = model_name
        self.tp_size = tp_size
        self.runtime = None
        
    def initialize_model(self):
        """Initialize SGLang runtime"""
        print(f"Initializing SGLang runtime: {self.model_name}")
        self.runtime = Runtime(
            model_path=self.model_name,
            tp_size=self.tp_size,
            mem_fraction_static=0.85,
            max_running_requests=200,
            context_length=4096
        )
        set_default_backend(self.runtime)
        print("SGLang runtime initialized successfully")
    
    def load_prompts(self, dataset_path):
        """Load prompts from JSONL file"""
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])
        return prompts[:1000]  # Limit to 1000 for throughput test
    
    @sgl.function
    def generate_response(self, s, prompt, max_tokens=512):
        """SGLang generation function - STANDARDIZED PARAMETERS"""
        s += sgl.user(prompt)
        s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=0.8, top_p=0.95, frequency_penalty=1.1))
    
    def run_throughput_test(self, prompts, batch_size=8, max_tokens=512):
        """Run throughput benchmark using SGLang"""
        results = []
        total_tokens = 0
        
        print(f"Running SGLang throughput test with {len(prompts)} prompts, batch_size={batch_size}")
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            start_time = time.time()
            
            # Create batch of generation tasks
            states = []
            for prompt in batch_prompts:
                state = self.generate_response.run(prompt=prompt, max_tokens=max_tokens)
                states.append(state)
            
            # Wait for all completions (SGLang handles batching internally)
            batch_tokens = 0
            for state in states:
                if hasattr(state, 'response') and state.response:
                    # Estimate token count from response length
                    response_tokens = len(state.response.split()) * 1.3  # Rough approximation
                    batch_tokens += int(response_tokens)
            
            batch_time = time.time() - start_time
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
            "engine": "sglang",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "tp_size": self.tp_size,
                "total_batches": len(results),
                "total_prompts": sum(r["batch_size"] for r in results),
                "radix_attention": True  # SGLang's key feature
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
        result_file = output_dir / f"sglang_s1_throughput_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        print(f"Average throughput: {avg_tokens_per_sec:.1f} tokens/second")
        
        return result_file
    
    def shutdown(self):
        """Cleanup SGLang runtime"""
        if self.runtime:
            self.runtime.shutdown()

def main():
    parser = argparse.ArgumentParser(description="SGLang S1 Throughput Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--dataset", default="./datasets/sharegpt_prompts.jsonl", help="Dataset path")
    parser.add_argument("--output", default="./results/sglang/s1_throughput", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = SGLangThroughputTest(args.model, args.tp_size)
        test.initialize_model()
        
        # Load prompts
        prompts = test.load_prompts(args.dataset)
        print(f"Loaded {len(prompts)} prompts")
        
        # Run test
        print("Starting SGLang throughput test...")
        results, total_tokens = test.run_throughput_test(
            prompts, args.batch_size, args.max_tokens
        )
        
        # Save results
        result_file = test.save_results(results, total_tokens, args.output)
        
        print("✅ SGLang S1 Throughput test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'test' in locals():
            test.shutdown()

if __name__ == "__main__":
    main()