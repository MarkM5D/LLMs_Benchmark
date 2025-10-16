#!/usr/bin/env python3
"""
TensorRT-LLM S1 Throughput Test
Tests high-throughput scenario using TensorRT-LLM optimized inference
"""

import json
import time
import numpy as np
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
import argparse
from pathlib import Path
import sys
import torch

class TensorRTLLMThroughputTest:
    def __init__(self, model_name="gpt-oss-20b", engine_path=None, world_size=1):
        self.model_name = model_name
        self.engine_path = engine_path or f"./engines/{model_name}/1-gpu"
        self.world_size = world_size
        self.runner = None
        
    def initialize_model(self):
        """Initialize TensorRT-LLM model runner"""
        print(f"Initializing TensorRT-LLM model: {self.model_name}")
        print(f"Engine path: {self.engine_path}")
        
        try:
            self.runner = ModelRunner.from_dir(
                engine_dir=self.engine_path,
                lora_dir=None,
                rank=0,
                world_size=self.world_size,
                max_batch_size=32,  # Fair batch size - aligned with other engines
                max_input_len=2048,
                max_output_len=512,
                max_beam_width=1
            )
            print("TensorRT-LLM model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize TensorRT-LLM model: {e}")
            print("Note: Ensure the TensorRT engine is built and available at the specified path")
            raise
    
    def load_prompts(self, dataset_path):
        """Load prompts from JSONL file"""
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])
        return prompts[:1000]  # Limit to 1000 for throughput test
    
    def run_throughput_test(self, prompts, batch_size=8, max_output_len=512):
        """Run throughput benchmark using TensorRT-LLM"""
        results = []
        total_tokens = 0
        
        print(f"Running TensorRT-LLM throughput test with {len(prompts)} prompts, batch_size={batch_size}")
        
        # Configure sampling parameters - STANDARDIZED FOR FAIR COMPARISON
        sampling_config = SamplingConfig(
            end_id=50256,  # Adjust based on tokenizer
            pad_id=50256,
            num_beams=1,
            temperature=0.8,  # Same as vLLM/SGLang
            top_k=50,
            top_p=0.95,      # Same as vLLM/SGLang
            repetition_penalty=1.1,  # Same as vLLM
            length_penalty=1.0
        )
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            try:
                start_time = time.time()
                
                # Generate outputs using TensorRT-LLM runner
                outputs = self.runner.generate(
                    batch_input_ids=None,  # Will be tokenized internally
                    batch_input_texts=batch_prompts,
                    max_new_tokens=max_output_len,
                    sampling_config=sampling_config,
                    output_sequence_lengths=True,
                    return_dict=True
                )
                
                batch_time = time.time() - start_time
                
                # Extract tokens and calculate metrics
                batch_tokens = 0
                if 'output_ids' in outputs:
                    for output_ids in outputs['output_ids']:
                        # Count generated tokens (excluding input tokens)
                        if isinstance(output_ids, torch.Tensor):
                            batch_tokens += output_ids.shape[-1]  # Approximate
                        else:
                            batch_tokens += len(output_ids)
                else:
                    # Fallback: estimate from output texts
                    output_texts = outputs.get('output_texts', [''] * len(batch_prompts))
                    for text in output_texts:
                        batch_tokens += len(text.split()) * 1.3  # Rough token estimate
                
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
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add error result to maintain batch tracking
                batch_result = {
                    "batch_id": i // batch_size,
                    "batch_size": len(batch_prompts),
                    "batch_time_seconds": 0,
                    "batch_tokens": 0,
                    "tokens_per_second": 0,
                    "prompts_per_second": 0,
                    "error": str(e)
                }
                results.append(batch_result)
                continue
        
        return results, total_tokens
    
    def save_results(self, results, total_tokens, output_dir):
        """Save benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics (exclude error batches)
        valid_results = [r for r in results if "error" not in r]
        total_time = sum(r["batch_time_seconds"] for r in valid_results)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        summary = {
            "test_type": "s1_throughput",
            "engine": "tensorrt-llm",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "engine_path": self.engine_path,
                "world_size": self.world_size,
                "total_batches": len(results),
                "successful_batches": len(valid_results),
                "total_prompts": sum(r["batch_size"] for r in valid_results),
                "tensorrt_optimizations": True
            },
            "metrics": {
                "total_tokens_generated": total_tokens,
                "total_time_seconds": total_time,
                "average_tokens_per_second": avg_tokens_per_sec,
                "peak_tokens_per_second": max([r["tokens_per_second"] for r in valid_results], default=0),
                "min_tokens_per_second": min([r["tokens_per_second"] for r in valid_results], default=0)
            },
            "batch_results": results
        }
        
        # Save detailed results
        result_file = output_dir / f"tensorrt_s1_throughput_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        print(f"Average throughput: {avg_tokens_per_sec:.1f} tokens/second")
        print(f"Successful batches: {len(valid_results)}/{len(results)}")
        
        return result_file

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM S1 Throughput Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--engine-path", help="Path to TensorRT engine directory")
    parser.add_argument("--dataset", default="./datasets/sharegpt_prompts.jsonl", help="Dataset path")
    parser.add_argument("--output", default="./results/tensorrt/s1_throughput", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-output-len", type=int, default=512, help="Max output length")
    parser.add_argument("--world-size", type=int, default=1, help="World size for multi-GPU")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = TensorRTLLMThroughputTest(args.model, args.engine_path, args.world_size)
        test.initialize_model()
        
        # Load prompts
        prompts = test.load_prompts(args.dataset)
        print(f"Loaded {len(prompts)} prompts")
        
        # Run test
        print("Starting TensorRT-LLM throughput test...")
        results, total_tokens = test.run_throughput_test(
            prompts, args.batch_size, args.max_output_len
        )
        
        # Save results
        result_file = test.save_results(results, total_tokens, args.output)
        
        print("✅ TensorRT-LLM S1 Throughput test completed successfully!")
        
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