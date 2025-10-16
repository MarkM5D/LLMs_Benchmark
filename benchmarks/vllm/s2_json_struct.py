#!/usr/bin/env python3
"""
vLLM S2 JSON Structure Test
Tests structured JSON output generation capabilities
"""

import json
import time
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"❌ vLLM import failed: {e}")
    print("Please install vLLM: pip install vllm")
    exit(1)
import argparse
from pathlib import Path
import sys
import re

class VLLMJSONStructTest:
    def __init__(self, model_name="openai/gpt-oss-20b", tensor_parallel_size=1):
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
            max_num_seqs=50,
            max_model_len=4096
        )
        print("Model initialized successfully")
    
    def get_json_prompts(self):
        """Generate prompts that require structured JSON responses"""
        prompts = [
            {
                "prompt": "Generate a JSON object describing a person with name, age, occupation, and skills array:",
                "expected_fields": ["name", "age", "occupation", "skills"]
            },
            {
                "prompt": "Create a JSON product catalog entry with id, name, price, description, and categories:",
                "expected_fields": ["id", "name", "price", "description", "categories"]
            },
            {
                "prompt": "Generate a JSON configuration object with database settings including host, port, username, and options:",
                "expected_fields": ["host", "port", "username", "options"]
            },
            {
                "prompt": "Create a JSON API response with status, message, data object, and timestamp:",
                "expected_fields": ["status", "message", "data", "timestamp"]
            },
            {
                "prompt": "Generate a JSON user profile with personal info, preferences, and activity history:",
                "expected_fields": ["personal_info", "preferences", "activity_history"]
            }
        ]
        
        # Repeat prompts to get more test cases
        extended_prompts = []
        for _ in range(200):  # 200 * 5 = 1000 test cases
            extended_prompts.extend(prompts)
        
        return extended_prompts
    
    def validate_json_output(self, text, expected_fields):
        """Validate if output contains valid JSON with expected fields"""
        try:
            # Try to extract JSON from the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return False, "No JSON object found"
            
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Check if it's a dictionary
            if not isinstance(parsed, dict):
                return False, "JSON is not an object"
            
            # Check for expected fields
            missing_fields = [field for field in expected_fields if field not in parsed]
            if missing_fields:
                return False, f"Missing fields: {missing_fields}"
            
            return True, parsed
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def run_json_struct_test(self, prompts, max_tokens=256):
        """Run JSON structure benchmark"""
        sampling_params = SamplingParams(
            temperature=0.3,  # Lower temperature for more consistent structure
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.1
        )
        
        results = []
        successful_generations = 0
        total_tokens = 0
        
        print(f"Running JSON structure test with {len(prompts)} prompts")
        
        # Process prompts in batches
        batch_size = 10
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_texts = [p["prompt"] + " Respond only with valid JSON:" for p in batch_prompts]
            
            start_time = time.time()
            outputs = self.llm.generate(batch_texts, sampling_params)
            generation_time = time.time() - start_time
            
            # Validate each output
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                tokens_generated = len(output.outputs[0].token_ids)
                total_tokens += tokens_generated
                
                # Validate JSON structure
                is_valid, validation_result = self.validate_json_output(
                    generated_text, 
                    batch_prompts[j]["expected_fields"]
                )
                
                if is_valid:
                    successful_generations += 1
                
                result = {
                    "prompt_id": i + j,
                    "prompt": batch_prompts[j]["prompt"],
                    "expected_fields": batch_prompts[j]["expected_fields"],
                    "generated_text": generated_text,
                    "tokens_generated": tokens_generated,
                    "generation_time_ms": (generation_time / len(outputs)) * 1000,
                    "is_valid_json": is_valid,
                    "validation_result": validation_result if is_valid else str(validation_result),
                    "success": is_valid
                }
                results.append(result)
            
            if (i // batch_size + 1) % 10 == 0:
                success_rate = successful_generations / len(results) * 100 if results else 0
                print(f"Processed {len(results)} prompts, success rate: {success_rate:.1f}%")
        
        return results, total_tokens, successful_generations
    
    def save_results(self, results, total_tokens, successful_generations, output_dir):
        """Save benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics
        total_prompts = len(results)
        success_rate = (successful_generations / total_prompts * 100) if total_prompts > 0 else 0
        avg_tokens_per_gen = total_tokens / total_prompts if total_prompts > 0 else 0
        avg_generation_time = sum(r["generation_time_ms"] for r in results) / total_prompts if total_prompts > 0 else 0
        
        summary = {
            "test_type": "s2_json_struct",
            "engine": "vllm",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "total_prompts": total_prompts,
                "temperature": 0.3
            },
            "metrics": {
                "total_prompts": total_prompts,
                "successful_generations": successful_generations,
                "success_rate_percent": success_rate,
                "total_tokens_generated": total_tokens,
                "average_tokens_per_generation": avg_tokens_per_gen,
                "average_generation_time_ms": avg_generation_time
            },
            "detailed_results": results
        }
        
        # Save detailed results
        result_file = output_dir / f"vllm_s2_json_struct_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        print(f"Success rate: {success_rate:.1f}% ({successful_generations}/{total_prompts})")
        print(f"Average generation time: {avg_generation_time:.1f}ms")
        
        return result_file

def main():
    parser = argparse.ArgumentParser(description="vLLM S2 JSON Structure Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--dataset", default="./datasets/sharegpt_prompts.jsonl", help="Dataset path")
    parser.add_argument("--output", default="./results/vllm/s2_json_struct", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--tensor-parallel", dest="tensor_parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to test")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = VLLMJSONStructTest(args.model, args.tensor_parallel)
        test.initialize_model()
        
        # Generate prompts
        all_prompts = test.get_json_prompts()
        prompts = all_prompts[:args.num_prompts]
        print(f"Testing with {len(prompts)} JSON structure prompts")
        
        # Run test
        print("Starting JSON structure test...")
        results, total_tokens, successful_generations = test.run_json_struct_test(
            prompts, args.max_tokens
        )
        
        # Save results
        result_file = test.save_results(results, total_tokens, successful_generations, args.output)
        
        print("✅ vLLM S2 JSON Structure test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()