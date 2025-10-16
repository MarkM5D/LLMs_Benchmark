#!/usr/bin/env python3
"""
TensorRT-LLM S2 JSON Structure Test
Tests structured JSON output generation using TensorRT-LLM's optimized inference
"""

import json
import time
import re
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
import argparse
from pathlib import Path
import sys
import torch

class TensorRTLLMJSONStructTest:
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
                max_batch_size=32,  # Moderate batch size for JSON tasks
                max_input_len=1024,
                max_output_len=256,
                max_beam_width=1
            )
            print("TensorRT-LLM model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize TensorRT-LLM model: {e}")
            print("Note: Ensure the TensorRT engine is built and available at the specified path")
            raise
    
    def get_json_prompts(self):
        """Generate prompts that require structured JSON responses"""
        prompts = [
            {
                "prompt": "Generate a JSON object describing a person with name, age, occupation, and skills array. Respond only with valid JSON:",
                "expected_fields": ["name", "age", "occupation", "skills"]
            },
            {
                "prompt": "Create a JSON product catalog entry with id, name, price, description, and categories. Respond only with valid JSON:",
                "expected_fields": ["id", "name", "price", "description", "categories"]
            },
            {
                "prompt": "Generate a JSON configuration object with database settings including host, port, username, and options. Respond only with valid JSON:",
                "expected_fields": ["host", "port", "username", "options"]
            },
            {
                "prompt": "Create a JSON API response with status, message, data object, and timestamp. Respond only with valid JSON:",
                "expected_fields": ["status", "message", "data", "timestamp"]
            },
            {
                "prompt": "Generate a JSON user profile with personal_info, preferences, and activity_history. Respond only with valid JSON:",
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
    
    def run_json_struct_test(self, prompts, max_output_len=256):
        """Run JSON structure benchmark"""
        # Configure sampling parameters for more deterministic JSON generation
        sampling_config = SamplingConfig(
            end_id=50256,  # Adjust based on tokenizer
            pad_id=50256,
            num_beams=1,
            temperature=0.3,  # Lower temperature for more consistent structure
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            length_penalty=1.0
        )
        
        results = []
        successful_generations = 0
        total_tokens = 0
        
        print(f"Running TensorRT-LLM JSON structure test with {len(prompts)} prompts")
        
        # Process prompts in batches
        batch_size = 10
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_texts = [p["prompt"] for p in batch_prompts]
            
            try:
                start_time = time.time()
                
                # Generate outputs using TensorRT-LLM runner
                outputs = self.runner.generate(
                    batch_input_ids=None,
                    batch_input_texts=batch_texts,
                    max_new_tokens=max_output_len,
                    sampling_config=sampling_config,
                    output_sequence_lengths=True,
                    return_dict=True
                )
                
                generation_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Process each output in the batch
                output_texts = outputs.get('output_texts', [''] * len(batch_texts))
                output_ids = outputs.get('output_ids', [])
                
                for j, (output_text, prompt_info) in enumerate(zip(output_texts, batch_prompts)):
                    # Calculate tokens generated
                    if j < len(output_ids) and isinstance(output_ids[j], torch.Tensor):
                        tokens_generated = output_ids[j].shape[-1]
                    else:
                        # Fallback: estimate from text
                        tokens_generated = len(output_text.split()) + len(output_text) // 4
                    
                    total_tokens += tokens_generated
                    
                    # Validate JSON structure
                    is_valid, validation_result = self.validate_json_output(
                        output_text, 
                        prompt_info["expected_fields"]
                    )
                    
                    if is_valid:
                        successful_generations += 1
                    
                    result = {
                        "prompt_id": i + j,
                        "prompt": prompt_info["prompt"],
                        "expected_fields": prompt_info["expected_fields"],
                        "generated_text": output_text,
                        "tokens_generated": tokens_generated,
                        "generation_time_ms": generation_time / len(batch_texts),
                        "is_valid_json": is_valid,
                        "validation_result": validation_result if is_valid else str(validation_result),
                        "success": is_valid
                    }
                    results.append(result)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add error results to maintain tracking
                for j, prompt_info in enumerate(batch_prompts):
                    result = {
                        "prompt_id": i + j,
                        "prompt": prompt_info["prompt"],
                        "expected_fields": prompt_info["expected_fields"],
                        "generated_text": "",
                        "tokens_generated": 0,
                        "generation_time_ms": 0,
                        "is_valid_json": False,
                        "validation_result": f"Generation error: {e}",
                        "success": False,
                        "error": str(e)
                    }
                    results.append(result)
                continue
            
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
        valid_results = [r for r in results if "error" not in r]
        total_prompts = len(results)
        success_rate = (successful_generations / total_prompts * 100) if total_prompts > 0 else 0
        avg_tokens_per_gen = total_tokens / len(valid_results) if valid_results else 0
        avg_generation_time = sum(r["generation_time_ms"] for r in valid_results) / len(valid_results) if valid_results else 0
        
        summary = {
            "test_type": "s2_json_struct",
            "engine": "tensorrt-llm",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "engine_path": self.engine_path,
                "world_size": self.world_size,
                "total_prompts": total_prompts,
                "successful_prompts": len(valid_results),
                "temperature": 0.3,
                "tensorrt_optimizations": True
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
        result_file = output_dir / f"tensorrt_s2_json_struct_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        print(f"Success rate: {success_rate:.1f}% ({successful_generations}/{total_prompts})")
        print(f"Average generation time: {avg_generation_time:.1f}ms")
        print(f"Successful generations: {len(valid_results)}/{total_prompts}")
        
        return result_file

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM S2 JSON Structure Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--engine-path", help="Path to TensorRT engine directory")
    parser.add_argument("--output", default="./results/tensorrt/s2_json_struct", help="Output directory")
    parser.add_argument("--max-output-len", type=int, default=256, help="Max output length")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to test")
    parser.add_argument("--world-size", type=int, default=1, help="World size for multi-GPU")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = TensorRTLLMJSONStructTest(args.model, args.engine_path, args.world_size)
        test.initialize_model()
        
        # Generate prompts
        all_prompts = test.get_json_prompts()
        prompts = all_prompts[:args.num_prompts]
        print(f"Testing with {len(prompts)} JSON structure prompts")
        
        # Run test
        print("Starting TensorRT-LLM JSON structure test...")
        results, total_tokens, successful_generations = test.run_json_struct_test(
            prompts, args.max_output_len
        )
        
        # Save results
        result_file = test.save_results(results, total_tokens, successful_generations, args.output)
        
        print("✅ TensorRT-LLM S2 JSON Structure test completed successfully!")
        
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