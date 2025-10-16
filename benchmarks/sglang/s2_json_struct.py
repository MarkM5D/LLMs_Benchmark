#!/usr/bin/env python3
"""
SGLang S2 JSON Structure Test
Tests structured JSON output generation using SGLang's structured generation capabilities
"""

import json
import time
import re
import sglang as sgl
from sglang import Runtime, set_default_backend
import argparse
from pathlib import Path
import sys

class SGLangJSONStructTest:
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
            max_running_requests=50,
            context_length=4096
        )
        set_default_backend(self.runtime)
        print("SGLang runtime initialized successfully")
    
    @sgl.function
    def generate_json_person(self, s):
        """Generate JSON for person object using SGLang structured generation"""
        s += sgl.user("Generate a JSON object describing a person with name, age, occupation, and skills array. Respond only with valid JSON:")
        s += sgl.assistant('{"name": "')
        s += sgl.gen("name", max_tokens=10, stop='"')
        s += '", "age": '
        s += sgl.gen("age", max_tokens=3, regex=r"\d+")
        s += ', "occupation": "'
        s += sgl.gen("occupation", max_tokens=15, stop='"')
        s += '", "skills": ['
        s += sgl.gen("skills", max_tokens=20, stop=']')
        s += ']}'
    
    @sgl.function
    def generate_json_product(self, s):
        """Generate JSON for product object"""
        s += sgl.user("Create a JSON product catalog entry with id, name, price, description, and categories. Respond only with valid JSON:")
        s += sgl.assistant('{"id": ')
        s += sgl.gen("id", max_tokens=5, regex=r"\d+")
        s += ', "name": "'
        s += sgl.gen("name", max_tokens=15, stop='"')
        s += '", "price": '
        s += sgl.gen("price", max_tokens=5, regex=r"\d+\.?\d*")
        s += ', "description": "'
        s += sgl.gen("description", max_tokens=20, stop='"')
        s += '", "categories": ['
        s += sgl.gen("categories", max_tokens=15, stop=']')
        s += ']}'
    
    @sgl.function
    def generate_json_config(self, s):
        """Generate JSON for configuration object"""
        s += sgl.user("Generate a JSON configuration object with database settings including host, port, username, and options. Respond only with valid JSON:")
        s += sgl.assistant('{"host": "')
        s += sgl.gen("host", max_tokens=10, stop='"')
        s += '", "port": '
        s += sgl.gen("port", max_tokens=5, regex=r"\d+")
        s += ', "username": "'
        s += sgl.gen("username", max_tokens=10, stop='"')
        s += '", "options": {'
        s += sgl.gen("options", max_tokens=25, stop='}')
        s += '}}'
    
    @sgl.function 
    def generate_json_api_response(self, s):
        """Generate JSON for API response"""
        s += sgl.user("Create a JSON API response with status, message, data object, and timestamp. Respond only with valid JSON:")
        s += sgl.assistant('{"status": ')
        s += sgl.gen("status", max_tokens=5, regex=r"\d+")
        s += ', "message": "'
        s += sgl.gen("message", max_tokens=15, stop='"')
        s += '", "data": {'
        s += sgl.gen("data", max_tokens=20, stop='}')
        s += '}, "timestamp": "'
        s += sgl.gen("timestamp", max_tokens=10, stop='"')
        s += '"}'
    
    @sgl.function
    def generate_json_user_profile(self, s):
        """Generate JSON for user profile"""
        s += sgl.user("Generate a JSON user profile with personal_info, preferences, and activity_history. Respond only with valid JSON:")
        s += sgl.assistant('{"personal_info": {')
        s += sgl.gen("personal_info", max_tokens=15, stop='}')
        s += '}, "preferences": {'
        s += sgl.gen("preferences", max_tokens=15, stop='}')
        s += '}, "activity_history": ['
        s += sgl.gen("activity_history", max_tokens=18, stop=']')
        s += ']}'
    
    def get_json_generators(self):
        """Get list of JSON generation functions and their expected fields"""
        return [
            {
                "generator": self.generate_json_person,
                "expected_fields": ["name", "age", "occupation", "skills"],
                "name": "person"
            },
            {
                "generator": self.generate_json_product,
                "expected_fields": ["id", "name", "price", "description", "categories"],
                "name": "product"
            },
            {
                "generator": self.generate_json_config,
                "expected_fields": ["host", "port", "username", "options"],
                "name": "config"
            },
            {
                "generator": self.generate_json_api_response,
                "expected_fields": ["status", "message", "data", "timestamp"],
                "name": "api_response"
            },
            {
                "generator": self.generate_json_user_profile,
                "expected_fields": ["personal_info", "preferences", "activity_history"],
                "name": "user_profile"
            }
        ]
    
    def validate_json_output(self, text, expected_fields):
        """Validate if output contains valid JSON with expected fields"""
        try:
            # The text should be the complete JSON since we structured the generation
            parsed = json.loads(text)
            
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
    
    def run_json_struct_test(self, num_iterations=1000):
        """Run JSON structure benchmark"""
        generators = self.get_json_generators()
        results = []
        successful_generations = 0
        total_tokens = 0
        
        print(f"Running SGLang JSON structure test with {num_iterations} iterations")
        
        # Distribute iterations across different generators
        iterations_per_generator = num_iterations // len(generators)
        
        for gen_info in generators:
            generator = gen_info["generator"]
            expected_fields = gen_info["expected_fields"]
            gen_name = gen_info["name"]
            
            print(f"Testing {gen_name} generator...")
            
            for i in range(iterations_per_generator):
                try:
                    start_time = time.time()
                    state = generator.run()
                    generation_time = (time.time() - start_time) * 1000  # ms
                    
                    # Reconstruct the full JSON from the state
                    # This is SGLang-specific - we need to combine the generated parts
                    if hasattr(state, 'name') and hasattr(state, 'age'):
                        # Person object
                        generated_json = f'{{"name": "{state.name}", "age": {state.age}, "occupation": "{state.occupation}", "skills": [{state.skills}]}}'
                    elif hasattr(state, 'id') and hasattr(state, 'price'):
                        # Product object  
                        generated_json = f'{{"id": {state.id}, "name": "{state.name}", "price": {state.price}, "description": "{state.description}", "categories": [{state.categories}]}}'
                    elif hasattr(state, 'host') and hasattr(state, 'port'):
                        # Config object
                        generated_json = f'{{"host": "{state.host}", "port": {state.port}, "username": "{state.username}", "options": {{{state.options}}}}}'
                    elif hasattr(state, 'status') and hasattr(state, 'message'):
                        # API response
                        generated_json = f'{{"status": {state.status}, "message": "{state.message}", "data": {{{state.data}}}, "timestamp": "{state.timestamp}"}}'
                    elif hasattr(state, 'personal_info'):
                        # User profile
                        generated_json = f'{{"personal_info": {{{state.personal_info}}}, "preferences": {{{state.preferences}}}, "activity_history": [{state.activity_history}]}}'
                    else:
                        generated_json = "{}"
                    
                    # Estimate token count
                    tokens_generated = len(generated_json.split()) + len(generated_json) // 4  # Rough estimate
                    total_tokens += tokens_generated
                    
                    # Validate JSON structure
                    is_valid, validation_result = self.validate_json_output(
                        generated_json, expected_fields
                    )
                    
                    if is_valid:
                        successful_generations += 1
                    
                    result = {
                        "iteration": len(results) + 1,
                        "generator_type": gen_name,
                        "expected_fields": expected_fields,
                        "generated_json": generated_json,
                        "tokens_generated": tokens_generated,
                        "generation_time_ms": generation_time,
                        "is_valid_json": is_valid,
                        "validation_result": validation_result if is_valid else str(validation_result),
                        "success": is_valid
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error in iteration {i + 1} for {gen_name}: {e}")
                    continue
            
            success_rate = successful_generations / len(results) * 100 if results else 0
            print(f"{gen_name}: {iterations_per_generator} iterations, current success rate: {success_rate:.1f}%")
        
        return results, total_tokens, successful_generations
    
    def save_results(self, results, total_tokens, successful_generations, output_dir):
        """Save benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics
        total_iterations = len(results)
        success_rate = (successful_generations / total_iterations * 100) if total_iterations > 0 else 0
        avg_tokens_per_gen = total_tokens / total_iterations if total_iterations > 0 else 0
        avg_generation_time = sum(r["generation_time_ms"] for r in results) / total_iterations if total_iterations > 0 else 0
        
        summary = {
            "test_type": "s2_json_struct",
            "engine": "sglang",
            "model": self.model_name,
            "timestamp": timestamp,
            "config": {
                "tp_size": self.tp_size,
                "total_iterations": total_iterations,
                "structured_generation": True,  # SGLang's key feature
                "regex_constraints": True
            },
            "metrics": {
                "total_iterations": total_iterations,
                "successful_generations": successful_generations,
                "success_rate_percent": success_rate,
                "total_tokens_generated": total_tokens,
                "average_tokens_per_generation": avg_tokens_per_gen,
                "average_generation_time_ms": avg_generation_time
            },
            "detailed_results": results
        }
        
        # Save detailed results
        result_file = output_dir / f"sglang_s2_json_struct_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        print(f"Success rate: {success_rate:.1f}% ({successful_generations}/{total_iterations})")
        print(f"Average generation time: {avg_generation_time:.1f}ms")
        
        return result_file
    
    def shutdown(self):
        """Cleanup SGLang runtime"""
        if self.runtime:
            self.runtime.shutdown()

def main():
    parser = argparse.ArgumentParser(description="SGLang S2 JSON Structure Test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--output", default="./results/sglang/s2_json_struct", help="Output directory")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of test iterations")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = SGLangJSONStructTest(args.model, args.tp_size)
        test.initialize_model()
        
        # Run test
        print("Starting SGLang JSON structure test...")
        results, total_tokens, successful_generations = test.run_json_struct_test(
            args.iterations
        )
        
        # Save results
        result_file = test.save_results(results, total_tokens, successful_generations, args.output)
        
        print("✅ SGLang S2 JSON Structure test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'test' in locals():
            test.shutdown()

if __name__ == "__main__":
    main()