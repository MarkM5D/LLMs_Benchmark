#!/usr/bin/env python3
"""
Test Fairness Analyzer
Analyzes all 9 benchmark tests for parameter fairness across engines
"""

import json
import os
import re
from pathlib import Path

class TestFairnessAnalyzer:
    def __init__(self, benchmarks_dir):
        self.benchmarks_dir = Path(benchmarks_dir)
        self.engines = ["vllm", "sglang", "tensorrt"]
        self.tests = ["s1_throughput", "s2_json_struct", "s3_low_latency"]
        
    def extract_parameters(self, file_path):
        """Extract key parameters from test files"""
        params = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract default batch sizes (avoid max_batch_size)
        batch_matches = re.findall(r'(?<!max_)batch_size=(\d+)', content)
        default_matches = re.findall(r'default=(\d+).*batch', content)
        if batch_matches:
            params['default_batch_size'] = int(batch_matches[0])
        elif default_matches:
            params['default_batch_size'] = int(default_matches[0])
            
        # Extract max tokens/output length  
        max_token_matches = re.findall(r'max_tokens=(\d+)', content)
        max_output_matches = re.findall(r'max_output_len=(\d+)', content)
        
        # For SGLang structured generation, sum up all max_tokens
        if 'sglang' in str(file_path) and 's2_json' in str(file_path):
            total_tokens = sum(int(match) for match in max_token_matches)
            if total_tokens > 0:
                params['max_tokens'] = total_tokens
        elif max_token_matches:
            params['max_tokens'] = int(max_token_matches[0])
        elif max_output_matches:
            params['max_tokens'] = int(max_output_matches[0])
            
        # Extract temperature
        temp_matches = re.findall(r'temperature=([0-9.]+)', content)
        if temp_matches:
            params['temperature'] = float(temp_matches[0])
            
        # Extract top_p
        top_p_matches = re.findall(r'top_p=([0-9.]+)', content)
        if top_p_matches:
            params['top_p'] = float(top_p_matches[0])
            
        # Extract repetition/frequency penalty
        rep_penalty = re.search(r'repetition_penalty=([0-9.]+)', content)
        freq_penalty = re.search(r'frequency_penalty=([0-9.]+)', content)
        if rep_penalty:
            params['penalty'] = float(rep_penalty.group(1))
        elif freq_penalty:
            params['penalty'] = float(freq_penalty.group(1))
            
        # Extract iterations for S3
        iter_matches = re.findall(r'iterations.*?default=(\d+)', content)
        if iter_matches:
            params['iterations'] = int(iter_matches[0])
            
        # Extract dataset limit for S1/S2
        limit_matches = re.findall(r'prompts\[:(\d+)\]', content)
        if limit_matches:
            params['dataset_limit'] = int(limit_matches[0])
            
        return params
    
    def analyze_fairness(self):
        """Analyze fairness across all tests"""
        results = {}
        
        for test in self.tests:
            results[test] = {}
            
            for engine in self.engines:
                file_path = self.benchmarks_dir / engine / f"{test}.py"
                if file_path.exists():
                    params = self.extract_parameters(file_path)
                    results[test][engine] = params
                else:
                    print(f"⚠️  Missing file: {file_path}")
                    
        return results
    
    def compare_parameters(self, results):
        """Compare parameters and identify unfairness"""
        unfair_tests = []
        
        print("🔍 FAIRNESS ANALYSIS RESULTS:")
        print("=" * 50)
        
        for test in self.tests:
            print(f"\n📊 {test.upper()}:")
            test_data = results[test]
            
            # Compare each parameter
            param_keys = set()
            for engine_params in test_data.values():
                param_keys.update(engine_params.keys())
            
            unfair_params = []
            
            for param in param_keys:
                values = {}
                for engine in self.engines:
                    if engine in test_data and param in test_data[engine]:
                        values[engine] = test_data[engine][param]
                
                if len(set(values.values())) > 1:  # Different values found
                    unfair_params.append((param, values))
                    print(f"  ❌ {param}: {values}")
                else:
                    if values:  # If parameter exists
                        print(f"  ✅ {param}: {list(values.values())[0]} (consistent)")
            
            if unfair_params:
                unfair_tests.append((test, unfair_params))
        
        return unfair_tests
    
    def generate_fixes(self, unfair_tests):
        """Generate recommended fixes"""
        print("\n🔧 RECOMMENDED FIXES:")
        print("=" * 50)
        
        standard_params = {
            's1_throughput': {
                'default_batch_size': 8,
                'max_tokens': 512,
                'temperature': 0.8,
                'top_p': 0.95,
                'penalty': 1.1,
                'dataset_limit': 1000
            },
            's2_json_struct': {
                'max_tokens': 256,
                'temperature': 0.3,
                'top_p': 0.9,
                'penalty': 1.1,
                'dataset_limit': 1000
            },
            's3_low_latency': {
                'iterations': 500,
                'max_tokens': 50,
                'temperature': 0.1,
                'top_p': 0.95,
                'penalty': 1.0
            }
        }
        
        for test, unfair_params in unfair_tests:
            print(f"\n📝 {test.upper()}:")
            for param, current_values in unfair_params:
                if param in standard_params[test]:
                    target_value = standard_params[test][param]
                    print(f"  🎯 {param} should be: {target_value}")
                    for engine, current_val in current_values.items():
                        if current_val != target_value:
                            print(f"    - Fix {engine}: {current_val} → {target_value}")

def main():
    analyzer = TestFairnessAnalyzer("./benchmarks")
    results = analyzer.analyze_fairness()
    unfair_tests = analyzer.compare_parameters(results)
    
    if unfair_tests:
        analyzer.generate_fixes(unfair_tests)
        print(f"\n🚨 Found {len(unfair_tests)} unfair test categories")
        return 1
    else:
        print("\n✅ All tests are fair!")
        return 0

if __name__ == "__main__":
    exit(main())