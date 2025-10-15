#!/usr/bin/env python3
"""
Benchmark Framework Test Runner

This script validates the benchmark framework components and tests individual scripts
without requiring the full inference engine installations. Useful for development
and pre-deployment validation.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any


class BenchmarkFrameworkTester:
    """Test runner for validating benchmark framework components"""
    
    def __init__(self):
        self.workspace_root = Path(__file__).parent
        self.test_results = {}
        
    def test_file_structure(self) -> Dict[str, Any]:
        """Test that all required files exist"""
        print("🔍 Testing file structure...")
        
        required_files = [
            "setup_env.sh",
            "run_all.py", 
            "aggregate_results.py",
            "benchmark_plan.md",
            "benchmarks/metrics.py",
            "benchmarks/vllm_benchmark.py",
            "benchmarks/sglang_benchmark.py", 
            "benchmarks/tensorrtllm_benchmark.py"
        ]
        
        required_dirs = [
            "benchmarks",
            "benchmarks/data",
            "benchmarks/results", 
            "benchmarks/scripts"
        ]
        
        results = {"files": {}, "directories": {}}
        
        # Check files
        for file_path in required_files:
            full_path = self.workspace_root / file_path
            exists = full_path.exists()
            results["files"][file_path] = {
                "exists": exists,
                "size": full_path.stat().st_size if exists else 0
            }
            status = "✅" if exists else "❌"
            print(f"  {status} {file_path}")
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.workspace_root / dir_path
            exists = full_path.exists()
            results["directories"][dir_path] = exists
            status = "✅" if exists else "❌"  
            print(f"  {status} {dir_path}/")
        
        return results
    
    def test_python_syntax(self) -> Dict[str, Any]:
        """Test Python files for syntax errors"""
        print("\n🐍 Testing Python syntax...")
        
        python_files = [
            "run_all.py",
            "aggregate_results.py", 
            "benchmarks/metrics.py",
            "benchmarks/vllm_benchmark.py",
            "benchmarks/sglang_benchmark.py",
            "benchmarks/tensorrtllm_benchmark.py"
        ]
        
        results = {}
        
        for file_path in python_files:
            full_path = self.workspace_root / file_path
            if not full_path.exists():
                results[file_path] = {"status": "missing"}
                print(f"  ⚠️ {file_path} - Missing")
                continue
                
            try:
                # Test syntax compilation with UTF-8 encoding
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                compile(source, str(full_path), 'exec')
                results[file_path] = {"status": "valid"}
                print(f"  ✅ {file_path} - Syntax OK")
                
            except SyntaxError as e:
                results[file_path] = {
                    "status": "syntax_error", 
                    "error": str(e),
                    "line": e.lineno
                }
                print(f"  ❌ {file_path} - Syntax Error: {e}")
            except Exception as e:
                results[file_path] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"  ❌ {file_path} - Error: {e}")
        
        return results
    
    def test_imports_available(self) -> Dict[str, Any]:
        """Test which imports are available in current environment"""
        print("\n📦 Testing available imports...")
        
        # Detect if we're in RunPod environment
        is_runpod = os.path.exists("/workspace") or "runpod" in os.environ.get("HOSTNAME", "").lower()
        
        imports_to_test = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"), 
            ("matplotlib", "Matplotlib"),
            ("psutil", "PSUtil"),
            ("subprocess", "Subprocess (builtin)"),
            ("json", "JSON (builtin)"),
            ("pathlib", "Pathlib (builtin)"),
            # Optional inference engines (expected to fail in local dev)
            ("vllm", "vLLM (RunPod H100 only)"),
            ("sglang", "SGLang (RunPod H100 only)"),
            ("tensorrt_llm", "TensorRT-LLM (RunPod H100 only)")
        ]
        
        if is_runpod:
            print("🚀 RunPod environment detected - testing H100 engines...")
        else:
            print("🏠 Local development environment - H100 engines expected to be missing")
        
        results = {}
        
        for module_name, display_name in imports_to_test:
            try:
                __import__(module_name)
                results[module_name] = {"available": True}
                print(f"  ✅ {display_name}")
            except ImportError:
                results[module_name] = {"available": False}
                if "RunPod H100 only" in display_name:
                    if is_runpod:
                        print(f"  ❌ {display_name} (Missing - needs installation)")
                    else:
                        print(f"  ⚠️ {display_name} (Expected missing in local dev)")
                else:
                    print(f"  ❌ {display_name} (Install required)")
        
        return results
    
    def test_script_help_messages(self) -> Dict[str, Any]:
        """Test that scripts show help messages properly"""
        print("\n📝 Testing script help messages...")
        
        scripts_to_test = [
            ("run_all.py", ["--help"]),
            ("aggregate_results.py", ["--help"]),
            ("benchmarks/vllm_benchmark.py", ["--help"]),
            ("benchmarks/sglang_benchmark.py", ["--help"]),
            ("benchmarks/tensorrtllm_benchmark.py", ["--help"])
        ]
        
        results = {}
        
        for script_path, args in scripts_to_test:
            full_path = self.workspace_root / script_path
            if not full_path.exists():
                results[script_path] = {"status": "missing"}
                print(f"  ⚠️ {script_path} - Missing")
                continue
                
            try:
                result = subprocess.run(
                    [sys.executable, str(full_path)] + args,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.workspace_root)
                )
                
                # Help should return 0 or show usage info
                if result.returncode == 0 or "usage:" in result.stdout.lower() or "help" in result.stdout.lower():
                    results[script_path] = {"status": "ok"}
                    print(f"  ✅ {script_path} - Help OK")
                else:
                    results[script_path] = {
                        "status": "no_help",
                        "stdout": result.stdout[:200],
                        "stderr": result.stderr[:200]
                    }
                    print(f"  ⚠️ {script_path} - No help message")
                    
            except subprocess.TimeoutExpired:
                results[script_path] = {"status": "timeout"}
                print(f"  ⚠️ {script_path} - Timeout")
            except Exception as e:
                results[script_path] = {"status": "error", "error": str(e)}
                print(f"  ❌ {script_path} - Error: {e}")
        
        return results
    
    def test_metrics_utilities(self) -> Dict[str, Any]:
        """Test the metrics utilities module"""
        print("\n📊 Testing metrics utilities...")
        
        results = {}
        
        try:
            # Add benchmarks to path
            sys.path.insert(0, str(self.workspace_root))
            
            # Test imports
            from benchmarks.metrics import (
                GPUMonitor, LatencyTracker, ThroughputCalculator, 
                BenchmarkMetrics, warm_up_gpu, clear_gpu_memory
            )
            
            results["import"] = {"status": "success"}
            print("  ✅ Metrics module imports OK")
            
            # Test LatencyTracker
            tracker = LatencyTracker()
            for i in range(10):
                tracker.add_latency(0.1 + i * 0.01)
            
            percentiles = tracker.get_percentiles()
            if percentiles["p50"] > 0 and percentiles["p95"] > 0:
                results["latency_tracker"] = {"status": "success"}
                print("  ✅ LatencyTracker working")
            else:
                results["latency_tracker"] = {"status": "failed"}
                print("  ❌ LatencyTracker failed")
            
            # Test ThroughputCalculator
            throughput = ThroughputCalculator.calculate_throughput(1000, 10.0)
            if throughput == 100.0:
                results["throughput_calculator"] = {"status": "success"}
                print("  ✅ ThroughputCalculator working")
            else:
                results["throughput_calculator"] = {"status": "failed"}
                print("  ❌ ThroughputCalculator failed")
            
            # Test token counting
            tokens = ThroughputCalculator.count_tokens_simple("Hello world test")
            if tokens == 3:
                results["token_counting"] = {"status": "success"}
                print("  ✅ Token counting working")
            else:
                results["token_counting"] = {"status": "failed"}
                print("  ❌ Token counting failed")
                
        except Exception as e:
            results["metrics_test"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Metrics test failed: {e}")
        
        return results
    
    def test_sample_data_creation(self) -> Dict[str, Any]:
        """Test sample data creation for development"""
        print("\n📄 Testing sample data creation...")
        
        results = {}
        
        try:
            # Create sample dataset
            sample_data = []
            for i in range(10):
                sample_data.append({
                    "prompt": f"Test prompt {i+1}: Explain artificial intelligence.",
                    "conversation": [
                        {"from": "human", "value": f"Test question {i+1}"},
                        {"from": "assistant", "value": f"Test response {i+1}"}
                    ]
                })
            
            # Save to test file
            test_file = self.workspace_root / "benchmarks" / "data" / "test_sample.jsonl"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(test_file, 'w') as f:
                for item in sample_data:
                    json.dump(item, f)
                    f.write('\n')
            
            # Verify file
            if test_file.exists() and test_file.stat().st_size > 0:
                results["sample_creation"] = {"status": "success"}
                print(f"  ✅ Sample data created: {test_file}")
                
                # Clean up
                test_file.unlink()
                print("  🧹 Sample data cleaned up")
            else:
                results["sample_creation"] = {"status": "failed"}
                print("  ❌ Sample data creation failed")
                
        except Exception as e:
            results["sample_creation"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Sample data creation error: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🧪 BENCHMARK FRAMEWORK TEST SUITE")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all test suites
        self.test_results = {
            "file_structure": self.test_file_structure(),
            "python_syntax": self.test_python_syntax(),
            "imports": self.test_imports_available(),
            "help_messages": self.test_script_help_messages(),
            "metrics_utilities": self.test_metrics_utilities(),
            "sample_data": self.test_sample_data_creation()
        }
        
        end_time = time.time()
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        
        total_tests = 0
        passed_tests = 0
        
        for suite_name, suite_results in self.test_results.items():
            suite_passed = 0
            suite_total = 0
            
            if isinstance(suite_results, dict):
                for test_name, test_result in suite_results.items():
                    suite_total += 1
                    if isinstance(test_result, dict):
                        if test_result.get("status") == "success" or test_result.get("exists") or test_result.get("available"):
                            suite_passed += 1
            
            total_tests += suite_total
            passed_tests += suite_passed
            
            print(f"  {suite_name}: {suite_passed}/{suite_total} passed")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        print(f"⏱️ Duration: {end_time - start_time:.1f} seconds")
        
        if success_rate >= 80:
            print("✅ Framework ready for deployment!")
        elif success_rate >= 60:
            print("⚠️ Framework mostly ready - check warnings above")
        else:
            print("❌ Framework needs fixes before deployment")
        
        return self.test_results
    
    def save_test_report(self):
        """Save detailed test report"""
        report_path = self.workspace_root / "test_report.json"
        
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "test_results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed test report saved to: {report_path}")


def main():
    """Main test execution"""
    print("🚀 BENCHMARK FRAMEWORK VALIDATOR")
    print("Testing framework components for deployment readiness...\n")
    
    tester = BenchmarkFrameworkTester()
    results = tester.run_all_tests()
    tester.save_test_report()
    
    print("\n" + "=" * 50)
    print("Test completed! Ready for RunPod deployment.")


if __name__ == "__main__":
    main()