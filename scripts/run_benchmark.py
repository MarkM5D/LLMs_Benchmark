#!/usr/bin/env python3
"""
Main Benchmark Runner - Orchestrates all LLM engine tests
Usage: python run_benchmark.py --engine vllm --test s1_throughput --runs 3
"""

import argparse
import sys
import subprocess
import time
import json
from pathlib import Path
import logging
from datetime import datetime

class BenchmarkOrchestrator:
    def __init__(self):
        self.engines = ["vllm", "sglang", "tensorrt"]
        self.tests = ["s1_throughput", "s2_json_struct", "s3_low_latency"]
        self.benchmarks_dir = Path("./benchmarks")
        self.results_dir = Path("./results")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"benchmark_run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Benchmark Orchestrator initialized")
    
    def validate_inputs(self, engine, test):
        """Validate engine and test inputs"""
        if engine not in self.engines:
            raise ValueError(f"Invalid engine: {engine}. Must be one of: {self.engines}")
        
        if test != "all" and test not in self.tests:
            raise ValueError(f"Invalid test: {test}. Must be one of: {self.tests} or 'all'")
        
        # Check if test file exists
        if test != "all":
            test_file = self.benchmarks_dir / engine / f"{test}.py"
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
    
    def run_single_test(self, engine, test, model, dataset_path, runs=1, **kwargs):
        """Run a single benchmark test"""
        test_file = self.benchmarks_dir / engine / f"{test}.py"
        output_dir = self.results_dir / engine / test.replace("_", "/")
        
        self.logger.info(f"Running {engine} {test} benchmark (runs: {runs})")
        
        results = []
        for run_id in range(1, runs + 1):
            self.logger.info(f"  Run {run_id}/{runs}")
            
            # Build command
            cmd = [
                sys.executable, str(test_file),
                "--model", model,
                "--output", str(output_dir)
            ]
            
            # Add dataset path if provided
            if dataset_path:
                cmd.extend(["--dataset", str(dataset_path)])
            
            # Add additional arguments
            for key, value in kwargs.items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            try:
                # Run the test
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    self.logger.info(f"  ✅ Run {run_id} completed successfully ({execution_time:.1f}s)")
                    
                    # Try to find the result file
                    result_files = list(output_dir.glob(f"{engine}_{test}_*.json"))
                    if result_files:
                        latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                        results.append({
                            "run_id": run_id,
                            "status": "success",
                            "execution_time": execution_time,
                            "result_file": str(latest_result),
                            "stdout": result.stdout.split('\n')[-10:]  # Last 10 lines
                        })
                    else:
                        results.append({
                            "run_id": run_id,
                            "status": "success_no_file",
                            "execution_time": execution_time,
                            "stdout": result.stdout.split('\n')[-10:]
                        })
                else:
                    self.logger.error(f"  ❌ Run {run_id} failed (exit code: {result.returncode})")
                    self.logger.error(f"  Error: {result.stderr}")
                    results.append({
                        "run_id": run_id,
                        "status": "failed",
                        "execution_time": execution_time,
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                        "stdout": result.stdout
                    })
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"  ⏰ Run {run_id} timed out after 1 hour")
                results.append({
                    "run_id": run_id,
                    "status": "timeout",
                    "execution_time": 3600
                })
            except Exception as e:
                self.logger.error(f"  💥 Run {run_id} crashed: {e}")
                results.append({
                    "run_id": run_id,
                    "status": "crashed",
                    "error": str(e)
                })
        
        return results
    
    def run_all_tests(self, engine, model, dataset_path, runs=1, **kwargs):
        """Run all tests for a single engine"""
        self.logger.info(f"Running ALL tests for engine: {engine}")
        
        all_results = {}
        for test in self.tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting {test.upper()}")
            self.logger.info(f"{'='*50}")
            
            test_results = self.run_single_test(
                engine, test, model, dataset_path, runs, **kwargs
            )
            all_results[test] = test_results
            
            # Brief summary
            successful_runs = len([r for r in test_results if r["status"] == "success"])
            self.logger.info(f"{test} Summary: {successful_runs}/{len(test_results)} runs successful")
        
        return all_results
    
    def save_orchestrator_results(self, engine, test_or_all, all_results, output_dir):
        """Save orchestrator-level results summary"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_dir / f"orchestrator_summary_{engine}_{test_or_all}_{timestamp}.json"
        
        summary = {
            "orchestrator_version": "1.0",
            "timestamp": timestamp,
            "engine": engine,
            "test_scope": test_or_all,
            "total_test_categories": len(all_results),
            "execution_summary": {}
        }
        
        total_runs = 0
        total_successful = 0
        
        for test_name, test_results in all_results.items():
            successful_runs = len([r for r in test_results if r["status"] == "success"])
            total_runs += len(test_results)
            total_successful += successful_runs
            
            summary["execution_summary"][test_name] = {
                "total_runs": len(test_results),
                "successful_runs": successful_runs,
                "success_rate": successful_runs / len(test_results) * 100,
                "avg_execution_time": sum(r.get("execution_time", 0) for r in test_results) / len(test_results)
            }
        
        summary["overall_summary"] = {
            "total_runs": total_runs,
            "total_successful": total_successful,
            "overall_success_rate": total_successful / total_runs * 100 if total_runs > 0 else 0
        }
        
        summary["detailed_results"] = all_results
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Orchestrator summary saved: {summary_file}")
        return summary_file
    
    def print_final_summary(self, engine, test_or_all, summary):
        """Print final execution summary"""
        print("\n" + "="*60)
        print(f"🎯 BENCHMARK EXECUTION SUMMARY")
        print("="*60)
        print(f"Engine: {engine}")
        print(f"Tests: {test_or_all}")
        print(f"Timestamp: {summary['timestamp']}")
        print()
        
        for test_name, test_summary in summary["execution_summary"].items():
            status_emoji = "✅" if test_summary["success_rate"] == 100 else "⚠️" if test_summary["success_rate"] > 0 else "❌"
            print(f"{status_emoji} {test_name.upper()}:")
            print(f"  Runs: {test_summary['successful_runs']}/{test_summary['total_runs']}")
            print(f"  Success Rate: {test_summary['success_rate']:.1f}%")
            print(f"  Avg Time: {test_summary['avg_execution_time']:.1f}s")
            print()
        
        overall = summary["overall_summary"]
        overall_emoji = "🎉" if overall["overall_success_rate"] == 100 else "⚠️" if overall["overall_success_rate"] > 50 else "💥"
        print(f"{overall_emoji} OVERALL: {overall['total_successful']}/{overall['total_runs']} successful ({overall['overall_success_rate']:.1f}%)")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Engine Benchmark Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --engine vllm --test s1_throughput --runs 3
  python run_benchmark.py --engine sglang --test all --model gpt-oss-20b --runs 1
  python run_benchmark.py --engine tensorrt --test s3_low_latency --iterations 1000
        """
    )
    
    parser.add_argument("--engine", required=True, choices=["vllm", "sglang", "tensorrt"],
                       help="LLM engine to benchmark")
    parser.add_argument("--test", default="all", 
                       help="Test to run: s1_throughput, s2_json_struct, s3_low_latency, or 'all'")
    parser.add_argument("--model", default="gpt-oss-20b", 
                       help="Model name to benchmark")
    parser.add_argument("--dataset", 
                       help="Path to dataset file (default: ./datasets/sharegpt_prompts.jsonl)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs for each test")
    parser.add_argument("--output", default="./results",
                       help="Output directory for results")
    
    # Additional test parameters
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--max-tokens", type=int, help="Max tokens override") 
    parser.add_argument("--iterations", type=int, help="Iterations override (for S3)")
    parser.add_argument("--tensor-parallel", type=int, help="Tensor parallel size")
    parser.add_argument("--tp-size", type=int, help="TP size (SGLang)")
    parser.add_argument("--engine-path", help="Engine path (TensorRT)")
    
    args = parser.parse_args()
    
    try:
        orchestrator = BenchmarkOrchestrator()
        
        # Validate inputs
        orchestrator.validate_inputs(args.engine, args.test)
        
        # Set default dataset if not provided
        dataset_path = args.dataset or "./datasets/sharegpt_prompts.jsonl"
        if not Path(dataset_path).exists():
            orchestrator.logger.error(f"Dataset file not found: {dataset_path}")
            return 1
        
        # Prepare additional kwargs
        kwargs = {}
        for param in ["batch_size", "max_tokens", "iterations", "tensor_parallel", "tp_size", "engine_path"]:
            value = getattr(args, param.replace("-", "_"), None)
            if value is not None:
                kwargs[param] = value
        
        # Run benchmarks
        if args.test == "all":
            all_results = orchestrator.run_all_tests(
                args.engine, args.model, dataset_path, args.runs, **kwargs
            )
        else:
            single_result = orchestrator.run_single_test(
                args.engine, args.test, args.model, dataset_path, args.runs, **kwargs
            )
            all_results = {args.test: single_result}
        
        # Save summary
        summary_file = orchestrator.save_orchestrator_results(
            args.engine, args.test, all_results, args.output
        )
        
        # Load and print final summary
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        orchestrator.print_final_summary(args.engine, args.test, summary)
        
        # Exit with appropriate code
        overall_success_rate = summary["overall_summary"]["overall_success_rate"]
        return 0 if overall_success_rate == 100 else 1
        
    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
        if hasattr(orchestrator, 'logger'):
            orchestrator.logger.error(f"Orchestrator failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())