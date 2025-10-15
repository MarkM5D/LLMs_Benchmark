#!/usr/bin/env python3
"""
LLM Inference Engine Benchmark Orchestrator

This script orchestrates the complete benchmarking pipeline for comparing
vLLM, SGLang, and TensorRT-LLM inference engines according to the benchmark plan.

Execution sequence:
1. Environment setup and validation
2. Dataset preparation  
3. Sequential engine benchmarking (vLLM → SGLang → TensorRT-LLM)
4. Results aggregation and analysis
5. Report generation

Parameters (consistent across all engines):
- Model: gpt-oss-20b
- Max tokens: 128, Batch size: 32, Concurrency: 16
- Temperature: 0.8, Top-p: 0.95
"""

import subprocess
import json
import platform
import torch
import psutil
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class BenchmarkOrchestrator:
    """Main benchmark orchestration class"""
    
    def __init__(self, results_dir: Optional[str] = None, skip_setup: bool = False):
        self.workspace_root = Path(__file__).parent
        self.benchmarks_dir = self.workspace_root / "benchmarks"
        self.results_dir = Path(results_dir) if results_dir else self.benchmarks_dir / "results"
        self.skip_setup = skip_setup
        
        # Engine configurations
        self.engines = {
            "vllm": {
                "script": self.benchmarks_dir / "vllm_benchmark.py",
                "name": "vLLM",
                "description": "High-performance LLM inference with PagedAttention"
            },
            "sglang": {
                "script": self.benchmarks_dir / "sglang_benchmark.py", 
                "name": "SGLang",
                "description": "Efficient LLM inference with RadixAttention"
            },
            "tensorrtllm": {
                "script": self.benchmarks_dir / "tensorrtllm_benchmark.py",
                "name": "TensorRT-LLM", 
                "description": "NVIDIA optimized LLM inference engine"
            }
        }
        
        # Results tracking
        self.benchmark_results = {}
        self.start_time = None
        self.end_time = None
    
    def check_environment(self) -> bool:
        """Check if the environment is properly set up"""
        print("🔍 Checking benchmark environment...")
        
        checks = {
            "Python": sys.version_info >= (3, 8),
            "CUDA": torch.cuda.is_available() if 'torch' in globals() else False,
            "Results Directory": self.results_dir.exists() or self._create_directories(),
            "Benchmarks Directory": self.benchmarks_dir.exists(),
            "Dataset": self._check_dataset(),
            "GPU Available": self._check_gpu()
        }
        
        for check, status in checks.items():
            status_symbol = "✅" if status else "❌"
            print(f"  {status_symbol} {check}")
            
        all_passed = all(checks.values())
        
        if not all_passed:
            print("\n⚠️ Environment check failed. Please run setup first:")
            print("   bash setup_env.sh")
            
        return all_passed
    
    def _create_directories(self) -> bool:
        """Create necessary directories"""
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            (self.benchmarks_dir / "data").mkdir(parents=True, exist_ok=True)
            (self.benchmarks_dir / "logs").mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False
    
    def _check_dataset(self) -> bool:
        """Check if dataset is available"""
        dataset_paths = [
            "/workspace/benchmarks/data/sharegpt-10k.jsonl",
            self.benchmarks_dir / "data" / "sharegpt-10k.jsonl"
        ]
        
        return any(Path(path).exists() for path in dataset_paths)
    
    def _check_gpu(self) -> bool:
        """Check GPU availability"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version information"""
        cuda_info = "Not available"
        
        # Try multiple methods to get CUDA version
        try:
            # Method 1: Try nvidia-smi for driver CUDA version
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                driver_version = result.stdout.strip().split('\n')[0]
                cuda_info = f"Driver: {driver_version}"
        except Exception:
            pass
        
        # Method 2: Try nvcc for toolkit version
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "release" in result.stdout:
                # Extract version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        import re
                        version_match = re.search(r'release\s+(\d+\.\d+)', line, re.IGNORECASE)
                        if version_match:
                            toolkit_version = version_match.group(1)
                            if cuda_info != "Not available":
                                cuda_info += f", Toolkit: {toolkit_version}"
                            else:
                                cuda_info = f"Toolkit: {toolkit_version}"
                            break
        except Exception:
            pass
        
        # Method 3: Check PyTorch CUDA compilation version (safely)
        try:
            if 'torch' in globals():
                # Try to access torch.version.cuda safely
                version_attr = getattr(torch, 'version', None)
                if version_attr and hasattr(version_attr, 'cuda'):
                    pytorch_cuda = getattr(version_attr, 'cuda', None)
                    if pytorch_cuda:
                        if cuda_info != "Not available":
                            cuda_info += f", PyTorch compiled with: {pytorch_cuda}"
                        else:
                            cuda_info = f"PyTorch compiled with CUDA: {pytorch_cuda}"
        except Exception:
            pass
        
        # Method 4: If PyTorch has CUDA but we couldn't get version info
        try:
            if cuda_info == "Not available" and 'torch' in globals() and torch.cuda.is_available():
                cuda_info = "CUDA available (version unknown)"
        except Exception:
            pass
        
        return cuda_info
    
    def collect_environment_info(self) -> dict:
        """Collect comprehensive environment information"""
        print("📋 Collecting environment information...")
        
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
                text=True, timeout=10
            ).strip().split("\n")

            gpus = []
            for g in gpu_info:
                parts = [x.strip() for x in g.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                        "compute_capability": parts[3] if len(parts) > 3 else "Unknown"
                    })
        except Exception as e:
            gpus = [{"error": f"GPU info collection failed: {e}"}]

        env_info = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "system": {
                "os": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "gpus": gpus
            },
            "software": {
                "torch_version": torch.__version__ if 'torch' in globals() else "Not available",
                "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
                "cuda_version": self._get_cuda_version()
            },
            "benchmark_config": {
                "model": "gpt-oss-20b",
                "max_tokens": 128,
                "batch_size": 32,
                "concurrency": 16,
                "temperature": 0.8,
                "top_p": 0.95
            }
        }

        # Save environment info
        env_file = self.results_dir / "environment_info.json"
        with open(env_file, "w") as f:
            json.dump(env_info, f, indent=2)

        print(f"✅ Environment info saved to {env_file}")
        return env_info
    
    def setup_environment(self) -> bool:
        """Run environment setup if needed"""
        if self.skip_setup:
            print("⏩ Skipping environment setup")
            return True
            
        print("🔧 Setting up benchmark environment...")
        
        setup_script = self.workspace_root / "setup_env.sh"
        
        if not setup_script.exists():
            print(f"❌ Setup script not found: {setup_script}")
            return False
        
        try:
            # Make setup script executable
            setup_script.chmod(0o755)
            
            # Run setup script
            print("Running setup_env.sh...")
            result = subprocess.run(
                ["bash", str(setup_script)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            print("✅ Environment setup completed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Setup failed: {e}")
            print(f"Setup output: {e.stdout}")
            print(f"Setup errors: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Setup timed out")
            return False
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return False
    
    def run_single_benchmark(self, engine_key: str) -> dict:
        """Run benchmark for a single engine"""
        engine = self.engines[engine_key]
        print(f"\n{'='*60}")
        print(f"🚀 Running {engine['name']} Benchmark")
        print(f"📝 {engine['description']}")
        print(f"{'='*60}")
        
        if not engine['script'].exists():
            error_msg = f"Benchmark script not found: {engine['script']}"
            print(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}
        
        try:
            start_time = time.time()
            
            # Run the benchmark script
            cmd = [
                sys.executable, str(engine['script']),
                "--output-dir", str(self.results_dir)
            ]
            
            print(f"📝 Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours timeout
                cwd=str(self.workspace_root)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ {engine['name']} benchmark completed in {duration:.1f} seconds")
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {engine['name']} benchmark failed: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        except subprocess.TimeoutExpired:
            print(f"❌ {engine['name']} benchmark timed out")
            return {"status": "timeout", "error": "Benchmark timed out"}
        except Exception as e:
            print(f"❌ {engine['name']} benchmark error: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_all_benchmarks(self) -> dict:
        """Run benchmarks for all engines sequentially"""
        print("\n🎯 Starting comprehensive benchmark execution...")
        self.start_time = time.time()
        
        # Clear GPU memory before starting
        self._clear_gpu_memory()
        
        # Run each engine benchmark
        for engine_key in ["vllm", "sglang", "tensorrtllm"]:
            engine_result = self.run_single_benchmark(engine_key)
            self.benchmark_results[engine_key] = engine_result
            
            # Clean up between benchmarks
            if engine_result["status"] == "success":
                print(f"✅ {self.engines[engine_key]['name']} completed successfully")
            else:
                print(f"⚠️ {self.engines[engine_key]['name']} failed: {engine_result.get('error', 'Unknown error')}")
            
            # Clear GPU memory between runs
            self._clear_gpu_memory()
            time.sleep(5)  # Brief pause between engines
        
        self.end_time = time.time()
        
        # Summary
        successful = sum(1 for result in self.benchmark_results.values() if result["status"] == "success")
        total = len(self.benchmark_results)
        
        print(f"\n📊 Benchmark execution completed: {successful}/{total} engines successful")
        
        return self.benchmark_results
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU memory cache cleared")
        except Exception as e:
            print(f"⚠️ Could not clear GPU memory: {e}")
    
    def aggregate_results(self) -> bool:
        """Run results aggregation and analysis"""
        print("\n📊 Aggregating and analyzing results...")
        
        aggregation_script = self.workspace_root / "aggregate_results.py"
        
        if not aggregation_script.exists():
            print(f"❌ Aggregation script not found: {aggregation_script}")
            return False
        
        try:
            cmd = [
                sys.executable, str(aggregation_script),
                "--results-dir", str(self.results_dir),
                "--output-dir", str(self.results_dir),
                "--create-charts"
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            print("✅ Results aggregation completed")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Results aggregation failed: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Aggregation error: {e}")
            return False
    
    def generate_final_report(self):
        """Generate and display final benchmark report"""
        print("\n" + "="*80)
        print("🏁 FINAL BENCHMARK REPORT")
        print("="*80)
        
        # Execution summary
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        print(f"⏱️ Total execution time: {total_time/60:.1f} minutes")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Engine results summary
        print(f"\n📊 Engine Results Summary:")
        for engine_key, result in self.benchmark_results.items():
            engine_name = self.engines[engine_key]["name"]
            status = result["status"]
            duration = result.get("duration_seconds", 0)
            
            status_symbol = "✅" if status == "success" else "❌"
            print(f"  {status_symbol} {engine_name}: {status} ({duration:.1f}s)")
        
        # Results location
        print(f"\n📁 Results saved to: {self.results_dir}")
        print(f"📄 Detailed report: {self.results_dir / 'benchmark_report.txt'}")
        print(f"📊 CSV results: {self.results_dir / 'benchmark_results.csv'}")
        
        # Quick recommendations
        try:
            summary_file = self.results_dir / "benchmark_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                best_engine = summary.get("summary", {}).get("best_engine")
                if best_engine:
                    print(f"\n🏆 Recommended Engine: {best_engine}")
        except Exception:
            pass
        
        print("\n" + "="*80)


def main():
    """Main orchestrator execution"""
    parser = argparse.ArgumentParser(
        description="LLM Inference Engine Benchmark Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Run full benchmark pipeline
  python run_all.py --skip-setup       # Skip environment setup
  python run_all.py --results-dir ./my_results  # Custom results directory
  python run_all.py --engine vllm      # Run only vLLM benchmark
        """
    )
    
    parser.add_argument("--results-dir", 
                       help="Directory to save results (default: benchmarks/results)")
    parser.add_argument("--skip-setup", action="store_true",
                       help="Skip environment setup phase")
    parser.add_argument("--engine", choices=["vllm", "sglang", "tensorrtllm"],
                       help="Run benchmark for single engine only")
    parser.add_argument("--no-aggregation", action="store_true",
                       help="Skip results aggregation phase")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 LLM INFERENCE ENGINE BENCHMARK ORCHESTRATOR")
    print("="*80)
    print("📋 Comparing vLLM, SGLang, and TensorRT-LLM performance")
    print("🎯 Model: gpt-oss-20b | Hardware: H100 80GB")
    print("="*80)
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(
        results_dir=args.results_dir,
        skip_setup=args.skip_setup
    )
    
    try:
        # Phase 1: Environment setup and validation
        if not orchestrator.skip_setup:
            if not orchestrator.setup_environment():
                print("❌ Environment setup failed")
                sys.exit(1)
        
        if not orchestrator.check_environment():
            print("❌ Environment validation failed")
            sys.exit(1)
        
        # Phase 2: Collect environment info
        orchestrator.collect_environment_info()
        
        # Phase 3: Run benchmarks
        if args.engine:
            # Run single engine
            print(f"\n🎯 Running benchmark for {args.engine} only")
            result = orchestrator.run_single_benchmark(args.engine)
            orchestrator.benchmark_results[args.engine] = result
        else:
            # Run all engines
            orchestrator.run_all_benchmarks()
        
        # Phase 4: Aggregate results
        if not args.no_aggregation:
            orchestrator.aggregate_results()
        
        # Phase 5: Final report
        orchestrator.generate_final_report()
        
        print("🎉 Benchmark orchestration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark orchestration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
