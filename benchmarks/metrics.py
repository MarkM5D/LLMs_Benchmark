"""
LLM Benchmark Metrics Collection Utility

This module provides shared utilities for measuring performance metrics
across different inference engines (vLLM, SGLang, TensorRT-LLM).

Metrics collected:
- Throughput (tokens/second)
- Latency (p50, p95 percentiles)
- GPU memory peak usage
- GPU utilization percentage
"""

import time
import json
import subprocess
import threading
import statistics
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import psutil
import os


class GPUMonitor:
    """Real-time GPU monitoring during benchmark execution"""
    
    def __init__(self, sampling_interval: float = 0.5):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.gpu_stats = []
        self.monitor_thread = None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_current_gpu_stats()
                if stats:
                    timestamp = time.time()
                    for i, stat in enumerate(stats):
                        stat['timestamp'] = timestamp
                        stat['gpu_id'] = i
                    self.gpu_stats.extend(stats)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"GPU monitoring error: {e}")
                
    def start_monitoring(self):
        """Start background GPU monitoring"""
        self.monitoring = True
        self.gpu_stats = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> List[Dict]:
        """Stop monitoring and return collected stats"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        return self.gpu_stats.copy()
    
    @staticmethod
    def get_current_gpu_stats() -> List[Dict]:
        """Get current GPU statistics"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True, timeout=10)
            
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            
            for line in lines:
                if not line.strip():
                    continue
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 3:
                    gpu_stats.append({
                        'utilization_percent': int(parts[0]) if parts[0] != 'N/A' else 0,
                        'memory_used_mb': int(parts[1]) if parts[1] != 'N/A' else 0,
                        'memory_total_mb': int(parts[2]) if parts[2] != 'N/A' else 0,
                        'temperature_c': int(parts[3]) if len(parts) > 3 and parts[3] != 'N/A' else 0,
                        'power_draw_w': float(parts[4]) if len(parts) > 4 and parts[4] != 'N/A' else 0
                    })
            
            return gpu_stats
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return []


class LatencyTracker:
    """Track request latencies for percentile calculation"""
    
    def __init__(self):
        self.latencies = []
    
    def add_latency(self, latency_seconds: float):
        """Add a latency measurement"""
        self.latencies.append(latency_seconds)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        
        return {
            "p50": float(np.percentile(self.latencies, 50)),
            "p95": float(np.percentile(self.latencies, 95)),
            "p99": float(np.percentile(self.latencies, 99)),
            "mean": float(np.mean(self.latencies)),
            "std": float(np.std(self.latencies)),
            "min": float(min(self.latencies)),
            "max": float(max(self.latencies))
        }
    
    def reset(self):
        """Reset latency tracking"""
        self.latencies = []


class ThroughputCalculator:
    """Calculate throughput metrics"""
    
    @staticmethod
    def calculate_throughput(total_tokens: int, total_time_seconds: float) -> float:
        """Calculate tokens per second"""
        if total_time_seconds <= 0:
            return 0.0
        return total_tokens / total_time_seconds
    
    @staticmethod
    def count_tokens_simple(text: str) -> int:
        """Simple token counting (approximate)"""
        # This is a rough approximation. For more accuracy, use actual tokenizer
        return len(text.split())


class BenchmarkMetrics:
    """Main metrics collection class"""
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.gpu_monitor = GPUMonitor()
        self.latency_tracker = LatencyTracker()
        self.start_time = None
        self.end_time = None
        self.total_tokens_generated = 0
        self.total_requests = 0
        self.warmup_completed = False
        
    def start_benchmark(self):
        """Start benchmark timing and GPU monitoring"""
        print(f"🚀 Starting benchmark for {self.engine_name}")
        self.start_time = time.time()
        self.gpu_monitor.start_monitoring()
        
    def complete_warmup(self):
        """Mark warmup phase as completed"""
        self.warmup_completed = True
        print(f"✅ Warmup completed for {self.engine_name}")
        
    def add_request_result(self, latency_seconds: float, tokens_generated: int):
        """Add results from a single request"""
        if self.warmup_completed:  # Only count post-warmup requests
            self.latency_tracker.add_latency(latency_seconds)
            self.total_tokens_generated += tokens_generated
            self.total_requests += 1
    
    def end_benchmark(self) -> Dict[str, Any]:
        """End benchmark and calculate final metrics"""
        self.end_time = time.time()
        gpu_stats = self.gpu_monitor.stop_monitoring()
        
        total_time = (self.end_time or 0.0) - (self.start_time or 0.0)
        
        # Calculate throughput
        throughput = ThroughputCalculator.calculate_throughput(
            self.total_tokens_generated, total_time
        )
        
        # Get latency percentiles
        latency_stats = self.latency_tracker.get_percentiles()
        
        # Analyze GPU usage
        gpu_analysis = self._analyze_gpu_stats(gpu_stats)
        
        # Compile results
        results = {
            "engine_name": self.engine_name,
            "timestamp": datetime.now().isoformat(),
            "benchmark_duration_seconds": total_time,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "throughput_tokens_per_second": throughput,
            "latency_statistics": latency_stats,
            "gpu_statistics": gpu_analysis,
            "system_info": self._get_system_info()
        }
        
        print(f"✅ Benchmark completed for {self.engine_name}")
        print(f"📊 Throughput: {throughput:.2f} tokens/sec")
        print(f"⏱️ Latency P95: {latency_stats['p95']:.3f}s")
        
        return results
    
    def _analyze_gpu_stats(self, gpu_stats: List[Dict]) -> Dict[str, Any]:
        """Analyze collected GPU statistics"""
        if not gpu_stats:
            return {"error": "No GPU stats collected"}
        
        # Group by GPU
        gpus = {}
        for stat in gpu_stats:
            gpu_id = stat.get('gpu_id', 0)
            if gpu_id not in gpus:
                gpus[gpu_id] = []
            gpus[gpu_id].append(stat)
        
        gpu_analysis = {}
        for gpu_id, stats in gpus.items():
            utilizations = [s['utilization_percent'] for s in stats]
            memory_used = [s['memory_used_mb'] for s in stats]
            memory_total = stats[0]['memory_total_mb'] if stats else 0
            
            gpu_analysis[f"gpu_{gpu_id}"] = {
                "utilization_mean_percent": np.mean(utilizations),
                "utilization_max_percent": max(utilizations),
                "memory_peak_mb": max(memory_used) if memory_used else 0,
                "memory_total_mb": memory_total,
                "memory_peak_percent": (max(memory_used) / memory_total * 100) if memory_total > 0 and memory_used else 0,
                "samples_collected": len(stats)
            }
        
        return gpu_analysis
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_pid": os.getpid(),
            "benchmark_time": datetime.now().isoformat()
        }


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)


def warm_up_gpu():
    """Warm up GPU with dummy computation"""
    print("🔥 Warming up GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            # Create dummy tensors and perform operations
            x = torch.randn(1000, 1000, device=device)
            for _ in range(10):
                y = torch.matmul(x, x.T)
                torch.cuda.synchronize()
            print(f"✅ GPU warmup completed on device {device}")
        else:
            print("⚠️ CUDA not available, skipping GPU warmup")
    except Exception as e:
        print(f"⚠️ GPU warmup failed: {e}")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cache cleared")
    except Exception as e:
        print(f"⚠️ Could not clear GPU memory: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing metrics collection utilities...")
    
    # Test GPU monitoring
    monitor = GPUMonitor()
    current_stats = monitor.get_current_gpu_stats()
    print(f"📊 Current GPU stats: {current_stats}")
    
    # Test brief monitoring
    monitor.start_monitoring()
    time.sleep(2)
    collected_stats = monitor.stop_monitoring()
    print(f"📈 Collected {len(collected_stats)} GPU measurements")
    
    # Test latency tracking
    tracker = LatencyTracker()
    for i in range(10):
        tracker.add_latency(0.1 + i * 0.01)
    percentiles = tracker.get_percentiles()
    print(f"⏱️ Latency percentiles: {percentiles}")
    
    print("✅ Metrics utilities test completed")