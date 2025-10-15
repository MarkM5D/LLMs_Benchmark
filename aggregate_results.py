#!/usr/bin/env python3
"""
Benchmark Results Aggregation Script

This script combines and analyzes results from all three inference engines:
- vLLM
- SGLang  
- TensorRT-LLM

It produces comparative tables, rankings, and summary reports in multiple formats.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional


class BenchmarkAggregator:
    """Aggregate and analyze benchmark results from all engines"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.engines = ["vllm", "sglang", "tensorrtllm"]
        self.results_data = {}
        
    def load_results(self) -> Dict[str, Any]:
        """Load results from all engines"""
        print("📊 Loading benchmark results...")
        
        for engine in self.engines:
            result_file = self.results_dir / f"{engine}_results.json"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    self.results_data[engine] = data
                    print(f"✅ Loaded {engine} results")
                except Exception as e:
                    print(f"❌ Error loading {engine} results: {e}")
                    self.results_data[engine] = None
            else:
                print(f"⚠️ {engine} results not found: {result_file}")
                self.results_data[engine] = None
        
        return self.results_data
    
    def extract_metrics(self) -> pd.DataFrame:
        """Extract key metrics into a comparison dataframe"""
        print("📈 Extracting performance metrics...")
        
        metrics_data = []
        
        for engine_name, data in self.results_data.items():
            if data is None:
                # Handle missing results
                metrics_data.append({
                    'Engine': engine_name.upper(),
                    'Status': 'Not Available',
                    'Throughput (tokens/sec)': 0,
                    'Latency P50 (ms)': 0,
                    'Latency P95 (ms)': 0,
                    'Latency Mean (ms)': 0,
                    'GPU Utilization (%)': 0,
                    'GPU Memory Peak (%)': 0,
                    'Total Requests': 0,
                    'Error': 'Results file missing'
                })
                continue
            
            # Handle different result structures
            if 'direct_benchmark' in data:
                # vLLM format with direct benchmark results
                benchmark_data = data['direct_benchmark']
            elif 'engine_name' in data:
                # Direct format (SGLang, TensorRT-LLM)
                benchmark_data = data
            else:
                # Fallback - use the data as-is
                benchmark_data = data
            
            # Extract metrics with safe defaults
            throughput = benchmark_data.get('throughput_tokens_per_second', 0)
            
            # Latency statistics
            latency_stats = benchmark_data.get('latency_statistics', {})
            latency_p50 = latency_stats.get('p50', 0) * 1000  # Convert to ms
            latency_p95 = latency_stats.get('p95', 0) * 1000  # Convert to ms
            latency_mean = latency_stats.get('mean', 0) * 1000  # Convert to ms
            
            # GPU statistics (use first GPU)
            gpu_stats = benchmark_data.get('gpu_statistics', {})
            gpu_util = 0
            gpu_memory = 0
            
            if gpu_stats:
                # Find first GPU stats
                first_gpu_key = next(iter(gpu_stats.keys()), None)
                if first_gpu_key and isinstance(gpu_stats[first_gpu_key], dict):
                    first_gpu = gpu_stats[first_gpu_key]
                    gpu_util = first_gpu.get('utilization_mean_percent', 0)
                    gpu_memory = first_gpu.get('memory_peak_percent', 0)
            
            # Status
            status = benchmark_data.get('status', 'Success')
            if status == 'not_available':
                status = 'Not Available'
            elif throughput > 0:
                status = 'Success'
            else:
                status = 'Failed'
            
            metrics_data.append({
                'Engine': engine_name.upper(),
                'Status': status,
                'Throughput (tokens/sec)': round(throughput, 2),
                'Latency P50 (ms)': round(latency_p50, 1),
                'Latency P95 (ms)': round(latency_p95, 1),
                'Latency Mean (ms)': round(latency_mean, 1),
                'GPU Utilization (%)': round(gpu_util, 1),
                'GPU Memory Peak (%)': round(gpu_memory, 1),
                'Total Requests': benchmark_data.get('total_requests', 0),
                'Error': benchmark_data.get('error', '')
            })
        
        df = pd.DataFrame(metrics_data)
        return df
    
    def create_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance rankings"""
        print("🏆 Creating performance rankings...")
        
        # Filter only successful runs
        successful_df = df[df['Status'] == 'Success'].copy()
        
        if len(successful_df) == 0:
            print("⚠️ No successful benchmark results to rank")
            return pd.DataFrame()
        
        # Calculate rankings (1 = best)
        successful_df['Throughput Rank'] = successful_df['Throughput (tokens/sec)'].rank(method='min', ascending=False)
        successful_df['Latency P50 Rank'] = successful_df['Latency P50 (ms)'].rank(method='min', ascending=True)
        successful_df['Latency P95 Rank'] = successful_df['Latency P95 (ms)'].rank(method='min', ascending=True)
        successful_df['GPU Efficiency Rank'] = (
            successful_df['Throughput (tokens/sec)'] / 
            (successful_df['GPU Memory Peak (%)'] + 1)  # +1 to avoid division by zero
        ).rank(method='min', ascending=False)
        
        # Overall score (lower is better)
        successful_df['Overall Score'] = (
            successful_df['Throughput Rank'] * 0.4 +  # 40% weight on throughput
            successful_df['Latency P50 Rank'] * 0.3 +  # 30% weight on latency
            successful_df['GPU Efficiency Rank'] * 0.3  # 30% weight on efficiency
        )
        
        successful_df['Overall Rank'] = successful_df['Overall Score'].rank(method='min')
        
        return successful_df.sort_values('Overall Rank')
    
    def generate_summary_report(self, df: pd.DataFrame, rankings: pd.DataFrame) -> str:
        """Generate a text summary report"""
        print("📝 Generating summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("🚀 LLM INFERENCE ENGINE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        if len(rankings) > 0:
            winner = rankings.iloc[0]
            report.append(f"🏆 Best Overall Engine: {winner['Engine']}")
            report.append(f"   - Throughput: {winner['Throughput (tokens/sec)']} tokens/sec")
            report.append(f"   - Latency P50: {winner['Latency P50 (ms)']} ms")
            report.append(f"   - GPU Memory: {winner['GPU Memory Peak (%)']}%")
            report.append("")
            
            # Performance comparison
            best_throughput = df[df['Throughput (tokens/sec)'] == df['Throughput (tokens/sec)'].max()]
            best_latency = df[df['Status'] == 'Success']['Latency P50 (ms)'].min() if len(df[df['Status'] == 'Success']) > 0 else 0
            
            if not best_throughput.empty:
                report.append(f"⚡ Highest Throughput: {best_throughput.iloc[0]['Engine']} "
                             f"({best_throughput.iloc[0]['Throughput (tokens/sec)']} tokens/sec)")
            
            if best_latency > 0:
                best_latency_engine = df[df['Latency P50 (ms)'] == best_latency].iloc[0]['Engine']
                report.append(f"🏃 Lowest Latency: {best_latency_engine} ({best_latency} ms)")
        else:
            report.append("⚠️ No successful benchmark results to analyze")
        
        report.append("")
        
        # Detailed Results
        report.append("📈 DETAILED RESULTS")
        report.append("-" * 40)
        
        for _, row in df.iterrows():
            report.append(f"\n🔧 {row['Engine']} ENGINE")
            report.append(f"   Status: {row['Status']}")
            if row['Status'] == 'Success':
                report.append(f"   Throughput: {row['Throughput (tokens/sec)']} tokens/sec")
                report.append(f"   Latency P50: {row['Latency P50 (ms)']} ms")
                report.append(f"   Latency P95: {row['Latency P95 (ms)']} ms")
                report.append(f"   GPU Utilization: {row['GPU Utilization (%)']}%")
                report.append(f"   GPU Memory Peak: {row['GPU Memory Peak (%)']}%")
                report.append(f"   Total Requests: {row['Total Requests']}")
            else:
                report.append(f"   Error: {row['Error']}")
        
        # Rankings
        if len(rankings) > 0:
            report.append("\n🏆 PERFORMANCE RANKINGS")
            report.append("-" * 40)
            for i, (_, row) in enumerate(rankings.iterrows(), 1):
                report.append(f"{i}. {row['Engine']}")
                report.append(f"   Overall Score: {row['Overall Score']:.2f}")
                report.append(f"   Throughput Rank: #{int(row['Throughput Rank'])}")
                report.append(f"   Latency Rank: #{int(row['Latency P50 Rank'])}")
                report.append(f"   Efficiency Rank: #{int(row['GPU Efficiency Rank'])}")
                report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if len(rankings) > 0:
            winner = rankings.iloc[0]
            report.append(f"For production deployment, {winner['Engine']} shows the best overall performance")
            report.append("based on the combination of throughput, latency, and GPU efficiency.")
            report.append("")
            
            # Specific use case recommendations
            if len(df[df['Status'] == 'Success']) > 1:
                highest_throughput = df.loc[df['Throughput (tokens/sec)'].idxmax()]
                lowest_latency_idx = df[df['Status'] == 'Success']['Latency P50 (ms)'].idxmin()
                lowest_latency = df.loc[lowest_latency_idx]
                
                report.append("Specific Use Cases:")
                report.append(f"- High-throughput batch processing: {highest_throughput['Engine']}")
                report.append(f"- Low-latency real-time inference: {lowest_latency['Engine']}")
        else:
            report.append("Unable to provide recommendations due to insufficient successful benchmark data.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Create performance visualization charts"""
        print("📊 Creating performance visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Filter successful results
        success_df = df[df['Status'] == 'Success']
        
        if len(success_df) == 0:
            print("⚠️ No successful results to visualize")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Inference Engine Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Throughput comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(success_df['Engine'], success_df['Throughput (tokens/sec)'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(success_df)])
        ax1.set_title('Throughput Comparison', fontweight='bold')
        ax1.set_ylabel('Tokens per Second')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 2. Latency comparison
        ax2 = axes[0, 1]
        x_pos = np.arange(len(success_df))
        width = 0.35
        
        bars2a = ax2.bar(x_pos - width/2, success_df['Latency P50 (ms)'], width, 
                        label='P50', alpha=0.8)
        bars2b = ax2.bar(x_pos + width/2, success_df['Latency P95 (ms)'], width,
                        label='P95', alpha=0.8)
        
        ax2.set_title('Latency Comparison', fontweight='bold')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(success_df['Engine'], rotation=45)
        ax2.legend()
        
        # 3. GPU utilization
        ax3 = axes[1, 0]
        bars3 = ax3.bar(success_df['Engine'], success_df['GPU Utilization (%)'],
                       color=['#d62728', '#9467bd', '#8c564b'][:len(success_df)])
        ax3.set_title('GPU Utilization', fontweight='bold')
        ax3.set_ylabel('GPU Utilization (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 4. GPU memory usage
        ax4 = axes[1, 1]
        bars4 = ax4.bar(success_df['Engine'], success_df['GPU Memory Peak (%)'],
                       color=['#17becf', '#bcbd22', '#e377c2'][:len(success_df)])
        ax4.set_title('Peak GPU Memory Usage', fontweight='bold')
        ax4.set_ylabel('Memory Usage (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        chart_path = output_dir / "performance_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {chart_path}")
    
    def save_results(self, df: pd.DataFrame, rankings: pd.DataFrame, 
                    report: str, output_dir: Path):
        """Save all results to files"""
        print("💾 Saving results...")
        
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed metrics CSV
        csv_path = output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"📄 Detailed results saved to: {csv_path}")
        
        # Save rankings CSV
        if len(rankings) > 0:
            rankings_path = output_dir / "performance_rankings.csv"
            rankings.to_csv(rankings_path, index=False)
            print(f"🏆 Rankings saved to: {rankings_path}")
        
        # Save text report
        report_path = output_dir / "benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📝 Report saved to: {report_path}")
        
        # Save summary JSON
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "engines_tested": len(df),
                "successful_engines": len(df[df['Status'] == 'Success']),
                "best_engine": rankings.iloc[0]['Engine'] if len(rankings) > 0 else None,
                "metrics": df.to_dict('records')
            }
        }
        
        json_path = output_dir / "benchmark_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"📊 Summary JSON saved to: {json_path}")


def main():
    """Main aggregation execution"""
    parser = argparse.ArgumentParser(description="Aggregate LLM Benchmark Results")
    parser.add_argument("--results-dir", default="/workspace/benchmarks/results",
                       help="Directory containing benchmark results")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results",
                       help="Directory to save aggregated results")
    parser.add_argument("--create-charts", action="store_true",
                       help="Create performance visualization charts")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 LLM Benchmark Results Aggregation")
    print("=" * 80)
    
    # Initialize aggregator
    aggregator = BenchmarkAggregator(args.results_dir)
    
    # Load and process results
    results_data = aggregator.load_results()
    
    if not any(data is not None for data in results_data.values()):
        print("❌ No benchmark results found!")
        return
    
    # Extract metrics
    metrics_df = aggregator.extract_metrics()
    print("\n📊 Metrics Summary:")
    print(metrics_df.to_string(index=False))
    
    # Create rankings
    rankings_df = aggregator.create_rankings(metrics_df)
    
    if len(rankings_df) > 0:
        print("\n🏆 Performance Rankings:")
        print(rankings_df[['Engine', 'Overall Rank', 'Throughput (tokens/sec)', 
                          'Latency P50 (ms)', 'Overall Score']].to_string(index=False))
    
    # Generate report
    report = aggregator.generate_summary_report(metrics_df, rankings_df)
    print("\n" + report)
    
    # Save results
    output_dir = Path(args.output_dir)
    aggregator.save_results(metrics_df, rankings_df, report, output_dir)
    
    # Create visualizations if requested
    if args.create_charts:
        try:
            aggregator.create_visualizations(metrics_df, output_dir)
        except Exception as e:
            print(f"⚠️ Could not create charts: {e}")
    
    print("\n✅ Benchmark aggregation completed!")
    print(f"📁 Results available in: {output_dir}")


if __name__ == "__main__":
    main()