#!/usr/bin/env python3
"""
LLM Benchmark Results Analyzer

Comprehensive analysis tool for comparing performance across different LLM inference engines.
Processes benchmark results, generates comparisons, visualizations, and detailed reports.

Author: AI Multiple LLM Benchmark Suite
"""

import os
import json
import sys
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics
from collections import defaultdict
import traceback

def setup_analysis_environment():
    """Setup analysis environment with optional visualization libraries."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        return True, plt, sns
    except ImportError:
        print("📊 Visualization libraries not available. Install with:")
        print("   pip install matplotlib seaborn")
        print("   (Continuing with text-based analysis only)")
        return False, None, None

class BenchmarkAnalyzer:
    """Main analyzer for processing and comparing benchmark results."""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.has_viz, self.plt, self.sns = setup_analysis_environment()
        self.engines = ["vllm", "sglang", "tensorrt"]
        self.test_types = ["s1_throughput", "s2_json_struct", "s3_low_latency"]
        
        # Analysis results storage
        self.results_data = {}
        self.comparison_data = {}
        
    def discover_results(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover all available benchmark result files."""
        discovered = defaultdict(lambda: defaultdict(list))
        
        print("🔍 Discovering benchmark results...")
        
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}")
            return discovered
        
        # Search for JSON result files
        pattern = str(self.results_dir / "**" / "*.json")
        result_files = glob.glob(pattern, recursive=True)
        
        for file_path in result_files:
            rel_path = Path(file_path).relative_to(self.results_dir)
            path_parts = rel_path.parts
            
            if len(path_parts) >= 2:
                engine = path_parts[0]
                test_type = path_parts[1] if len(path_parts) >= 3 else "unknown"
                
                if engine in self.engines:
                    discovered[engine][test_type].append(file_path)
        
        # Print discovery summary
        total_files = 0
        for engine, tests in discovered.items():
            for test_type, files in tests.items():
                count = len(files)
                total_files += count
                print(f"  📁 {engine}/{test_type}: {count} result files")
        
        print(f"✅ Discovered {total_files} result files across {len(discovered)} engines")
        return dict(discovered)
    
    def load_result_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load and validate a single result file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation
            if 'metrics' not in data:
                print(f"⚠️  Invalid result file (no metrics): {file_path}")
                return None
                
            return data
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def extract_key_metrics(self, result_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics from result data."""
        metrics = result_data.get('metrics', {})
        extracted = {}
        
        # Throughput metrics
        if 'average_tokens_per_second' in metrics:
            extracted['throughput'] = float(metrics['average_tokens_per_second'])
        
        # Success rate
        if 'success_rate_percent' in metrics:
            extracted['success_rate'] = float(metrics['success_rate_percent'])
        
        # Latency metrics
        lat_stats = metrics.get('latency_statistics', {})
        if 'total_latency' in lat_stats:
            total_lat = lat_stats['total_latency']
            extracted['latency_mean'] = total_lat.get('mean_ms', 0)
            extracted['latency_p50'] = total_lat.get('p50_ms', 0)
            extracted['latency_p95'] = total_lat.get('p95_ms', 0)
            extracted['latency_p99'] = total_lat.get('p99_ms', 0)
        
        # Generation metrics
        if 'generation_statistics' in lat_stats:
            gen_stats = lat_stats['generation_statistics']
            extracted['generation_mean'] = gen_stats.get('mean_ms', 0)
            extracted['generation_p95'] = gen_stats.get('p95_ms', 0)
        
        # Token metrics
        if 'total_output_tokens' in metrics:
            extracted['total_tokens'] = float(metrics['total_output_tokens'])
        
        # JSON validation (for structured generation tests)
        if 'json_validation' in metrics:
            json_val = metrics['json_validation']
            extracted['json_success_rate'] = float(json_val.get('success_rate_percent', 0))
        
        return extracted
    
    def aggregate_results(self, file_paths: List[str]) -> Dict[str, Any]:
        """Aggregate results from multiple files for the same test."""
        all_metrics = []
        
        for file_path in file_paths:
            result_data = self.load_result_file(file_path)
            if result_data:
                metrics = self.extract_key_metrics(result_data)
                if metrics:
                    all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {}
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m and m[key] is not None]
            if values:
                aggregated[f"{key}_mean"] = statistics.mean(values)
                aggregated[f"{key}_median"] = statistics.median(values)
                if len(values) > 1:
                    aggregated[f"{key}_stdev"] = statistics.stdev(values)
                else:
                    aggregated[f"{key}_stdev"] = 0
                aggregated[f"{key}_min"] = min(values)
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_count"] = len(values)
        
        return aggregated
    
    def process_all_results(self):
        """Process all discovered benchmark results."""
        print("\n📊 Processing benchmark results...")
        
        discovered = self.discover_results()
        
        for engine in self.engines:
            if engine not in discovered:
                print(f"⚠️  No results found for engine: {engine}")
                continue
            
            self.results_data[engine] = {}
            
            for test_type in self.test_types:
                if test_type in discovered[engine]:
                    file_paths = discovered[engine][test_type]
                    print(f"  🔄 Processing {engine}/{test_type} ({len(file_paths)} files)...")
                    
                    aggregated = self.aggregate_results(file_paths)
                    if aggregated:
                        self.results_data[engine][test_type] = aggregated
                        print(f"     ✅ Processed successfully")
                    else:
                        print(f"     ❌ Processing failed")
                else:
                    print(f"  ⚠️  No results for {engine}/{test_type}")
    
    def generate_comparison_table(self) -> str:
        """Generate a comparison table of key metrics across engines."""
        if not self.results_data:
            return "No results data available for comparison."
        
        # Define key metrics to compare
        key_metrics = [
            ('throughput_mean', 'Throughput (tokens/s)', '{:.1f}'),
            ('success_rate_mean', 'Success Rate (%)', '{:.1f}'),
            ('latency_mean_mean', 'Avg Latency (ms)', '{:.1f}'),
            ('latency_p95_mean', 'P95 Latency (ms)', '{:.1f}'),
        ]
        
        table_lines = []
        
        for test_type in self.test_types:
            table_lines.append(f"\n## {test_type.upper()} Results")
            table_lines.append("")
            
            # Create table header
            header = "| Engine | " + " | ".join([metric[1] for metric in key_metrics]) + " |"
            separator = "|" + "|".join(["-" * (len(col) + 2) for col in ["Engine"] + [m[1] for m in key_metrics]]) + "|"
            
            table_lines.append(header)
            table_lines.append(separator)
            
            # Add data rows
            for engine in self.engines:
                if engine in self.results_data and test_type in self.results_data[engine]:
                    data = self.results_data[engine][test_type]
                    row_values = [engine.upper()]
                    
                    for metric_key, _, format_str in key_metrics:
                        value = data.get(metric_key, 0)
                        formatted_value = format_str.format(value) if value > 0 else "N/A"
                        row_values.append(formatted_value)
                    
                    row = "| " + " | ".join(row_values) + " |"
                    table_lines.append(row)
                else:
                    # Engine has no data for this test
                    row_values = [engine.upper()] + ["N/A"] * len(key_metrics)
                    row = "| " + " | ".join(row_values) + " |"
                    table_lines.append(row)
            
            table_lines.append("")
        
        return "\n".join(table_lines)
    
    def calculate_performance_rankings(self) -> Dict[str, Dict[str, int]]:
        """Calculate performance rankings for each test type."""
        rankings = {}
        
        for test_type in self.test_types:
            rankings[test_type] = {}
            
            # Collect engine performance data
            engine_scores = []
            for engine in self.engines:
                if (engine in self.results_data and 
                    test_type in self.results_data[engine]):
                    
                    data = self.results_data[engine][test_type]
                    
                    # Calculate composite score (higher is better)
                    throughput = data.get('throughput_mean', 0)
                    success_rate = data.get('success_rate_mean', 0)
                    # For latency, lower is better, so invert it
                    latency = data.get('latency_mean_mean', float('inf'))
                    latency_score = 1000 / latency if latency > 0 else 0
                    
                    composite_score = (throughput * 0.4 + 
                                     success_rate * 0.3 + 
                                     latency_score * 0.3)
                    
                    engine_scores.append((engine, composite_score))
            
            # Sort by score (descending)
            engine_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Assign rankings
            for rank, (engine, score) in enumerate(engine_scores, 1):
                rankings[test_type][engine] = rank
        
        return rankings
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        report_lines = [
            "# LLM Benchmark Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Engines Tested:** {', '.join(self.engines)}",
            f"**Test Types:** {', '.join(self.test_types)}",
            ""
        ]
        
        # Executive Summary
        report_lines.extend([
            "## Executive Summary",
            ""
        ])
        
        rankings = self.calculate_performance_rankings()
        
        # Overall winner analysis
        overall_scores = defaultdict(int)
        for test_type, test_rankings in rankings.items():
            for engine, rank in test_rankings.items():
                # Lower rank = better performance, so invert for scoring
                overall_scores[engine] += (len(self.engines) + 1 - rank)
        
        if overall_scores:
            winner = max(overall_scores.items(), key=lambda x: x[1])
            report_lines.append(f"🏆 **Overall Best Performance:** {winner[0].upper()}")
            
            # Test-specific winners
            report_lines.append("")
            report_lines.append("### Test-Specific Performance Leaders:")
            report_lines.append("")
            
            for test_type in self.test_types:
                if test_type in rankings:
                    test_winner = min(rankings[test_type].items(), key=lambda x: x[1])
                    report_lines.append(f"- **{test_type.upper()}:** {test_winner[0].upper()}")
        
        report_lines.append("")
        
        # Detailed comparison table
        report_lines.extend([
            "## Performance Comparison",
            "",
            self.generate_comparison_table()
        ])
        
        # Rankings section
        report_lines.extend([
            "\n## Performance Rankings",
            ""
        ])
        
        for test_type in self.test_types:
            if test_type in rankings and rankings[test_type]:
                report_lines.append(f"### {test_type.upper()}")
                report_lines.append("")
                
                sorted_engines = sorted(rankings[test_type].items(), key=lambda x: x[1])
                for rank, (engine, _) in enumerate(sorted_engines, 1):
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                    report_lines.append(f"{rank}. {medal} {engine.upper()}")
                
                report_lines.append("")
        
        # Technical insights
        report_lines.extend([
            "## Technical Insights",
            ""
        ])
        
        insights = self.generate_technical_insights()
        report_lines.extend(insights)
        
        return "\n".join(report_lines)
    
    def generate_technical_insights(self) -> List[str]:
        """Generate technical insights from the analysis."""
        insights = []
        
        if not self.results_data:
            return ["No data available for technical analysis."]
        
        # Throughput analysis
        throughput_data = {}
        for engine in self.engines:
            if engine in self.results_data:
                for test_type in self.test_types:
                    if test_type in self.results_data[engine]:
                        key = f"{engine}_{test_type}"
                        throughput_data[key] = self.results_data[engine][test_type].get('throughput_mean', 0)
        
        if throughput_data:
            max_throughput = max(throughput_data.items(), key=lambda x: x[1])
            min_throughput = min(throughput_data.items(), key=lambda x: x[1])
            
            insights.extend([
                "### Throughput Analysis",
                "",
                f"- **Highest Throughput:** {max_throughput[0]} - {max_throughput[1]:.1f} tokens/s",
                f"- **Lowest Throughput:** {min_throughput[0]} - {min_throughput[1]:.1f} tokens/s",
                f"- **Performance Gap:** {((max_throughput[1] - min_throughput[1]) / min_throughput[1] * 100):.1f}% difference",
                ""
            ])
        
        # Latency analysis
        latency_data = {}
        for engine in self.engines:
            if engine in self.results_data:
                for test_type in self.test_types:
                    if test_type in self.results_data[engine]:
                        key = f"{engine}_{test_type}"
                        latency_data[key] = self.results_data[engine][test_type].get('latency_p95_mean', float('inf'))
        
        if latency_data:
            min_latency = min(latency_data.items(), key=lambda x: x[1])
            max_latency = max(latency_data.items(), key=lambda x: x[1])
            
            insights.extend([
                "### Latency Analysis",
                "",
                f"- **Lowest P95 Latency:** {min_latency[0]} - {min_latency[1]:.1f}ms",
                f"- **Highest P95 Latency:** {max_latency[0]} - {max_latency[1]:.1f}ms",
                ""
            ])
        
        # Reliability analysis
        success_rates = {}
        for engine in self.engines:
            if engine in self.results_data:
                rates = []
                for test_type in self.test_types:
                    if test_type in self.results_data[engine]:
                        rate = self.results_data[engine][test_type].get('success_rate_mean', 0)
                        rates.append(rate)
                if rates:
                    success_rates[engine] = statistics.mean(rates)
        
        if success_rates:
            most_reliable = max(success_rates.items(), key=lambda x: x[1])
            insights.extend([
                "### Reliability Analysis",
                "",
                f"- **Most Reliable Engine:** {most_reliable[0].upper()} - {most_reliable[1]:.1f}% avg success rate",
                ""
            ])
            
            for engine, rate in sorted(success_rates.items(), key=lambda x: x[1], reverse=True):
                insights.append(f"- {engine.upper()}: {rate:.1f}% average success rate")
            insights.append("")
        
        # Recommendations
        insights.extend([
            "### Recommendations",
            "",
            "Based on the analysis results:",
            ""
        ])
        
        if throughput_data and latency_data and success_rates:
            # Find best engine for each use case
            best_throughput = max(throughput_data.items(), key=lambda x: x[1])[0].split('_')[0]
            best_latency = min(latency_data.items(), key=lambda x: x[1])[0].split('_')[0]
            best_reliability = max(success_rates.items(), key=lambda x: x[1])[0]
            
            insights.extend([
                f"- **For High Throughput Workloads:** Consider {best_throughput.upper()}",
                f"- **For Low Latency Requirements:** Consider {best_latency.upper()}",
                f"- **For Maximum Reliability:** Consider {best_reliability.upper()}",
                ""
            ])
        
        insights.extend([
            "- Monitor GPU utilization and memory usage during production deployment",
            "- Consider load balancing between multiple engines based on workload characteristics",
            "- Regularly benchmark with your specific prompts and use cases",
            ""
        ])
        
        return insights
    
    def create_visualizations(self, output_dir: str = "./analysis_output"):
        """Create performance visualization charts (if matplotlib available)."""
        if not self.has_viz:
            print("📊 Skipping visualizations (matplotlib not available)")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("📈 Creating performance visualizations...")
        
        try:
            # Set style
            self.sns.set_style("whitegrid")
            
            # 1. Throughput comparison chart
            self._create_throughput_chart(output_path)
            
            # 2. Latency comparison chart  
            self._create_latency_chart(output_path)
            
            # 3. Success rate comparison
            self._create_success_rate_chart(output_path)
            
            print(f"✅ Visualizations saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def _create_throughput_chart(self, output_path: Path):
        """Create throughput comparison chart."""
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        engines_data = []
        test_types_data = []
        throughput_values = []
        
        for test_type in self.test_types:
            for engine in self.engines:
                if (engine in self.results_data and 
                    test_type in self.results_data[engine]):
                    
                    throughput = self.results_data[engine][test_type].get('throughput_mean', 0)
                    engines_data.append(engine.upper())
                    test_types_data.append(test_type.replace('_', ' ').title())
                    throughput_values.append(throughput)
        
        if throughput_values:
            # Create grouped bar chart
            import numpy as np
            
            x = np.arange(len(self.test_types))
            width = 0.25
            
            for i, engine in enumerate(self.engines):
                engine_throughputs = []
                for test_type in self.test_types:
                    if (engine in self.results_data and 
                        test_type in self.results_data[engine]):
                        throughput = self.results_data[engine][test_type].get('throughput_mean', 0)
                    else:
                        throughput = 0
                    engine_throughputs.append(throughput)
                
                ax.bar(x + i * width, engine_throughputs, width, label=engine.upper())
            
            ax.set_xlabel('Test Types')
            ax.set_ylabel('Throughput (tokens/second)')
            ax.set_title('Throughput Comparison Across Engines')
            ax.set_xticks(x + width)
            ax.set_xticklabels([t.replace('_', ' ').title() for t in self.test_types])
            ax.legend()
            
            self.plt.tight_layout()
            self.plt.savefig(output_path / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
            self.plt.close()
    
    def _create_latency_chart(self, output_path: Path):
        """Create latency comparison chart."""
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        import numpy as np
        
        x = np.arange(len(self.test_types))
        width = 0.25
        
        for i, engine in enumerate(self.engines):
            engine_latencies = []
            for test_type in self.test_types:
                if (engine in self.results_data and 
                    test_type in self.results_data[engine]):
                    latency = self.results_data[engine][test_type].get('latency_p95_mean', 0)
                else:
                    latency = 0
                engine_latencies.append(latency)
            
            ax.bar(x + i * width, engine_latencies, width, label=engine.upper())
        
        ax.set_xlabel('Test Types')
        ax.set_ylabel('P95 Latency (ms)')
        ax.set_title('P95 Latency Comparison Across Engines')
        ax.set_xticks(x + width)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in self.test_types])
        ax.legend()
        
        self.plt.tight_layout()
        self.plt.savefig(output_path / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        self.plt.close()
    
    def _create_success_rate_chart(self, output_path: Path):
        """Create success rate comparison chart."""
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        import numpy as np
        
        x = np.arange(len(self.test_types))
        width = 0.25
        
        for i, engine in enumerate(self.engines):
            engine_success_rates = []
            for test_type in self.test_types:
                if (engine in self.results_data and 
                    test_type in self.results_data[engine]):
                    success_rate = self.results_data[engine][test_type].get('success_rate_mean', 0)
                else:
                    success_rate = 0
                engine_success_rates.append(success_rate)
            
            ax.bar(x + i * width, engine_success_rates, width, label=engine.upper())
        
        ax.set_xlabel('Test Types')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Comparison Across Engines')
        ax.set_xticks(x + width)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in self.test_types])
        ax.legend()
        ax.set_ylim(0, 105)  # Set y-axis limit for percentage
        
        self.plt.tight_layout()
        self.plt.savefig(output_path / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        self.plt.close()


def main():
    """Main function for running benchmark analysis."""
    parser = argparse.ArgumentParser(description='Analyze LLM benchmark results')
    parser.add_argument('--results-dir', default='./results',
                       help='Directory containing benchmark results')
    parser.add_argument('--output-dir', default='./analysis_output',
                       help='Directory for analysis output')
    parser.add_argument('--format', choices=['markdown', 'json', 'both'], 
                       default='both',
                       help='Output format for analysis report')
    parser.add_argument('--visualizations', action='store_true',
                       help='Generate performance visualization charts')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🚀 LLM Benchmark Results Analyzer")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = BenchmarkAnalyzer(args.results_dir)
        
        # Process all results
        analyzer.process_all_results()
        
        if not analyzer.results_data:
            print("❌ No valid benchmark results found to analyze")
            sys.exit(1)
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📝 Generating analysis reports...")
        
        # Generate summary report
        if args.format in ['markdown', 'both']:
            report = analyzer.generate_summary_report()
            
            report_file = output_path / 'benchmark_analysis_report.md'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ Markdown report saved: {report_file}")
        
        # Generate JSON output
        if args.format in ['json', 'both']:
            json_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'results_data': analyzer.results_data,
                'rankings': analyzer.calculate_performance_rankings(),
                'summary': {
                    'engines_tested': analyzer.engines,
                    'test_types': analyzer.test_types,
                    'total_results': len(analyzer.results_data)
                }
            }
            
            json_file = output_path / 'benchmark_analysis_data.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"✅ JSON data saved: {json_file}")
        
        # Generate visualizations
        if args.visualizations:
            analyzer.create_visualizations(args.output_dir)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Output directory: {output_path.absolute()}")
        
        # Show quick summary
        print(f"\n📊 Quick Summary:")
        rankings = analyzer.calculate_performance_rankings()
        for test_type in analyzer.test_types:
            if test_type in rankings and rankings[test_type]:
                winner = min(rankings[test_type].items(), key=lambda x: x[1])
                print(f"   {test_type.upper()}: {winner[0].upper()} (Rank #{winner[1]})")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()