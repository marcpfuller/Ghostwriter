"""
Metrics collection framework for passive voice detection performance testing.

This module provides a unified framework for collecting, analyzing, and reporting
performance metrics with multi-run averaging and baseline comparisons.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """
    Centralized metrics collection and reporting.

    Handles multi-run execution, statistical analysis, baseline comparison,
    and report generation with no duplicate logic.
    """

    def __init__(self, run_number: int, results_dir: Path):
        """
        Initialize collector for a specific run.

        Args:
            run_number: Current run iteration (1-indexed)
            results_dir: Directory to store results
        """
        self.run_number = run_number
        self.results_dir = results_dir
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.baselines: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Load baselines once per collector instance
        baselines_path = results_dir / "baselines.json"
        if baselines_path.exists():
            with open(baselines_path, encoding="utf-8") as f:
                self.baselines = json.load(f)

    def record(self, category: str, metric_name: str, value: float) -> None:
        """
        Record a single metric measurement.

        Args:
            category: Metric category (startup, processing, api, optimization)
            metric_name: Name of the metric
            value: Measured value
        """
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][metric_name] = value

    def save_run_results(self) -> Path:
        """
        Save current run results to file.

        Returns:
            Path to saved results file
        """
        run_file = self.results_dir / f"run_{self.run_number:03d}_results.txt"

        with open(run_file, "w", encoding="utf-8") as f:
            f.write(f"Performance Metrics - Run {self.run_number}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            for category, metrics in self.metrics.items():
                f.write(f"{category.upper()}\n")
                f.write("-" * 70 + "\n")
                for metric_name, value in metrics.items():
                    baseline_info = self._get_baseline_info(category, metric_name)
                    f.write(f"  {metric_name}: {value:.6f} {baseline_info['unit']}\n")
                f.write("\n")

        return run_file

    def _get_baseline_info(self, category: str, metric_name: str) -> Dict[str, Any]:
        """Get baseline information for a metric (internal helper)."""
        if category in self.baselines and metric_name in self.baselines[category]:
            return self.baselines[category][metric_name]
        return {"unit": "", "target": None, "warning": None}

    @staticmethod
    def calculate_stats(values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics from multiple run values.

        Args:
            values: List of metric values from all runs

        Returns:
            Dictionary with mean, median, stdev, p95, p99
        """
        if not values:
            return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if n > 1 else 0.0,
            "p95": sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
            "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
        }

    def get_baseline_status(
        self, category: str, metric_name: str, value: float
    ) -> tuple[str, str]:
        """
        Determine if metric meets baseline expectations.

        Args:
            category: Metric category
            metric_name: Name of the metric
            value: Measured value

        Returns:
            Tuple of (status, symbol) where status is PASS/WARNING/FAIL
        """
        baseline = self._get_baseline_info(category, metric_name)

        if baseline["target"] is None:
            return "PASS", "✓"

        target = baseline["target"]
        warning = baseline["warning"]

        # Handle ratio metrics (higher is better)
        if baseline["unit"] == "ratio":
            if value >= target:
                return "PASS", "✓"
            if value >= warning:
                return "WARNING", "⚠"
            return "FAIL", "✗"

        # Handle throughput metrics (higher is better)
        if "throughput" in metric_name or "requests" in baseline["unit"]:
            if value >= target:
                return "PASS", "✓"
            if value >= warning:
                return "WARNING", "⚠"
            return "FAIL", "✗"

        # Default: lower is better (time, memory)
        if value <= target:
            return "PASS", "✓"
        if value <= warning:
            return "WARNING", "⚠"
        return "FAIL", "✗"

    @classmethod
    def generate_summary_report(
        cls, all_metrics: Dict[str, List[float]], results_dir: Path
    ) -> Path:
        """
        Generate summary report with statistics and baseline comparison.

        Args:
            all_metrics: Dict mapping "category.metric_name" to list of values
            results_dir: Directory to save summary

        Returns:
            Path to summary report file
        """
        summary_file = results_dir / "summary_report.txt"

        # Load baselines for status determination
        baselines_path = results_dir / "baselines.json"
        baselines = {}
        if baselines_path.exists():
            with open(baselines_path, encoding="utf-8") as f:
                baselines = json.load(f)

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Performance Metrics Summary Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Runs: {len(next(iter(all_metrics.values())))}\n")
            f.write("=" * 80 + "\n\n")

            # Group by category
            by_category: Dict[str, Dict[str, List[float]]] = {}
            for key, values in all_metrics.items():
                category, metric_name = key.split(".", 1)
                if category not in by_category:
                    by_category[category] = {}
                by_category[category][metric_name] = values

            # Track failures for summary
            failures = []

            for category, metrics in by_category.items():
                f.write(f"{category.upper()}\n")
                f.write("-" * 80 + "\n\n")

                for metric_name, values in metrics.items():
                    stats = cls.calculate_stats(values)

                    # Get baseline info
                    baseline_info = {}
                    if category in baselines and metric_name in baselines[category]:
                        baseline_info = baselines[category][metric_name]

                    unit = baseline_info.get("unit", "")

                    # Determine status based on mean
                    temp_collector = cls(1, results_dir)
                    temp_collector.baselines = baselines
                    status, symbol = temp_collector.get_baseline_status(
                        category, metric_name, stats["mean"]
                    )

                    if status in ["WARNING", "FAIL"]:
                        failures.append((category, metric_name, status, symbol, stats["mean"], unit))

                    f.write(f"  {metric_name} {symbol} {status}\n")
                    f.write(f"    Mean:   {stats['mean']:.6f} {unit}\n")
                    f.write(f"    Median: {stats['median']:.6f} {unit}\n")
                    f.write(f"    StdDev: {stats['stdev']:.6f} {unit}\n")
                    f.write(f"    P95:    {stats['p95']:.6f} {unit}\n")
                    f.write(f"    P99:    {stats['p99']:.6f} {unit}\n")

                    if baseline_info:
                        f.write(f"    Target: {baseline_info.get('target', 'N/A')} {unit}\n")
                        f.write(f"    Warning: {baseline_info.get('warning', 'N/A')} {unit}\n")

                    f.write("\n")

                f.write("\n")

            # Highlight failures at the end
            if failures:
                f.write("ATTENTION: Metrics Not Meeting Baselines\n")
                f.write("=" * 80 + "\n\n")
                for category, metric_name, status, symbol, value, unit in failures:
                    f.write(f"{symbol} {status}: {category}.{metric_name}\n")
                    f.write(f"    Value: {value:.6f} {unit}\n\n")

        return summary_file


class MetricTest:
    """
    Base class for performance tests with timing utilities.

    Provides reusable timing context manager and metric recording.
    """

    def __init__(self, collector: MetricsCollector):
        """
        Initialize test with collector.

        Args:
            collector: MetricsCollector instance for this run
        """
        self.collector = collector

    class Timer:
        """Context manager for timing code blocks."""

        def __init__(self, callback):
            """
            Initialize timer with callback.

            Args:
                callback: Function to call with elapsed time
            """
            self.callback = callback
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start_time
            self.callback(elapsed)

    def time_operation(self, category: str, metric_name: str):
        """
        Context manager to time an operation and record result.

        Args:
            category: Metric category
            metric_name: Name of the metric

        Returns:
            Timer context manager

        Example:
            with self.time_operation("processing", "short_text_time"):
                result = detector.detect_passive_sentences("text")
        """
        def record(elapsed):
            self.collector.record(category, metric_name, elapsed)

        return self.Timer(record)
    class MemoryMonitor:
        """Context manager for measuring memory usage."""

        def __init__(self, callback, process):
            """
            Initialize memory monitor.

            Args:
                callback: Function to call with memory delta in MB
                process: psutil.Process instance
            """
            self.callback = callback
            self.process = process
            self.start_mem = None

        def __enter__(self):
            import gc
            gc.collect()  # Force GC before measurement
            self.start_mem = self.process.memory_info().rss
            return self

        def __exit__(self, *args):
            import gc
            gc.collect()  # Force GC after operation
            end_mem = self.process.memory_info().rss
            delta_mb = (end_mem - self.start_mem) / 1024 / 1024
            self.callback(delta_mb)

    def memory_operation(self, category: str, metric_name: str, process):
        """
        Context manager to measure memory usage and record result.

        Args:
            category: Metric category
            metric_name: Name of the metric
            process: psutil.Process instance

        Returns:
            MemoryMonitor context manager

        Example:
            import psutil
            process = psutil.Process()
            with self.memory_operation("processing", "text_memory", process):
                result = detector.detect_passive_sentences("text")
        """
        def record(delta_mb):
            self.collector.record(category, metric_name, delta_mb)

        return self.MemoryMonitor(record, process)