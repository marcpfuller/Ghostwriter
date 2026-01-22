"""
Performance tests for passive voice detector with varying text lengths.

Tests detector behavior with text ranging from small to maximum allowed size (100,000 chars).
Each test size runs 10 iterations and outputs metrics to files.
"""

import gc
import json
import statistics
import time
from pathlib import Path

import psutil
from django.test import TestCase

from ghostwriter.modules.passive_voice.detector import PassiveVoiceDetector


class TextLengthPerformanceTests(TestCase):
    """Test passive voice detector performance with varying text lengths."""

    @classmethod
    def setUpClass(cls):
        """Pre-warm spaCy model and prepare test data."""
        super().setUpClass()
        cls.detector = PassiveVoiceDetector()

        # Pre-warm the model
        cls.detector.detect_passive_sentences("The test was initialized.")
        print("\n✓ spaCy model pre-warmed for performance testing")

        # Create output directory
        cls.output_dir = Path(__file__).parent / "metrics_results"
        cls.output_dir.mkdir(exist_ok=True)

        # Storage for all results
        cls.all_results = []

    def _generate_test_text(self, target_length: int) -> str:
        """
        Generate realistic test text with passive voice examples.

        Args:
            target_length: Target character count

        Returns:
            String of approximately target_length characters with passive voice
        """
        # Base sentences with mix of active and passive voice
        sentences = [
            "The security assessment was conducted over a two-week period. ",
            "Multiple vulnerabilities were discovered during the testing phase. ",
            "The team analyzed the network infrastructure thoroughly. ",
            "SQL injection flaws were identified in the web application. ",
            "These issues have been documented and will be reported to management. ",
            "The authentication mechanism was found to be vulnerable to bypass attacks. ",
            "Session tokens were transmitted over unencrypted channels. ",
            "Administrators configured the firewall to block suspicious traffic. ",
            "Critical data was exposed through misconfigured API endpoints. ",
            "The penetration test revealed several high-severity findings. ",
        ]

        text = ""
        while len(text) < target_length:
            for sentence in sentences:
                text += sentence
                if len(text) >= target_length:
                    break

        return text[:target_length]

    def _measure_detection_performance(self, text: str, run_number: int) -> dict:
        """
        Measure detector performance for a single run.

        Args:
            text: Text to analyze
            run_number: Current iteration number

        Returns:
            Dictionary with performance metrics
        """
        # Force garbage collection for consistent memory measurements
        gc.collect()

        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024  # MB

        # Measure detection time
        start_time = time.perf_counter()
        try:
            ranges = self.detector.detect_passive_sentences(text)
            duration = time.perf_counter() - start_time
            success = True
            error = None
        except Exception as e:
            duration = time.perf_counter() - start_time
            ranges = []
            success = False
            error = str(e)

        end_mem = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_mem - start_mem

        return {
            "run_number": run_number,
            "text_length": len(text),
            "duration_seconds": duration,
            "duration_ms": duration * 1000,
            "passive_sentences_found": len(ranges),
            "memory_delta_mb": memory_delta,
            "memory_end_mb": end_mem,
            "success": success,
            "error": error,
            "chars_per_second": len(text) / duration if duration > 0 else 0,
        }

    def _run_length_test(self, text_length: int, num_runs: int = 10):
        """
        Run performance test for specific text length.

        Args:
            text_length: Character count for test text
            num_runs: Number of iterations to run (default: 10)
        """
        print(f"\n{'=' * 70}")
        print(f"Testing text length: {text_length:,} characters ({num_runs} runs)")
        print('=' * 70)

        # Generate test text
        text = self._generate_test_text(text_length)
        actual_length = len(text)

        # Run multiple iterations
        run_results = []
        for i in range(1, num_runs + 1):
            metrics = self._measure_detection_performance(text, i)
            run_results.append(metrics)

            # Print progress
            if i % 2 == 0 or i == num_runs:
                print(f"  Run {i}/{num_runs}: {metrics['duration_ms']:.2f}ms, "
                      f"{metrics['passive_sentences_found']} passive sentences, "
                      f"{metrics['chars_per_second']:,.0f} chars/sec")

        # Calculate statistics
        durations = [r["duration_seconds"] for r in run_results if r["success"]]
        durations_ms = [r["duration_ms"] for r in run_results if r["success"]]
        memory_deltas = [r["memory_delta_mb"] for r in run_results if r["success"]]
        chars_per_sec = [r["chars_per_second"] for r in run_results if r["success"]]

        if not durations:
            print(f"\n  ❌ ALL RUNS FAILED for length {text_length}")
            return

        summary = {
            "text_length": actual_length,
            "num_runs": num_runs,
            "successful_runs": len(durations),
            "failed_runs": num_runs - len(durations),
            "duration_stats": {
                "mean_seconds": statistics.mean(durations),
                "median_seconds": statistics.median(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "stdev_seconds": statistics.stdev(durations) if len(durations) > 1 else 0,
                "mean_ms": statistics.mean(durations_ms),
                "median_ms": statistics.median(durations_ms),
            },
            "memory_stats": {
                "mean_delta_mb": statistics.mean(memory_deltas),
                "median_delta_mb": statistics.median(memory_deltas),
                "min_delta_mb": min(memory_deltas),
                "max_delta_mb": max(memory_deltas),
            },
            "throughput_stats": {
                "mean_chars_per_sec": statistics.mean(chars_per_sec),
                "median_chars_per_sec": statistics.median(chars_per_sec),
                "min_chars_per_sec": min(chars_per_sec),
                "max_chars_per_sec": max(chars_per_sec),
            },
            "passive_sentences_found": run_results[0]["passive_sentences_found"],
            "individual_runs": run_results,
        }

        # Print summary
        print(f"\n  Summary for {actual_length:,} characters:")
        print(f"    Mean duration: {summary['duration_stats']['mean_ms']:.2f}ms")
        print(f"    Median duration: {summary['duration_stats']['median_ms']:.2f}ms")
        print(f"    Min/Max: {summary['duration_stats']['min_seconds']*1000:.2f}ms / "
              f"{summary['duration_stats']['max_seconds']*1000:.2f}ms")
        print(f"    Throughput: {summary['throughput_stats']['mean_chars_per_sec']:,.0f} chars/sec")
        print(f"    Memory delta: {summary['memory_stats']['mean_delta_mb']:.2f} MB (avg)")
        print(f"    Passive sentences: {summary['passive_sentences_found']}")

        # Save individual run results
        results_file = self.output_dir / f"text_length_{actual_length}_runs.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  ✓ Results saved to: {results_file}")

        # Store for final summary
        self.all_results.append(summary)

    def test_text_length_performance(self):
        """
        Test detector performance with various text lengths.

        Tests text sizes from 1KB to 100KB (max allowed).
        Each size runs 10 iterations to get statistical averages.
        """
        # Define test lengths (in characters)
        # Approximately: 1KB, 5KB, 10KB, 25KB, 50KB, 75KB, 100KB
        test_lengths = [
            1_000,      # ~1KB
            5_000,      # ~5KB
            10_000,     # ~10KB
            25_000,     # ~25KB
            50_000,     # ~50KB
            75_000,     # ~75KB
            100_000,    # ~100KB (maximum)
        ]

        print("\n" + "=" * 70)
        print("TEXT LENGTH PERFORMANCE TEST")
        print("Testing passive voice detection with varying text sizes")
        print("=" * 70)

        # Run tests for each length
        for length in test_lengths:
            self._run_length_test(length, num_runs=10)

        # Generate final summary
        self._generate_final_summary()

    def _generate_final_summary(self):
        """Generate overall summary comparing all text lengths."""
        print("\n" + "=" * 70)
        print("FINAL SUMMARY - All Text Lengths")
        print("=" * 70)

        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("TEXT LENGTH PERFORMANCE SUMMARY")
        summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("=" * 70)
        summary_lines.append("")

        # Create comparison table
        summary_lines.append(f"{'Length (chars)':<15} {'Mean (ms)':<12} {'Median (ms)':<12} "
                           f"{'Throughput':<15} {'Memory (MB)':<12} {'Passive':<8}")
        summary_lines.append("-" * 70)

        for result in self.all_results:
            length = result["text_length"]
            mean_ms = result["duration_stats"]["mean_ms"]
            median_ms = result["duration_stats"]["median_ms"]
            throughput = result["throughput_stats"]["mean_chars_per_sec"]
            memory = result["memory_stats"]["mean_delta_mb"]
            passive_count = result["passive_sentences_found"]

            line = (f"{length:<15,} {mean_ms:<12.2f} {median_ms:<12.2f} "
                   f"{throughput:<15,.0f} {memory:<12.2f} {passive_count:<8}")
            summary_lines.append(line)
            print(f"  {line}")

        summary_lines.append("")
        summary_lines.append("=" * 70)
        summary_lines.append("KEY INSIGHTS")
        summary_lines.append("=" * 70)

        # Calculate scaling metrics
        if len(self.all_results) >= 2:
            first = self.all_results[0]
            last = self.all_results[-1]

            length_ratio = last["text_length"] / first["text_length"]
            time_ratio = (last["duration_stats"]["mean_seconds"] /
                         first["duration_stats"]["mean_seconds"])

            summary_lines.append(f"Length increase: {length_ratio:.1f}x "
                               f"({first['text_length']:,} → {last['text_length']:,} chars)")
            summary_lines.append(f"Time increase: {time_ratio:.2f}x "
                               f"({first['duration_stats']['mean_ms']:.2f}ms → "
                               f"{last['duration_stats']['mean_ms']:.2f}ms)")
            summary_lines.append(f"Scaling efficiency: {(time_ratio/length_ratio)*100:.1f}% "
                               f"(lower is better)")

            # Check if scaling is linear
            if time_ratio / length_ratio < 1.2:
                summary_lines.append("✓ Performance scales near-linearly with text size")
            elif time_ratio / length_ratio < 2.0:
                summary_lines.append("⚠ Performance scales sub-linearly (acceptable)")
            else:
                summary_lines.append("❌ Performance degrades significantly with text size")

        summary_lines.append("")

        # Save summary
        summary_file = self.output_dir / "text_length_performance_summary.txt"
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))

        print(f"\n✓ Final summary saved to: {summary_file}")
        print("=" * 70)
