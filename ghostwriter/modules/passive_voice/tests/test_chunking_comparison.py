"""
Comparison test: Single-pass vs Chunked vs Parallel Chunked processing.

Tests whether breaking text into chunks improves passive voice detection performance.
"""

import gc
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from django.test import TestCase

from ghostwriter.modules.passive_voice.detector import PassiveVoiceDetector


class ChunkingComparisonTests(TestCase):
    """Compare performance of different text processing strategies."""

    @classmethod
    def setUpClass(cls):
        """Pre-warm spaCy model and prepare test infrastructure."""
        super().setUpClass()
        cls.detector = PassiveVoiceDetector()
        
        # Pre-warm the model
        cls.detector.detect_passive_sentences("The test was initialized.")
        print("\n✓ spaCy model pre-warmed for chunking comparison")
        
        # Create output directory
        cls.output_dir = Path(__file__).parent / "metrics_results"
        cls.output_dir.mkdir(exist_ok=True)
        
        # Storage for all results
        cls.comparison_results = []

    @staticmethod
    def _paired_ttest(sample1, sample2):
        """
        Perform paired t-test (manually implemented).
        
        Tests H0: mean(sample1) <= mean(sample2)
        Against H1: mean(sample1) > mean(sample2)
        
        Args:
            sample1: First sample (e.g., chunked times)
            sample2: Second sample (e.g., single-pass times)
            
        Returns:
            (t_statistic, p_value_one_tailed)
        """
        n = len(sample1)
        if n != len(sample2):
            raise ValueError("Samples must have same length")
        
        # Calculate differences
        differences = [s1 - s2 for s1, s2 in zip(sample1, sample2)]
        
        # Mean and standard deviation of differences
        mean_diff = statistics.mean(differences)
        if n < 2:
            return float('nan'), 1.0
        
        stdev_diff = statistics.stdev(differences)
        
        # Standard error
        se = stdev_diff / math.sqrt(n)
        
        # T-statistic
        if se == 0:
            t_stat = float('inf') if mean_diff > 0 else float('-inf')
        else:
            t_stat = mean_diff / se
        
        # Degrees of freedom
        df = n - 1
        
        # P-value (one-tailed) using t-distribution approximation
        # For large n (>30), t-distribution approximates normal distribution
        # Using cumulative distribution function approximation
        p_value = ChunkingComparisonTests._t_distribution_cdf(-abs(t_stat), df)
        if mean_diff > 0:
            p_value = 1 - ChunkingComparisonTests._t_distribution_cdf(t_stat, df)
        
        return t_stat, p_value

    @staticmethod
    def _t_distribution_cdf(t, df):
        """
        Approximate cumulative distribution function for t-distribution.
        
        Uses normal approximation for df >= 25 (sufficient accuracy for our purposes).
        
        Args:
            t: t-statistic
            df: degrees of freedom
            
        Returns:
            Approximate p-value
        """
        # For df >= 25, t-distribution closely approximates normal distribution
        # At df=29 (n=30), the difference is negligible for practical significance testing
        return ChunkingComparisonTests._normal_cdf(t)

    @staticmethod
    def _normal_cdf(x):
        """Approximate standard normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

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

    def _chunk_by_sentences(self, text: str, max_chunk_size: int = 5000) -> list:
        """
        Split text into chunks of complete sentences.
        
        Ensures chunks don't exceed max_chunk_size and always end on sentence boundaries.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split on sentence boundaries (period, exclamation, question mark followed by space/end)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit and we have content, start new chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _detect_single_pass(self, text: str) -> tuple:
        """
        Detect passive voice using single-pass processing (current approach).
        
        Args:
            text: Full text to analyze
            
        Returns:
            (duration_seconds, passive_ranges, num_passive)
        """
        gc.collect()
        start_time = time.perf_counter()
        passive_ranges = self.detector.detect_passive_sentences(text)
        duration = time.perf_counter() - start_time
        return duration, passive_ranges, len(passive_ranges)

    def _detect_chunked_sequential(self, text: str, chunk_size: int = 5000) -> tuple:
        """
        Detect passive voice by processing chunks sequentially.
        
        Args:
            text: Full text to analyze
            chunk_size: Maximum characters per chunk
            
        Returns:
            (duration_seconds, passive_ranges, num_passive)
        """
        gc.collect()
        chunks = self._chunk_by_sentences(text, chunk_size)
        
        start_time = time.perf_counter()
        all_ranges = []
        char_offset = 0
        
        for chunk in chunks:
            # Detect passive sentences in chunk
            chunk_ranges = self.detector.detect_passive_sentences(chunk)
            
            # Adjust ranges to account for position in full text
            adjusted_ranges = [
                (start + char_offset, end + char_offset)
                for start, end in chunk_ranges
            ]
            all_ranges.extend(adjusted_ranges)
            
            # Update offset (including space between chunks)
            char_offset += len(chunk) + 1
        
        duration = time.perf_counter() - start_time
        return duration, all_ranges, len(all_ranges)

    def _detect_chunk_worker(self, chunk_data: tuple) -> tuple:
        """
        Worker function for parallel chunk processing.
        
        Args:
            chunk_data: (chunk_text, char_offset)
            
        Returns:
            (passive_ranges_adjusted, num_passive)
        """
        chunk, char_offset = chunk_data
        chunk_ranges = self.detector.detect_passive_sentences(chunk)
        
        # Adjust ranges for position in full text
        adjusted_ranges = [
            (start + char_offset, end + char_offset)
            for start, end in chunk_ranges
        ]
        
        return adjusted_ranges, len(adjusted_ranges)

    def _detect_chunked_parallel(self, text: str, chunk_size: int = 5000, max_workers: int = 4) -> tuple:
        """
        Detect passive voice by processing chunks in parallel.
        
        Args:
            text: Full text to analyze
            chunk_size: Maximum characters per chunk
            max_workers: Number of parallel threads
            
        Returns:
            (duration_seconds, passive_ranges, num_passive)
        """
        gc.collect()
        chunks = self._chunk_by_sentences(text, chunk_size)
        
        # Prepare chunk data with offsets
        chunk_data = []
        char_offset = 0
        for chunk in chunks:
            chunk_data.append((chunk, char_offset))
            char_offset += len(chunk) + 1
        
        start_time = time.perf_counter()
        all_ranges = []
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._detect_chunk_worker, data)
                for data in chunk_data
            ]
            
            for future in as_completed(futures):
                ranges, _ = future.result()
                all_ranges.extend(ranges)
        
        # Sort ranges by start position (parallel processing may return out of order)
        all_ranges.sort(key=lambda x: x[0])
        
        duration = time.perf_counter() - start_time
        return duration, all_ranges, len(all_ranges)

    def _run_comparison(self, text_length: int, num_runs: int = 30, chunk_size: int = 5000):
        """
        Run comparison test for specific text length.
        
        Args:
            text_length: Character count for test text
            num_runs: Number of iterations per approach (default: 30 for statistical power)
            chunk_size: Size of chunks for chunked approaches
        """
        print(f"\n{'=' * 70}")
        print(f"Testing: {text_length:,} characters | Chunk size: {chunk_size:,} | {num_runs} runs each")
        print('=' * 70)
        
        # Generate test text
        text = self._generate_test_text(text_length)
        actual_length = len(text)
        
        # Test single-pass approach
        print("\n  [1/3] Single-pass (current approach)...")
        single_pass_times = []
        single_pass_passive = 0
        for i in range(num_runs):
            duration, ranges, num_passive = self._detect_single_pass(text)
            single_pass_times.append(duration)
            single_pass_passive = num_passive
        
        single_pass_mean = statistics.mean(single_pass_times)
        single_pass_median = statistics.median(single_pass_times)
        single_pass_stdev = statistics.stdev(single_pass_times)
        print(f"    Mean: {single_pass_mean*1000:.2f}ms, Median: {single_pass_median*1000:.2f}ms, "
              f"Stdev: {single_pass_stdev*1000:.2f}ms, Passive: {single_pass_passive}")
        
        # Test chunked sequential approach
        print("\n  [2/3] Chunked sequential...")
        chunked_seq_times = []
        chunked_seq_passive = 0
        for i in range(num_runs):
            duration, ranges, num_passive = self._detect_chunked_sequential(text, chunk_size)
            chunked_seq_times.append(duration)
            chunked_seq_passive = num_passive
        
        chunked_seq_mean = statistics.mean(chunked_seq_times)
        chunked_seq_median = statistics.median(chunked_seq_times)
        chunked_seq_stdev = statistics.stdev(chunked_seq_times)
        print(f"    Mean: {chunked_seq_mean*1000:.2f}ms, Median: {chunked_seq_median*1000:.2f}ms, "
              f"Stdev: {chunked_seq_stdev*1000:.2f}ms, Passive: {chunked_seq_passive}")
        
        # Test chunked parallel approach
        print("\n  [3/3] Chunked parallel (4 threads)...")
        chunked_par_times = []
        chunked_par_passive = 0
        for i in range(num_runs):
            duration, ranges, num_passive = self._detect_chunked_parallel(text, chunk_size, max_workers=4)
            chunked_par_times.append(duration)
            chunked_par_passive = num_passive
        
        chunked_par_mean = statistics.mean(chunked_par_times)
        chunked_par_median = statistics.median(chunked_par_times)
        chunked_par_stdev = statistics.stdev(chunked_par_times)
        print(f"    Mean: {chunked_par_mean*1000:.2f}ms, Median: {chunked_par_median*1000:.2f}ms, "
              f"Stdev: {chunked_par_stdev*1000:.2f}ms, Passive: {chunked_par_passive}")
        
        # Perform paired t-tests (one-tailed: H0: chunked <= single, H1: chunked > single)
        # Note: We're testing if chunked is slower (greater than) single-pass
        t_stat_seq, p_value_seq = self._paired_ttest(chunked_seq_times, single_pass_times)
        t_stat_par, p_value_par = self._paired_ttest(chunked_par_times, single_pass_times)
        
        # Calculate performance ratios
        seq_ratio = chunked_seq_mean / single_pass_mean
        par_ratio = chunked_par_mean / single_pass_mean
        
        print(f"\n  Performance vs Single-pass:")
        print(f"    Sequential chunked: {seq_ratio:.2f}x ({'SLOWER' if seq_ratio > 1 else 'FASTER'})")
        print(f"    Parallel chunked:   {par_ratio:.2f}x ({'SLOWER' if par_ratio > 1 else 'FASTER'})")
        
        print(f"\n  Statistical Significance (paired t-test, one-tailed):")
        print(f"    Sequential vs Single: t={t_stat_seq:.3f}, p={p_value_seq:.4f} "
              f"({'SIGNIFICANT' if p_value_seq < 0.05 else 'NOT significant'})")
        print(f"    Parallel vs Single:   t={t_stat_par:.3f}, p={p_value_par:.4f} "
              f"({'SIGNIFICANT' if p_value_par < 0.05 else 'NOT significant'})")
        
        # Store results
        result = {
            "text_length": actual_length,
            "chunk_size": chunk_size,
            "num_runs": num_runs,
            "single_pass": {
                "mean_ms": single_pass_mean * 1000,
                "median_ms": single_pass_median * 1000,
                "stdev_ms": single_pass_stdev * 1000,
                "passive_count": single_pass_passive,
                "raw_times": single_pass_times,
            },
            "chunked_sequential": {
                "mean_ms": chunked_seq_mean * 1000,
                "median_ms": chunked_seq_median * 1000,
                "stdev_ms": chunked_seq_stdev * 1000,
                "passive_count": chunked_seq_passive,
                "vs_single_pass": seq_ratio,
                "t_statistic": t_stat_seq,
                "p_value": p_value_seq,
                "significant": p_value_seq < 0.05,
                "raw_times": chunked_seq_times,
            },
            "chunked_parallel": {
                "mean_ms": chunked_par_mean * 1000,
                "median_ms": chunked_par_median * 1000,
                "stdev_ms": chunked_par_stdev * 1000,
                "passive_count": chunked_par_passive,
                "vs_single_pass": par_ratio,
                "t_statistic": t_stat_par,
                "p_value": p_value_par,
                "significant": p_value_par < 0.05,
                "raw_times": chunked_par_times,
            },
        }
        
        self.comparison_results.append(result)

    def test_chunking_strategies(self):
        """
        Compare single-pass vs chunked processing strategies.
        
        Tests multiple text sizes with different chunk sizes to determine
        if chunking provides any performance benefit.
        """
        print("\n" + "=" * 70)
        print("CHUNKING STRATEGY COMPARISON")
        print("Testing whether text chunking improves performance")
        print("=" * 70)
        
        # Test configurations: (text_length, chunk_size)
        test_configs = [
            (10_000, 2_000),   # 10KB text, 2KB chunks (5 chunks)
            (25_000, 5_000),   # 25KB text, 5KB chunks (5 chunks)
            (50_000, 5_000),   # 50KB text, 5KB chunks (10 chunks)
            (100_000, 10_000), # 100KB text, 10KB chunks (10 chunks)
        ]
        
        for text_length, chunk_size in test_configs:
            self._run_comparison(text_length, num_runs=30, chunk_size=chunk_size)
        
        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate final summary comparing all approaches."""
        print("\n" + "=" * 70)
        print("FINAL SUMMARY - Chunking Performance Analysis")
        print("=" * 70)
        
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("CHUNKING STRATEGY COMPARISON SUMMARY")
        summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Sample size: {self.comparison_results[0]['num_runs']} runs per test")
        summary_lines.append("=" * 70)
        summary_lines.append("")
        summary_lines.append("Comparison: Single-pass vs Sequential Chunked vs Parallel Chunked")
        summary_lines.append("")
        
        # Table header
        summary_lines.append(f"{'Text Size':<12} {'Chunk':<8} {'Single (ms)':<13} "
                           f"{'Seq (ms)':<13} {'Par (ms)':<13} {'Seq vs':<10} {'Par vs':<10}")
        summary_lines.append("-" * 70)
        
        for result in self.comparison_results:
            length = result["text_length"]
            chunk = result["chunk_size"]
            single_ms = result["single_pass"]["mean_ms"]
            seq_ms = result["chunked_sequential"]["mean_ms"]
            par_ms = result["chunked_parallel"]["mean_ms"]
            seq_vs = result["chunked_sequential"]["vs_single_pass"]
            par_vs = result["chunked_parallel"]["vs_single_pass"]
            
            line = (f"{length:<12,} {chunk:<8,} {single_ms:<13.2f} "
                   f"{seq_ms:<13.2f} {par_ms:<13.2f} {seq_vs:<10.2f}x {par_vs:<10.2f}x")
            summary_lines.append(line)
            print(f"  {line}")
        
        summary_lines.append("")
        summary_lines.append("=" * 70)
        summary_lines.append("STATISTICAL SIGNIFICANCE (Paired t-test, α=0.05)")
        summary_lines.append("=" * 70)
        summary_lines.append("")
        summary_lines.append("H0: Chunked performance <= Single-pass (no worse)")
        summary_lines.append("H1: Chunked performance > Single-pass (slower)")
        summary_lines.append("")
        
        # Show statistical results for each test
        for result in self.comparison_results:
            length = result["text_length"]
            seq_t = result["chunked_sequential"]["t_statistic"]
            seq_p = result["chunked_sequential"]["p_value"]
            seq_sig = result["chunked_sequential"]["significant"]
            par_t = result["chunked_parallel"]["t_statistic"]
            par_p = result["chunked_parallel"]["p_value"]
            par_sig = result["chunked_parallel"]["significant"]
            
            summary_lines.append(f"{length:,} characters:")
            summary_lines.append(f"  Sequential: t={seq_t:.3f}, p={seq_p:.4f} "
                               f"({'✗ SIGNIFICANTLY SLOWER' if seq_sig else '✓ not significantly different'})")
            summary_lines.append(f"  Parallel:   t={par_t:.3f}, p={par_p:.4f} "
                               f"({'✗ SIGNIFICANTLY SLOWER' if par_sig else '✓ not significantly different'})")
            summary_lines.append("")
            
            print(f"  {length:,} chars - Seq: p={seq_p:.4f} {'✗ SIG' if seq_sig else '✓'}, "
                  f"Par: p={par_p:.4f} {'✗ SIG' if par_sig else '✓'}")
        
        summary_lines.append("=" * 70)
        summary_lines.append("KEY FINDINGS")
        summary_lines.append("=" * 70)
        
        # Calculate averages
        avg_seq_ratio = statistics.mean([r["chunked_sequential"]["vs_single_pass"] for r in self.comparison_results])
        avg_par_ratio = statistics.mean([r["chunked_parallel"]["vs_single_pass"] for r in self.comparison_results])
        
        # Count how many tests show significant difference
        seq_sig_count = sum(1 for r in self.comparison_results if r["chunked_sequential"]["significant"])
        par_sig_count = sum(1 for r in self.comparison_results if r["chunked_parallel"]["significant"])
        total_tests = len(self.comparison_results)
        
        summary_lines.append(f"Average performance vs single-pass:")
        summary_lines.append(f"  Sequential chunked: {avg_seq_ratio:.2f}x ({'SLOWER' if avg_seq_ratio > 1 else 'FASTER'})")
        summary_lines.append(f"  Parallel chunked:   {avg_par_ratio:.2f}x ({'SLOWER' if avg_par_ratio > 1 else 'FASTER'})")
        summary_lines.append("")
        summary_lines.append(f"Statistically significant differences (p < 0.05):")
        summary_lines.append(f"  Sequential chunked: {seq_sig_count}/{total_tests} tests show significant slowdown")
        summary_lines.append(f"  Parallel chunked:   {par_sig_count}/{total_tests} tests show significant slowdown")
        summary_lines.append("")
        
        # Recommendation
        if seq_sig_count > 0 or par_sig_count > 0:
            summary_lines.append("✓ RECOMMENDATION: Keep single-pass approach")
            summary_lines.append("  Statistical evidence shows chunking degrades performance.")
            summary_lines.append("  spaCy is optimized for full-document processing.")
        elif avg_seq_ratio > 1 and avg_par_ratio > 1:
            summary_lines.append("✓ RECOMMENDATION: Keep single-pass approach")
            summary_lines.append("  Both chunking strategies are slower (though not statistically significant).")
            summary_lines.append("  Single-pass approach is simpler and performs better.")
        elif avg_par_ratio < 0.8:
            summary_lines.append("✓ RECOMMENDATION: Use parallel chunked approach")
            summary_lines.append(f"  Parallel processing provides {(1-avg_par_ratio)*100:.1f}% speedup.")
        else:
            summary_lines.append("⚠ RECOMMENDATION: Single-pass approach is optimal")
            summary_lines.append("  No significant performance benefit from chunking.")
        
        summary_lines.append("")
        
        # Save summary
        summary_file = self.output_dir / "chunking_comparison_summary.txt"
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))
        
        print(f"\n✓ Summary saved to: {summary_file}")
        print("=" * 70)
