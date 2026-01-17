"""
Performance metrics tests for passive voice detection.

Runs multiple iterations of each test to calculate statistical averages
and compare against baseline performance expectations.
"""

import gc
import json
import time
from pathlib import Path

import psutil
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model

from ghostwriter.modules.passive_voice.detector import get_detector
from .metrics_runner import MetricsCollector, MetricTest

User = get_user_model()


class PassiveVoiceMetricsTests(TestCase):
    """Performance metrics for passive voice detection system."""

    # Test configuration
    NUM_RUNS = 10
    RESULTS_DIR = Path(__file__).parent / "metrics_results"

    @classmethod
    def setUpClass(cls):
        """Initialize metrics collection framework."""
        super().setUpClass()

        # Create results directory
        cls.RESULTS_DIR.mkdir(exist_ok=True)

        # Create run-specific subdirectory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cls.RUN_DIR = cls.RESULTS_DIR / f"run_{timestamp}"
        cls.RUN_DIR.mkdir(exist_ok=True)

        # Storage for all run metrics and collectors per run
        cls.all_metrics = {}
        cls.collectors = {}  # Keyed by run_number

    def _get_or_create_collector(self, run_num: int) -> MetricsCollector:
        """Get existing collector for a run or create new one."""
        if run_num not in self.collectors:
            self.collectors[run_num] = MetricsCollector(run_num, self.RUN_DIR)
        return self.collectors[run_num]

    def _run_test_multiple_times(self, test_func, test_name):
        """
        Execute a test function multiple times and collect metrics.

        Args:
            test_func: Function to execute (should accept collector)
            test_name: Name of the test for reporting
        """
        print(f"\nRunning {test_name} ({self.NUM_RUNS} iterations)...")

        for run_num in range(1, self.NUM_RUNS + 1):
            # Force garbage collection before each run for consistency
            gc.collect()

            # Get or create collector for this run (shared across all tests in the run)
            collector = self._get_or_create_collector(run_num)

            # Execute test
            test_func(collector)

            # Aggregate metrics
            for category, metrics in collector.metrics.items():
                for metric_name, value in metrics.items():
                    key = f"{category}.{metric_name}"
                    if key not in self.all_metrics:
                        self.all_metrics[key] = []
                    # Avoid duplicate entries if already added
                    if len(self.all_metrics[key]) < run_num:
                        self.all_metrics[key].append(value)

            print(f"  Completed run {run_num}/{self.NUM_RUNS}")

    def test_startup_metrics(self):
        """Measure model loading and singleton access performance."""

        def run_startup_test(collector):
            test = MetricTest(collector)

            # Clear singleton to force reload
            from ghostwriter.modules.passive_voice import detector as detector_module
            detector_module._detector_instance = None
            gc.collect()

            # Measure first load
            with test.time_operation("startup", "model_load_time"):
                detector = get_detector()

            # Measure subsequent access (should be instant)
            with test.time_operation("startup", "singleton_access_time"):
                detector = get_detector()

        self._run_test_multiple_times(run_startup_test, "Startup Metrics")

    def test_processing_metrics(self):
        """Measure text processing performance across different sizes."""

        # Prepare test texts with complex passive voice constructions
        short_text = (
            "The vulnerability was discovered during the penetration test. "
            "Multiple SQL injection flaws were identified in the login form. "
            "The database credentials were hardcoded in the source code. "
            "Access controls were bypassed using privilege escalation. "
            "The findings have been documented and will be presented to management. "
        )  # ~50 words, varied passive constructions

        medium_text = (
            "During the assessment, several critical vulnerabilities were identified across "
            "the target infrastructure. The web application was found to be susceptible to "
            "cross-site scripting attacks, which could be exploited by malicious actors. "
            "Authentication tokens were transmitted over unencrypted connections, exposing "
            "them to potential interception. The firewall rules had been misconfigured, "
            "allowing unauthorized access to internal services. Database backups were stored "
            "without encryption, and default credentials were being used on multiple systems. "
            "These issues were documented thoroughly, and remediation steps have been provided. "
            "The risks were assessed as high priority and should be addressed immediately. "
            "Regular security audits are recommended to prevent similar issues in the future. "
        ) * 10  # ~900 words, realistic report language

        long_text = (
            "The security assessment was conducted over a two-week period from January 5th "
            "to January 19th. Multiple attack vectors were tested, including external "
            "reconnaissance, network scanning, and application-layer exploitation. "
            "A total of fifteen critical vulnerabilities were discovered during the engagement. "
            "SQL injection flaws were found in three separate web applications, allowing "
            "unauthorized database access. Cross-site scripting vulnerabilities were identified "
            "in user input fields that had not been properly sanitized. Session management "
            "weaknesses were exploited to hijack authenticated user sessions. "
            "The internal network was compromised through phishing attacks that were launched "
            "against employees with elevated privileges. Once inside, lateral movement was "
            "achieved by exploiting unpatched systems running outdated software versions. "
            "Domain administrator credentials were obtained from memory dumps of compromised "
            "systems. Sensitive data, including customer records and financial information, "
            "was exfiltrated to demonstrate the potential impact of the vulnerabilities. "
            "All activities were documented in real-time using the operations logging system. "
            "Screenshots and proof-of-concept code were captured for each finding. "
            "The remediation timeline was established in collaboration with the IT team. "
            "High-risk issues are expected to be resolved within 30 days, while medium-risk "
            "findings should be addressed within 90 days. A follow-up assessment will be "
            "scheduled to verify that all vulnerabilities have been properly remediated. "
        ) * 50  # ~5000 words, complex nested passive constructions

        def run_processing_test(collector):
            test = MetricTest(collector)
            detector = get_detector()

            # Get process for memory measurements
            process = psutil.Process()

            # Short text
            with test.time_operation("processing", "short_text_time"):
                with test.memory_operation("processing", "short_text_memory", process):
                    detector.detect_passive_sentences(short_text)

            # Medium text
            with test.time_operation("processing", "medium_text_time"):
                with test.memory_operation("processing", "medium_text_memory", process):
                    detector.detect_passive_sentences(medium_text)

            # Long text
            with test.time_operation("processing", "long_text_time"):
                with test.memory_operation("processing", "long_text_memory", process):
                    detector.detect_passive_sentences(long_text)

            # Overall memory overhead (detector + model loaded)
            mem_info = process.memory_info()
            collector.record("processing", "memory_overhead", mem_info.rss / 1024 / 1024)  # MB

        self._run_test_multiple_times(run_processing_test, "Processing Metrics")

    @override_settings(DEBUG=False)
    def test_api_metrics(self):
        """Measure API endpoint performance."""

        def run_api_test(collector):
            test = MetricTest(collector)

            # Create test user
            user = User.objects.create_user(
                username=f"metrics_user_{collector.run_number}",
                password="testpass123",
            )

            # Login
            self.client.force_login(user)

            # Prepare payload with realistic report excerpt
            payload = {
                "text": (
                    "The network infrastructure was thoroughly examined during the assessment. "
                    "Multiple misconfigured services were discovered on externally facing systems. "
                    "Remote code execution vulnerabilities were identified in the content management "
                    "system, which could be leveraged to gain initial access. Administrative interfaces "
                    "were exposed without proper authentication requirements. The identified risks have "
                    "been categorized based on their severity and likelihood of exploitation. "
                    "Detailed remediation guidance will be provided in the final deliverable. "
                )  # ~80 words with varied passive constructions
            }

            # Measure API response time
            with test.time_operation("api", "api_response_time"):
                response = self.client.post(
                    "/api/v1/passive-voice/detect",
                    data=json.dumps(payload),
                    content_type="application/json",
                )

            self.assertEqual(response.status_code, 200)

            # Cleanup
            user.delete()

        self._run_test_multiple_times(run_api_test, "API Metrics")

    def test_optimization_metrics(self):
        """Compare optimized vs full pipeline performance."""

        def run_optimization_test(collector):
            test = MetricTest(collector)

            # Use realistic mixed passive/active text
            test_text = (
                "The security team performed a comprehensive analysis of the target environment. "
                "Critical vulnerabilities were discovered in the authentication mechanism. "
                "We exploited several misconfigurations to gain elevated privileges. "
                "Sensitive data was extracted from the database without triggering alarms. "
                "The findings demonstrate significant security gaps that must be addressed. "
                "Network segmentation should be implemented to limit lateral movement. "
                "Multi-factor authentication was not enforced for administrative accounts. "
                "Regular patching cycles are recommended to reduce the attack surface. "
                "The client's security posture was evaluated against industry best practices. "
                "Several compliance violations were noted during the review process. "
            ) * 7  # ~490 words with realistic mix of active and passive voice

            # Measure optimized pipeline (current implementation)
            detector_optimized = get_detector()
            with test.time_operation("optimization", "_optimized_time"):
                detector_optimized.detect_passive_sentences(test_text)

            # For comparison: load full pipeline
            import spacy
            from django.conf import settings

            nlp_full = spacy.load(settings.SPACY_MODEL)  # All components enabled

            start_time = time.perf_counter()
            doc = nlp_full(test_text)
            # Count passive sentences (same logic)
            count = sum(
                1 for sent in doc.sents
                if any(token.dep_ == "auxpass" for token in sent)
            )
            full_time = time.perf_counter() - start_time

            collector.record("optimization", "_full_time", full_time)

            # Calculate speedup ratio
            optimized_time = collector.metrics["optimization"]["_optimized_time"]
            speedup = full_time / optimized_time if optimized_time > 0 else 0
            collector.record("optimization", "optimized_vs_full_speedup", speedup)

        self._run_test_multiple_times(run_optimization_test, "Optimization Metrics")

    @classmethod
    def tearDownClass(cls):
        """Generate summary report after all tests complete."""
        super().tearDownClass()

        # Save all run results now that all tests have executed
        for run_num, collector in cls.collectors.items():
            collector.save_run_results()

        if cls.all_metrics:
            print("\n" + "=" * 80)
            print("Generating summary report...")
            summary_path = MetricsCollector.generate_summary_report(
                cls.all_metrics, cls.RUN_DIR
            )
            print(f"Summary saved to: {summary_path}")
            print("=" * 80)

            # Print summary to console
            with open(summary_path, encoding="utf-8") as f:
                print(f.read())
