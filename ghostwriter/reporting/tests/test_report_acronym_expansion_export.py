"""Tests for acronym expansion deduplication in Word document export."""

# Standard Libraries
import os
import tempfile
import re

# Django Imports
from django.test import TestCase

# Third-Party Libraries
from docx import Document

# Ghostwriter Libraries
from ghostwriter.factories import (
    ClientFactory,
    ProjectFactory,
    ReportFactory,
    ReportFindingLinkFactory,
    ReportAcronymExpansionFactory,
    ReportTemplateFactory,
    SeverityFactory,
    FindingTypeFactory,
    UserFactory,
)
from ghostwriter.modules.reportwriter.report.docx import ExportReportDocx
from ghostwriter.reporting.models import ReportAcronymExpansion


class ReportAcronymExpansionExportTests(TestCase):
    """Test acronym expansion deduplication in Word document export."""

    @classmethod
    def setUpTestData(cls):
        cls.user = UserFactory(password="testpass123", role="manager")
        cls.client = ClientFactory()
        cls.project = ProjectFactory(client=cls.client)
        cls.report = ReportFactory(project=cls.project)
        cls.report_template = ReportTemplateFactory()
        cls.severity = SeverityFactory(severity="High", weight=1, color="ff0000")
        cls.finding_type = FindingTypeFactory(finding_type="Technical")

    def test_deduplicate_acronyms_with_single_expansion_in_database(self):
        """
        Test that acronyms tracked in ReportAcronymExpansion are deduplicated
        in exported Word documents - subsequent uses should not include (ACRONYM).
        """
        # Record that "ACL" has been expanded in this report
        ReportAcronymExpansionFactory(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )

        # Create a finding with multiple ACL references
        finding = ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="ACL Vulnerability",
            description=(
                "<p>The Access Control List (ACL) was misconfigured. "
                "The Access Control List (ACL) allows unauthorized access. "
                "Review the Access Control List (ACL) settings.</p>"
            ),
        )

        # Create the DOCX exporter
        exporter = ExportReportDocx(self.report, report_template=self.report_template)

        # Map rich texts (creates context with HTML)
        rich_text_context = exporter.map_rich_texts()

        # Apply deduplication (this is what we'll implement)
        deduplicated_context = exporter.deduplicate_acronyms(rich_text_context)

        # Get the finding's description HTML after deduplication
        finding_description = None
        for finding_data in deduplicated_context["findings"]:
            if finding_data["title"] == "ACL Vulnerability":
                # Extract HTML from the RichTextBase object
                finding_description = str(finding_data["description_rt"].__html__())
                break

        # Verify deduplication:
        # First occurrence should keep the full expansion with acronym
        # Subsequent occurrences should be replaced with just the acronym
        self.assertIsNotNone(finding_description)

        # Should have exactly 1 occurrence with full expansion (the first one)
        full_count = finding_description.count("Access Control List (ACL)")
        self.assertEqual(full_count, 1, "First occurrence should keep 'Access Control List (ACL)'")

        # Should have 2 occurrences of just 'ACL' (subsequent instances)
        # But need to count 'ACL' that's NOT inside 'Access Control List (ACL)'
        just_acl_count = finding_description.count("ACL") - finding_description.count("(ACL)")
        self.assertEqual(just_acl_count, 2, "Should have 2 subsequent instances with just 'ACL'")

    def test_deduplicate_acronyms_across_multiple_findings(self):
        """
        Test that acronym deduplication is consistent across multiple findings
        in the same report.
        """
        # Record two acronyms
        ReportAcronymExpansionFactory(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )
        ReportAcronymExpansionFactory(
            report=self.report,
            acronym="API",
            expansion="Application Programming Interface",
        )

        # Create findings with references to both acronyms
        ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="Finding 1",
            position=1,
            description=(
                "<p>The Access Control List (ACL) protects the "
                "Application Programming Interface (API).</p>"
            ),
        )

        ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="Finding 2",
            position=2,
            description=(
                "<p>The Access Control List (ACL) and "
                "Application Programming Interface (API) were tested.</p>"
            ),
        )

        # Export and deduplicate
        exporter = ExportReportDocx(self.report, report_template=self.report_template)
        rich_text_context = exporter.map_rich_texts()
        deduplicated_context = exporter.deduplicate_acronyms(rich_text_context)

        # Check Finding 1 - should have both acronyms deduplicated
        finding1_desc = None
        finding2_desc = None
        for finding_data in deduplicated_context["findings"]:
            desc_html = str(finding_data["description_rt"].__html__())
            if finding_data["title"] == "Finding 1":
                finding1_desc = desc_html
            elif finding_data["title"] == "Finding 2":
                finding2_desc = desc_html

        self.assertIsNotNone(finding1_desc)
        self.assertIsNotNone(finding2_desc)

        # Finding 1: Should have both full expansions (first occurrence of each in the report)
        self.assertIn("Access Control List (ACL)", finding1_desc)
        self.assertIn("Application Programming Interface (API)", finding1_desc)

        # Finding 2: Should have just the acronyms (subsequent occurrences)
        self.assertNotIn("Access Control List (ACL)", finding2_desc)
        self.assertNotIn("Application Programming Interface (API)", finding2_desc)
        # Should contain just the acronyms
        self.assertIn("ACL", finding2_desc)
        self.assertIn("API", finding2_desc)

    def test_no_deduplication_when_acronym_not_in_database(self):
        """
        Test that acronyms NOT in ReportAcronymExpansion are left unchanged.
        """
        # Don't record any expansions

        # Create finding with acronym that hasn't been expanded yet
        ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="Unexpanded Acronym",
            description=(
                "<p>The Access Control List (ACL) was tested. "
                "The Access Control List (ACL) is important.</p>"
            ),
        )

        # Export and deduplicate
        exporter = ExportReportDocx(self.report, report_template=self.report_template)
        rich_text_context = exporter.map_rich_texts()
        deduplicated_context = exporter.deduplicate_acronyms(rich_text_context)

        # Get finding description
        finding_desc = None
        for finding_data in deduplicated_context["findings"]:
            if finding_data["title"] == "Unexpanded Acronym":
                finding_desc = str(finding_data["description_rt"].__html__())
                break

        self.assertIsNotNone(finding_desc)

        # Both occurrences should still have the full form with (ACL)
        self.assertEqual(
            finding_desc.count("Access Control List (ACL)"),
            2,
            "Should preserve both occurrences when not in database"
        )

    def test_deduplicate_acronyms_in_multiple_finding_fields(self):
        """
        Test that deduplication works across all rich text fields
        (description, impact, mitigation, etc.).
        """
        ReportAcronymExpansionFactory(
            report=self.report,
            acronym="SQL",
            expansion="Structured Query Language",
        )

        ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="SQL Injection",
            description="<p>Structured Query Language (SQL) injection found.</p>",
            impact="<p>The Structured Query Language (SQL) database is at risk.</p>",
            mitigation="<p>Sanitize Structured Query Language (SQL) inputs.</p>",
        )

        exporter = ExportReportDocx(self.report, report_template=self.report_template)
        rich_text_context = exporter.map_rich_texts()
        deduplicated_context = exporter.deduplicate_acronyms(rich_text_context)

        # Extract all three fields
        finding_data = deduplicated_context["findings"][0]
        description = str(finding_data["description_rt"].__html__())
        impact = str(finding_data["impact_rt"].__html__())
        mitigation = str(finding_data["mitigation_rt"].__html__())

        # First field (description) should keep the full expansion with acronym
        self.assertIn("Structured Query Language (SQL)", description)

        # Subsequent fields (impact, mitigation) should have just the acronym
        self.assertNotIn("Structured Query Language (SQL)", impact)
        self.assertNotIn("Structured Query Language (SQL)", mitigation)
        self.assertIn("SQL", impact)
        self.assertIn("SQL", mitigation)

    def test_case_sensitive_acronym_matching(self):
        """
        Test that acronym matching is case-sensitive (ACL != acl).
        """
        # Record uppercase ACL
        ReportAcronymExpansionFactory(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )

        ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="Case Test",
            description=(
                "<p>The Access Control List (ACL) and some text (acl) here.</p>"
            ),
        )

        exporter = ExportReportDocx(self.report, report_template=self.report_template)
        rich_text_context = exporter.map_rich_texts()
        deduplicated_context = exporter.deduplicate_acronyms(rich_text_context)

        finding_desc = str(deduplicated_context["findings"][0]["description_rt"].__html__())

        # Uppercase ACL should be present (first occurrence keeps acronym)
        self.assertIn("Access Control List (ACL)", finding_desc)
        # Lowercase acl should remain unchanged (not tracked in database)
        self.assertIn("(acl)", finding_desc)

    def test_full_document_generation_with_acronym_deduplication(self):
        """
        Integration test: Generate an actual Word document with multiple findings
        containing the same acronym expansion, then parse the document to verify
        deduplication worked correctly.
        """
        # Record that "ACL" has been expanded in this report
        ReportAcronymExpansionFactory(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )

        # Create first finding with ACL
        finding1 = ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="First Finding",
            description="<p>test Access Control List (ACL)</p>",
        )

        # Create second finding with ACL (should be deduplicated)
        finding2 = ReportFindingLinkFactory(
            report=self.report,
            severity=self.severity,
            finding_type=self.finding_type,
            assigned_to=self.user,
            title="Second Finding",
            description="<p>test 2 Access Control List (ACL)</p>",
        )

        # Generate the Word document
        exporter = ExportReportDocx(self.report, report_template=self.report_template)

        # Create temporary file for the document
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Generate the document (returns BytesIO)
            docx_bytes = exporter.run()

            # Write BytesIO to file
            with open(tmp_path, 'wb') as f:
                f.write(docx_bytes.getvalue())

            # Parse the generated document
            doc = Document(tmp_path)

            # Extract all text from paragraphs
            all_text = "\n".join([para.text for para in doc.paragraphs])

            # Count occurrences of the full expansion with acronym
            full_expansion_pattern = r"Access Control List \(ACL\)"
            full_matches = re.findall(full_expansion_pattern, all_text)

            # Count occurrences of just 'ACL' (not inside the full expansion)
            # Look for ACL that's not preceded by 'List ('
            just_acl_pattern = r"(?<!List \()ACL(?!\))"
            just_acl_matches = re.findall(just_acl_pattern, all_text)

            # Debug output
            print(f"\n=== Acronym Deduplication Analysis ===")
            print(f"Full expansion 'Access Control List (ACL)' count: {len(full_matches)}")
            print(f"Just acronym 'ACL' count: {len(just_acl_matches)}")
            print(f"\nAll text excerpts containing 'ACL':")
            for line in all_text.split("\n"):
                if "ACL" in line:
                    print(f"  - {line.strip()}")

            # Verification:
            # - Should have exactly 1 occurrence of "Access Control List (ACL)" (first instance)
            # - Should have at least 1 occurrence of just "ACL" (deduplicated subsequent instances)
            self.assertEqual(
                len(full_matches),
                1,
                f"Expected exactly 1 occurrence of 'Access Control List (ACL)', found {len(full_matches)}. "
                f"Deduplication should keep only the first instance with full expansion.",
            )

            self.assertGreaterEqual(
                len(just_acl_matches),
                1,
                f"Expected at least 1 occurrence of just 'ACL', found {len(just_acl_matches)}. "
                f"Subsequent occurrences should be replaced with just the acronym.",
            )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
