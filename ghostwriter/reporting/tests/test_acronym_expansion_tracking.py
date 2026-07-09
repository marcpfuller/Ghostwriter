"""Tests for report-level acronym expansion tracking."""

# Django Imports
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

# Ghostwriter Libraries
from ghostwriter.factories import ReportFactory, UserFactory
from ghostwriter.reporting.models import ReportAcronymExpansion

User = get_user_model()


class ReportAcronymExpansionModelTests(TestCase):
    """Test ReportAcronymExpansion model."""

    @classmethod
    def setUpTestData(cls):
        cls.user = UserFactory()
        cls.report = ReportFactory()

    def test_model_creation(self):
        """Test creating an expansion record."""
        expansion = ReportAcronymExpansion.objects.create(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )

        self.assertEqual(expansion.acronym, "ACL")
        self.assertEqual(expansion.expansion, "Access Control List")
        self.assertEqual(expansion.report, self.report)
        self.assertIsNotNone(expansion.expanded_at)

    def test_unique_together_constraint(self):
        """Test that (report, acronym) must be unique."""
        ReportAcronymExpansion.objects.create(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )

        # Try to create duplicate - should raise IntegrityError
        with self.assertRaises(IntegrityError):
            ReportAcronymExpansion.objects.create(
                report=self.report,
                acronym="ACL",
                expansion="Another Expansion",
            )

    def test_different_reports_same_acronym(self):
        """Test same acronym can have different expansions in different reports."""
        report2 = ReportFactory()

        exp1 = ReportAcronymExpansion.objects.create(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )

        exp2 = ReportAcronymExpansion.objects.create(
            report=report2,
            acronym="ACL",
            expansion="Anterior Cruciate Ligament",
        )

        self.assertNotEqual(exp1.expansion, exp2.expansion)
        self.assertEqual(
            ReportAcronymExpansion.objects.filter(acronym="ACL").count(), 2
        )

    def test_cascade_delete_on_report_delete(self):
        """Test expansions are deleted when report is deleted."""
        ReportAcronymExpansion.objects.create(
            report=self.report,
            acronym="ACL",
            expansion="Access Control List",
        )
        ReportAcronymExpansion.objects.create(
            report=self.report,
            acronym="DNS",
            expansion="Domain Name System",
        )

        self.assertEqual(ReportAcronymExpansion.objects.count(), 2)

        self.report.delete()

        self.assertEqual(ReportAcronymExpansion.objects.count(), 0)

    def test_string_representation(self):
        """Test the string representation of expansion."""
        expansion = ReportAcronymExpansion.objects.create(
            report=self.report,
            acronym="API",
            expansion="Application Programming Interface",
        )

        expected = f"API: Application Programming Interface (Report #{self.report.id})"
        self.assertEqual(str(expansion), expected)




class ReportAcronymExpansionQueryTests(TestCase):
    """Test querying expansion records."""

    @classmethod
    def setUpTestData(cls):
        cls.user = UserFactory()
        cls.report1 = ReportFactory()
        cls.report2 = ReportFactory()

        # Create expansions for report1
        ReportAcronymExpansion.objects.create(
            report=cls.report1,
            acronym="ACL",
            expansion="Access Control List",
        )
        ReportAcronymExpansion.objects.create(
            report=cls.report1,
            acronym="API",
            expansion="Application Programming Interface",
        )

        # Create expansion for report2
        ReportAcronymExpansion.objects.create(
            report=cls.report2,
            acronym="DNS",
            expansion="Domain Name System",
        )

    def test_get_expansions_for_report(self):
        """Test retrieving all expansions for a specific report."""
        expansions = ReportAcronymExpansion.objects.filter(report=self.report1)

        self.assertEqual(expansions.count(), 2)
        acronyms = set(expansions.values_list("acronym", flat=True))
        self.assertEqual(acronyms, {"ACL", "API"})

    def test_check_if_acronym_expanded_in_report(self):
        """Test checking if specific acronym was expanded in report."""
        exists = ReportAcronymExpansion.objects.filter(
            report=self.report1, acronym="ACL"
        ).exists()

        self.assertTrue(exists)

        exists = ReportAcronymExpansion.objects.filter(
            report=self.report1, acronym="DNS"
        ).exists()

        self.assertFalse(exists)

    def test_get_expansion_for_acronym(self):
        """Test getting the expansion text for a specific acronym."""
        expansion = ReportAcronymExpansion.objects.get(
            report=self.report1, acronym="ACL"
        )

        self.assertEqual(expansion.expansion, "Access Control List")

    def test_get_all_expansions_as_dict(self):
        """Test retrieving expansions as a dictionary (for API response)."""
        expansions = ReportAcronymExpansion.objects.filter(report=self.report1)
        expansion_dict = {exp.acronym: exp.expansion for exp in expansions}

        expected = {
            "ACL": "Access Control List",
            "API": "Application Programming Interface",
        }
        self.assertEqual(expansion_dict, expected)
