"""Tests for report acronym expansion tracking API endpoints."""

# Standard Libraries
import json

# Django Imports
from django.test import Client, TestCase
from django.urls import reverse

# Ghostwriter Libraries
from ghostwriter.factories import (
    ReportAcronymExpansionFactory,
    ReportFactory,
    UserFactory,
)
from ghostwriter.reporting.models import ReportAcronymExpansion


class AcronymExpansionAPITests(TestCase):
    """Test API endpoints for acronym expansion tracking."""

    @classmethod
    def setUpTestData(cls):
        cls.user = UserFactory(password="testpass", role="manager")
        cls.report = ReportFactory()
        cls.client = Client()

    def setUp(self):
        self.client.login(username=self.user.username, password="testpass")

    def test_get_expansions_empty_report(self):
        """Test getting expansions for report with no expansions."""
        url = reverse(
            "api:ajax_get_report_acronym_expansions",
            kwargs={"report_id": self.report.id},
        )
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, {})

    def test_get_expansions_with_data(self):
        """Test getting expansions for report with existing expansions."""
        # Create some expansions
        ReportAcronymExpansionFactory(
            report=self.report, acronym="ACL", expansion="Access Control List"
        )
        ReportAcronymExpansionFactory(
            report=self.report, acronym="API", expansion="Application Programming Interface"
        )

        url = reverse(
            "api:ajax_get_report_acronym_expansions",
            kwargs={"report_id": self.report.id},
        )
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(
            data,
            {
                "ACL": "Access Control List",
                "API": "Application Programming Interface",
            },
        )

    def test_get_expansions_requires_auth(self):
        """Test that unauthenticated requests are rejected."""
        self.client.logout()

        url = reverse(
            "api:ajax_get_report_acronym_expansions",
            kwargs={"report_id": self.report.id},
        )
        response = self.client.get(url)

        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_post_new_expansion(self):
        """Test recording a new acronym expansion."""
        url = reverse(
            "api:ajax_record_report_acronym_expansion",
            kwargs={"report_id": self.report.id},
        )

        data = {"acronym": "DNS", "expansion": "Domain Name System"}

        response = self.client.post(
            url, json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 201)
        response_data = response.json()
        self.assertEqual(response_data["result"], "success")

        # Verify it was created
        expansion = ReportAcronymExpansion.objects.get(
            report=self.report, acronym="DNS"
        )
        self.assertEqual(expansion.expansion, "Domain Name System")

    def test_post_duplicate_acronym_ignored(self):
        """Test that posting a duplicate acronym returns existing expansion."""
        # Create existing expansion
        ReportAcronymExpansionFactory(
            report=self.report, acronym="ACL", expansion="Access Control List"
        )

        url = reverse(
            "api:ajax_record_report_acronym_expansion",
            kwargs={"report_id": self.report.id},
        )

        data = {"acronym": "ACL", "expansion": "Another Expansion"}

        response = self.client.post(
            url, json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["result"], "exists")
        self.assertEqual(response_data["expansion"], "Access Control List")

        # Verify no new record created
        self.assertEqual(
            ReportAcronymExpansion.objects.filter(
                report=self.report, acronym="ACL"
            ).count(),
            1,
        )

    def test_post_expansion_missing_fields(self):
        """Test that missing required fields return error."""
        url = reverse(
            "api:ajax_record_report_acronym_expansion",
            kwargs={"report_id": self.report.id},
        )

        # Missing expansion field
        data = {"acronym": "DNS"}

        response = self.client.post(
            url, json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertEqual(response_data["result"], "error")
        self.assertIn("required", response_data["message"].lower())

    def test_post_expansion_invalid_json(self):
        """Test that invalid JSON returns error."""
        url = reverse(
            "api:ajax_record_report_acronym_expansion",
            kwargs={"report_id": self.report.id},
        )

        response = self.client.post(url, "invalid json", content_type="application/json")

        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertEqual(response_data["result"], "error")

    def test_post_expansion_requires_auth(self):
        """Test that unauthenticated POST requests are rejected."""
        self.client.logout()

        url = reverse(
            "api:ajax_record_report_acronym_expansion",
            kwargs={"report_id": self.report.id},
        )

        data = {"acronym": "DNS", "expansion": "Domain Name System"}

        response = self.client.post(
            url, json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_get_expansions_nonexistent_report(self):
        """Test getting expansions for non-existent report returns 404."""
        url = reverse(
            "api:ajax_get_report_acronym_expansions",
            kwargs={"report_id": 99999},
        )
        response = self.client.get(url)

        self.assertEqual(response.status_code, 404)

    def test_post_expansion_nonexistent_report(self):
        """Test posting expansion for non-existent report returns 404."""
        url = reverse(
            "api:ajax_record_report_acronym_expansion",
            kwargs={"report_id": 99999},
        )

        data = {"acronym": "DNS", "expansion": "Domain Name System"}

        response = self.client.post(
            url, json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 404)

    def test_expansions_isolated_by_report(self):
        """Test that expansions are properly isolated between reports."""
        report2 = ReportFactory()

        # Create expansion for report1
        ReportAcronymExpansionFactory(
            report=self.report, acronym="ACL", expansion="Access Control List"
        )

        # Create expansion for report2
        ReportAcronymExpansionFactory(
            report=report2, acronym="ACL", expansion="Anterior Cruciate Ligament"
        )

        # Get expansions for report1
        url1 = reverse(
            "api:ajax_get_report_acronym_expansions",
            kwargs={"report_id": self.report.id},
        )
        response1 = self.client.get(url1)
        data1 = response1.json()

        # Get expansions for report2
        url2 = reverse(
            "api:ajax_get_report_acronym_expansions",
            kwargs={"report_id": report2.id},
        )
        response2 = self.client.get(url2)
        data2 = response2.json()

        self.assertEqual(data1["ACL"], "Access Control List")
        self.assertEqual(data2["ACL"], "Anterior Cruciate Ligament")
