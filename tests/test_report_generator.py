"""
Tests for the ReportGenerator.
"""

import tempfile
from pathlib import Path

import pytest

from src.reporting.report_generator import ReportGenerator
from src.world_model.graph import EdgeType, NodeType, WorldModel


class TestReportGeneratorBasics:
    """Test basic report generator functionality."""

    def test_create_report_generator(self):
        """Test creating a report generator."""
        wm = WorldModel()
        rg = ReportGenerator(wm)

        assert rg.world_model == wm
        assert rg.min_confidence == 0.7
        assert rg.max_discoveries == 5

    def test_create_with_custom_params(self):
        """Test creating with custom parameters."""
        wm = WorldModel()
        rg = ReportGenerator(
            wm,
            min_confidence=0.8,
            max_discoveries=3,
        )

        assert rg.min_confidence == 0.8
        assert rg.max_discoveries == 3


class TestFindingExtraction:
    """Test finding extraction and ranking."""

    def test_extract_high_confidence_findings(self):
        """Test extracting high-confidence findings."""
        wm = WorldModel()

        # Add findings with various confidence levels
        f1 = wm.add_finding("High confidence finding", confidence=0.95)
        f2 = wm.add_finding("Medium confidence finding", confidence=0.75)
        f3 = wm.add_finding("Low confidence finding", confidence=0.5)

        rg = ReportGenerator(wm, min_confidence=0.7)
        findings = rg.extract_high_confidence_findings()

        # Should only get f1 and f2
        assert len(findings) == 2
        finding_ids = [f["node_id"] for f in findings]
        assert f1 in finding_ids
        assert f2 in finding_ids
        assert f3 not in finding_ids

    def test_findings_sorted_by_confidence(self):
        """Test that findings are sorted by confidence."""
        wm = WorldModel()

        wm.add_finding("Medium finding", confidence=0.75)
        wm.add_finding("High finding", confidence=0.95)
        wm.add_finding("Higher finding", confidence=0.85)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()

        # Should be sorted by confidence descending
        assert findings[0]["confidence"] >= findings[1]["confidence"]
        assert findings[1]["confidence"] >= findings[2]["confidence"]


class TestNoveltyAssessment:
    """Test novelty assessment."""

    def test_novelty_with_no_papers(self):
        """Test novelty when there are no supporting papers."""
        wm = WorldModel()
        f1 = wm.add_finding("Novel finding", confidence=0.9)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()

        # Should have high novelty (no literature)
        assert findings[0]["novelty_score"] >= 0.3

    def test_novelty_with_question(self):
        """Test novelty when finding answers a question."""
        wm = WorldModel()

        q1 = wm.add_question("What causes X?")
        f1 = wm.add_finding("Finding that answers question", confidence=0.9)

        # Link question to finding
        wm.add_edge(q1, f1, EdgeType.RELATES_TO)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()

        # Should have moderate novelty (answers question)
        assert findings[0]["novelty_score"] >= 0.3

    def test_novelty_with_refuting_paper(self):
        """Test novelty when finding contradicts a paper."""
        wm = WorldModel()

        p1 = wm.add_paper(
            text="Paper summary",
            title="Existing Research",
            authors=["Smith, J."],
            year=2020,
        )
        f1 = wm.add_finding("Contradictory finding", confidence=0.9)

        # Link paper as refuting the finding
        wm.add_edge(p1, f1, EdgeType.REFUTES)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()

        # Should have high novelty (contradicts literature)
        assert findings[0]["novelty_score"] >= 0.5


class TestDiscoveryGrouping:
    """Test grouping findings into discoveries."""

    def test_group_single_finding(self):
        """Test grouping a single isolated finding."""
        wm = WorldModel()
        f1 = wm.add_finding("Isolated finding", confidence=0.9)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        assert len(discoveries) == 1
        assert len(discoveries[0].findings) == 1

    def test_group_connected_findings(self):
        """Test grouping connected findings."""
        wm = WorldModel()

        # Create connected findings
        f1 = wm.add_finding("Finding 1", confidence=0.9)
        f2 = wm.add_finding("Finding 2", confidence=0.85)
        f3 = wm.add_finding("Finding 3", confidence=0.8)

        # Connect them
        wm.add_edge(f1, f2, EdgeType.SUPPORTS)
        wm.add_edge(f2, f3, EdgeType.SUPPORTS)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        # Should group all three into one discovery
        assert len(discoveries) == 1
        assert len(discoveries[0].findings) == 3

    def test_group_separate_clusters(self):
        """Test grouping separate clusters of findings."""
        wm = WorldModel()

        # Cluster 1
        f1 = wm.add_finding("Finding 1a", confidence=0.9)
        f2 = wm.add_finding("Finding 1b", confidence=0.85)
        wm.add_edge(f1, f2, EdgeType.SUPPORTS)

        # Cluster 2 (unconnected)
        f3 = wm.add_finding("Finding 2a", confidence=0.8)
        f4 = wm.add_finding("Finding 2b", confidence=0.75)
        wm.add_edge(f3, f4, EdgeType.SUPPORTS)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        # Should create two separate discoveries
        assert len(discoveries) == 2

    def test_discovery_includes_papers(self):
        """Test that discoveries include supporting papers."""
        wm = WorldModel()

        p1 = wm.add_paper(
            text="Paper summary",
            title="Supporting Research",
            authors=["Smith, J."],
            year=2020,
        )
        f1 = wm.add_finding("Finding", confidence=0.9)

        # Link paper as supporting
        wm.add_edge(p1, f1, EdgeType.SUPPORTS)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        assert len(discoveries) == 1
        assert len(discoveries[0].papers) == 1
        assert discoveries[0].papers[0]["node_id"] == p1

    def test_max_discoveries_limit(self):
        """Test that max_discoveries limits output."""
        wm = WorldModel()

        # Create 10 isolated findings
        for i in range(10):
            wm.add_finding(f"Finding {i}", confidence=0.9)

        rg = ReportGenerator(wm, max_discoveries=3)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        # Should limit to 3
        assert len(discoveries) <= 3


class TestCitations:
    """Test citation system."""

    def test_create_paper_citation(self):
        """Test creating a paper citation."""
        wm = WorldModel()
        rg = ReportGenerator(wm)

        citation = rg._create_citation("paper", {
            "title": "Test Paper",
            "authors": ["Smith, J.", "Doe, A."],
            "year": 2023,
            "doi": "10.1234/test",
        })

        assert citation.citation_type == "paper"
        assert citation.citation_id == "1"
        assert "[1]" in citation.format_inline()
        assert "Smith, J. et al." in citation.format_bibliography()

    def test_create_code_citation(self):
        """Test creating a code citation."""
        wm = WorldModel()
        rg = ReportGenerator(wm)

        citation = rg._create_citation("code", {
            "provenance": "analysis.py:42",
            "node_id": "finding123",
        })

        assert citation.citation_type == "code"
        assert "Analysis" in citation.format_inline()
        assert "analysis.py:42" in citation.format_bibliography()

    def test_citation_deduplication(self):
        """Test that duplicate citations are not created."""
        wm = WorldModel()
        rg = ReportGenerator(wm)

        content = {
            "title": "Test Paper",
            "authors": ["Smith, J."],
            "year": 2023,
        }

        cit1 = rg._create_citation("paper", content)
        cit2 = rg._create_citation("paper", content)

        # Should be the same citation
        assert cit1.citation_id == cit2.citation_id


class TestReportGeneration:
    """Test full report generation."""

    def test_generate_report_basic(self, tmp_path):
        """Test generating a basic report."""
        wm = WorldModel()

        # Add some findings
        f1 = wm.add_finding(
            "Temperature has increased",
            code_link="analysis.py:10",
            confidence=0.95,
        )
        f2 = wm.add_finding(
            "CO2 levels are rising",
            code_link="analysis.py:20",
            confidence=0.90,
        )

        # Connect them
        wm.add_edge(f1, f2, EdgeType.RELATES_TO)

        rg = ReportGenerator(wm)
        output_path = tmp_path / "report.md"

        result = rg.generate_report(
            output_path,
            include_appendix=True,
            generate_narratives=False,  # Use templates
        )

        # Check files were created
        assert result["report"].exists()
        assert result["appendix"].exists()

        # Check report content
        report_content = result["report"].read_text()
        assert "Discovery Report" in report_content
        assert "Executive Summary" in report_content
        assert "Discovery 1:" in report_content
        assert "Temperature has increased" in report_content
        assert "CO2 levels are rising" in report_content

    def test_generate_report_with_papers(self, tmp_path):
        """Test generating a report with paper citations."""
        wm = WorldModel()

        p1 = wm.add_paper(
            text="Climate research summary",
            title="Climate Change Study",
            authors=["Smith, J.", "Doe, A."],
            year=2023,
            doi="10.1234/climate",
        )

        f1 = wm.add_finding(
            "Temperature anomaly detected",
            confidence=0.95,
        )

        wm.add_edge(p1, f1, EdgeType.SUPPORTS)

        rg = ReportGenerator(wm)
        output_path = tmp_path / "report.md"

        result = rg.generate_report(
            output_path,
            include_appendix=False,
            generate_narratives=False,
        )

        report_content = result["report"].read_text()

        # Check for paper citation
        assert "[1]" in report_content
        assert "References" in report_content
        assert "Climate Change Study" in report_content
        assert "Smith, J. and Doe, A." in report_content or "Smith, J. et al." in report_content

    def test_report_without_appendix(self, tmp_path):
        """Test generating report without appendix."""
        wm = WorldModel()
        f1 = wm.add_finding("Finding", confidence=0.9)

        rg = ReportGenerator(wm)
        output_path = tmp_path / "report.md"

        result = rg.generate_report(
            output_path,
            include_appendix=False,
        )

        assert "report" in result
        assert "appendix" not in result

    def test_appendix_includes_all_findings(self, tmp_path):
        """Test that appendix includes all high-confidence findings."""
        wm = WorldModel()

        for i in range(5):
            wm.add_finding(f"Finding {i}", confidence=0.9)

        rg = ReportGenerator(wm)
        output_path = tmp_path / "report.md"

        result = rg.generate_report(
            output_path,
            include_appendix=True,
            generate_narratives=False,
        )

        appendix_content = result["appendix"].read_text()

        # Check all findings are mentioned
        for i in range(5):
            assert f"Finding {i}" in appendix_content

    def test_empty_world_model(self, tmp_path):
        """Test generating report from empty world model."""
        wm = WorldModel()
        rg = ReportGenerator(wm)
        output_path = tmp_path / "report.md"

        result = rg.generate_report(
            output_path,
            generate_narratives=False,
        )

        # Should still create files
        assert result["report"].exists()

        report_content = result["report"].read_text()
        assert "Discovery Report" in report_content
        assert "Executive Summary" in report_content


class TestTemplateNarrative:
    """Test template-based narrative generation."""

    def test_template_narrative_basic(self):
        """Test basic template narrative."""
        wm = WorldModel()
        f1 = wm.add_finding("Test finding", confidence=0.9)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        narrative = rg._generate_template_narrative(discoveries[0])

        assert "finding" in narrative.lower()
        assert "confidence" in narrative.lower()

    def test_template_narrative_with_papers(self):
        """Test template narrative includes paper count."""
        wm = WorldModel()

        p1 = wm.add_paper(text="Paper 1", title="Title 1")
        p2 = wm.add_paper(text="Paper 2", title="Title 2")
        f1 = wm.add_finding("Finding", confidence=0.9)

        wm.add_edge(p1, f1, EdgeType.SUPPORTS)
        wm.add_edge(p2, f1, EdgeType.SUPPORTS)

        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        narrative = rg._generate_template_narrative(discoveries[0])

        assert "2 papers" in narrative or "2 paper" in narrative

    def test_template_narrative_mentions_novelty(self):
        """Test template narrative mentions high novelty."""
        wm = WorldModel()
        f1 = wm.add_finding("Novel finding", confidence=0.9)

        # Make it novel (no papers)
        rg = ReportGenerator(wm)
        findings = rg.extract_high_confidence_findings()
        discoveries = rg.group_findings_into_discoveries(findings)

        # Manually set high novelty for test
        discoveries[0].novelty_score = 0.8

        narrative = rg._generate_template_narrative(discoveries[0])

        assert "novel" in narrative.lower()
