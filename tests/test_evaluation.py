"""
Tests for the evaluation module (claim_extractor and evaluation_interface).
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from src.evaluation.claim_extractor import (
    Claim,
    ClaimType,
    ClaimExtractor,
)
from src.evaluation.evaluation_interface import (
    Evaluation,
    EvaluationInterface,
    Verdict,
)


# ==================== ClaimType Tests ====================


class TestClaimTypeEnum:
    """Test ClaimType enumeration."""

    def test_claim_type_values(self):
        """Test all claim types are defined."""
        assert ClaimType.DATA_ANALYSIS.value == "data_analysis"
        assert ClaimType.LITERATURE.value == "literature"
        assert ClaimType.INTERPRETATION.value == "interpretation"


# ==================== Claim Tests ====================


class TestClaimDataclass:
    """Test Claim dataclass."""

    def test_create_claim_basic(self):
        """Test creating a basic claim."""
        claim = Claim(
            claim_id="claim_001",
            text="The correlation is significant",
            claim_type=ClaimType.DATA_ANALYSIS,
            discovery_title="Test Discovery"
        )

        assert claim.claim_id == "claim_001"
        assert claim.text == "The correlation is significant"
        assert claim.claim_type == ClaimType.DATA_ANALYSIS
        assert claim.discovery_title == "Test Discovery"
        assert claim.confidence is None
        assert claim.context is None
        assert claim.source_section is None
        assert claim.metadata == {}

    def test_create_claim_full(self):
        """Test creating a claim with all fields."""
        claim = Claim(
            claim_id="claim_002",
            text="Data shows 15% increase",
            claim_type=ClaimType.DATA_ANALYSIS,
            discovery_title="Growth Analysis",
            confidence=0.95,
            context="During Q4...",
            source_section="findings",
            metadata={"verified": True}
        )

        assert claim.claim_id == "claim_002"
        assert claim.confidence == 0.95
        assert claim.context == "During Q4..."
        assert claim.source_section == "findings"
        assert claim.metadata == {"verified": True}

    def test_claim_to_dict(self):
        """Test converting claim to dictionary."""
        claim = Claim(
            claim_id="claim_003",
            text="Test claim",
            claim_type=ClaimType.LITERATURE,
            discovery_title="Test",
            confidence=0.8
        )

        result = claim.to_dict()

        assert result["claim_id"] == "claim_003"
        assert result["text"] == "Test claim"
        assert result["claim_type"] == "literature"
        assert result["discovery_title"] == "Test"
        assert result["confidence"] == 0.8

    def test_claim_from_dict(self):
        """Test creating claim from dictionary."""
        data = {
            "claim_id": "claim_004",
            "text": "Test from dict",
            "claim_type": "interpretation",
            "discovery_title": "Test Discovery",
            "confidence": 0.7,
            "context": None,
            "source_section": None,
            "metadata": {}
        }

        claim = Claim.from_dict(data)

        assert claim.claim_id == "claim_004"
        assert claim.text == "Test from dict"
        assert claim.claim_type == ClaimType.INTERPRETATION
        assert claim.confidence == 0.7


# ==================== ClaimExtractor Tests ====================


class TestClaimExtractor:
    """Test ClaimExtractor functionality."""

    def test_create_extractor(self):
        """Test creating a claim extractor."""
        extractor = ClaimExtractor()

        assert extractor.claim_counter == 0

    def test_extract_claims_file_not_found(self, tmp_path):
        """Test extracting claims from nonexistent file."""
        extractor = ClaimExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_claims(tmp_path / "nonexistent.md")

    def test_extract_claims_from_report(self, tmp_path):
        """Test extracting claims from a markdown report."""
        extractor = ClaimExtractor()

        # Create a sample report
        report_content = """
# Discovery Report

## Discovery 1: Climate Temperature Analysis

**Confidence:** 0.85
**Novelty Score:** 0.7

The analysis reveals a significant increase in global temperatures.
Data shows a clear correlation between CO2 levels and temperature rise.
This finding is consistent with previous research on climate change.

### Supporting Evidence

**Data Analysis:**

- Temperature increased by 2°C over 50 years (confidence: 0.90)
- CO2 levels rose by 40% since 1980 (confidence: 0.95)

**Related Hypotheses:**

- Rising CO2 causes temperature increase (confidence: 0.80)

---

## Discovery 2: Economic Impact Study

**Confidence:** 0.75
**Novelty Score:** 0.5

Economic indicators suggest a strong relationship between climate and GDP.
The findings imply that climate policies may affect economic growth.

### Supporting Evidence

**Data Analysis:**

- GDP correlation with temperature anomalies: r=0.65 (confidence: 0.78)

---
"""

        report_path = tmp_path / "report.md"
        report_path.write_text(report_content)

        claims = extractor.extract_claims(report_path)

        # Should extract multiple claims
        assert len(claims) > 0

        # Check claim attributes
        for claim in claims:
            assert claim.claim_id is not None
            assert claim.text is not None
            assert claim.claim_type in ClaimType
            assert claim.discovery_title is not None

    def test_categorize_claim_data(self):
        """Test categorizing data analysis claims."""
        extractor = ClaimExtractor()

        data_claims = [
            "The correlation between X and Y is 0.85",
            "We observed a statistically significant increase p<0.01",
            "The mean value was 42.5 with standard deviation of 3.2",
            "Results show a 15% increase in sales",
            "Values were higher than expected in all groups",
        ]

        for text in data_claims:
            claim_type = extractor._categorize_claim(text)
            assert claim_type == ClaimType.DATA_ANALYSIS, f"Failed for: {text}"

    def test_categorize_claim_literature(self):
        """Test categorizing literature claims."""
        extractor = ClaimExtractor()

        lit_claims = [
            "According to previous research, this effect is well-known",
            "This finding is consistent with prior studies",
            "Literature shows similar patterns in other domains",
            "As reported by Smith et al., the phenomenon is documented",
        ]

        for text in lit_claims:
            claim_type = extractor._categorize_claim(text)
            assert claim_type == ClaimType.LITERATURE, f"Failed for: {text}"

    def test_categorize_claim_interpretation(self):
        """Test categorizing interpretation claims."""
        extractor = ClaimExtractor()

        interp_claims = [
            "This suggests that the mechanism involves feedback loops",
            "The results may indicate a causal relationship",
            "Therefore, we conclude that the hypothesis is supported",
            "This finding appears to explain the observed anomaly",
        ]

        for text in interp_claims:
            claim_type = extractor._categorize_claim(text)
            assert claim_type == ClaimType.INTERPRETATION, f"Failed for: {text}"

    def test_split_sentences(self):
        """Test sentence splitting."""
        extractor = ClaimExtractor()

        text = "First sentence. Second sentence! Third sentence? Last one."
        sentences = extractor._split_sentences(text)

        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]

    def test_split_sentences_with_abbreviations(self):
        """Test sentence splitting with abbreviations."""
        extractor = ClaimExtractor()

        text = "The U.S. has high rates. The value is 3.14 percent. Dr. Smith reported results."
        sentences = extractor._split_sentences(text)

        # Should handle abbreviations correctly
        assert len(sentences) >= 2

    def test_save_claims(self, tmp_path):
        """Test saving claims to JSON."""
        extractor = ClaimExtractor()

        claims = [
            Claim(
                claim_id="claim_001",
                text="Test claim 1",
                claim_type=ClaimType.DATA_ANALYSIS,
                discovery_title="Discovery 1",
                confidence=0.9
            ),
            Claim(
                claim_id="claim_002",
                text="Test claim 2",
                claim_type=ClaimType.LITERATURE,
                discovery_title="Discovery 1",
                confidence=0.8
            ),
        ]

        output_path = tmp_path / "claims.json"
        extractor.save_claims(claims, output_path)

        # Verify file was created
        assert output_path.exists()

        # Load and verify content
        with open(output_path) as f:
            data = json.load(f)

        assert data["total_claims"] == 2
        assert data["claims_by_type"]["data_analysis"] == 1
        assert data["claims_by_type"]["literature"] == 1
        assert len(data["claims"]) == 2

    def test_load_claims(self, tmp_path):
        """Test loading claims from JSON."""
        extractor = ClaimExtractor()

        # Create JSON file
        claims_data = {
            "total_claims": 2,
            "claims_by_type": {"data_analysis": 1, "interpretation": 1},
            "claims": [
                {
                    "claim_id": "claim_001",
                    "text": "Claim 1",
                    "claim_type": "data_analysis",
                    "discovery_title": "Test",
                    "confidence": 0.9,
                    "context": None,
                    "source_section": None,
                    "metadata": {}
                },
                {
                    "claim_id": "claim_002",
                    "text": "Claim 2",
                    "claim_type": "interpretation",
                    "discovery_title": "Test",
                    "confidence": 0.8,
                    "context": None,
                    "source_section": None,
                    "metadata": {}
                }
            ]
        }

        input_path = tmp_path / "claims.json"
        with open(input_path, 'w') as f:
            json.dump(claims_data, f)

        claims = extractor.load_claims(input_path)

        assert len(claims) == 2
        assert claims[0].claim_id == "claim_001"
        assert claims[0].claim_type == ClaimType.DATA_ANALYSIS
        assert claims[1].claim_id == "claim_002"
        assert claims[1].claim_type == ClaimType.INTERPRETATION


# ==================== Verdict Tests ====================


class TestVerdictEnum:
    """Test Verdict enumeration."""

    def test_verdict_values(self):
        """Test all verdict values are defined."""
        assert Verdict.SUPPORTED.value == "supported"
        assert Verdict.REFUTED.value == "refuted"
        assert Verdict.UNCLEAR.value == "unclear"
        assert Verdict.PARTIALLY_SUPPORTED.value == "partially_supported"


# ==================== Evaluation Tests ====================


class TestEvaluationDataclass:
    """Test Evaluation dataclass."""

    def test_create_evaluation_basic(self):
        """Test creating a basic evaluation."""
        evaluation = Evaluation(
            claim_id="claim_001",
            verdict=Verdict.SUPPORTED,
            evaluator_id="expert_1"
        )

        assert evaluation.claim_id == "claim_001"
        assert evaluation.verdict == Verdict.SUPPORTED
        assert evaluation.evaluator_id == "expert_1"
        assert evaluation.evaluation_id is not None  # Auto-generated
        assert evaluation.timestamp is not None  # Auto-generated

    def test_create_evaluation_full(self):
        """Test creating evaluation with all fields."""
        evaluation = Evaluation(
            evaluation_id="eval_001",
            claim_id="claim_002",
            verdict=Verdict.REFUTED,
            evaluator_id="expert_2",
            notes="This claim is incorrect",
            confidence_in_verdict=0.95,
            metadata={"source": "domain_review"}
        )

        assert evaluation.evaluation_id == "eval_001"
        assert evaluation.verdict == Verdict.REFUTED
        assert evaluation.notes == "This claim is incorrect"
        assert evaluation.confidence_in_verdict == 0.95
        assert evaluation.metadata["source"] == "domain_review"

    def test_evaluation_to_dict(self):
        """Test converting evaluation to dictionary."""
        evaluation = Evaluation(
            claim_id="claim_003",
            verdict=Verdict.PARTIALLY_SUPPORTED,
            evaluator_id="expert_3",
            notes="Partially correct"
        )

        result = evaluation.to_dict()

        assert result["claim_id"] == "claim_003"
        assert result["verdict"] == "partially_supported"
        assert result["evaluator_id"] == "expert_3"
        assert result["notes"] == "Partially correct"
        assert "timestamp" in result

    def test_evaluation_without_verdict(self):
        """Test evaluation without verdict."""
        evaluation = Evaluation(
            claim_id="claim_004",
            evaluator_id="expert_4"
        )

        result = evaluation.to_dict()

        assert result["verdict"] is None


# ==================== EvaluationInterface Tests ====================


class TestEvaluationInterface:
    """Test EvaluationInterface functionality."""

    def test_create_interface(self, tmp_path):
        """Test creating an evaluation interface."""
        db_path = tmp_path / "evaluations.db"
        interface = EvaluationInterface(db_path)

        assert interface.db_path == db_path
        assert db_path.exists()

    def test_database_initialization(self, tmp_path):
        """Test that database is properly initialized."""
        db_path = tmp_path / "eval" / "test.db"
        interface = EvaluationInterface(db_path)

        # Parent directory should be created
        assert db_path.parent.exists()
        assert db_path.exists()

    def test_store_claims(self, tmp_path):
        """Test storing claims in database."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        claims = [
            Claim(
                claim_id="claim_001",
                text="Test claim 1",
                claim_type=ClaimType.DATA_ANALYSIS,
                discovery_title="Test",
                confidence=0.9
            ),
            Claim(
                claim_id="claim_002",
                text="Test claim 2",
                claim_type=ClaimType.LITERATURE,
                discovery_title="Test",
                confidence=0.8
            )
        ]

        interface.store_claims(claims)

        # Should not raise an error
        # Can verify by getting unevaluated claims
        unevaluated = interface.get_unevaluated_claims()
        assert len(unevaluated) == 2

    def test_save_and_get_evaluations(self, tmp_path):
        """Test saving and retrieving evaluations."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        evaluation = Evaluation(
            claim_id="claim_001",
            verdict=Verdict.SUPPORTED,
            evaluator_id="expert_1",
            notes="Correct claim",
            confidence_in_verdict=0.9
        )

        interface.save_evaluation(evaluation)

        # Retrieve evaluations
        evaluations = interface.get_evaluations(claim_id="claim_001")

        assert len(evaluations) == 1
        assert evaluations[0].claim_id == "claim_001"
        assert evaluations[0].verdict == Verdict.SUPPORTED
        assert evaluations[0].notes == "Correct claim"

    def test_save_multiple_evaluations(self, tmp_path):
        """Test saving multiple evaluations."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        evaluations = [
            Evaluation(
                claim_id="claim_001",
                verdict=Verdict.SUPPORTED,
                evaluator_id="expert_1"
            ),
            Evaluation(
                claim_id="claim_002",
                verdict=Verdict.REFUTED,
                evaluator_id="expert_1"
            ),
            Evaluation(
                claim_id="claim_003",
                verdict=Verdict.UNCLEAR,
                evaluator_id="expert_2"
            )
        ]

        interface.save_evaluations(evaluations)

        # Get all evaluations
        all_evals = interface.get_evaluations()
        assert len(all_evals) == 3

    def test_filter_evaluations_by_evaluator(self, tmp_path):
        """Test filtering evaluations by evaluator."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        evaluations = [
            Evaluation(claim_id="c1", verdict=Verdict.SUPPORTED, evaluator_id="expert_1"),
            Evaluation(claim_id="c2", verdict=Verdict.REFUTED, evaluator_id="expert_1"),
            Evaluation(claim_id="c3", verdict=Verdict.UNCLEAR, evaluator_id="expert_2"),
        ]

        interface.save_evaluations(evaluations)

        # Filter by evaluator
        expert1_evals = interface.get_evaluations(evaluator_id="expert_1")
        expert2_evals = interface.get_evaluations(evaluator_id="expert_2")

        assert len(expert1_evals) == 2
        assert len(expert2_evals) == 1

    def test_filter_evaluations_by_verdict(self, tmp_path):
        """Test filtering evaluations by verdict."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        evaluations = [
            Evaluation(claim_id="c1", verdict=Verdict.SUPPORTED, evaluator_id="e1"),
            Evaluation(claim_id="c2", verdict=Verdict.SUPPORTED, evaluator_id="e2"),
            Evaluation(claim_id="c3", verdict=Verdict.REFUTED, evaluator_id="e3"),
        ]

        interface.save_evaluations(evaluations)

        supported_evals = interface.get_evaluations(verdict=Verdict.SUPPORTED)
        refuted_evals = interface.get_evaluations(verdict=Verdict.REFUTED)

        assert len(supported_evals) == 2
        assert len(refuted_evals) == 1

    def test_skip_evaluations_without_verdict(self, tmp_path):
        """Test that evaluations without verdict are skipped."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        evaluations = [
            Evaluation(claim_id="c1", verdict=Verdict.SUPPORTED, evaluator_id="e1"),
            Evaluation(claim_id="c2", verdict=None, evaluator_id="e1"),  # No verdict
        ]

        interface.save_evaluations(evaluations)

        all_evals = interface.get_evaluations()

        # Should only save the one with verdict
        assert len(all_evals) == 1

    def test_get_unevaluated_claims(self, tmp_path):
        """Test getting unevaluated claims."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        # Store some claims
        claims = [
            Claim(claim_id="c1", text="Claim 1", claim_type=ClaimType.DATA_ANALYSIS, discovery_title="T"),
            Claim(claim_id="c2", text="Claim 2", claim_type=ClaimType.LITERATURE, discovery_title="T"),
            Claim(claim_id="c3", text="Claim 3", claim_type=ClaimType.INTERPRETATION, discovery_title="T"),
        ]
        interface.store_claims(claims)

        # Evaluate one claim
        evaluation = Evaluation(claim_id="c1", verdict=Verdict.SUPPORTED, evaluator_id="default")
        interface.save_evaluation(evaluation)

        # Get unevaluated claims
        unevaluated = interface.get_unevaluated_claims("default")

        assert len(unevaluated) == 2
        claim_ids = [c.claim_id for c in unevaluated]
        assert "c1" not in claim_ids
        assert "c2" in claim_ids
        assert "c3" in claim_ids

    def test_collect_verdict_non_interactive(self, tmp_path):
        """Test collecting verdict in non-interactive mode."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        claim = Claim(
            claim_id="claim_001",
            text="Test claim",
            claim_type=ClaimType.DATA_ANALYSIS,
            discovery_title="Test"
        )

        evaluation = interface.collect_verdict(
            claim,
            evaluator_id="test_evaluator",
            interactive=False
        )

        assert evaluation.claim_id == "claim_001"
        assert evaluation.evaluator_id == "test_evaluator"
        assert evaluation.verdict is None  # No verdict in non-interactive mode

    def test_export_evaluations(self, tmp_path):
        """Test exporting evaluations to JSON."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        # Add some evaluations
        evaluations = [
            Evaluation(claim_id="c1", verdict=Verdict.SUPPORTED, evaluator_id="e1"),
            Evaluation(claim_id="c2", verdict=Verdict.REFUTED, evaluator_id="e1"),
        ]
        interface.save_evaluations(evaluations)

        # Export
        export_path = tmp_path / "export.json"
        interface.export_evaluations(export_path)

        # Verify export
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)

        assert data["total_evaluations"] == 2
        assert "export_timestamp" in data
        assert len(data["evaluations"]) == 2


class TestPresentClaim:
    """Test claim presentation."""

    def test_present_claim(self, tmp_path, capsys):
        """Test presenting a claim."""
        db_path = tmp_path / "test.db"
        interface = EvaluationInterface(db_path)

        claim = Claim(
            claim_id="claim_001",
            text="The data shows a significant correlation",
            claim_type=ClaimType.DATA_ANALYSIS,
            discovery_title="Correlation Analysis",
            confidence=0.85,
            context="In the study of X and Y..."
        )

        interface.present_claim(claim, show_confidence=True)

        captured = capsys.readouterr()

        assert "CLAIM claim_001" in captured.out
        assert "Correlation Analysis" in captured.out
        assert "data_analysis" in captured.out
        assert "0.85" in captured.out
        assert "significant correlation" in captured.out


class TestIntegration:
    """Integration tests for claim extraction and evaluation."""

    def test_full_workflow(self, tmp_path):
        """Test full workflow from extraction to evaluation."""
        # Create extractor and interface
        extractor = ClaimExtractor()
        db_path = tmp_path / "workflow.db"
        interface = EvaluationInterface(db_path)

        # Create sample report
        report_content = """
# Discovery Report

## Discovery 1: Test Analysis

**Confidence:** 0.90
**Novelty Score:** 0.8

This analysis shows a strong correlation between variables.
The data suggests that the relationship is causal.

### Supporting Evidence

**Data Analysis:**

- Mean value is 42.5 (confidence: 0.95)
- Correlation r=0.85 (confidence: 0.90)
"""

        report_path = tmp_path / "report.md"
        report_path.write_text(report_content)

        # Extract claims
        claims = extractor.extract_claims(report_path)
        assert len(claims) > 0

        # Store claims
        interface.store_claims(claims)

        # Check unevaluated claims
        unevaluated = interface.get_unevaluated_claims()
        assert len(unevaluated) == len(claims)

        # Create evaluations for all claims
        evaluations = []
        for claim in claims:
            evaluation = Evaluation(
                claim_id=claim.claim_id,
                verdict=Verdict.SUPPORTED,
                evaluator_id="test_expert",
                confidence_in_verdict=0.9
            )
            evaluations.append(evaluation)

        interface.save_evaluations(evaluations)

        # Verify all evaluated
        remaining = interface.get_unevaluated_claims("test_expert")
        assert len(remaining) == 0

        # Export results
        export_path = tmp_path / "results.json"
        interface.export_evaluations(export_path)
        assert export_path.exists()
