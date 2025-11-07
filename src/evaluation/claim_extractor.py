"""
Claim Extractor - Extracts testable claims from generated reports.

This module parses markdown reports and identifies factual claims that can be
evaluated by domain experts, categorizing them by type (data analysis, literature,
or interpretation).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of claims that can be extracted."""
    DATA_ANALYSIS = "data_analysis"  # Claims about data/statistics
    LITERATURE = "literature"  # Claims about existing research
    INTERPRETATION = "interpretation"  # Claims about meaning/implications


@dataclass
class Claim:
    """Represents a single testable claim extracted from a report."""
    claim_id: str
    text: str
    claim_type: ClaimType
    discovery_title: str
    confidence: Optional[float] = None
    context: Optional[str] = None  # Surrounding text for context
    source_section: Optional[str] = None  # Which section of the report
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary for storage."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "discovery_title": self.discovery_title,
            "confidence": self.confidence,
            "context": self.context,
            "source_section": self.source_section,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create claim from dictionary."""
        data_copy = data.copy()
        data_copy["claim_type"] = ClaimType(data_copy["claim_type"])
        return cls(**data_copy)


class ClaimExtractor:
    """
    Extracts testable claims from generated reports.

    The extractor:
    1. Parses markdown report structure
    2. Identifies factual claims in findings and narratives
    3. Categorizes claims by type
    4. Extracts confidence scores and context
    """

    # Patterns for identifying different types of claims
    DATA_PATTERNS = [
        r"(?:correlation|association|relationship) (?:of|between|with)",
        r"(?:significant|statistically significant|p\s*[<>=])",
        r"(?:mean|median|average|distribution|variance|standard deviation)",
        r"(?:increase|decrease|change|trend|pattern) (?:in|of|by)",
        r"(?:\d+(?:\.\d+)?%|proportion|percentage|rate)",
        r"(?:higher|lower|greater|less) than",
        r"(?:positively|negatively) (?:correlated|associated)",
    ]

    LITERATURE_PATTERNS = [
        r"(?:according to|as reported by|consistent with|contradicts|supports)",
        r"(?:previous (?:research|studies|work)|prior (?:research|studies))",
        r"(?:literature|papers|studies) (?:show|suggest|indicate|demonstrate)",
        r"(?:well-established|known|documented) (?:finding|result|phenomenon)",
    ]

    INTERPRETATION_PATTERNS = [
        r"(?:suggests? that|indicates? that|implies? that|means? that)",
        r"(?:may|might|could|appears? to) (?:be|indicate|suggest|explain)",
        r"(?:important|significant|notable|interesting) (?:implication|finding)",
        r"(?:therefore|thus|hence|consequently)",
        r"(?:explains?|accounts? for|results? in|leads? to)",
    ]

    def __init__(self):
        """Initialize the claim extractor."""
        self.claim_counter = 0

    def extract_claims(self, report_path: Path) -> List[Claim]:
        """
        Extract testable claims from a markdown report.

        Args:
            report_path: Path to the markdown report file

        Returns:
            List of extracted claims
        """
        logger.info(f"Extracting claims from report: {report_path}")

        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")

        # Read report
        with open(report_path, 'r') as f:
            report_text = f.read()

        # Parse report structure
        discoveries = self._parse_report_structure(report_text)

        # Extract claims from each discovery
        all_claims = []
        for discovery in discoveries:
            claims = self._extract_claims_from_discovery(discovery)
            all_claims.extend(claims)

        logger.info(f"Extracted {len(all_claims)} claims from report")
        return all_claims

    def _parse_report_structure(self, report_text: str) -> List[Dict[str, Any]]:
        """
        Parse the markdown report into structured discoveries.

        Returns:
            List of discovery dictionaries with title, narrative, findings, etc.
        """
        discoveries = []

        # Split by discovery sections (## Discovery N:)
        discovery_pattern = r'## Discovery \d+: (.+?)\n'
        discovery_matches = list(re.finditer(discovery_pattern, report_text))

        for i, match in enumerate(discovery_matches):
            # Get discovery title
            title = match.group(1).strip()

            # Get content from this discovery to the next (or end)
            start = match.end()
            end = discovery_matches[i + 1].start() if i + 1 < len(discovery_matches) else len(report_text)
            content = report_text[start:end]

            # Extract confidence and novelty
            confidence = None
            novelty = None
            conf_match = re.search(r'\*\*Confidence:\*\*\s*([\d.]+)', content)
            nov_match = re.search(r'\*\*Novelty Score:\*\*\s*([\d.]+)', content)
            if conf_match:
                confidence = float(conf_match.group(1))
            if nov_match:
                novelty = float(nov_match.group(1))

            # Extract narrative section
            narrative_match = re.search(
                r'\*\*Novelty Score:\*\*.*?\n\n(.*?)(?=### Supporting Evidence|\Z)',
                content,
                re.DOTALL
            )
            narrative = narrative_match.group(1).strip() if narrative_match else ""

            # Extract data findings
            findings = []
            findings_section = re.search(
                r'\*\*Data Analysis:\*\*\n\n(.*?)(?=\n\n\*\*|---|\Z)',
                content,
                re.DOTALL
            )
            if findings_section:
                finding_lines = findings_section.group(1).strip().split('\n')
                for line in finding_lines:
                    # Parse finding: "- Text (confidence: X.XX)"
                    finding_match = re.match(r'^-\s*(.+?)\s*(?:\[.*?\])?\s*\(confidence:\s*([\d.]+)\)', line)
                    if finding_match:
                        findings.append({
                            "text": finding_match.group(1).strip(),
                            "confidence": float(finding_match.group(2))
                        })

            # Extract related hypotheses
            hypotheses = []
            hyp_section = re.search(
                r'\*\*Related Hypotheses:\*\*\n\n(.*?)(?=\n\n\*\*|---|\Z)',
                content,
                re.DOTALL
            )
            if hyp_section:
                hyp_lines = hyp_section.group(1).strip().split('\n')
                for line in hyp_lines:
                    hyp_match = re.match(r'^-\s*(.+?)(?:\s*\(confidence:\s*([\d.]+)\))?$', line)
                    if hyp_match:
                        hyp_conf = float(hyp_match.group(2)) if hyp_match.group(2) else None
                        hypotheses.append({
                            "text": hyp_match.group(1).strip(),
                            "confidence": hyp_conf
                        })

            discoveries.append({
                "title": title,
                "confidence": confidence,
                "novelty": novelty,
                "narrative": narrative,
                "findings": findings,
                "hypotheses": hypotheses,
            })

        logger.info(f"Parsed {len(discoveries)} discoveries from report")
        return discoveries

    def _extract_claims_from_discovery(self, discovery: Dict[str, Any]) -> List[Claim]:
        """Extract claims from a single discovery."""
        claims = []

        # Extract from narrative
        if discovery.get("narrative"):
            narrative_claims = self._extract_claims_from_text(
                discovery["narrative"],
                discovery["title"],
                "narrative",
                discovery.get("confidence")
            )
            claims.extend(narrative_claims)

        # Extract from findings
        for finding in discovery.get("findings", []):
            finding_claims = self._extract_claims_from_text(
                finding["text"],
                discovery["title"],
                "data_finding",
                finding.get("confidence")
            )
            claims.extend(finding_claims)

        # Extract from hypotheses
        for hypothesis in discovery.get("hypotheses", []):
            hyp_claims = self._extract_claims_from_text(
                hypothesis["text"],
                discovery["title"],
                "hypothesis",
                hypothesis.get("confidence")
            )
            claims.extend(hyp_claims)

        return claims

    def _extract_claims_from_text(
        self,
        text: str,
        discovery_title: str,
        source_section: str,
        confidence: Optional[float] = None
    ) -> List[Claim]:
        """
        Extract individual claims from a text passage.

        Splits on sentence boundaries and categorizes each claim.
        """
        claims = []

        # Split into sentences
        sentences = self._split_sentences(text)

        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue

            # Categorize claim type
            claim_type = self._categorize_claim(sentence)

            # Create claim
            self.claim_counter += 1
            claim = Claim(
                claim_id=f"claim_{self.claim_counter:04d}",
                text=sentence,
                claim_type=claim_type,
                discovery_title=discovery_title,
                confidence=confidence,
                source_section=source_section,
                metadata={
                    "original_text": text,
                }
            )

            claims.append(claim)

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (can be improved with NLTK/spaCy)
        # Handle common abbreviations
        text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '.<ABBREV>', text)  # U.S.
        text = re.sub(r'(?<=\d)\.(?=\d)', '.<DECIMAL>', text)  # 3.14
        text = re.sub(r'(?<= [a-z])\.(?= [a-z])', '.<ABBREV>', text)  # e.g., i.e.

        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<ABBREV>', '.').replace('<DECIMAL>', '.') for s in sentences]

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _categorize_claim(self, text: str) -> ClaimType:
        """
        Categorize a claim based on its content.

        Returns the most likely claim type based on pattern matching.
        """
        text_lower = text.lower()

        # Count matches for each category
        data_score = sum(1 for pattern in self.DATA_PATTERNS
                        if re.search(pattern, text_lower))
        lit_score = sum(1 for pattern in self.LITERATURE_PATTERNS
                       if re.search(pattern, text_lower))
        interp_score = sum(1 for pattern in self.INTERPRETATION_PATTERNS
                          if re.search(pattern, text_lower))

        # Return category with highest score (default to interpretation)
        scores = [
            (data_score, ClaimType.DATA_ANALYSIS),
            (lit_score, ClaimType.LITERATURE),
            (interp_score, ClaimType.INTERPRETATION),
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        # If highest score is 0, default to DATA_ANALYSIS for findings
        if scores[0][0] == 0:
            return ClaimType.DATA_ANALYSIS

        return scores[0][1]

    def save_claims(self, claims: List[Claim], output_path: Path) -> None:
        """
        Save extracted claims to JSON file.

        Args:
            claims: List of claims to save
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        claims_data = [claim.to_dict() for claim in claims]

        with open(output_path, 'w') as f:
            json.dump({
                "total_claims": len(claims),
                "claims_by_type": {
                    ct.value: len([c for c in claims if c.claim_type == ct])
                    for ct in ClaimType
                },
                "claims": claims_data,
            }, f, indent=2)

        logger.info(f"Saved {len(claims)} claims to {output_path}")

    def load_claims(self, input_path: Path) -> List[Claim]:
        """
        Load claims from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            List of claims
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        claims = [Claim.from_dict(claim_data) for claim_data in data["claims"]]

        logger.info(f"Loaded {len(claims)} claims from {input_path}")
        return claims
