"""
Hypothesis Generation Agent - Analyzes findings and generates testable hypotheses.

This agent uses Claude API to analyze research findings and generate novel,
testable hypotheses that can drive further research.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic

from src.world_model.graph import EdgeType, NodeType, WorldModel


@dataclass
class HypothesisResult:
    """Result from hypothesis generation."""

    hypotheses_generated: int
    hypothesis_ids: List[str]
    cost: float
    raw_hypotheses: List[Dict[str, Any]]


class HypothesisAgent:
    """
    AI-powered hypothesis generation agent.

    Uses Claude API to analyze findings from the world model and generate
    novel, testable hypotheses automatically.

    Features:
    - Analyzes recent findings and literature
    - Generates testable hypotheses
    - Validates novelty and testability
    - Integrates with world model graph
    - Tracks provenance and metadata
    """

    def __init__(
        self,
        world_model: WorldModel,
        api_key: Optional[str] = None,
        max_hypotheses: int = 5,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
    ):
        """
        Initialize the hypothesis generation agent.

        Args:
            world_model: WorldModel graph for querying findings and storing hypotheses
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_hypotheses: Maximum number of hypotheses to generate per call
            model: Claude model to use
            max_tokens: Maximum tokens for API response
        """
        self.world_model = world_model
        self.max_hypotheses = max_hypotheses
        self.model = model
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided in constructor or environment")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_hypotheses(
        self,
        current_cycle: Optional[int] = None,
        min_finding_confidence: float = 0.6,
        max_findings: int = 20,
    ) -> HypothesisResult:
        """
        Generate novel hypotheses based on recent findings.

        Args:
            current_cycle: Current research cycle (for provenance tracking)
            min_finding_confidence: Minimum confidence threshold for findings
            max_findings: Maximum number of findings to analyze

        Returns:
            HypothesisResult with generated hypotheses and metadata
        """
        # Query world model for recent findings
        findings = self._get_recent_findings(
            max_count=max_findings,
            min_confidence=min_finding_confidence,
        )

        # Get existing hypotheses to avoid duplication
        existing_hypotheses = self._get_existing_hypotheses()

        # Get related papers and claims
        papers = self._get_related_papers(max_count=10)

        # Build context string for Claude
        context = self._build_context(
            findings=findings,
            existing_hypotheses=existing_hypotheses,
            papers=papers,
        )

        # Create prompt for hypothesis generation
        prompt = self._create_hypothesis_prompt(
            context=context,
            max_hypotheses=self.max_hypotheses,
        )

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract hypotheses from response
            raw_hypotheses = self._parse_response(response)

            # Validate and filter hypotheses
            valid_hypotheses = []
            for hyp in raw_hypotheses:
                if self._validate_hypothesis(hyp, existing_hypotheses):
                    valid_hypotheses.append(hyp)

            # Add validated hypotheses to world model
            hypothesis_ids = self._add_to_world_model(
                hypotheses=valid_hypotheses,
                findings=findings,
                current_cycle=current_cycle,
            )

            # Calculate cost (approximate based on tokens)
            # TODO: Get actual token usage from API response
            estimated_cost = self._estimate_cost(response)

            return HypothesisResult(
                hypotheses_generated=len(hypothesis_ids),
                hypothesis_ids=hypothesis_ids,
                cost=estimated_cost,
                raw_hypotheses=valid_hypotheses,
            )

        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            return HypothesisResult(
                hypotheses_generated=0,
                hypothesis_ids=[],
                cost=0.0,
                raw_hypotheses=[],
            )

    def _get_recent_findings(
        self,
        max_count: int,
        min_confidence: float,
    ) -> List[Dict[str, Any]]:
        """
        Query world model for recent findings.

        Args:
            max_count: Maximum number of findings to retrieve
            min_confidence: Minimum confidence threshold

        Returns:
            List of finding data dictionaries
        """
        findings = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            # Filter for findings with sufficient confidence
            if data.get("node_type") == NodeType.FINDING.value:
                confidence = data.get("confidence", 0.0)
                if confidence is not None and confidence >= min_confidence:
                    findings.append(
                        {
                            "id": node_id,
                            "text": data.get("text", ""),
                            "confidence": confidence,
                            "provenance": data.get("provenance", ""),
                            "metadata": data.get("metadata", {}),
                        }
                    )

        # Sort by confidence (descending) and limit
        findings.sort(key=lambda f: f.get("confidence", 0.0), reverse=True)
        return findings[:max_count]

    def _get_existing_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Get existing hypotheses from world model.

        Returns:
            List of hypothesis data dictionaries
        """
        hypotheses = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == NodeType.HYPOTHESIS.value:
                hypotheses.append(
                    {
                        "id": node_id,
                        "text": data.get("text", ""),
                        "confidence": data.get("confidence", 0.0),
                    }
                )

        return hypotheses

    def _get_related_papers(self, max_count: int) -> List[Dict[str, Any]]:
        """
        Get related papers from world model.

        Args:
            max_count: Maximum number of papers to retrieve

        Returns:
            List of paper data dictionaries
        """
        papers = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == NodeType.PAPER.value:
                metadata = data.get("metadata", {})
                papers.append(
                    {
                        "id": node_id,
                        "title": metadata.get("title", ""),
                        "authors": metadata.get("authors", []),
                        "year": metadata.get("year"),
                        "text": data.get("text", ""),  # Summary/claim from paper
                    }
                )

        return papers[:max_count]

    def _build_context(
        self,
        findings: List[Dict[str, Any]],
        existing_hypotheses: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build structured context for hypothesis generation.

        Args:
            findings: List of findings
            existing_hypotheses: List of existing hypotheses
            papers: List of papers

        Returns:
            Context dictionary
        """
        return {
            "findings": findings,
            "existing_hypotheses": existing_hypotheses,
            "papers": papers,
            "total_findings": len(findings),
            "total_hypotheses": len(existing_hypotheses),
            "total_papers": len(papers),
        }

    def _create_hypothesis_prompt(
        self,
        context: Dict[str, Any],
        max_hypotheses: int,
    ) -> str:
        """
        Create the Claude prompt for hypothesis generation.

        Args:
            context: Context dictionary with findings, hypotheses, and papers
            max_hypotheses: Maximum number of hypotheses to generate

        Returns:
            Prompt string
        """
        findings = context["findings"]
        existing_hypotheses = context["existing_hypotheses"]
        papers = context["papers"]

        # Format findings
        findings_text = "\n".join(
            [
                f"- {f['text']} (confidence: {f['confidence']:.2f})"
                for f in findings[:15]  # Limit to avoid token bloat
            ]
        )

        # Format existing hypotheses
        hypotheses_text = "\n".join(
            [f"- {h['text']}" for h in existing_hypotheses[:10]]  # Limit to avoid duplication
        )

        # Format papers
        papers_text = "\n".join(
            [f"- {p['title']}: {p['text']}" for p in papers[:8]]  # Limit to most relevant
        )

        prompt = f"""You are a scientific research assistant specialized in hypothesis generation.

Your task is to analyze research findings and generate novel, testable hypotheses.

## Research Findings

Based on recent data analysis, we have identified the following findings:

{findings_text if findings_text else "No recent findings available."}

## Literature Context

Relevant papers and claims from the literature:

{papers_text if papers_text else "No literature context available."}

## Existing Hypotheses

To avoid duplication, here are hypotheses we've already generated:

{hypotheses_text if hypotheses_text else "No existing hypotheses."}

## Task

Generate up to {max_hypotheses} novel, testable hypotheses based on the findings and literature.

For each hypothesis, provide:
1. **Statement**: A clear, concise hypothesis statement
2. **Rationale**: Why this hypothesis is plausible given the findings (2-3 sentences)
3. **Testability**: How this hypothesis could be tested (specific approach)
4. **Novelty Score**: A score from 0.0 to 1.0 indicating how novel this hypothesis is
   - 1.0 = Completely novel, not similar to existing hypotheses
   - 0.5 = Moderately novel, extends existing ideas
   - 0.0 = Not novel, duplicates existing hypothesis

## Requirements

- Hypotheses must be **testable** with concrete variables/relationships
- Hypotheses must be **novel** and not duplicate existing ones
- Hypotheses should be **grounded** in the provided findings
- Hypotheses should be **specific** and falsifiable

## Output Format

Provide your response as a JSON array of hypothesis objects:

```json
[
  {{
    "statement": "Clear hypothesis statement",
    "rationale": "Why this hypothesis is plausible based on findings...",
    "testability": "How to test this hypothesis...",
    "novelty_score": 0.85
  }}
]
```

Respond ONLY with the JSON array, no additional text before or after.
"""

        return prompt

    def _parse_response(self, response: anthropic.types.Message) -> List[Dict[str, Any]]:
        """
        Parse Claude API response to extract hypotheses.

        Args:
            response: Anthropic API response

        Returns:
            List of hypothesis dictionaries
        """
        # Extract text from response
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Try to extract JSON
        try:
            # Look for JSON code block
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_text = text[start:end].strip()
            else:
                # Try to parse entire response as JSON
                json_text = text.strip()

            hypotheses = json.loads(json_text)

            # Ensure it's a list
            if isinstance(hypotheses, dict):
                hypotheses = [hypotheses]

            return hypotheses

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {text[:500]}")
            return []

    def _validate_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        existing_hypotheses: List[Dict[str, Any]],
    ) -> bool:
        """
        Validate a hypothesis for quality and novelty.

        Args:
            hypothesis: Hypothesis dictionary to validate
            existing_hypotheses: List of existing hypotheses

        Returns:
            True if hypothesis is valid, False otherwise
        """
        # Check required fields
        required_fields = ["statement", "rationale", "testability", "novelty_score"]
        if not all(field in hypothesis for field in required_fields):
            return False

        # Check statement is not empty
        statement = hypothesis.get("statement", "").strip()
        if not statement or len(statement) < 10:
            return False

        # Check novelty score is valid
        novelty_score = hypothesis.get("novelty_score", 0.0)
        if not isinstance(novelty_score, (int, float)) or not 0.0 <= novelty_score <= 1.0:
            return False

        # Check for duplicates (simple text similarity)
        statement_lower = statement.lower()
        for existing in existing_hypotheses:
            existing_statement = existing.get("text", "").lower()
            # Simple duplicate check - if statements are very similar
            if statement_lower == existing_statement:
                return False
            # Check if one statement contains the other (substring match)
            if len(statement_lower) > 20 and len(existing_statement) > 20:
                if statement_lower in existing_statement or existing_statement in statement_lower:
                    return False

        # Check testability is not empty
        testability = hypothesis.get("testability", "").strip()
        if not testability or len(testability) < 10:
            return False

        return True

    def _add_to_world_model(
        self,
        hypotheses: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        current_cycle: Optional[int] = None,
    ) -> List[str]:
        """
        Add validated hypotheses to the world model.

        Args:
            hypotheses: List of validated hypothesis dictionaries
            findings: List of findings that informed the hypotheses
            current_cycle: Current research cycle number

        Returns:
            List of hypothesis node IDs
        """
        hypothesis_ids = []

        for hyp in hypotheses:
            # Create metadata
            metadata = {
                "rationale": hyp.get("rationale", ""),
                "testability": hyp.get("testability", ""),
                "novelty_score": hyp.get("novelty_score", 0.0),
            }
            if current_cycle is not None:
                metadata["cycle"] = current_cycle

            # Add hypothesis node
            hypothesis_id = self.world_model.add_hypothesis(
                text=hyp.get("statement", ""),
                confidence=hyp.get("novelty_score", 0.5),  # Use novelty as confidence
                metadata=metadata,
            )

            # Link to supporting findings
            # Link to top 3 most confident findings as supporting evidence
            top_findings = sorted(findings, key=lambda f: f.get("confidence", 0.0), reverse=True)[
                :3
            ]
            for finding in top_findings:
                finding_id = finding.get("id")
                if finding_id:
                    try:
                        self.world_model.add_edge(
                            source=hypothesis_id,
                            target=finding_id,
                            edge_type=EdgeType.DERIVES_FROM,
                            metadata={"generated_from_finding": True},
                        )
                    except ValueError:
                        # Edge might already exist or node not found
                        pass

            hypothesis_ids.append(hypothesis_id)

        return hypothesis_ids

    def _estimate_cost(self, response: anthropic.types.Message) -> float:
        """
        Estimate API cost based on token usage.

        Args:
            response: Anthropic API response

        Returns:
            Estimated cost in dollars
        """
        # Extract token usage from response
        usage = response.usage

        # Claude Sonnet 4 pricing (as of early 2025)
        # Input: $3 per million tokens
        # Output: $15 per million tokens
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0

        return input_cost + output_cost
