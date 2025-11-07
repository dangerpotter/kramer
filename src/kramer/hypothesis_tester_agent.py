"""
Hypothesis Tester Agent - Validates hypotheses through data analysis and literature search.

This agent uses Claude API to assess testability, performs statistical tests using
DataAnalysisAgent, searches literature for supporting/refuting evidence, and synthesizes
results into a final verdict.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from src.kramer.data_analysis_agent import AgentConfig, DataAnalysisAgent
from src.world_model.graph import EdgeType, NodeType, WorldModel
from src.utils.cost_tracker import CostTracker


@dataclass
class TestResult:
    """Result from hypothesis testing."""

    hypothesis_id: str
    test_type: str  # "data_driven" | "literature_based" | "comprehensive"
    outcome: str  # "supported" | "refuted" | "inconclusive"
    confidence: float  # 0.0 to 1.0
    evidence: List[Dict[str, Any]]
    statistical_metrics: Dict[str, float]  # p-values, effect sizes, etc.
    cost: float
    reasoning: str
    raw_data: Dict[str, Any] = field(default_factory=dict)


class HypothesisTesterAgent:
    """
    AI-powered hypothesis testing agent.

    Uses Claude API with extended thinking to determine how to test hypotheses,
    performs statistical tests using DataAnalysisAgent, searches literature for
    evidence, and synthesizes results into a validation verdict.

    Features:
    - Assesses hypothesis testability
    - Data-driven testing via statistical analysis
    - Literature-based testing via paper search
    - Evidence synthesis and confidence scoring
    - World model integration
    """

    def __init__(
        self,
        world_model: WorldModel,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        use_extended_thinking: bool = True,
        max_tokens: int = 8000,
    ):
        """
        Initialize the hypothesis tester agent.

        Args:
            world_model: WorldModel graph for querying data and storing results
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            use_extended_thinking: Whether to use extended thinking mode
            max_tokens: Maximum tokens for API response
        """
        self.world_model = world_model
        self.model = model
        self.use_extended_thinking = use_extended_thinking
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided in constructor or environment")

        self.client = anthropic.Anthropic(api_key=api_key)

        # Will be initialized when needed
        self._data_agent = None

    def test_hypothesis(
        self,
        hypothesis_id: str,
        dataset_path: Optional[str] = None,
        test_approaches: List[str] = None,
    ) -> TestResult:
        """
        Test a hypothesis using available evidence.

        Args:
            hypothesis_id: ID of the hypothesis node in world model
            dataset_path: Optional path to dataset for data-driven testing
            test_approaches: List of approaches to use: ["data", "literature", "both"]
                           Defaults to ["both"]

        Returns:
            TestResult with testing outcome and evidence
        """
        test_approaches = test_approaches or ["both"]
        total_cost = 0.0

        # Retrieve hypothesis from world model
        hypothesis_data = self._get_hypothesis_data(hypothesis_id)
        if not hypothesis_data:
            return TestResult(
                hypothesis_id=hypothesis_id,
                test_type="none",
                outcome="inconclusive",
                confidence=0.0,
                evidence=[],
                statistical_metrics={},
                cost=0.0,
                reasoning="Hypothesis not found in world model",
            )

        # Assess testability
        testability_assessment = self._assess_testability(hypothesis_data, dataset_path)
        total_cost += testability_assessment.get("cost", 0.0)

        # Determine which tests to run
        run_data_test = "data" in test_approaches or "both" in test_approaches
        run_literature_test = "literature" in test_approaches or "both" in test_approaches

        # Adjust based on testability assessment
        if not testability_assessment.get("can_test_with_data", False):
            run_data_test = False
        if not testability_assessment.get("can_test_with_literature", True):
            run_literature_test = False

        # Collect evidence
        evidence = []
        statistical_metrics = {}

        # Data-driven testing
        if run_data_test and dataset_path:
            try:
                data_results = self._test_with_data(
                    hypothesis_data, dataset_path, testability_assessment
                )
                evidence.extend(data_results.get("evidence", []))
                statistical_metrics.update(data_results.get("metrics", {}))
                total_cost += data_results.get("cost", 0.0)
            except Exception as e:
                print(f"Error in data-driven testing: {e}")
                evidence.append(
                    {
                        "type": "data_test_error",
                        "source": "data_analysis",
                        "message": str(e),
                        "supports": False,
                    }
                )

        # Literature-based testing
        if run_literature_test:
            try:
                literature_results = self._test_with_literature(hypothesis_data)
                evidence.extend(literature_results.get("evidence", []))
                total_cost += literature_results.get("cost", 0.0)
            except Exception as e:
                print(f"Error in literature-based testing: {e}")
                evidence.append(
                    {
                        "type": "literature_test_error",
                        "source": "literature",
                        "message": str(e),
                        "supports": False,
                    }
                )

        # Synthesize test results
        synthesis = self._synthesize_test_results(
            hypothesis_data, evidence, statistical_metrics
        )
        total_cost += synthesis.get("cost", 0.0)

        # Determine test type
        if run_data_test and run_literature_test:
            test_type = "comprehensive"
        elif run_data_test:
            test_type = "data_driven"
        else:
            test_type = "literature_based"

        return TestResult(
            hypothesis_id=hypothesis_id,
            test_type=test_type,
            outcome=synthesis.get("outcome", "inconclusive"),
            confidence=synthesis.get("confidence", 0.0),
            evidence=evidence,
            statistical_metrics=statistical_metrics,
            cost=total_cost,
            reasoning=synthesis.get("reasoning", ""),
            raw_data={
                "testability_assessment": testability_assessment,
                "synthesis": synthesis,
            },
        )

    def _get_hypothesis_data(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve hypothesis data from world model.

        Args:
            hypothesis_id: ID of the hypothesis node

        Returns:
            Dictionary with hypothesis data or None if not found
        """
        if not self.world_model.graph.has_node(hypothesis_id):
            return None

        node_data = self.world_model.graph.nodes[hypothesis_id]

        # Get connected findings (evidence that led to this hypothesis)
        supporting_findings = []
        for source, target, edge_data in self.world_model.graph.in_edges(
            hypothesis_id, data=True
        ):
            if edge_data.get("edge_type") == EdgeType.DERIVES_FROM.value:
                finding_data = self.world_model.graph.nodes.get(source, {})
                if finding_data.get("node_type") == NodeType.FINDING.value:
                    supporting_findings.append(
                        {
                            "id": source,
                            "text": finding_data.get("text", ""),
                            "confidence": finding_data.get("confidence", 0.0),
                        }
                    )

        return {
            "id": hypothesis_id,
            "text": node_data.get("text", ""),
            "confidence": node_data.get("confidence", 0.0),
            "metadata": node_data.get("metadata", {}),
            "supporting_findings": supporting_findings,
        }

    def _assess_testability(
        self, hypothesis_data: Dict[str, Any], dataset_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Use LLM to assess how to test the hypothesis.

        Args:
            hypothesis_data: Dictionary with hypothesis information
            dataset_path: Optional path to dataset

        Returns:
            Dictionary with testability assessment and suggested approaches
        """
        hypothesis_text = hypothesis_data.get("text", "")
        metadata = hypothesis_data.get("metadata", {})
        supporting_findings = hypothesis_data.get("supporting_findings", [])

        # Build context
        findings_text = "\n".join(
            [f"- {f['text']} (confidence: {f['confidence']:.2f})" for f in supporting_findings]
        )

        prompt = f"""You are a scientific research assistant evaluating the testability of a hypothesis.

## Hypothesis

{hypothesis_text}

## Background

This hypothesis was generated based on the following findings:

{findings_text if findings_text else "No supporting findings available."}

Rationale: {metadata.get('rationale', 'Not provided')}

## Available Resources

- Dataset: {"Available at " + dataset_path if dataset_path else "No dataset available"}
- Literature search: Available

## Task

Assess how this hypothesis can be tested and provide specific testing strategies.

For each testing approach (data-driven and literature-based), determine:

1. **Feasibility**: Can this hypothesis be tested with this approach? (yes/no)
2. **Test Strategy**: If yes, describe the specific approach
3. **Required Variables**: What variables or measurements are needed?
4. **Expected Evidence**: What would constitute supporting or refuting evidence?

## Output Format

Provide your response as a JSON object:

```json
{{
  "can_test_with_data": true/false,
  "data_test_strategy": {{
    "approach": "Description of statistical or computational approach",
    "required_variables": ["variable1", "variable2"],
    "statistical_tests": ["test_name1", "test_name2"],
    "supporting_evidence_criteria": "What results would support the hypothesis",
    "refuting_evidence_criteria": "What results would refute the hypothesis"
  }},
  "can_test_with_literature": true/false,
  "literature_test_strategy": {{
    "search_terms": ["term1", "term2"],
    "supporting_evidence_criteria": "What findings in papers would support",
    "refuting_evidence_criteria": "What findings in papers would refute"
  }},
  "overall_testability": "high" | "medium" | "low",
  "reasoning": "Brief explanation of testability assessment"
}}
```

Respond ONLY with the JSON object, no additional text.
"""

        try:
            # Call Claude API
            thinking_param = {"type": "enabled", "budget_tokens": 2000} if self.use_extended_thinking else {"type": "disabled"}

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=1.0,
                thinking=thinking_param,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            assessment = self._parse_json_response(response)
            assessment["cost"] = CostTracker.track_call(self.model, response)

            return assessment

        except Exception as e:
            print(f"Error assessing testability: {e}")
            return {
                "can_test_with_data": False,
                "can_test_with_literature": True,
                "overall_testability": "low",
                "reasoning": f"Error during assessment: {str(e)}",
                "cost": 0.0,
            }

    def _test_with_data(
        self,
        hypothesis_data: Dict[str, Any],
        dataset_path: str,
        testability_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Test hypothesis using statistical/computational analysis.

        Args:
            hypothesis_data: Dictionary with hypothesis information
            dataset_path: Path to dataset
            testability_assessment: Assessment from _assess_testability

        Returns:
            Dictionary with evidence and metrics
        """
        # Initialize DataAnalysisAgent if needed
        if self._data_agent is None:
            config = AgentConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=self.model,
                use_extended_thinking=self.use_extended_thinking,
            )
            self._data_agent = DataAnalysisAgent(config=config)

        # Create analysis objective from hypothesis
        hypothesis_text = hypothesis_data.get("text", "")
        test_strategy = testability_assessment.get("data_test_strategy", {})
        approach = test_strategy.get("approach", "Perform statistical analysis")

        objective = f"""Test the following hypothesis using the dataset:

Hypothesis: {hypothesis_text}

Testing Approach: {approach}

Required Variables: {', '.join(test_strategy.get('required_variables', []))}

Perform appropriate statistical tests and provide:
1. Descriptive statistics for relevant variables
2. Statistical test results (p-values, effect sizes, confidence intervals)
3. Visualizations if appropriate
4. Clear interpretation of whether results support or refute the hypothesis

Supporting Evidence Criteria: {test_strategy.get('supporting_evidence_criteria', 'Not specified')}
Refuting Evidence Criteria: {test_strategy.get('refuting_evidence_criteria', 'Not specified')}
"""

        # Run analysis
        try:
            final_result = self._data_agent.analyze(
                objective=objective,
                dataset_path=dataset_path,
                world_model_context={"hypothesis_id": hypothesis_data.get("id")},
            )

            # Extract evidence from analysis results
            evidence = []
            metrics = {}

            # Parse findings from analysis
            for finding in final_result.parsed_results.findings:
                # Determine if finding supports or refutes hypothesis
                supports = self._evaluate_finding_support(
                    finding.get("text", ""),
                    hypothesis_text,
                    test_strategy,
                )

                evidence.append(
                    {
                        "type": "statistical_analysis",
                        "source": "data_analysis_agent",
                        "finding": finding.get("text", ""),
                        "confidence": finding.get("confidence", 0.0),
                        "supports": supports,
                        "metadata": finding,
                    }
                )

            # Extract statistical metrics
            for stat in final_result.parsed_results.statistics:
                metric_name = stat.get("name", "unknown")
                metric_value = stat.get("value")
                if metric_value is not None:
                    metrics[metric_name] = metric_value

            return {
                "evidence": evidence,
                "metrics": metrics,
                "cost": final_result.total_cost,
                "notebook_path": str(final_result.notebook_path),
            }

        except Exception as e:
            print(f"Error in data analysis: {e}")
            return {
                "evidence": [
                    {
                        "type": "error",
                        "source": "data_analysis_agent",
                        "message": str(e),
                        "supports": False,
                    }
                ],
                "metrics": {},
                "cost": 0.0,
            }

    def _evaluate_finding_support(
        self, finding_text: str, hypothesis_text: str, test_strategy: Dict[str, Any]
    ) -> bool:
        """
        Evaluate whether a finding supports the hypothesis.

        Uses simple heuristics based on keywords and statistical significance.

        Args:
            finding_text: Text of the finding
            hypothesis_text: Text of the hypothesis
            test_strategy: Test strategy from testability assessment

        Returns:
            True if finding supports hypothesis, False otherwise
        """
        finding_lower = finding_text.lower()

        # Check for statistical significance indicators
        if "p-value" in finding_lower or "p <" in finding_lower or "significant" in finding_lower:
            # If significant result is mentioned, it likely supports the hypothesis
            if "significant" in finding_lower and "not" not in finding_lower:
                return True
            if "p <" in finding_lower or "p=" in finding_lower:
                # Try to extract p-value
                import re

                p_match = re.search(r"p\s*[<=]\s*(0\.\d+)", finding_lower)
                if p_match:
                    p_value = float(p_match.group(1))
                    if p_value < 0.05:
                        return True

        # Check for positive correlation/association keywords
        positive_keywords = [
            "positive correlation",
            "associated with",
            "increases",
            "higher",
            "effect",
            "relationship",
        ]
        if any(keyword in finding_lower for keyword in positive_keywords):
            return True

        # Check against supporting criteria if provided
        supporting_criteria = test_strategy.get("supporting_evidence_criteria", "")
        if supporting_criteria:
            # Simple keyword matching
            criteria_keywords = supporting_criteria.lower().split()
            if any(keyword in finding_lower for keyword in criteria_keywords if len(keyword) > 4):
                return True

        # Default to neutral/unsupporting if unclear
        return False

    def _test_with_literature(self, hypothesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test hypothesis by searching for supporting/refuting evidence in papers.

        Args:
            hypothesis_data: Dictionary with hypothesis information

        Returns:
            Dictionary with evidence from literature
        """
        hypothesis_text = hypothesis_data.get("text", "")

        # Query world model for related papers
        papers = []
        for node_id, node_data in self.world_model.graph.nodes(data=True):
            if node_data.get("node_type") == NodeType.PAPER.value:
                metadata = node_data.get("metadata", {})
                papers.append(
                    {
                        "id": node_id,
                        "title": metadata.get("title", ""),
                        "text": node_data.get("text", ""),
                        "year": metadata.get("year"),
                        "authors": metadata.get("authors", []),
                    }
                )

        if not papers:
            return {
                "evidence": [
                    {
                        "type": "literature_review",
                        "source": "world_model",
                        "message": "No papers available in world model for literature-based testing",
                        "supports": None,
                    }
                ],
                "cost": 0.0,
            }

        # Use LLM to evaluate papers against hypothesis
        papers_text = "\n\n".join(
            [
                f"**Paper {i+1}**: {p['title']}\n"
                f"Authors: {', '.join(p.get('authors', [])) if isinstance(p.get('authors'), list) else p.get('authors', 'Unknown')}\n"
                f"Year: {p.get('year', 'Unknown')}\n"
                f"Key Finding: {p['text']}"
                for i, p in enumerate(papers[:10])  # Limit to 10 papers to avoid token bloat
            ]
        )

        prompt = f"""You are evaluating whether existing literature supports or refutes a hypothesis.

## Hypothesis

{hypothesis_text}

## Available Papers

{papers_text}

## Task

For each paper that is relevant to the hypothesis, determine:

1. **Relevance**: Is this paper relevant to the hypothesis? (1-5 scale)
2. **Verdict**: Does it support, refute, or is neutral regarding the hypothesis?
3. **Confidence**: How confident is this verdict? (0.0 to 1.0)
4. **Reasoning**: Brief explanation (1-2 sentences)

## Output Format

Provide your response as a JSON array:

```json
[
  {{
    "paper_number": 1,
    "relevance": 4,
    "verdict": "supports" | "refutes" | "neutral",
    "confidence": 0.75,
    "reasoning": "Brief explanation"
  }}
]
```

Only include papers with relevance >= 3. If no papers are relevant, return an empty array [].

Respond ONLY with the JSON array, no additional text.
"""

        try:
            # Call Claude API
            thinking_param = {"type": "enabled", "budget_tokens": 2000} if self.use_extended_thinking else {"type": "disabled"}

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=1.0,
                thinking=thinking_param,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            paper_evaluations = self._parse_json_response(response)
            cost = CostTracker.track_call(self.model, response)

            # Convert to evidence format
            evidence = []
            for eval_data in paper_evaluations:
                if isinstance(eval_data, dict):
                    paper_idx = eval_data.get("paper_number", 1) - 1
                    if 0 <= paper_idx < len(papers):
                        paper = papers[paper_idx]
                        verdict = eval_data.get("verdict", "neutral")

                        evidence.append(
                            {
                                "type": "literature_review",
                                "source": "paper",
                                "paper_id": paper["id"],
                                "paper_title": paper["title"],
                                "verdict": verdict,
                                "supports": verdict == "supports",
                                "confidence": eval_data.get("confidence", 0.5),
                                "reasoning": eval_data.get("reasoning", ""),
                                "relevance": eval_data.get("relevance", 3),
                            }
                        )

            return {
                "evidence": evidence,
                "cost": cost,
            }

        except Exception as e:
            print(f"Error in literature testing: {e}")
            return {
                "evidence": [
                    {
                        "type": "error",
                        "source": "literature_review",
                        "message": str(e),
                        "supports": False,
                    }
                ],
                "cost": 0.0,
            }

    def _synthesize_test_results(
        self,
        hypothesis_data: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        statistical_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Synthesize all evidence into final verdict.

        Args:
            hypothesis_data: Dictionary with hypothesis information
            evidence: List of evidence items
            statistical_metrics: Dictionary of statistical metrics

        Returns:
            Dictionary with outcome, confidence, and reasoning
        """
        # Count supporting vs refuting evidence
        supporting = [e for e in evidence if e.get("supports") is True]
        refuting = [e for e in evidence if e.get("supports") is False]
        neutral = [e for e in evidence if e.get("supports") is None]

        # Calculate weighted confidence
        confidence = self._calculate_test_confidence(supporting, refuting, statistical_metrics)

        # Determine outcome
        if len(supporting) > len(refuting) and confidence > 0.6:
            outcome = "supported"
        elif len(refuting) > len(supporting) and confidence > 0.6:
            outcome = "refuted"
        else:
            outcome = "inconclusive"

        # Generate reasoning
        reasoning = self._generate_reasoning(
            hypothesis_data, outcome, supporting, refuting, neutral, statistical_metrics
        )

        return {
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": reasoning,
            "supporting_count": len(supporting),
            "refuting_count": len(refuting),
            "neutral_count": len(neutral),
            "cost": 0.0,  # No API call needed for synthesis
        }

    def _calculate_test_confidence(
        self,
        supporting: List[Dict[str, Any]],
        refuting: List[Dict[str, Any]],
        statistical_metrics: Dict[str, float],
    ) -> float:
        """
        Calculate confidence in test results.

        Args:
            supporting: List of supporting evidence
            refuting: List of refuting evidence
            statistical_metrics: Dictionary of statistical metrics

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not supporting and not refuting:
            return 0.0

        # Base confidence on evidence count and quality
        total_evidence = len(supporting) + len(refuting)
        stronger_side = max(len(supporting), len(refuting))

        # Ratio of stronger side to total
        evidence_ratio = stronger_side / total_evidence if total_evidence > 0 else 0.0

        # Weight by individual confidence scores
        if len(supporting) > len(refuting):
            avg_confidence = (
                sum(e.get("confidence", 0.5) for e in supporting) / len(supporting)
                if supporting
                else 0.5
            )
        else:
            avg_confidence = (
                sum(e.get("confidence", 0.5) for e in refuting) / len(refuting)
                if refuting
                else 0.5
            )

        # Boost confidence if we have statistical significance
        stat_boost = 0.0
        if "p_value" in statistical_metrics:
            p_value = statistical_metrics["p_value"]
            if p_value < 0.01:
                stat_boost = 0.2
            elif p_value < 0.05:
                stat_boost = 0.1

        # Combine factors
        confidence = min(1.0, (evidence_ratio * 0.5 + avg_confidence * 0.5) + stat_boost)

        return round(confidence, 3)

    def _generate_reasoning(
        self,
        hypothesis_data: Dict[str, Any],
        outcome: str,
        supporting: List[Dict[str, Any]],
        refuting: List[Dict[str, Any]],
        neutral: List[Dict[str, Any]],
        statistical_metrics: Dict[str, float],
    ) -> str:
        """
        Generate human-readable reasoning for the test outcome.

        Args:
            hypothesis_data: Dictionary with hypothesis information
            outcome: Test outcome (supported/refuted/inconclusive)
            supporting: List of supporting evidence
            refuting: List of refuting evidence
            neutral: List of neutral evidence
            statistical_metrics: Dictionary of statistical metrics

        Returns:
            Reasoning string
        """
        reasoning_parts = []

        # Overall verdict
        if outcome == "supported":
            reasoning_parts.append(
                f"The hypothesis is SUPPORTED by the available evidence ({len(supporting)} supporting, {len(refuting)} refuting)."
            )
        elif outcome == "refuted":
            reasoning_parts.append(
                f"The hypothesis is REFUTED by the available evidence ({len(refuting)} refuting, {len(supporting)} supporting)."
            )
        else:
            reasoning_parts.append(
                f"The hypothesis testing is INCONCLUSIVE ({len(supporting)} supporting, {len(refuting)} refuting, {len(neutral)} neutral)."
            )

        # Add statistical evidence if available
        if statistical_metrics:
            metrics_str = ", ".join(
                [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in statistical_metrics.items()]
            )
            reasoning_parts.append(f"Statistical metrics: {metrics_str}.")

        # Summarize key supporting evidence
        if supporting:
            top_support = supporting[0]
            reasoning_parts.append(
                f"Key supporting evidence: {top_support.get('finding', top_support.get('reasoning', 'Statistical analysis'))}"
            )

        # Summarize key refuting evidence
        if refuting:
            top_refute = refuting[0]
            reasoning_parts.append(
                f"Key refuting evidence: {top_refute.get('finding', top_refute.get('reasoning', 'Analysis results'))}"
            )

        return " ".join(reasoning_parts)

    def _parse_json_response(self, response: anthropic.types.Message) -> Any:
        """
        Parse Claude API response to extract JSON.

        Args:
            response: Anthropic API response

        Returns:
            Parsed JSON object (dict or list)
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

            return json.loads(json_text)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {text[:500]}")
            # Return empty dict/list as fallback
            return {} if "{" in text else []
