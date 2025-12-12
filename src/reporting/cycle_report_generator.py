"""
Cycle Report Generator - Generates lightweight reports at the end of each discovery cycle.

This module creates cycle reports that:
1. Summarize what was accomplished during the cycle
2. Provide a compact summary for LLM context in subsequent cycles
3. Save full markdown reports for user visibility
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from src.world_model.graph import WorldModel
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


@dataclass
class CycleReportContent:
    """Content for a cycle report."""
    summary: str          # Compact (~500 tokens) for LLM context
    full_content: str     # Full markdown report
    tasks_completed: int
    findings_count: int
    hypotheses_count: int
    papers_count: int
    generation_cost: float


class CycleReportGenerator:
    """
    Generates lightweight reports at the end of each discovery cycle.

    Reports include:
    - Cycle overview (tasks, time, budget)
    - Key findings discovered during the cycle
    - Hypotheses generated and tested
    - Papers found
    - Progress assessment (AI-generated)
    """

    def __init__(
        self,
        world_model: WorldModel,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the cycle report generator.

        Args:
            world_model: The world model containing discoveries
            anthropic_api_key: API key for Claude (optional, uses env var if not provided)
            model: Claude model to use for generating summaries
        """
        self.world_model = world_model
        self.model = model

        # Initialize Claude client
        self.client = None
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key and HAS_ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=api_key)

    def _get_cycle_findings(
        self,
        cycle_started_at: datetime,
        cycle_completed_at: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get findings created during this cycle."""
        findings = []
        end_time = cycle_completed_at or datetime.utcnow()

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") != "finding":
                continue

            node_created = data.get("created_at")
            if not node_created:
                continue

            # Parse datetime if string
            if isinstance(node_created, str):
                try:
                    node_created = datetime.fromisoformat(node_created)
                except (ValueError, AttributeError):
                    continue

            if cycle_started_at <= node_created <= end_time:
                findings.append({
                    "id": node_id,
                    "text": data.get("text", "")[:200],  # Truncate for summary
                    "confidence": data.get("confidence", 0.0),
                    "novelty": data.get("novelty", 0.0),
                })

        # Sort by confidence
        findings.sort(key=lambda f: f.get("confidence", 0.0), reverse=True)
        return findings

    def _get_cycle_hypotheses(
        self,
        cycle_started_at: datetime,
        cycle_completed_at: Optional[datetime] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get hypotheses created and tested during this cycle."""
        generated = []
        tested = []
        end_time = cycle_completed_at or datetime.utcnow()

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") != "hypothesis":
                continue

            metadata = data.get("metadata", {})

            # Check if hypothesis was CREATED during this cycle
            node_created = data.get("created_at")
            if node_created:
                if isinstance(node_created, str):
                    try:
                        node_created = datetime.fromisoformat(node_created)
                    except (ValueError, AttributeError):
                        node_created = None

                if node_created and cycle_started_at <= node_created <= end_time:
                    generated.append({
                        "id": node_id,
                        "text": data.get("text", "")[:150],
                        "confidence": data.get("confidence", 0.0),
                        "tested": metadata.get("tested", False),
                    })

            # Check if hypothesis was TESTED during this cycle (using tested_at)
            tested_at = metadata.get("tested_at")
            if tested_at:
                if isinstance(tested_at, str):
                    try:
                        tested_at = datetime.fromisoformat(tested_at)
                    except (ValueError, AttributeError):
                        tested_at = None

                if tested_at and cycle_started_at <= tested_at <= end_time:
                    tested.append({
                        "id": node_id,
                        "text": data.get("text", "")[:150],
                        "confidence": data.get("confidence", 0.0),
                        "outcome": metadata.get("test_outcome", "unknown"),
                        "test_confidence": metadata.get("test_confidence", 0.0),
                    })

        return {"generated": generated, "tested": tested}

    def _get_cycle_papers(
        self,
        cycle_started_at: datetime,
        cycle_completed_at: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get papers discovered during this cycle."""
        papers = []
        end_time = cycle_completed_at or datetime.utcnow()

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") != "paper":
                continue

            node_created = data.get("created_at")
            if not node_created:
                continue

            if isinstance(node_created, str):
                try:
                    node_created = datetime.fromisoformat(node_created)
                except (ValueError, AttributeError):
                    continue

            if cycle_started_at <= node_created <= end_time:
                metadata = data.get("metadata", {})
                papers.append({
                    "id": node_id,
                    "title": metadata.get("title", "Unknown")[:100],
                    "authors": metadata.get("authors", [])[:3],  # First 3 authors
                    "year": metadata.get("year"),
                    "relevance": metadata.get("relevance_score", 0.0),
                })

        # Sort by relevance
        papers.sort(key=lambda p: p.get("relevance", 0.0), reverse=True)
        return papers

    def _generate_progress_assessment(
        self,
        cycle_number: int,
        objective: str,
        findings: List[Dict[str, Any]],
        hypotheses: Dict[str, List[Dict[str, Any]]],
        papers: List[Dict[str, Any]],
        tasks_completed: int,
        budget_used: float,
    ) -> tuple[str, float]:
        """
        Generate an AI-powered progress assessment for the cycle.

        Returns:
            Tuple of (assessment_text, generation_cost)
        """
        if not self.client:
            # Fallback to template-based assessment
            assessment = self._generate_template_assessment(
                cycle_number, findings, hypotheses, papers, tasks_completed
            )
            return assessment, 0.0

        # Build context for Claude
        findings_summary = "\n".join([
            f"- {f['text'][:100]}... (confidence: {f['confidence']:.2f})"
            for f in findings[:5]
        ]) or "No new findings"

        hyp_generated = len(hypotheses.get("generated", []))
        hyp_tested = len(hypotheses.get("tested", []))

        prompt = f"""You are summarizing research progress for a discovery cycle. Be concise and factual.

Research Objective: {objective}
Cycle Number: {cycle_number}

This Cycle's Accomplishments:
- Tasks completed: {tasks_completed}
- Budget used: ${budget_used:.2f}
- New findings: {len(findings)}
- Hypotheses generated: {hyp_generated}
- Hypotheses tested: {hyp_tested}
- Papers found: {len(papers)}

Top Findings:
{findings_summary}

Write a 2-3 sentence progress assessment. Focus on:
1. What was accomplished
2. Key discoveries or insights
3. Direction for next cycle (if applicable)

Keep it under 100 words. Be factual, not speculative."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            assessment = response.content[0].text.strip()
            cost = CostTracker.track_call(self.model, response)

            return assessment, cost

        except Exception as e:
            logger.warning(f"Failed to generate AI assessment: {e}")
            assessment = self._generate_template_assessment(
                cycle_number, findings, hypotheses, papers, tasks_completed
            )
            return assessment, 0.0

    def _generate_template_assessment(
        self,
        cycle_number: int,
        findings: List[Dict[str, Any]],
        hypotheses: Dict[str, List[Dict[str, Any]]],
        papers: List[Dict[str, Any]],
        tasks_completed: int,
    ) -> str:
        """Generate a template-based assessment when AI is unavailable."""
        hyp_generated = len(hypotheses.get("generated", []))
        hyp_tested = len(hypotheses.get("tested", []))

        parts = [f"Cycle {cycle_number} completed {tasks_completed} tasks."]

        if findings:
            avg_conf = sum(f.get("confidence", 0) for f in findings) / len(findings)
            parts.append(f"Generated {len(findings)} findings (avg confidence: {avg_conf:.2f}).")

        if hyp_generated:
            parts.append(f"Created {hyp_generated} hypotheses, tested {hyp_tested}.")

        if papers:
            parts.append(f"Found {len(papers)} relevant papers.")

        return " ".join(parts)

    def _generate_compact_summary(
        self,
        cycle_number: int,
        objective: str,
        findings: List[Dict[str, Any]],
        hypotheses: Dict[str, List[Dict[str, Any]]],
        papers: List[Dict[str, Any]],
        tasks_completed: int,
        budget_used: float,
        assessment: str,
    ) -> str:
        """Generate a compact summary for LLM context (~500 tokens max)."""
        hyp_generated = len(hypotheses.get("generated", []))
        hyp_tested = len(hypotheses.get("tested", []))

        lines = [
            f"## Cycle {cycle_number} Summary",
            f"Tasks: {tasks_completed} | Budget: ${budget_used:.2f}",
            f"Findings: {len(findings)} | Hypotheses: {hyp_generated} (tested: {hyp_tested}) | Papers: {len(papers)}",
            "",
            assessment,
        ]

        # Add top 3 findings
        if findings:
            lines.append("")
            lines.append("Key findings:")
            for f in findings[:3]:
                conf = f.get("confidence", 0)
                lines.append(f"- [{conf:.1f}] {f['text'][:80]}...")

        # Add untested hypotheses for next cycle planning
        untested = [h for h in hypotheses.get("generated", []) if not h.get("tested")]
        if untested:
            lines.append("")
            lines.append(f"Untested hypotheses: {len(untested)}")

        return "\n".join(lines)

    def _generate_full_report(
        self,
        cycle_number: int,
        cycle_id: str,
        objective: str,
        cycle_started_at: datetime,
        cycle_completed_at: Optional[datetime],
        findings: List[Dict[str, Any]],
        hypotheses: Dict[str, List[Dict[str, Any]]],
        papers: List[Dict[str, Any]],
        tasks_completed: int,
        budget_used: float,
        assessment: str,
    ) -> str:
        """Generate the full markdown report for the cycle."""
        completed_at = cycle_completed_at or datetime.utcnow()
        duration = (completed_at - cycle_started_at).total_seconds()

        lines = [
            f"# Cycle {cycle_number} Report",
            "",
            f"**Cycle ID:** {cycle_id}",
            f"**Started:** {cycle_started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Completed:** {completed_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {duration/60:.1f} minutes",
            f"**Budget Used:** ${budget_used:.2f}",
            "",
            "## Objective",
            "",
            objective,
            "",
            "## Progress Assessment",
            "",
            assessment,
            "",
            "## Metrics",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Tasks Completed | {tasks_completed} |",
            f"| Findings Generated | {len(findings)} |",
            f"| Hypotheses Generated | {len(hypotheses.get('generated', []))} |",
            f"| Hypotheses Tested | {len(hypotheses.get('tested', []))} |",
            f"| Papers Found | {len(papers)} |",
            "",
        ]

        # Findings section
        if findings:
            lines.extend([
                "## Key Findings",
                "",
            ])
            for i, f in enumerate(findings[:10], 1):
                conf = f.get("confidence", 0)
                novelty = f.get("novelty", 0)
                lines.append(f"### {i}. Finding (Confidence: {conf:.2f}, Novelty: {novelty:.2f})")
                lines.append("")
                lines.append(f"{f['text']}")
                lines.append("")

        # Hypotheses section
        generated = hypotheses.get("generated", [])
        if generated:
            lines.extend([
                "## Hypotheses",
                "",
            ])
            for h in generated:
                status = "Tested" if h.get("tested") else "Pending"
                lines.append(f"- **[{status}]** {h['text']}")
            lines.append("")

        # Papers section
        if papers:
            lines.extend([
                "## Papers Found",
                "",
            ])
            for p in papers[:10]:
                authors = p.get("authors", [])
                author_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    author_str += " et al."
                year = p.get("year", "n.d.")
                rel = p.get("relevance", 0)
                lines.append(f"- **{p['title']}** ({author_str}, {year}) [relevance: {rel:.2f}]")
            lines.append("")

        lines.extend([
            "---",
            f"*Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
        ])

        return "\n".join(lines)

    async def generate_cycle_report(
        self,
        cycle_id: str,
        cycle_number: int,
        objective: str,
        cycle_started_at: datetime,
        cycle_completed_at: Optional[datetime],
        tasks_completed: int,
        budget_used: float,
    ) -> CycleReportContent:
        """
        Generate a complete cycle report.

        Args:
            cycle_id: Unique identifier for the cycle
            cycle_number: The cycle number (1-indexed)
            objective: The research objective
            cycle_started_at: When the cycle started
            cycle_completed_at: When the cycle completed (or None if just finishing)
            tasks_completed: Number of tasks completed in the cycle
            budget_used: Budget spent during the cycle

        Returns:
            CycleReportContent with summary and full report
        """
        # Gather cycle data
        findings = self._get_cycle_findings(cycle_started_at, cycle_completed_at)
        hypotheses = self._get_cycle_hypotheses(cycle_started_at, cycle_completed_at)
        papers = self._get_cycle_papers(cycle_started_at, cycle_completed_at)

        # Generate AI assessment
        assessment, generation_cost = self._generate_progress_assessment(
            cycle_number=cycle_number,
            objective=objective,
            findings=findings,
            hypotheses=hypotheses,
            papers=papers,
            tasks_completed=tasks_completed,
            budget_used=budget_used,
        )

        # Generate compact summary for LLM context
        summary = self._generate_compact_summary(
            cycle_number=cycle_number,
            objective=objective,
            findings=findings,
            hypotheses=hypotheses,
            papers=papers,
            tasks_completed=tasks_completed,
            budget_used=budget_used,
            assessment=assessment,
        )

        # Generate full markdown report
        full_content = self._generate_full_report(
            cycle_number=cycle_number,
            cycle_id=cycle_id,
            objective=objective,
            cycle_started_at=cycle_started_at,
            cycle_completed_at=cycle_completed_at,
            findings=findings,
            hypotheses=hypotheses,
            papers=papers,
            tasks_completed=tasks_completed,
            budget_used=budget_used,
            assessment=assessment,
        )

        return CycleReportContent(
            summary=summary,
            full_content=full_content,
            tasks_completed=tasks_completed,
            findings_count=len(findings),
            hypotheses_count=len(hypotheses.get("generated", [])),
            papers_count=len(papers),
            generation_cost=generation_cost,
        )

    def save_report_to_file(
        self,
        report: CycleReportContent,
        output_dir: Path,
        cycle_number: int,
    ) -> Path:
        """
        Save the cycle report to a markdown file.

        Args:
            report: The generated report content
            output_dir: Directory to save the report
            cycle_number: The cycle number for filename

        Returns:
            Path to the saved report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"cycle_{cycle_number:02d}_report.md"
        filepath = output_dir / filename

        filepath.write_text(report.full_content, encoding="utf-8")
        logger.info(f"Saved cycle report to {filepath}")

        return filepath
