"""
Report Generator - Synthesizes world model findings into publication-quality reports.

The ReportGenerator analyzes the world model to identify high-confidence discoveries,
groups related findings, and generates narrative reports with proper citations.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from src.world_model.graph import EdgeType, NodeType, WorldModel

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation (paper or code)."""
    citation_id: str
    citation_type: str  # "paper" or "code"
    content: Dict[str, Any]

    def format_inline(self) -> str:
        """Format citation for inline use."""
        if self.citation_type == "paper":
            return f"[{self.citation_id}]"
        else:  # code
            return f"[Analysis {self.citation_id}]"

    def format_bibliography(self) -> str:
        """Format citation for bibliography."""
        if self.citation_type == "paper":
            c = self.content
            authors = c.get("authors", "Unknown")
            if isinstance(authors, list):
                if len(authors) == 1:
                    authors = authors[0]
                elif len(authors) == 2:
                    authors = f"{authors[0]} and {authors[1]}"
                else:
                    authors = f"{authors[0]} et al."

            title = c.get("title", "Untitled")
            year = c.get("year", "n.d.")
            doi = c.get("doi", c.get("url", ""))

            bib = f"[{self.citation_id}] {authors}. ({year}). {title}."
            if doi:
                bib += f" {doi}"
            return bib
        else:  # code
            provenance = self.content.get("provenance", "unknown")
            node_id = self.content.get("node_id", "unknown")
            return f"[Analysis {self.citation_id}] Code: {provenance} (Finding {node_id[:8]})"


@dataclass
class Discovery:
    """Represents a grouped set of findings that form a coherent discovery."""
    title: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    papers: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    novelty_score: float = 0.0
    narrative: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)

    def __post_init__(self):
        """Calculate aggregate metrics."""
        if self.findings:
            confidences = [f.get("confidence", 0.0) for f in self.findings if f.get("confidence")]
            self.confidence = sum(confidences) / len(confidences) if confidences else 0.0


class ReportGenerator:
    """
    Generates publication-quality reports from world model findings.

    The generator:
    1. Extracts high-confidence findings (>0.7)
    2. Groups related findings into discoveries
    3. Generates narrative descriptions using Claude API
    4. Formats as markdown with citations
    5. Produces both main report and appendix
    """

    def __init__(
        self,
        world_model: WorldModel,
        anthropic_api_key: Optional[str] = None,
        min_confidence: float = 0.7,
        max_discoveries: int = 5,
    ):
        """
        Initialize the report generator.

        Args:
            world_model: The world model to generate reports from
            anthropic_api_key: API key for Claude (optional, uses env var if not provided)
            min_confidence: Minimum confidence threshold for findings
            max_discoveries: Maximum number of discoveries to include in report
        """
        self.world_model = world_model
        self.min_confidence = min_confidence
        self.max_discoveries = max_discoveries

        # Initialize Claude client if API key provided
        self.anthropic_client = None
        if anthropic_api_key:
            if not HAS_ANTHROPIC:
                logger.warning("anthropic package not installed. Install with: pip install anthropic")
            else:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Track citations
        self.citations: Dict[str, Citation] = {}
        self.citation_counter = 0

    def extract_high_confidence_findings(self) -> List[Dict[str, Any]]:
        """
        Extract findings with confidence > threshold.

        Returns:
            List of high-confidence finding nodes
        """
        high_confidence_findings = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") != NodeType.FINDING.value:
                continue

            confidence = data.get("confidence")
            if confidence is None or confidence < self.min_confidence:
                continue

            # Check if supported by multiple sources
            supporters = self._get_supporting_nodes(node_id)

            # Check novelty
            novelty_score = self._assess_novelty(node_id, data)

            high_confidence_findings.append({
                "node_id": node_id,
                "text": data.get("text"),
                "confidence": confidence,
                "provenance": data.get("provenance"),
                "metadata": data.get("metadata", {}),
                "supporters": supporters,
                "novelty_score": novelty_score,
            })

        # Sort by confidence and novelty
        high_confidence_findings.sort(
            key=lambda x: (x["confidence"], x["novelty_score"]),
            reverse=True
        )

        logger.info(f"Extracted {len(high_confidence_findings)} high-confidence findings")
        return high_confidence_findings

    def _get_supporting_nodes(self, node_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get nodes that support this finding."""
        supporters = {
            "findings": [],
            "papers": [],
            "hypotheses": [],
        }

        # Get incoming SUPPORTS edges
        for source, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
            if edge_data.get("edge_type") != EdgeType.SUPPORTS.value:
                continue

            source_node = self.world_model.get_node(source)
            if not source_node:
                continue

            node_type = source_node.get("node_type")
            if node_type == NodeType.FINDING.value:
                supporters["findings"].append(source_node)
            elif node_type == NodeType.PAPER.value:
                supporters["papers"].append(source_node)
            elif node_type == NodeType.HYPOTHESIS.value:
                supporters["hypotheses"].append(source_node)

        return supporters

    def _assess_novelty(self, node_id: str, node_data: Dict[str, Any]) -> float:
        """
        Assess novelty of a finding.

        Novelty is high if:
        - It contradicts literature (has REFUTES edges from papers)
        - It fills a gap (related to questions)
        - It has few supporting papers

        Returns:
            Novelty score from 0.0 to 1.0
        """
        novelty = 0.0

        # Check for contradictions with literature
        refuting_papers = []
        for source, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
            if edge_data.get("edge_type") == EdgeType.REFUTES.value:
                source_node = self.world_model.get_node(source)
                if source_node and source_node.get("node_type") == NodeType.PAPER.value:
                    refuting_papers.append(source_node)

        if refuting_papers:
            novelty += 0.5  # Contradicts existing literature

        # Check if it answers a question
        for source, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
            source_node = self.world_model.get_node(source)
            if source_node and source_node.get("node_type") == NodeType.QUESTION.value:
                novelty += 0.3  # Answers a research question
                break

        # Check literature coverage
        supporting_papers = []
        for source, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
            if edge_data.get("edge_type") == EdgeType.SUPPORTS.value:
                source_node = self.world_model.get_node(source)
                if source_node and source_node.get("node_type") == NodeType.PAPER.value:
                    supporting_papers.append(source_node)

        if len(supporting_papers) == 0:
            novelty += 0.3  # No existing literature support
        elif len(supporting_papers) <= 2:
            novelty += 0.1  # Limited literature coverage

        return min(1.0, novelty)

    def group_findings_into_discoveries(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[Discovery]:
        """
        Group related findings into coherent discoveries.

        Uses graph connectivity and semantic similarity to group findings.

        Args:
            findings: List of high-confidence findings

        Returns:
            List of Discovery objects
        """
        if not findings:
            return []

        # Build connectivity graph
        finding_ids = {f["node_id"] for f in findings}
        clusters = []
        processed = set()

        for finding in findings:
            if finding["node_id"] in processed:
                continue

            # Start a new cluster with this finding
            cluster = self._expand_cluster(finding["node_id"], finding_ids, processed)
            if cluster:
                clusters.append(cluster)

        # Convert clusters to Discovery objects
        discoveries = []
        for cluster in clusters:
            discovery = self._create_discovery_from_cluster(cluster, findings)
            if discovery:
                discoveries.append(discovery)

        # Sort by confidence and novelty
        discoveries.sort(
            key=lambda d: (d.confidence, d.novelty_score),
            reverse=True
        )

        logger.info(f"Grouped findings into {len(discoveries)} discoveries")
        return discoveries[:self.max_discoveries]

    def _expand_cluster(
        self,
        start_node_id: str,
        candidate_nodes: Set[str],
        processed: Set[str]
    ) -> Set[str]:
        """Expand a cluster using graph connectivity."""
        cluster = {start_node_id}
        processed.add(start_node_id)
        queue = [start_node_id]

        while queue:
            current = queue.pop(0)

            # Find connected nodes
            neighbors = []

            # Outgoing edges
            for _, target in self.world_model.graph.out_edges(current):
                if target in candidate_nodes and target not in processed:
                    neighbors.append(target)

            # Incoming edges
            for source, _ in self.world_model.graph.in_edges(current):
                if source in candidate_nodes and source not in processed:
                    neighbors.append(source)

            for neighbor in neighbors:
                cluster.add(neighbor)
                processed.add(neighbor)
                queue.append(neighbor)

        return cluster

    def _create_discovery_from_cluster(
        self,
        cluster: Set[str],
        all_findings: List[Dict[str, Any]]
    ) -> Optional[Discovery]:
        """Create a Discovery object from a cluster of findings."""
        # Get finding data
        cluster_findings = [f for f in all_findings if f["node_id"] in cluster]
        if not cluster_findings:
            return None

        # Collect related papers and hypotheses
        papers = []
        hypotheses = []

        for node_id in cluster:
            # Get supporting papers
            for source, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
                source_node = self.world_model.get_node(source)
                if not source_node:
                    continue

                if source_node.get("node_type") == NodeType.PAPER.value:
                    if source_node not in papers:
                        papers.append(source_node)
                elif source_node.get("node_type") == NodeType.HYPOTHESIS.value:
                    if source_node not in hypotheses:
                        hypotheses.append(source_node)

        # Generate title from first finding
        title = self._generate_discovery_title(cluster_findings)

        # Calculate novelty score
        novelty_scores = [f.get("novelty_score", 0.0) for f in cluster_findings]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        return Discovery(
            title=title,
            findings=cluster_findings,
            papers=papers,
            hypotheses=hypotheses,
            novelty_score=avg_novelty,
        )

    def _generate_discovery_title(self, findings: List[Dict[str, Any]]) -> str:
        """Generate a title for a discovery based on its findings."""
        if not findings:
            return "Untitled Discovery"

        # Use the first finding text, truncated
        first_text = findings[0].get("text", "Untitled Discovery")

        # Extract key terms (simple heuristic)
        if len(first_text) > 60:
            # Try to find a good breaking point
            truncated = first_text[:60]
            last_space = truncated.rfind(" ")
            if last_space > 30:
                return truncated[:last_space] + "..."
            return truncated + "..."

        return first_text

    def generate_narrative(self, discovery: Discovery) -> str:
        """
        Generate narrative description for a discovery using Claude API.

        Args:
            discovery: The discovery to generate narrative for

        Returns:
            Narrative text (markdown formatted)
        """
        if not self.anthropic_client:
            # Fallback to template-based narrative
            return self._generate_template_narrative(discovery)

        # Build prompt for Claude
        prompt = self._build_narrative_prompt(discovery)

        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            narrative = message.content[0].text
            return narrative

        except Exception as e:
            logger.error(f"Failed to generate narrative with Claude: {e}")
            return self._generate_template_narrative(discovery)

    def _build_narrative_prompt(self, discovery: Discovery) -> str:
        """Build prompt for Claude to generate narrative."""
        findings_text = "\n".join([
            f"- {f.get('text')} (confidence: {f.get('confidence', 0):.2f})"
            for f in discovery.findings
        ])

        papers_text = "None"
        if discovery.papers:
            papers_text = "\n".join([
                f"- {p.get('metadata', {}).get('title', 'Untitled')} "
                f"by {p.get('metadata', {}).get('authors', 'Unknown')}"
                for p in discovery.papers[:3]
            ])

        hypotheses_text = "None"
        if discovery.hypotheses:
            hypotheses_text = "\n".join([
                f"- {h.get('text')}"
                for h in discovery.hypotheses
            ])

        prompt = f"""You are a scientific writer helping to generate a discovery report.
Write a clear, engaging narrative description of the following scientific discovery.

Discovery Title: {discovery.title}

Findings from Data Analysis:
{findings_text}

Related Literature:
{papers_text}

Related Hypotheses:
{hypotheses_text}

Please write a 2-3 paragraph narrative that:
1. Describes the discovery in clear, accessible language
2. Explains the significance and implications
3. Describes the supporting evidence (both data analysis and literature)
4. Mentions any limitations or uncertainties
5. Uses a professional, scientific tone

Do not include citations in the narrative text - they will be added separately.
Write in third person, present tense where appropriate.
"""
        return prompt

    def _generate_template_narrative(self, discovery: Discovery) -> str:
        """Generate a template-based narrative (fallback when Claude API unavailable)."""
        confidence_desc = "high" if discovery.confidence > 0.9 else "moderate"

        narrative = f"This discovery is based on {len(discovery.findings)} "
        narrative += f"finding{'s' if len(discovery.findings) > 1 else ''} "
        narrative += f"with {confidence_desc} confidence ({discovery.confidence:.2f}). "

        # Main finding
        if discovery.findings:
            narrative += f"{discovery.findings[0].get('text')} "

        # Supporting evidence
        if discovery.papers:
            narrative += f"\n\nThis finding is supported by {len(discovery.papers)} "
            narrative += f"paper{'s' if len(discovery.papers) > 1 else ''} from the literature. "

        if discovery.novelty_score > 0.5:
            narrative += f"\n\nNotably, this discovery appears to be novel "
            narrative += f"(novelty score: {discovery.novelty_score:.2f}), "
            narrative += f"either contradicting existing literature or filling a knowledge gap. "

        # Limitations
        low_conf_findings = [f for f in discovery.findings if f.get("confidence", 1.0) < 0.8]
        if low_conf_findings:
            narrative += f"\n\nSome findings in this discovery have moderate confidence "
            narrative += f"and may require further validation. "

        return narrative

    def _create_citation(
        self,
        citation_type: str,
        content: Dict[str, Any]
    ) -> Citation:
        """Create and register a citation."""
        # Check if we already have this citation
        citation_key = f"{citation_type}:{json.dumps(content, sort_keys=True)}"

        for cit in self.citations.values():
            cit_key = f"{cit.citation_type}:{json.dumps(cit.content, sort_keys=True)}"
            if cit_key == citation_key:
                return cit

        # Create new citation
        if citation_type == "paper":
            self.citation_counter += 1
            citation_id = str(self.citation_counter)
        else:  # code
            # Use a reference ID
            self.citation_counter += 1
            citation_id = f"r{self.citation_counter}"

        citation = Citation(
            citation_id=citation_id,
            citation_type=citation_type,
            content=content,
        )

        self.citations[citation_id] = citation
        return citation

    def _add_citations_to_discovery(self, discovery: Discovery) -> None:
        """Add citations to a discovery."""
        citations = []

        # Add paper citations
        for paper in discovery.papers:
            citation = self._create_citation("paper", {
                "title": paper.get("metadata", {}).get("title", "Untitled"),
                "authors": paper.get("metadata", {}).get("authors", "Unknown"),
                "year": paper.get("metadata", {}).get("year"),
                "doi": paper.get("metadata", {}).get("doi"),
                "url": paper.get("metadata", {}).get("url"),
            })
            citations.append(citation)

        # Add code citations for each finding
        for finding in discovery.findings:
            if finding.get("provenance"):
                citation = self._create_citation("code", {
                    "provenance": finding.get("provenance"),
                    "node_id": finding.get("node_id"),
                })
                citations.append(citation)

        discovery.citations = citations

    def generate_report(
        self,
        output_path: Path,
        include_appendix: bool = True,
        generate_narratives: bool = True,
    ) -> Dict[str, Path]:
        """
        Generate the full report.

        Args:
            output_path: Path for the main report file
            include_appendix: Whether to generate appendix
            generate_narratives: Whether to generate AI narratives (requires API key)

        Returns:
            Dictionary mapping report types to file paths
        """
        logger.info("Starting report generation...")

        # Reset citations
        self.citations = {}
        self.citation_counter = 0

        # Extract findings
        findings = self.extract_high_confidence_findings()

        # Group into discoveries
        discoveries = self.group_findings_into_discoveries(findings)

        # Generate narratives if requested
        if generate_narratives:
            logger.info("Generating narratives with Claude API...")
            for discovery in discoveries:
                discovery.narrative = self.generate_narrative(discovery)
                self._add_citations_to_discovery(discovery)
        else:
            logger.info("Using template-based narratives...")
            for discovery in discoveries:
                discovery.narrative = self._generate_template_narrative(discovery)
                self._add_citations_to_discovery(discovery)

        # Generate main report
        logger.info(f"Writing main report to {output_path}...")
        self._write_main_report(output_path, discoveries)

        result = {"report": output_path}

        # Generate appendix if requested
        if include_appendix:
            appendix_path = output_path.parent / f"{output_path.stem}_appendix.md"
            logger.info(f"Writing appendix to {appendix_path}...")
            self._write_appendix(appendix_path, findings, discoveries)
            result["appendix"] = appendix_path

        logger.info("Report generation complete!")
        return result

    def _write_main_report(self, output_path: Path, discoveries: List[Discovery]) -> None:
        """Write the main report markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Title
            f.write("# Discovery Report\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents {len(discoveries)} key discoveries "
                   f"extracted from the research world model. ")
            f.write(f"All findings have confidence scores above {self.min_confidence:.2f} "
                   f"and are supported by data analysis")
            if any(d.papers for d in discoveries):
                f.write(" and literature review")
            f.write(".\n\n")

            stats = self.world_model.get_stats()
            f.write(f"**World Model Statistics:**\n")
            f.write(f"- Total nodes: {stats['total_nodes']}\n")
            f.write(f"- Findings: {stats['node_types'].get('finding', 0)}\n")
            f.write(f"- Papers: {stats['node_types'].get('paper', 0)}\n")
            f.write(f"- Hypotheses: {stats['node_types'].get('hypothesis', 0)}\n\n")

            # Discoveries
            for i, discovery in enumerate(discoveries, 1):
                f.write(f"## Discovery {i}: {discovery.title}\n\n")

                # Confidence and novelty badges
                f.write(f"**Confidence:** {discovery.confidence:.2f} | ")
                f.write(f"**Novelty Score:** {discovery.novelty_score:.2f}\n\n")

                # Narrative
                if discovery.narrative:
                    f.write(discovery.narrative)
                    f.write("\n\n")

                # Supporting Evidence
                f.write("### Supporting Evidence\n\n")

                # Data findings
                f.write("**Data Analysis:**\n\n")
                for finding in discovery.findings:
                    f.write(f"- {finding.get('text')}")

                    # Add code citation
                    code_cit = None
                    for cit in discovery.citations:
                        if (cit.citation_type == "code" and
                            cit.content.get("node_id") == finding.get("node_id")):
                            code_cit = cit
                            break

                    if code_cit:
                        f.write(f" {code_cit.format_inline()}")

                    f.write(f" (confidence: {finding.get('confidence', 0):.2f})")
                    f.write("\n")

                f.write("\n")

                # Literature support
                if discovery.papers:
                    f.write("**Literature Support:**\n\n")
                    paper_citations = [c for c in discovery.citations if c.citation_type == "paper"]
                    if paper_citations:
                        cit_str = ", ".join([c.format_inline() for c in paper_citations])
                        f.write(f"This discovery is supported by {len(discovery.papers)} "
                               f"papers from the literature {cit_str}.\n\n")

                # Related hypotheses
                if discovery.hypotheses:
                    f.write("**Related Hypotheses:**\n\n")
                    for hyp in discovery.hypotheses:
                        conf = hyp.get("confidence")
                        conf_str = f" (confidence: {conf:.2f})" if conf else ""
                        f.write(f"- {hyp.get('text')}{conf_str}\n")
                    f.write("\n")

                f.write("---\n\n")

            # Bibliography
            if self.citations:
                f.write("## References\n\n")

                # Papers first
                paper_citations = [c for c in self.citations.values() if c.citation_type == "paper"]
                if paper_citations:
                    f.write("### Literature\n\n")
                    for citation in sorted(paper_citations, key=lambda c: int(c.citation_id)):
                        f.write(f"{citation.format_bibliography()}\n\n")

                # Code references
                code_citations = [c for c in self.citations.values() if c.citation_type == "code"]
                if code_citations:
                    f.write("### Code References\n\n")
                    for citation in sorted(code_citations, key=lambda c: c.citation_id):
                        f.write(f"{citation.format_bibliography()}\n\n")

    def _write_appendix(
        self,
        output_path: Path,
        findings: List[Dict[str, Any]],
        discoveries: List[Discovery]
    ) -> None:
        """Write the appendix markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Appendix: Detailed Methods and Findings\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            # Methods
            f.write("## Methods\n\n")
            f.write(f"### Discovery Extraction\n\n")
            f.write(f"- Minimum confidence threshold: {self.min_confidence}\n")
            f.write(f"- Maximum discoveries: {self.max_discoveries}\n")
            f.write(f"- Total findings analyzed: {len(findings)}\n")
            f.write(f"- Discoveries generated: {len(discoveries)}\n\n")

            f.write("### Novelty Assessment\n\n")
            f.write("Novelty scores are calculated based on:\n")
            f.write("- Contradiction with existing literature (+0.5)\n")
            f.write("- Answers to research questions (+0.3)\n")
            f.write("- Limited or no literature coverage (+0.1 to +0.3)\n\n")

            # All high-confidence findings
            f.write("## All High-Confidence Findings\n\n")
            for i, finding in enumerate(findings, 1):
                f.write(f"### Finding {i}\n\n")
                f.write(f"**Text:** {finding.get('text')}\n\n")
                f.write(f"**Confidence:** {finding.get('confidence', 0):.2f}\n\n")
                f.write(f"**Novelty Score:** {finding.get('novelty_score', 0):.2f}\n\n")

                if finding.get("provenance"):
                    f.write(f"**Code:** `{finding.get('provenance')}`\n\n")

                if finding.get("metadata"):
                    f.write(f"**Metadata:** {json.dumps(finding.get('metadata'), indent=2)}\n\n")

                # Supporting nodes
                supporters = finding.get("supporters", {})
                if any(supporters.values()):
                    f.write("**Supporting Evidence:**\n\n")

                    if supporters.get("papers"):
                        f.write(f"- {len(supporters['papers'])} papers\n")
                    if supporters.get("findings"):
                        f.write(f"- {len(supporters['findings'])} other findings\n")
                    if supporters.get("hypotheses"):
                        f.write(f"- {len(supporters['hypotheses'])} hypotheses\n")
                    f.write("\n")

                f.write("---\n\n")

            # World model statistics
            f.write("## World Model Statistics\n\n")
            stats = self.world_model.get_stats()
            f.write(f"```json\n{json.dumps(stats, indent=2)}\n```\n\n")

            # Discovery groupings
            f.write("## Discovery Groupings\n\n")
            for i, discovery in enumerate(discoveries, 1):
                f.write(f"### Discovery {i}: {discovery.title}\n\n")
                f.write(f"- Findings: {len(discovery.findings)}\n")
                f.write(f"- Papers: {len(discovery.papers)}\n")
                f.write(f"- Hypotheses: {len(discovery.hypotheses)}\n")
                f.write(f"- Average Confidence: {discovery.confidence:.2f}\n")
                f.write(f"- Novelty Score: {discovery.novelty_score:.2f}\n\n")

                f.write("**Finding IDs:**\n")
                for finding in discovery.findings:
                    node_id = finding.get("node_id", "unknown")
                    f.write(f"- `{node_id}`\n")
                f.write("\n")
