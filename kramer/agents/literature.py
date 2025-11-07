"""Literature agent for searching papers and extracting claims."""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import anthropic

from kramer.world_model import WorldModel, Finding, Node, NodeType
from kramer.api_clients.semantic_scholar import SemanticScholarClient, PaperMetadata
from src.utils.cost_tracker import CostTracker


@dataclass
class ExtractedClaim:
    """A claim extracted from a paper with full citation."""
    claim_text: str
    paper_title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    confidence: float  # How central the claim is to the paper (0-1)
    paper_id: str  # Semantic Scholar paper ID


class LiteratureAgent:
    """
    Agent that searches for papers and extracts claims using Claude.

    The agent:
    1. Searches Semantic Scholar for relevant papers
    2. Fetches paper metadata and abstracts
    3. Uses Claude to extract 3-5 key claims from each abstract
    4. Stores papers as nodes and claims as findings in the world model
    """

    def __init__(
        self,
        world_model: WorldModel,
        anthropic_api_key: str,
        semantic_scholar_api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the Literature Agent.

        Args:
            world_model: World model to store papers and claims
            anthropic_api_key: API key for Claude
            semantic_scholar_api_key: Optional API key for Semantic Scholar
            model: Claude model to use for claim extraction
        """
        self.world_model = world_model
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.ss_client = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.model = model
        self.total_cost: float = 0.0  # Track total API costs

    async def __aenter__(self):
        await self.ss_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ss_client.__aexit__(exc_type, exc_val, exc_tb)

    async def search_and_extract(
        self,
        query: str,
        max_papers: int = 10,
        hypotheses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for papers and extract claims.

        Args:
            query: Search query for papers
            max_papers: Maximum number of papers to process
            hypotheses: Optional list of current hypotheses to consider

        Returns:
            Dictionary with:
                - papers: List of paper metadata
                - claims: List of extracted claims with full citations
                - cost: Total API cost in dollars
        """
        # Step 1: Search for papers
        print(f"Searching for papers on: {query}")
        papers = await self.ss_client.search_papers(query, limit=max_papers)
        print(f"Found {len(papers)} papers")

        if not papers:
            print("No papers found")
            return {
                "papers": [],
                "claims": [],
                "cost": self.total_cost,
            }

        # Step 2: Process each paper
        all_claims = []
        for i, paper in enumerate(papers, 1):
            print(f"\nProcessing paper {i}/{len(papers)}: {paper.title}")

            # Skip papers without abstracts
            if not paper.abstract or len(paper.abstract.strip()) < 50:
                print("  Skipping: No abstract available")
                continue

            # Step 3: Add paper to world model
            paper_node = self._add_paper_to_world_model(paper)

            # Step 4: Extract claims using Claude
            try:
                claims = await self._extract_claims_from_abstract(
                    paper=paper,
                    hypotheses=hypotheses
                )
                print(f"  Extracted {len(claims)} claims")

                # Step 5: Store claims as findings
                for claim in claims:
                    finding = Finding(
                        claim_text=claim.claim_text,
                        source_node_id=paper_node.id,
                        confidence=claim.confidence,
                        metadata={
                            "paper_title": paper.title,
                            "authors": paper.authors,
                            "year": paper.year,
                            "doi": paper.doi,
                            "paper_id": paper.paper_id
                        }
                    )
                    self.world_model.add_finding(finding)

                all_claims.extend(claims)

            except Exception as e:
                print(f"  Error extracting claims: {e}")
                continue

            # Small delay between papers to avoid rate limits
            await asyncio.sleep(0.5)

        print(f"\n✓ Total claims extracted: {len(all_claims)}")
        print(f"✓ World model: {self.world_model.summary()}")

        return {
            "papers": papers,
            "claims": all_claims,
            "cost": self.total_cost,
        }

    def _add_paper_to_world_model(self, paper: PaperMetadata) -> Node:
        """Add a paper to the world model as a node."""
        return self.world_model.create_paper_node(
            title=paper.title,
            authors=paper.authors,
            year=paper.year or 0,
            doi=paper.doi or "",
            abstract=paper.abstract or "",
            paper_id=paper.paper_id,
            citation_count=paper.citation_count,
            influential_citation_count=paper.influential_citation_count,
            url=paper.url,
            venue=paper.venue
        )

    async def _extract_claims_from_abstract(
        self,
        paper: PaperMetadata,
        hypotheses: Optional[List[str]] = None
    ) -> List[ExtractedClaim]:
        """
        Extract key claims from a paper abstract using Claude.

        Args:
            paper: Paper metadata including abstract
            hypotheses: Optional current hypotheses for context

        Returns:
            List of extracted claims
        """
        # Build prompt for Claude
        prompt = self._build_extraction_prompt(paper, hypotheses)

        # Call Claude API
        response = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track API cost
        cost = CostTracker.track_call(self.model, response)
        self.total_cost += cost

        # Parse response
        response_text = response.content[0].text

        # Extract claims from structured response
        claims = self._parse_claims_from_response(response_text, paper)

        return claims

    def _build_extraction_prompt(
        self,
        paper: PaperMetadata,
        hypotheses: Optional[List[str]] = None
    ) -> str:
        """Build the prompt for Claude to extract claims."""
        hypotheses_context = ""
        if hypotheses:
            hypotheses_context = f"\n\nCurrent hypotheses to consider:\n" + "\n".join(
                f"- {h}" for h in hypotheses
            )

        prompt = f"""Extract 3-5 key factual claims from the following scientific paper abstract.

Paper: {paper.title}
Authors: {", ".join(paper.authors)}
Year: {paper.year}

Abstract:
{paper.abstract}
{hypotheses_context}

For each claim, provide:
1. The claim text (one clear, factual statement)
2. A confidence score (0.0-1.0) indicating how central/important this claim is to the paper

Requirements:
- Focus on concrete findings, not background or speculation
- Claims should be self-contained and understandable
- Prioritize novel findings over known background
- Each claim should be a single factual statement
- Confidence should reflect how prominently the claim features in the abstract

Format your response as:

CLAIM 1:
Text: [claim text]
Confidence: [0.0-1.0]

CLAIM 2:
Text: [claim text]
Confidence: [0.0-1.0]

...and so on."""

        return prompt

    def _parse_claims_from_response(
        self,
        response: str,
        paper: PaperMetadata
    ) -> List[ExtractedClaim]:
        """Parse claims from Claude's response."""
        claims = []
        lines = response.strip().split("\n")

        current_claim_text = None
        current_confidence = None

        for line in lines:
            line = line.strip()

            if line.startswith("Text:"):
                current_claim_text = line[5:].strip()
            elif line.startswith("Confidence:"):
                confidence_str = line[11:].strip()
                try:
                    current_confidence = float(confidence_str)
                except ValueError:
                    current_confidence = 0.5

                # When we have both text and confidence, create a claim
                if current_claim_text and current_confidence is not None:
                    claims.append(ExtractedClaim(
                        claim_text=current_claim_text,
                        paper_title=paper.title,
                        authors=paper.authors,
                        year=paper.year,
                        doi=paper.doi,
                        confidence=current_confidence,
                        paper_id=paper.paper_id
                    ))

                    # Reset for next claim
                    current_claim_text = None
                    current_confidence = None

        return claims

    def format_citation(self, claim: ExtractedClaim) -> str:
        """Format a citation in standard format."""
        authors_str = ", ".join(claim.authors[:3])
        if len(claim.authors) > 3:
            authors_str += " et al."

        year_str = f"({claim.year})" if claim.year else ""

        citation = f"{authors_str} {year_str}. {claim.paper_title}."

        if claim.doi:
            citation += f" DOI: {claim.doi}"

        return citation

    async def close(self):
        """Close API clients."""
        await self.ss_client.close()
