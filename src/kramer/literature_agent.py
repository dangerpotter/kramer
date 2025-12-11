"""Real literature agent that searches Semantic Scholar, arXiv, and extracts claims using Claude."""

import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import httpx
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """A scientific paper with metadata."""
    paper_id: str
    title: str
    authors: str  # Comma-separated author names
    year: Optional[int]
    abstract: Optional[str]
    citation_count: int
    url: Optional[str]
    relevance_score: float = 0.0
    source: str = "unknown"  # "semantic_scholar" or "arxiv"


class LiteratureAgent:
    """
    Agent that searches real papers from Semantic Scholar, arXiv and extracts claims using Claude.

    Provides the same interface as the mock agent but with real functionality.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        search_arxiv: bool = True,
        search_semantic_scholar: bool = True
    ):
        """
        Initialize the Literature Agent.

        Args:
            anthropic_api_key: Claude API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use for claim extraction
            search_arxiv: Whether to search arXiv (default True)
            search_semantic_scholar: Whether to search Semantic Scholar (default True)
        """
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.model = model
        self.total_cost = 0.0
        self.search_arxiv = search_arxiv
        self.search_semantic_scholar = search_semantic_scholar

    async def search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on Semantic Scholar and arXiv.

        Args:
            query: Search query
            max_results: Maximum number of papers to return (split between sources)

        Returns:
            List of paper dictionaries
        """
        papers = []

        # Split results between sources
        per_source = max_results // 2 if (self.search_arxiv and self.search_semantic_scholar) else max_results

        # Search both sources concurrently
        tasks = []
        if self.search_semantic_scholar:
            tasks.append(self._search_semantic_scholar(query, per_source))
        if self.search_arxiv:
            tasks.append(self._search_arxiv(query, per_source))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    papers.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Search error: {result}")

        # Sort by relevance/citation count and limit
        papers.sort(key=lambda p: (p.citation_count, p.relevance_score), reverse=True)
        papers = papers[:max_results]

        # Convert to dict format expected by coordinator
        return [
            {
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "abstract": p.abstract or "",
                "citation_count": p.citation_count,
                "url": p.url,
                "relevance_score": p.relevance_score,
                "paper_id": p.paper_id,
                "source": p.source
            }
            for p in papers
        ]

    async def search_for_hypothesis(
        self,
        hypothesis: str,
        max_papers: int = 5
    ) -> Dict[str, Any]:
        """
        Search for papers related to a hypothesis and extract relevant findings.

        Args:
            hypothesis: The hypothesis to search for
            max_papers: Maximum number of papers to retrieve

        Returns:
            Dictionary with papers and extracted findings
        """
        # Search for papers from both sources
        papers_data = await self.search(hypothesis, max_papers)

        # Convert back to Paper objects for processing
        papers = [
            Paper(
                paper_id=p["paper_id"],
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                abstract=p["abstract"],
                citation_count=p["citation_count"],
                url=p["url"],
                relevance_score=p["relevance_score"],
                source=p["source"]
            )
            for p in papers_data
        ]

        # Extract findings/claims from paper abstracts using Claude
        findings = []
        if self.anthropic_client and papers:
            findings = await self._extract_claims_from_papers(papers, hypothesis)

        return {
            "task": "literature_search",
            "hypothesis": hypothesis,
            "papers": papers_data,
            "findings": findings,
            "cost": self.total_cost
        }

    async def search_topic(
        self,
        topic: str,
        max_papers: int = 10
    ) -> Dict[str, Any]:
        """
        Search for papers on a topic and extract key insights.

        Args:
            topic: Topic to search for
            max_papers: Maximum number of papers

        Returns:
            Dictionary with papers and insights
        """
        papers_data = await self.search(topic, max_papers)

        # Convert back to Paper objects
        papers = [
            Paper(
                paper_id=p["paper_id"],
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                abstract=p["abstract"],
                citation_count=p["citation_count"],
                url=p["url"],
                relevance_score=p["relevance_score"],
                source=p["source"]
            )
            for p in papers_data
        ]

        # Extract insights using Claude if available
        insights = []
        if self.anthropic_client and papers:
            insights = await self._synthesize_topic_insights(papers, topic)

        return {
            "task": "topic_search",
            "topic": topic,
            "papers": papers_data,
            "insights": insights,
            "cost": self.total_cost
        }

    async def _search_semantic_scholar(
        self,
        query: str,
        max_results: int
    ) -> List[Paper]:
        """
        Search Semantic Scholar API for papers.

        Args:
            query: Search query
            max_results: Max papers to return

        Returns:
            List of Paper objects
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": query,
            "limit": min(max_results, 100),  # API limit
            "fields": "paperId,title,authors,year,abstract,citationCount,url"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                papers = []
                for item in data.get("data", []):
                    authors_list = [a.get("name", "Unknown") for a in item.get("authors", [])]
                    authors_str = ", ".join(authors_list) if authors_list else "Unknown"

                    paper = Paper(
                        paper_id=item.get("paperId", ""),
                        title=item.get("title", "Untitled"),
                        authors=authors_str,
                        year=item.get("year"),
                        abstract=item.get("abstract"),
                        citation_count=item.get("citationCount", 0),
                        url=item.get("url"),
                        relevance_score=0.8,  # Semantic Scholar doesn't provide this
                        source="semantic_scholar"
                    )
                    papers.append(paper)

                logger.info(f"Found {len(papers)} papers on Semantic Scholar for: {query}")
                return papers

            except httpx.HTTPError as e:
                logger.error(f"Error searching Semantic Scholar: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error in Semantic Scholar search: {e}")
                return []

    async def _search_arxiv(
        self,
        query: str,
        max_results: int
    ) -> List[Paper]:
        """
        Search arXiv API for papers.

        Args:
            query: Search query
            max_results: Max papers to return

        Returns:
            List of Paper objects
        """
        # arXiv API endpoint (must use https)
        url = "https://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # arXiv requests 3 second delay between calls
                await asyncio.sleep(3)

                response = await client.get(url, params=params)
                response.raise_for_status()

                # Parse Atom XML response
                papers = self._parse_arxiv_response(response.text)

                logger.info(f"Found {len(papers)} papers on arXiv for: {query}")
                return papers

            except httpx.HTTPError as e:
                logger.error(f"Error searching arXiv: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error in arXiv search: {e}")
                return []

    def _parse_arxiv_response(self, xml_text: str) -> List[Paper]:
        """
        Parse arXiv Atom XML response into Paper objects.

        Args:
            xml_text: Raw XML response from arXiv API

        Returns:
            List of Paper objects
        """
        papers = []

        try:
            # Parse XML
            root = ET.fromstring(xml_text)

            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            # Find all entry elements
            for entry in root.findall('atom:entry', ns):
                # Extract basic fields
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Untitled"

                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None else None

                # Extract ID and URL
                id_elem = entry.find('atom:id', ns)
                paper_url = id_elem.text if id_elem is not None else None

                # Extract arXiv ID from URL (e.g., http://arxiv.org/abs/1234.5678)
                paper_id = paper_url.split('/abs/')[-1] if paper_url else ""

                # Extract authors
                author_names = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        author_names.append(name_elem.text)
                authors_str = ", ".join(author_names) if author_names else "Unknown"

                # Extract publication date to get year
                published_elem = entry.find('atom:published', ns)
                year = None
                if published_elem is not None:
                    try:
                        date_str = published_elem.text
                        year = int(date_str[:4])  # Extract year from ISO date
                    except:
                        pass

                # arXiv doesn't provide citation counts, use 0
                # Also doesn't provide relevance scores
                paper = Paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors_str,
                    year=year,
                    abstract=abstract,
                    citation_count=0,  # arXiv doesn't track citations
                    url=paper_url,
                    relevance_score=0.7,  # Default relevance
                    source="arxiv"
                )
                papers.append(paper)

            return papers

        except ET.ParseError as e:
            logger.error(f"Error parsing arXiv XML: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing arXiv response: {e}")
            return []

    async def _extract_claims_from_papers(
        self,
        papers: List[Paper],
        hypothesis: str
    ) -> List[str]:
        """
        Use Claude to extract relevant claims from paper abstracts.

        Args:
            papers: List of papers
            hypothesis: The hypothesis context

        Returns:
            List of extracted claim strings
        """
        if not self.anthropic_client:
            return []

        # Build prompt with paper abstracts
        papers_text = "\n\n".join([
            f"Paper {i+1}: {p.title} ({p.year}) [Source: {p.source}]\n"
            f"Authors: {p.authors}\n"
            f"Abstract: {p.abstract or 'No abstract available'}"
            for i, p in enumerate(papers) if p.abstract
        ])

        prompt = f"""Given this hypothesis:
"{hypothesis}"

And these research papers from arXiv and Semantic Scholar:

{papers_text}

Extract 3-5 key claims or findings from these papers that are relevant to the hypothesis. For each claim:
- State it clearly and concisely
- Indicate which paper it comes from
- Note whether it supports, refutes, or is neutral to the hypothesis

Format each claim as a bullet point."""

        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Track cost (approximate)
            self.total_cost += 0.015  # Rough estimate

            claims_text = response.content[0].text

            # Split into individual claims
            claims = [
                line.strip().lstrip("-•*").strip()
                for line in claims_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            return claims[:10]  # Limit to 10 claims

        except Exception as e:
            logger.error(f"Error extracting claims with Claude: {e}")
            return []

    async def _synthesize_topic_insights(
        self,
        papers: List[Paper],
        topic: str
    ) -> List[str]:
        """
        Use Claude to synthesize key insights about a topic from papers.

        Args:
            papers: List of papers
            topic: The research topic

        Returns:
            List of insight strings
        """
        if not self.anthropic_client:
            return []

        papers_text = "\n\n".join([
            f"Paper {i+1}: {p.title} ({p.year}) [Source: {p.source}]\n"
            f"Authors: {p.authors}\n"
            f"Citations: {p.citation_count if p.source == 'semantic_scholar' else 'N/A (arXiv)'}\n"
            f"Abstract: {p.abstract or 'No abstract available'}"
            for i, p in enumerate(papers) if p.abstract
        ])

        prompt = f"""Analyze these research papers on the topic of "{topic}":

{papers_text}

Provide 3-5 key insights or trends from this literature. Focus on:
- Common themes or findings across papers
- Important methodologies or approaches
- Current state of knowledge
- Gaps or contradictions

Format each insight as a clear bullet point."""

        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            self.total_cost += 0.015

            insights_text = response.content[0].text
            insights = [
                line.strip().lstrip("-•*").strip()
                for line in insights_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            return insights[:8]

        except Exception as e:
            logger.error(f"Error synthesizing insights with Claude: {e}")
            return []
