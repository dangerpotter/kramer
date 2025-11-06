"""Semantic Scholar API client for literature search."""

import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


@dataclass
class PaperMetadata:
    """Metadata for a scientific paper."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]
    citation_count: int = 0
    influential_citation_count: int = 0
    url: Optional[str] = None
    venue: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "PaperMetadata":
        """Parse paper metadata from Semantic Scholar API response."""
        authors = [
            author.get("name", "Unknown")
            for author in data.get("authors", [])
        ]

        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            doi=data.get("externalIds", {}).get("DOI"),
            abstract=data.get("abstract"),
            citation_count=data.get("citationCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            url=data.get("url"),
            venue=data.get("venue"),
        )


class SemanticScholarClient:
    """
    Client for Semantic Scholar API.

    API Documentation: https://api.semanticscholar.org/api-docs/
    Rate limits: No API key required for basic usage, but rate limited.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize Semantic Scholar client.

        Args:
            api_key: Optional API key for higher rate limits
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        headers = {"User-Agent": "Kramer/0.1.0"}
        if api_key:
            headers["x-api-key"] = api_key

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            follow_redirects=True
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json: JSON body

        Returns:
            Response data as dictionary

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.TimeoutException: For timeouts
        """
        url = f"{self.BASE_URL}{endpoint}"

        response = await self.client.request(
            method=method,
            url=url,
            params=params,
            json=json
        )
        response.raise_for_status()
        return response.json()

    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
        year_filter: Optional[str] = None
    ) -> List[PaperMetadata]:
        """
        Search for papers by query string.

        Args:
            query: Search query
            limit: Maximum number of results (max 100)
            fields: List of fields to return (default: all useful fields)
            year_filter: Year filter string (e.g., "2020-", "2015-2020")

        Returns:
            List of paper metadata
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "authors",
                "year",
                "abstract",
                "citationCount",
                "influentialCitationCount",
                "externalIds",
                "url",
                "venue"
            ]

        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(fields)
        }

        if year_filter:
            params["year"] = year_filter

        # Add small delay to respect rate limits
        await asyncio.sleep(0.1)

        data = await self._make_request("GET", "/paper/search", params=params)

        papers = []
        for item in data.get("data", []):
            try:
                paper = PaperMetadata.from_api_response(item)
                papers.append(paper)
            except Exception as e:
                # Skip papers that fail to parse
                print(f"Warning: Failed to parse paper: {e}")
                continue

        return papers

    async def get_paper_details(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None
    ) -> Optional[PaperMetadata]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID or DOI
            fields: List of fields to return

        Returns:
            Paper metadata or None if not found
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "authors",
                "year",
                "abstract",
                "citationCount",
                "influentialCitationCount",
                "externalIds",
                "url",
                "venue"
            ]

        params = {"fields": ",".join(fields)}

        # Add small delay to respect rate limits
        await asyncio.sleep(0.1)

        try:
            data = await self._make_request("GET", f"/paper/{paper_id}", params=params)
            return PaperMetadata.from_api_response(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100
    ) -> List[PaperMetadata]:
        """
        Get papers that cite a given paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to return

        Returns:
            List of citing papers
        """
        params = {
            "limit": min(limit, 1000),
            "fields": "paperId,title,authors,year,citationCount"
        }

        await asyncio.sleep(0.1)

        data = await self._make_request(
            "GET",
            f"/paper/{paper_id}/citations",
            params=params
        )

        papers = []
        for item in data.get("data", []):
            citing_paper_data = item.get("citingPaper", {})
            try:
                paper = PaperMetadata.from_api_response(citing_paper_data)
                papers.append(paper)
            except Exception:
                continue

        return papers
