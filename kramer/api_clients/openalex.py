"""OpenAlex API client for literature search."""

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

from kramer.api_clients.semantic_scholar import PaperMetadata


class OpenAlexClient:
    """
    Client for OpenAlex API.

    API Documentation: https://docs.openalex.org/
    No authentication required, but include email for "polite pool" (faster responses).
    Rate limits: ~100K requests/day recommended.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        email: str = "research@capella.edu",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize OpenAlex client.

        Args:
            email: Contact email for polite pool access
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.email = email
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": f"Kramer/0.1.0 (mailto:{email})"},
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
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP GET request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.TimeoutException: For timeouts
        """
        url = f"{self.BASE_URL}{endpoint}"

        # Add email for polite pool
        if params is None:
            params = {}
        params["mailto"] = self.email

        # Add small delay to respect rate limits
        await asyncio.sleep(0.1)

        response = await self.client.get(url, params=params)

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"OpenAlex rate limited, waiting {retry_after}s...")
            await asyncio.sleep(retry_after)
            response = await self.client.get(url, params=params)

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        """
        Reconstruct abstract from OpenAlex's inverted index format.

        OpenAlex stores abstracts as {word: [positions]} mapping.
        This reconstructs the original text.

        Args:
            inverted_index: Dictionary mapping words to their positions

        Returns:
            Reconstructed abstract text, or None if no index provided
        """
        if not inverted_index:
            return None

        # Build list of (position, word) tuples
        position_word_pairs = []
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word_pairs.append((pos, word))

        # Sort by position and join
        position_word_pairs.sort(key=lambda x: x[0])
        return " ".join(word for _, word in position_word_pairs)

    @staticmethod
    def _extract_pdf_url(work: Dict[str, Any]) -> Optional[str]:
        """
        Extract best PDF URL from work data.

        Priority:
        1. best_oa_location.pdf_url
        2. open_access.oa_url (if it looks like a PDF)
        3. First location with pdf_url

        Args:
            work: OpenAlex work object

        Returns:
            PDF URL or None if not available
        """
        # Try best_oa_location first
        best_oa = work.get("best_oa_location")
        if best_oa:
            pdf_url = best_oa.get("pdf_url")
            if pdf_url:
                return pdf_url

        # Try open_access.oa_url
        open_access = work.get("open_access", {})
        oa_url = open_access.get("oa_url")
        if oa_url and oa_url.lower().endswith(".pdf"):
            return oa_url

        # Try locations array
        for location in work.get("locations", []):
            pdf_url = location.get("pdf_url")
            if pdf_url:
                return pdf_url

        return None

    def _parse_work(self, work: Dict[str, Any]) -> PaperMetadata:
        """
        Parse OpenAlex work into PaperMetadata.

        Args:
            work: OpenAlex work object

        Returns:
            PaperMetadata instance
        """
        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)

        # Extract IDs
        ids = work.get("ids", {})
        doi = ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None

        # Extract year from publication_date
        pub_date = work.get("publication_date", "")
        year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

        # Get venue from primary_location
        venue = None
        primary_location = work.get("primary_location")
        if primary_location:
            source = primary_location.get("source")
            if source:
                venue = source.get("display_name")

        # Reconstruct abstract
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))

        # Get PDF URL
        pdf_url = self._extract_pdf_url(work)

        return PaperMetadata(
            paper_id=work.get("id", "").replace("https://openalex.org/", ""),
            title=work.get("title", ""),
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            citation_count=work.get("cited_by_count", 0),
            influential_citation_count=0,  # OpenAlex doesn't have this metric
            url=pdf_url or work.get("doi"),
            venue=venue
        )

    async def search_works(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, str]] = None
    ) -> List[PaperMetadata]:
        """
        Search for works by query string.

        Args:
            query: Search query (searches title, abstract, fulltext)
            limit: Maximum number of results (max 200 per page)
            filters: Optional filters (e.g., {"publication_year": ">2020"})

        Returns:
            List of paper metadata
        """
        params = {
            "search": query,
            "per_page": min(limit, 200),
            "select": "id,title,authorships,publication_date,ids,abstract_inverted_index,"
                      "cited_by_count,open_access,best_oa_location,locations,primary_location"
        }

        # Add filters
        if filters:
            filter_parts = [f"{k}:{v}" for k, v in filters.items()]
            params["filter"] = ",".join(filter_parts)

        try:
            data = await self._make_request("/works", params=params)

            papers = []
            for work in data.get("results", []):
                try:
                    paper = self._parse_work(work)
                    papers.append(paper)
                except Exception as e:
                    print(f"Warning: Failed to parse OpenAlex work: {e}")
                    continue

            return papers

        except httpx.HTTPStatusError as e:
            print(f"OpenAlex search failed: {e}")
            return []

    async def get_work(self, work_id: str) -> Optional[PaperMetadata]:
        """
        Get details for a specific work.

        Args:
            work_id: OpenAlex work ID (e.g., "W2741809807") or full URL

        Returns:
            Paper metadata or None if not found
        """
        # Normalize ID
        if work_id.startswith("https://openalex.org/"):
            work_id = work_id.replace("https://openalex.org/", "")

        params = {
            "select": "id,title,authorships,publication_date,ids,abstract_inverted_index,"
                      "cited_by_count,open_access,best_oa_location,locations,primary_location"
        }

        try:
            data = await self._make_request(f"/works/{work_id}", params=params)
            return self._parse_work(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_work_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """
        Get work by DOI.

        Args:
            doi: DOI string (with or without https://doi.org/ prefix)

        Returns:
            Paper metadata or None if not found
        """
        # Normalize DOI
        if not doi.startswith("https://doi.org/"):
            doi = f"https://doi.org/{doi}"

        return await self.get_work(doi)
