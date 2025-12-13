"""CORE API client for literature search."""

import asyncio
from typing import List, Dict, Optional, Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from kramer.api_clients.semantic_scholar import PaperMetadata


class COREClient:
    """
    Client for CORE API v3.

    API Documentation: https://api.core.ac.uk/docs/v3
    Authentication: Bearer token required (register at https://core.ac.uk/services/api)
    Rate limits: Token-based, varies by user type (1K-5K tokens/day for free tiers)
    """

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize CORE client.

        Args:
            api_key: CORE API key (required)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if not api_key:
            raise ValueError("CORE API key is required")

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "Kramer/0.1.0",
                "Authorization": f"Bearer {api_key}"
            },
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
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON body for POST requests

        Returns:
            Response data as dictionary

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.TimeoutException: For timeouts
        """
        url = f"{self.BASE_URL}{endpoint}"

        # Conservative delay to respect rate limits
        await asyncio.sleep(0.5)

        if method.upper() == "GET":
            response = await self.client.get(url, params=params)
        else:
            response = await self.client.post(url, params=params, json=json_data)

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get("X-RateLimit-Retry-After")
            wait_time = int(retry_after) if retry_after else 60
            print(f"CORE rate limited, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)

            if method.upper() == "GET":
                response = await self.client.get(url, params=params)
            else:
                response = await self.client.post(url, params=params, json=json_data)

        response.raise_for_status()
        return response.json()

    def _parse_work(self, work: Dict[str, Any]) -> PaperMetadata:
        """
        Parse CORE work/output into PaperMetadata.

        Args:
            work: CORE work or output object

        Returns:
            PaperMetadata instance
        """
        # Extract authors
        authors = []
        for author in work.get("authors", []):
            if isinstance(author, dict):
                name = author.get("name")
            else:
                name = str(author)
            if name:
                authors.append(name)

        # Get year
        year = work.get("yearPublished")
        if not year:
            # Try to extract from publishedDate
            pub_date = work.get("publishedDate", "")
            if pub_date and len(pub_date) >= 4:
                try:
                    year = int(pub_date[:4])
                except ValueError:
                    pass

        # Get DOI
        doi = work.get("doi")
        if not doi:
            # Check identifiers
            identifiers = work.get("identifiers", [])
            for ident in identifiers:
                if isinstance(ident, dict) and ident.get("type") == "doi":
                    doi = ident.get("identifier")
                    break

        # Get URL (prefer downloadUrl, fallback to other URLs)
        url = work.get("downloadUrl")
        if not url:
            urls = work.get("sourceFulltextUrls") or work.get("urls") or []
            if urls:
                url = urls[0] if isinstance(urls[0], str) else urls[0].get("url")

        # Get venue from journals
        venue = None
        journals = work.get("journals", [])
        if journals:
            journal = journals[0]
            if isinstance(journal, dict):
                venue = journal.get("title")
            else:
                venue = str(journal)

        return PaperMetadata(
            paper_id=f"CORE:{work.get('id', '')}",
            title=work.get("title", ""),
            authors=authors,
            year=year,
            doi=doi,
            abstract=work.get("abstract"),
            citation_count=work.get("citationCount", 0),
            influential_citation_count=0,
            url=url,
            venue=venue
        )

    async def search_works(
        self,
        query: str,
        limit: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[PaperMetadata]:
        """
        Search CORE works (deduplicated records).

        Args:
            query: Search query (supports CORE query syntax)
            limit: Maximum number of results
            year_from: Filter by minimum year
            year_to: Filter by maximum year

        Returns:
            List of paper metadata
        """
        # Build query with year filters
        full_query = query
        if year_from:
            full_query += f" AND yearPublished>={year_from}"
        if year_to:
            full_query += f" AND yearPublished<={year_to}"

        params = {
            "q": full_query,
            "limit": min(limit, 100)
        }

        try:
            data = await self._make_request("GET", "/search/works", params=params)

            papers = []
            for work in data.get("results", []):
                try:
                    paper = self._parse_work(work)
                    papers.append(paper)
                except Exception as e:
                    print(f"Warning: Failed to parse CORE work: {e}")
                    continue

            return papers

        except httpx.HTTPStatusError as e:
            print(f"CORE search failed: {e}")
            return []

    async def search_outputs(
        self,
        query: str,
        limit: int = 10
    ) -> List[PaperMetadata]:
        """
        Search CORE outputs (individual repository items).

        Outputs are not deduplicated, so the same paper may appear
        multiple times from different repositories.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of paper metadata
        """
        params = {
            "q": query,
            "limit": min(limit, 100)
        }

        try:
            data = await self._make_request("GET", "/search/outputs", params=params)

            papers = []
            for output in data.get("results", []):
                try:
                    paper = self._parse_work(output)
                    papers.append(paper)
                except Exception as e:
                    print(f"Warning: Failed to parse CORE output: {e}")
                    continue

            return papers

        except httpx.HTTPStatusError as e:
            print(f"CORE output search failed: {e}")
            return []

    async def get_work(self, core_id: str) -> Optional[PaperMetadata]:
        """
        Get work by CORE ID.

        Args:
            core_id: CORE work ID (numeric string)

        Returns:
            Paper metadata or None if not found
        """
        # Normalize ID
        if core_id.startswith("CORE:"):
            core_id = core_id[5:]

        try:
            data = await self._make_request("GET", f"/works/{core_id}")
            return self._parse_work(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_fulltext(self, core_id: str) -> Optional[str]:
        """
        Get full text content for a work if available.

        CORE often includes full text directly in the API response.

        Args:
            core_id: CORE work ID

        Returns:
            Full text content or None if not available
        """
        # Normalize ID
        if core_id.startswith("CORE:"):
            core_id = core_id[5:]

        try:
            # Fetch the work with full text
            data = await self._make_request("GET", f"/works/{core_id}")
            return data.get("fullText")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"Failed to get CORE full text: {e}")
            return None

    async def get_output_fulltext(self, output_id: str) -> Optional[str]:
        """
        Get full text content for an output.

        Args:
            output_id: CORE output ID

        Returns:
            Full text content or None if not available
        """
        try:
            data = await self._make_request("GET", f"/outputs/{output_id}")
            return data.get("fullText")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"Failed to get CORE output full text: {e}")
            return None

    async def search_with_fulltext(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search works and return those with full text available.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of dicts with 'paper' (PaperMetadata) and 'fulltext' (str) keys
        """
        # Search for works that have full text
        full_query = f"({query}) AND _exists_:fullText"

        params = {
            "q": full_query,
            "limit": min(limit, 100)
        }

        try:
            data = await self._make_request("GET", "/search/works", params=params)

            results = []
            for work in data.get("results", []):
                try:
                    paper = self._parse_work(work)
                    fulltext = work.get("fullText")
                    if fulltext:
                        results.append({
                            "paper": paper,
                            "fulltext": fulltext
                        })
                except Exception as e:
                    print(f"Warning: Failed to parse CORE work: {e}")
                    continue

            return results

        except httpx.HTTPStatusError as e:
            print(f"CORE fulltext search failed: {e}")
            return []
