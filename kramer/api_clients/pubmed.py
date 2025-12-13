"""PubMed E-utilities API client for literature search."""

import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from kramer.api_clients.semantic_scholar import PaperMetadata


class PubMedClient:
    """
    Client for PubMed E-utilities API.

    API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    Rate limits: 3 requests/second without API key, 10/sec with key.
    Required params: tool, email
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: str = "research@capella.edu",
        tool: str = "kramer",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize PubMed client.

        Args:
            api_key: Optional NCBI API key for higher rate limits
            email: Contact email (required by NCBI)
            tool: Tool name for identification (required by NCBI)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.email = email
        self.tool = tool
        self.timeout = timeout
        self.max_retries = max_retries

        # Rate limit: 3/sec without key, 10/sec with key
        self.request_delay = 0.35 if not api_key else 0.1

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": f"{tool}/0.1.0 (mailto:{email})"},
            follow_redirects=True
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _get_base_params(self) -> Dict[str, str]:
        """Get base parameters required for all NCBI requests."""
        params = {
            "tool": self.tool,
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Make HTTP GET request with retry logic.

        Args:
            endpoint: E-utility endpoint (esearch.fcgi, efetch.fcgi, etc.)
            params: Query parameters

        Returns:
            Response text (XML or other format)

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.TimeoutException: For timeouts
        """
        url = f"{self.EUTILS_BASE}/{endpoint}"

        # Add base params
        all_params = {**self._get_base_params(), **params}

        # Respect rate limits
        await asyncio.sleep(self.request_delay)

        response = await self.client.get(url, params=all_params)

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"PubMed rate limited, waiting {retry_after}s...")
            await asyncio.sleep(retry_after)
            response = await self.client.get(url, params=all_params)

        response.raise_for_status()
        return response.text

    async def search_pubmed(
        self,
        query: str,
        limit: int = 10,
        sort: str = "relevance"
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs.

        Args:
            query: Search query (supports PubMed query syntax)
            limit: Maximum number of results (max 10000)
            sort: Sort order ("relevance", "pub_date", "author")

        Returns:
            List of PubMed IDs (PMIDs)
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(limit, 10000),
            "sort": sort,
            "retmode": "xml"
        }

        try:
            xml_text = await self._make_request("esearch.fcgi", params)
            root = ET.fromstring(xml_text)

            pmids = []
            id_list = root.find("IdList")
            if id_list is not None:
                for id_elem in id_list.findall("Id"):
                    if id_elem.text:
                        pmids.append(id_elem.text)

            return pmids

        except ET.ParseError as e:
            print(f"Failed to parse PubMed search response: {e}")
            return []
        except httpx.HTTPStatusError as e:
            print(f"PubMed search failed: {e}")
            return []

    async def fetch_abstracts(self, pmids: List[str]) -> List[PaperMetadata]:
        """
        Fetch paper details for list of PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of paper metadata
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml"
        }

        try:
            xml_text = await self._make_request("efetch.fcgi", params)
            return self._parse_pubmed_xml(xml_text)
        except ET.ParseError as e:
            print(f"Failed to parse PubMed fetch response: {e}")
            return []
        except httpx.HTTPStatusError as e:
            print(f"PubMed fetch failed: {e}")
            return []

    def _parse_pubmed_xml(self, xml_text: str) -> List[PaperMetadata]:
        """
        Parse PubMed XML response into PaperMetadata objects.

        Args:
            xml_text: XML response from efetch

        Returns:
            List of paper metadata
        """
        papers = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                print(f"Warning: Failed to parse PubMed article: {e}")
                continue

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[PaperMetadata]:
        """
        Parse single PubMed article XML element.

        Args:
            article: PubmedArticle XML element

        Returns:
            PaperMetadata or None if parsing fails
        """
        medline = article.find("MedlineCitation")
        if medline is None:
            return None

        # Get PMID
        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None
        if not pmid:
            return None

        article_data = medline.find("Article")
        if article_data is None:
            return None

        # Title
        title_elem = article_data.find("ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Authors
        authors = []
        author_list = article_data.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                if last_name is not None and last_name.text:
                    name = last_name.text
                    if fore_name is not None and fore_name.text:
                        name = f"{fore_name.text} {last_name.text}"
                    authors.append(name)

        # Year
        year = None
        pub_date = article_data.find(".//PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass
            # Try MedlineDate if Year not found
            if year is None:
                medline_date = pub_date.find("MedlineDate")
                if medline_date is not None and medline_date.text:
                    # Extract first 4 digits as year
                    import re
                    match = re.search(r'\d{4}', medline_date.text)
                    if match:
                        year = int(match.group())

        # Abstract
        abstract = None
        abstract_elem = article_data.find("Abstract")
        if abstract_elem is not None:
            abstract_texts = []
            for abstract_text in abstract_elem.findall("AbstractText"):
                if abstract_text.text:
                    # Handle labeled sections
                    label = abstract_text.get("Label")
                    if label:
                        abstract_texts.append(f"{label}: {abstract_text.text}")
                    else:
                        abstract_texts.append(abstract_text.text)
            abstract = " ".join(abstract_texts) if abstract_texts else None

        # DOI
        doi = None
        article_id_list = article.find(".//ArticleIdList")
        if article_id_list is not None:
            for article_id in article_id_list.findall("ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

        # Venue (Journal)
        venue = None
        journal = article_data.find("Journal")
        if journal is not None:
            journal_title = journal.find("Title")
            if journal_title is not None:
                venue = journal_title.text

        return PaperMetadata(
            paper_id=f"PMID:{pmid}",
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            citation_count=0,  # PubMed doesn't provide citation counts
            influential_citation_count=0,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            venue=venue
        )

    async def search_and_fetch(
        self,
        query: str,
        limit: int = 10
    ) -> List[PaperMetadata]:
        """
        Combined search and fetch operation.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of paper metadata with abstracts
        """
        pmids = await self.search_pubmed(query, limit)
        if not pmids:
            return []

        return await self.fetch_abstracts(pmids)

    async def check_pmc_availability(self, pmid: str) -> Optional[str]:
        """
        Check if a PMID has full text available in PMC.

        Args:
            pmid: PubMed ID

        Returns:
            PMCID if available, None otherwise
        """
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "xml"
        }

        try:
            xml_text = await self._make_request("elink.fcgi", params)
            root = ET.fromstring(xml_text)

            # Look for PMC link
            link_set = root.find(".//LinkSetDb")
            if link_set is not None:
                link = link_set.find(".//Link/Id")
                if link is not None and link.text:
                    return f"PMC{link.text}"

            return None

        except Exception as e:
            print(f"Failed to check PMC availability: {e}")
            return None

    async def get_pmc_fulltext(self, pmcid: str) -> Optional[str]:
        """
        Get full text from PMC (open access articles only).

        Args:
            pmcid: PMC ID (with or without "PMC" prefix)

        Returns:
            Full text content or None if not available
        """
        # Normalize PMCID
        if pmcid.startswith("PMC"):
            pmcid = pmcid[3:]

        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "full",
            "retmode": "xml"
        }

        try:
            xml_text = await self._make_request("efetch.fcgi", params)
            return self._extract_text_from_pmc_xml(xml_text)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Article might not be open access
                return None
            raise
        except Exception as e:
            print(f"Failed to get PMC full text: {e}")
            return None

    def _extract_text_from_pmc_xml(self, xml_text: str) -> Optional[str]:
        """
        Extract plain text from PMC full-text XML.

        Args:
            xml_text: PMC XML content

        Returns:
            Extracted plain text
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None

        # Find body content
        body = root.find(".//body")
        if body is None:
            return None

        # Extract all text content
        text_parts = []

        def extract_text(elem):
            if elem.text:
                text_parts.append(elem.text.strip())
            for child in elem:
                extract_text(child)
                if child.tail:
                    text_parts.append(child.tail.strip())

        extract_text(body)

        return " ".join(filter(None, text_parts))
