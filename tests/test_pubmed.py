"""Tests for PubMed E-utilities API client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from kramer.api_clients.pubmed import PubMedClient
from kramer.api_clients.semantic_scholar import PaperMetadata


class TestPubMedClient:
    """Tests for PubMedClient."""

    @pytest.fixture
    def client(self):
        """Create PubMed client for testing."""
        return PubMedClient(email="test@example.com")

    @pytest.fixture
    def sample_esearch_response(self):
        """Sample ESearch XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <eSearchResult>
            <Count>3</Count>
            <RetMax>3</RetMax>
            <IdList>
                <Id>12345678</Id>
                <Id>23456789</Id>
                <Id>34567890</Id>
            </IdList>
        </eSearchResult>"""

    @pytest.fixture
    def sample_efetch_response(self):
        """Sample EFetch XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                            <Author>
                                <LastName>Doe</LastName>
                                <ForeName>Jane</ForeName>
                            </Author>
                        </AuthorList>
                        <Journal>
                            <Title>Nature Medicine</Title>
                        </Journal>
                        <Abstract>
                            <AbstractText Label="BACKGROUND">This is the background.</AbstractText>
                            <AbstractText Label="METHODS">These are the methods.</AbstractText>
                            <AbstractText Label="RESULTS">These are the results.</AbstractText>
                        </Abstract>
                        <ArticleDate>
                            <Year>2023</Year>
                            <Month>06</Month>
                            <Day>15</Day>
                        </ArticleDate>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="pubmed">12345678</ArticleId>
                        <ArticleId IdType="doi">10.1234/test.123</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>"""

    def test_get_base_params(self, client):
        """Test base parameters generation."""
        params = client._get_base_params()

        assert "tool" in params
        assert "email" in params
        assert params["email"] == "test@example.com"
        assert params["tool"] == "kramer"

    def test_get_base_params_with_api_key(self):
        """Test base parameters with API key."""
        client = PubMedClient(api_key="test_key", email="test@example.com")
        params = client._get_base_params()

        assert "api_key" in params
        assert params["api_key"] == "test_key"

    @pytest.mark.asyncio
    async def test_search_pubmed(self, client, sample_esearch_response):
        """Test search_pubmed method."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_esearch_response

            pmids = await client.search_pubmed("cancer treatment", limit=10)

            assert len(pmids) == 3
            assert "12345678" in pmids
            assert "23456789" in pmids
            assert "34567890" in pmids
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_abstracts(self, client, sample_efetch_response):
        """Test fetch_abstracts method."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_efetch_response

            papers = await client.fetch_abstracts(["12345678"])

            assert len(papers) == 1
            paper = papers[0]
            assert paper.paper_id == "PMID:12345678"
            assert paper.title == "Test Article Title"
            assert "John Smith" in paper.authors
            assert "Jane Doe" in paper.authors
            assert paper.doi == "10.1234/test.123"
            assert paper.venue == "Nature Medicine"
            assert "BACKGROUND: This is the background" in paper.abstract

    @pytest.mark.asyncio
    async def test_search_and_fetch(self, client, sample_esearch_response, sample_efetch_response):
        """Test combined search_and_fetch method."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            # First call returns search results, second call returns fetch results
            mock_request.side_effect = [sample_esearch_response, sample_efetch_response]

            papers = await client.search_and_fetch("cancer treatment", limit=10)

            assert len(papers) == 1
            assert papers[0].paper_id == "PMID:12345678"

    def test_parse_pubmed_xml(self, client, sample_efetch_response):
        """Test XML parsing."""
        papers = client._parse_pubmed_xml(sample_efetch_response)

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, PaperMetadata)
        assert paper.paper_id == "PMID:12345678"

    def test_parse_pubmed_xml_empty(self, client):
        """Test XML parsing with empty response."""
        xml = """<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>"""
        papers = client._parse_pubmed_xml(xml)
        assert len(papers) == 0

    def test_parse_pubmed_xml_invalid(self, client):
        """Test XML parsing with invalid XML."""
        papers = client._parse_pubmed_xml("not valid xml")
        assert len(papers) == 0

    def test_parse_article_with_medline_date(self, client):
        """Test parsing article with MedlineDate instead of Year."""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>99999999</PMID>
                    <Article>
                        <ArticleTitle>Old Article</ArticleTitle>
                        <AuthorList></AuthorList>
                        <Journal><Title>Old Journal</Title></Journal>
                        <Abstract>
                            <AbstractText>Simple abstract.</AbstractText>
                        </Abstract>
                        <ArticleDate>
                            <MedlineDate>1998 Jan-Feb</MedlineDate>
                        </ArticleDate>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""

        papers = client._parse_pubmed_xml(xml)
        # Year extraction from MedlineDate may or may not work depending on structure
        assert len(papers) == 1

    @pytest.mark.asyncio
    async def test_check_pmc_availability_found(self, client):
        """Test PMC availability check when PMC ID exists."""
        pmc_response = """<?xml version="1.0"?>
        <eLinkResult>
            <LinkSet>
                <LinkSetDb>
                    <DbTo>pmc</DbTo>
                    <Link>
                        <Id>7654321</Id>
                    </Link>
                </LinkSetDb>
            </LinkSet>
        </eLinkResult>"""

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pmc_response

            pmcid = await client.check_pmc_availability("12345678")

            assert pmcid == "PMC7654321"

    @pytest.mark.asyncio
    async def test_check_pmc_availability_not_found(self, client):
        """Test PMC availability check when no PMC ID."""
        pmc_response = """<?xml version="1.0"?>
        <eLinkResult>
            <LinkSet>
            </LinkSet>
        </eLinkResult>"""

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pmc_response

            pmcid = await client.check_pmc_availability("12345678")

            assert pmcid is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with PubMedClient(email="test@example.com") as client:
            assert client is not None
            assert hasattr(client, 'client')

    def test_rate_limit_delay_without_key(self):
        """Test rate limit delay without API key."""
        client = PubMedClient(email="test@example.com")
        assert client.request_delay == 0.35  # 3 req/sec limit

    def test_rate_limit_delay_with_key(self):
        """Test rate limit delay with API key."""
        client = PubMedClient(api_key="test_key", email="test@example.com")
        assert client.request_delay == 0.1  # 10 req/sec limit
