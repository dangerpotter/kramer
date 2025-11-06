"""Tests for Semantic Scholar API client."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from kramer.api_clients.semantic_scholar import (
    SemanticScholarClient,
    PaperMetadata
)


class TestPaperMetadata:
    """Tests for PaperMetadata class."""

    def test_from_api_response(self):
        """Test parsing paper metadata from API response."""
        api_data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "authors": [
                {"name": "Alice Smith"},
                {"name": "Bob Jones"}
            ],
            "year": 2023,
            "abstract": "This is a test abstract.",
            "citationCount": 10,
            "influentialCitationCount": 5,
            "externalIds": {"DOI": "10.1234/test"},
            "url": "https://example.com/paper",
            "venue": "Test Conference"
        }

        paper = PaperMetadata.from_api_response(api_data)

        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.year == 2023
        assert paper.abstract == "This is a test abstract."
        assert paper.citation_count == 10
        assert paper.influential_citation_count == 5
        assert paper.doi == "10.1234/test"
        assert paper.url == "https://example.com/paper"
        assert paper.venue == "Test Conference"

    def test_from_api_response_missing_fields(self):
        """Test parsing with missing optional fields."""
        api_data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "authors": []
        }

        paper = PaperMetadata.from_api_response(api_data)

        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert paper.authors == []
        assert paper.year is None
        assert paper.doi is None
        assert paper.abstract is None


class TestSemanticScholarClient:
    """Tests for SemanticScholarClient."""

    def test_init(self):
        """Test client initialization."""
        client = SemanticScholarClient()
        assert client.timeout == 30.0
        assert client.max_retries == 3

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = SemanticScholarClient(api_key="test-key")
        assert "x-api-key" in client.client.headers
        assert client.client.headers["x-api-key"] == "test-key"

    @pytest.mark.asyncio
    async def test_search_papers_mock(self):
        """Test searching for papers with mocked response."""
        client = SemanticScholarClient()

        mock_response = {
            "data": [
                {
                    "paperId": "paper1",
                    "title": "Test Paper 1",
                    "authors": [{"name": "Author 1"}],
                    "year": 2023,
                    "abstract": "Abstract 1",
                    "citationCount": 10,
                    "influentialCitationCount": 5,
                    "externalIds": {"DOI": "10.1234/1"},
                    "url": "https://example.com/1",
                    "venue": "Conference 1"
                },
                {
                    "paperId": "paper2",
                    "title": "Test Paper 2",
                    "authors": [{"name": "Author 2"}],
                    "year": 2022,
                    "abstract": "Abstract 2",
                    "citationCount": 20,
                    "influentialCitationCount": 10,
                    "externalIds": {"DOI": "10.1234/2"},
                    "url": "https://example.com/2",
                    "venue": "Conference 2"
                }
            ]
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            papers = await client.search_papers("test query", limit=10)

            assert len(papers) == 2
            assert papers[0].title == "Test Paper 1"
            assert papers[1].title == "Test Paper 2"
            assert papers[0].authors == ["Author 1"]

        await client.close()

    @pytest.mark.asyncio
    async def test_search_papers_empty_results(self):
        """Test searching with no results."""
        client = SemanticScholarClient()

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": []}

            papers = await client.search_papers("nonexistent query")

            assert len(papers) == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_get_paper_details_mock(self):
        """Test getting paper details with mocked response."""
        client = SemanticScholarClient()

        mock_response = {
            "paperId": "paper1",
            "title": "Test Paper",
            "authors": [{"name": "Author"}],
            "year": 2023,
            "abstract": "Test abstract",
            "citationCount": 5,
            "influentialCitationCount": 2,
            "externalIds": {"DOI": "10.1234/test"},
            "url": "https://example.com/paper",
            "venue": "Test Venue"
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            paper = await client.get_paper_details("paper1")

            assert paper is not None
            assert paper.paper_id == "paper1"
            assert paper.title == "Test Paper"

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as context manager."""
        async with SemanticScholarClient() as client:
            assert client.client is not None
