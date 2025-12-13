"""Tests for OpenAlex API client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from kramer.api_clients.openalex import OpenAlexClient
from kramer.api_clients.semantic_scholar import PaperMetadata


class TestOpenAlexClient:
    """Tests for OpenAlexClient."""

    @pytest.fixture
    def client(self):
        """Create OpenAlex client for testing."""
        return OpenAlexClient(email="test@example.com")

    def test_reconstruct_abstract(self):
        """Test abstract reconstruction from inverted index."""
        # Simple inverted index
        inverted_index = {
            "This": [0],
            "is": [1],
            "a": [2],
            "test": [3],
            "abstract": [4]
        }

        result = OpenAlexClient._reconstruct_abstract(inverted_index)
        assert result == "This is a test abstract"

    def test_reconstruct_abstract_with_repeated_words(self):
        """Test abstract reconstruction with repeated words."""
        inverted_index = {
            "The": [0, 4],
            "cat": [1],
            "sat": [2],
            "on": [3],
            "mat": [5]
        }

        result = OpenAlexClient._reconstruct_abstract(inverted_index)
        assert result == "The cat sat on The mat"

    def test_reconstruct_abstract_empty(self):
        """Test abstract reconstruction with empty/None input."""
        assert OpenAlexClient._reconstruct_abstract(None) is None
        assert OpenAlexClient._reconstruct_abstract({}) is None

    def test_extract_pdf_url_from_best_oa_location(self):
        """Test PDF URL extraction from best_oa_location."""
        work = {
            "best_oa_location": {
                "pdf_url": "https://example.com/paper.pdf"
            },
            "open_access": {
                "oa_url": "https://example.com/landing"
            }
        }

        result = OpenAlexClient._extract_pdf_url(work)
        assert result == "https://example.com/paper.pdf"

    def test_extract_pdf_url_from_oa_url(self):
        """Test PDF URL extraction from open_access.oa_url."""
        work = {
            "best_oa_location": None,
            "open_access": {
                "oa_url": "https://example.com/paper.pdf"
            }
        }

        result = OpenAlexClient._extract_pdf_url(work)
        assert result == "https://example.com/paper.pdf"

    def test_extract_pdf_url_from_locations(self):
        """Test PDF URL extraction from locations array."""
        work = {
            "best_oa_location": None,
            "open_access": {},
            "locations": [
                {"pdf_url": None},
                {"pdf_url": "https://example.com/paper2.pdf"}
            ]
        }

        result = OpenAlexClient._extract_pdf_url(work)
        assert result == "https://example.com/paper2.pdf"

    def test_extract_pdf_url_not_found(self):
        """Test PDF URL extraction when not available."""
        work = {
            "best_oa_location": None,
            "open_access": {},
            "locations": []
        }

        result = OpenAlexClient._extract_pdf_url(work)
        assert result is None

    def test_parse_work(self, client):
        """Test parsing OpenAlex work into PaperMetadata."""
        work = {
            "id": "https://openalex.org/W2741809807",
            "title": "Test Paper Title",
            "authorships": [
                {"author": {"display_name": "John Doe"}},
                {"author": {"display_name": "Jane Smith"}}
            ],
            "publication_date": "2023-06-15",
            "ids": {
                "doi": "https://doi.org/10.1234/test.123",
                "pmid": "12345678"
            },
            "abstract_inverted_index": {
                "Test": [0],
                "abstract": [1]
            },
            "cited_by_count": 42,
            "primary_location": {
                "source": {"display_name": "Nature"}
            },
            "best_oa_location": {
                "pdf_url": "https://example.com/paper.pdf"
            }
        }

        paper = client._parse_work(work)

        assert isinstance(paper, PaperMetadata)
        assert paper.paper_id == "W2741809807"
        assert paper.title == "Test Paper Title"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.year == 2023
        assert paper.doi == "10.1234/test.123"
        assert paper.abstract == "Test abstract"
        assert paper.citation_count == 42
        assert paper.venue == "Nature"
        assert paper.url == "https://example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_search_works(self, client):
        """Test search_works method."""
        mock_response = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "authorships": [{"author": {"display_name": "Author"}}],
                    "publication_date": "2023-01-01",
                    "ids": {"doi": "10.1234/test"},
                    "abstract_inverted_index": {"test": [0]},
                    "cited_by_count": 10,
                    "primary_location": None,
                    "best_oa_location": None,
                    "open_access": {},
                    "locations": []
                }
            ]
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            papers = await client.search_works("machine learning", limit=10)

            assert len(papers) == 1
            assert papers[0].paper_id == "W123"
            assert papers[0].title == "Test Paper"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_work(self, client):
        """Test get_work method."""
        mock_response = {
            "id": "https://openalex.org/W123",
            "title": "Single Paper",
            "authorships": [],
            "publication_date": "2023-01-01",
            "ids": {},
            "abstract_inverted_index": None,
            "cited_by_count": 5,
            "primary_location": None,
            "best_oa_location": None,
            "open_access": {},
            "locations": []
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            paper = await client.get_work("W123")

            assert paper is not None
            assert paper.title == "Single Paper"

    @pytest.mark.asyncio
    async def test_get_work_not_found(self, client):
        """Test get_work method when paper not found."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = httpx.HTTPStatusError(
                "Not Found",
                request=MagicMock(),
                response=MagicMock(status_code=404)
            )

            paper = await client.get_work("W999999999")
            assert paper is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with OpenAlexClient(email="test@example.com") as client:
            assert client is not None
            assert hasattr(client, 'client')
