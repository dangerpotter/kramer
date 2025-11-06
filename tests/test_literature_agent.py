"""Tests for Literature Agent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from kramer.agents.literature import LiteratureAgent, ExtractedClaim
from kramer.world_model import WorldModel, NodeType
from kramer.api_clients.semantic_scholar import PaperMetadata


class TestExtractedClaim:
    """Tests for ExtractedClaim dataclass."""

    def test_create_claim(self):
        """Test creating an extracted claim."""
        claim = ExtractedClaim(
            claim_text="This is a key finding",
            paper_title="Test Paper",
            authors=["Alice", "Bob"],
            year=2023,
            doi="10.1234/test",
            confidence=0.9,
            paper_id="paper123"
        )

        assert claim.claim_text == "This is a key finding"
        assert claim.paper_title == "Test Paper"
        assert claim.authors == ["Alice", "Bob"]
        assert claim.confidence == 0.9


class TestLiteratureAgent:
    """Tests for LiteratureAgent."""

    def test_init(self):
        """Test agent initialization."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        assert agent.world_model == world
        assert agent.model == "claude-3-5-sonnet-20241022"

    def test_add_paper_to_world_model(self):
        """Test adding a paper to the world model."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        paper = PaperMetadata(
            paper_id="paper123",
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2023,
            doi="10.1234/test",
            abstract="This is a test abstract",
            citation_count=10,
            influential_citation_count=5,
            url="https://example.com/paper",
            venue="Test Conference"
        )

        node = agent._add_paper_to_world_model(paper)

        assert node.type == NodeType.PAPER
        assert node.content == "Test Paper"
        assert node.metadata["authors"] == ["Alice", "Bob"]
        assert node.metadata["year"] == 2023
        assert node.metadata["doi"] == "10.1234/test"
        assert node.metadata["paper_id"] == "paper123"

        # Check it was added to world model
        assert len(world.get_nodes_by_type(NodeType.PAPER)) == 1

    def test_parse_claims_from_response(self):
        """Test parsing claims from Claude's response."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        paper = PaperMetadata(
            paper_id="paper123",
            title="Test Paper",
            authors=["Alice"],
            year=2023,
            doi="10.1234/test",
            abstract="Abstract"
        )

        response = """
CLAIM 1:
Text: Protein X is involved in DNA repair.
Confidence: 0.9

CLAIM 2:
Text: The mutation increases enzyme activity by 50%.
Confidence: 0.8

CLAIM 3:
Text: The pathway is conserved across species.
Confidence: 0.7
"""

        claims = agent._parse_claims_from_response(response, paper)

        assert len(claims) == 3
        assert claims[0].claim_text == "Protein X is involved in DNA repair."
        assert claims[0].confidence == 0.9
        assert claims[1].claim_text == "The mutation increases enzyme activity by 50%."
        assert claims[1].confidence == 0.8
        assert claims[2].confidence == 0.7

    def test_format_citation(self):
        """Test formatting citations."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        claim = ExtractedClaim(
            claim_text="Test claim",
            paper_title="A Study of Something Important",
            authors=["Smith", "Jones", "Brown"],
            year=2023,
            doi="10.1234/test",
            confidence=0.9,
            paper_id="paper123"
        )

        citation = agent.format_citation(claim)

        assert "Smith, Jones, Brown" in citation
        assert "(2023)" in citation
        assert "A Study of Something Important" in citation
        assert "10.1234/test" in citation

    def test_format_citation_many_authors(self):
        """Test formatting citations with many authors."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        claim = ExtractedClaim(
            claim_text="Test claim",
            paper_title="Test Paper",
            authors=["Author1", "Author2", "Author3", "Author4", "Author5"],
            year=2023,
            doi="10.1234/test",
            confidence=0.9,
            paper_id="paper123"
        )

        citation = agent.format_citation(claim)

        assert "et al." in citation
        assert "Author1, Author2, Author3" in citation

    def test_build_extraction_prompt(self):
        """Test building the extraction prompt."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        paper = PaperMetadata(
            paper_id="paper123",
            title="Test Paper",
            authors=["Alice"],
            year=2023,
            doi=None,
            abstract="This is a test abstract about DNA repair."
        )

        prompt = agent._build_extraction_prompt(paper, hypotheses=["Hypothesis 1"])

        assert "Test Paper" in prompt
        assert "Alice" in prompt
        assert "2023" in prompt
        assert "This is a test abstract about DNA repair." in prompt
        assert "Hypothesis 1" in prompt
        assert "3-5 key factual claims" in prompt

    @pytest.mark.asyncio
    async def test_search_and_extract_integration(self):
        """Test full search and extract workflow with mocks."""
        world = WorldModel()
        agent = LiteratureAgent(
            world_model=world,
            anthropic_api_key="test-key"
        )

        # Mock paper
        mock_paper = PaperMetadata(
            paper_id="paper123",
            title="Test Paper",
            authors=["Alice"],
            year=2023,
            doi="10.1234/test",
            abstract="This paper describes important findings about DNA repair mechanisms.",
            citation_count=10
        )

        # Mock Semantic Scholar response
        with patch.object(agent.ss_client, 'search_papers', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_paper]

            # Mock Claude response
            mock_response = Mock()
            mock_response.content = [Mock(text="""
CLAIM 1:
Text: DNA repair is important for cellular health.
Confidence: 0.9

CLAIM 2:
Text: The mechanism involves multiple proteins.
Confidence: 0.8
""")]

            with patch.object(agent.anthropic_client.messages, 'create') as mock_claude:
                mock_claude.return_value = mock_response

                claims = await agent.search_and_extract("DNA repair", max_papers=1)

                assert len(claims) == 2
                assert claims[0].claim_text == "DNA repair is important for cellular health."
                assert claims[1].confidence == 0.8

                # Check world model was updated
                papers = world.get_nodes_by_type(NodeType.PAPER)
                assert len(papers) == 1

                claim_nodes = world.get_nodes_by_type(NodeType.CLAIM)
                assert len(claim_nodes) == 2

                findings = world.get_findings_for_paper(papers[0].id)
                assert len(findings) == 2
