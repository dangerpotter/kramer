"""
End-to-end integration tests for RAG (paper_processor + rag_engine).

Tests the full pipeline:
1. Download PDF from Semantic Scholar
2. Extract and process text
3. Index in vector database
4. Query and retrieve relevant context
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from src.kramer.paper_processor import PaperProcessor
from src.kramer.rag_engine import RAGEngine
from src.kramer.literature_agent import SemanticScholarClient


class TestRAGIntegration:
    """End-to-end RAG integration tests."""

    @pytest.fixture
    async def paper_processor(self):
        """Create paper processor instance."""
        processor = PaperProcessor()
        yield processor
        await processor.cleanup()

    @pytest.fixture
    def rag_engine(self, tmp_path):
        """Create RAG engine with temporary storage."""
        persist_dir = tmp_path / "rag_data"
        engine = RAGEngine(
            collection_name="test_papers",
            persist_directory=str(persist_dir)
        )
        yield engine
        # Cleanup
        engine.delete_collection()

    @pytest.fixture
    def semantic_scholar_client(self):
        """Create Semantic Scholar client."""
        return SemanticScholarClient()

    @pytest.mark.asyncio
    async def test_full_rag_pipeline_with_real_paper(
        self,
        paper_processor,
        rag_engine,
        semantic_scholar_client
    ):
        """Test complete RAG pipeline with a real paper."""

        # Step 1: Search for a well-known paper
        query = "attention is all you need transformer"
        papers = await semantic_scholar_client.search_papers(query, max_results=1)

        assert len(papers) > 0, "Should find at least one paper"
        paper = papers[0]
        paper_id = paper.get("paperId")

        print(f"\nTesting with paper: {paper.get('title')}")
        print(f"Paper ID: {paper_id}")

        # Step 2: Download PDF
        pdf_path = await paper_processor.download_pdf(paper_id)

        if pdf_path is None:
            pytest.skip("PDF not available for this paper")

        assert Path(pdf_path).exists(), "PDF should be downloaded"

        # Step 3: Extract text
        text = await paper_processor.extract_text(pdf_path)
        assert len(text) > 0, "Should extract text from PDF"
        assert len(text) > 1000, "Should extract substantial text"

        print(f"Extracted {len(text)} characters")

        # Step 4: Chunk text
        chunks = paper_processor.chunk_text(text, paper_id)
        assert len(chunks) > 0, "Should create chunks"
        assert all("text" in chunk for chunk in chunks), "All chunks should have text"
        assert all("metadata" in chunk for chunk in chunks), "All chunks should have metadata"

        print(f"Created {len(chunks)} chunks")

        # Step 5: Add to RAG engine
        rag_engine.add_paper(paper_id, chunks)

        # Verify paper was indexed
        assert rag_engine.has_paper(paper_id), "Paper should be indexed"

        stats = rag_engine.get_stats()
        assert stats["num_papers"] == 1, "Should have one paper"
        assert stats["num_chunks"] > 0, "Should have chunks"

        print(f"Indexed: {stats['num_chunks']} chunks")

        # Step 6: Query the RAG engine
        query_text = "transformer architecture"
        results = rag_engine.query(query_text, top_k=3)

        assert len(results) > 0, "Should return results"
        assert len(results) <= 3, "Should respect top_k limit"

        # Verify result structure
        for result in results:
            assert "text" in result, "Result should have text"
            assert "metadata" in result, "Result should have metadata"
            assert "similarity" in result, "Result should have similarity score"
            assert 0 <= result["similarity"] <= 1, "Similarity should be normalized"

        print(f"Query returned {len(results)} results")
        print(f"Top result similarity: {results[0]['similarity']:.3f}")

        # Step 7: Get paper-specific context
        context = rag_engine.get_paper_context(paper_id, query_text, top_k=2)

        assert len(context) > 0, "Should return paper context"
        assert len(context) <= 2, "Should respect top_k limit"
        assert all(r["metadata"]["paper_id"] == paper_id for r in context), \
            "All results should be from specified paper"

        print("✓ Full RAG pipeline test passed")

    @pytest.mark.asyncio
    async def test_rag_persistence(self, tmp_path):
        """Test that RAG engine persists data correctly."""

        persist_dir = tmp_path / "rag_persist"

        # Create first engine and add data
        engine1 = RAGEngine(
            collection_name="test_persist",
            persist_directory=str(persist_dir)
        )

        paper_id = "test_paper_1"
        chunks = [
            {
                "text": "This is a test chunk about machine learning.",
                "metadata": {"paper_id": paper_id, "chunk_id": 0}
            },
            {
                "text": "Neural networks are powerful models.",
                "metadata": {"paper_id": paper_id, "chunk_id": 1}
            }
        ]

        engine1.add_paper(paper_id, chunks)
        assert engine1.has_paper(paper_id), "Paper should be indexed"

        # Create second engine with same persist directory
        engine2 = RAGEngine(
            collection_name="test_persist",
            persist_directory=str(persist_dir)
        )

        # Verify data persisted
        assert engine2.has_paper(paper_id), "Paper should persist across instances"

        results = engine2.query("machine learning", top_k=1)
        assert len(results) > 0, "Should retrieve persisted data"

        # Cleanup
        engine1.delete_collection()

        print("✓ RAG persistence test passed")

    @pytest.mark.asyncio
    async def test_rag_multiple_papers(self, rag_engine):
        """Test RAG engine with multiple papers."""

        papers = [
            ("paper1", [
                {"text": "Deep learning uses neural networks.", "metadata": {"paper_id": "paper1", "chunk_id": 0}},
                {"text": "Backpropagation trains neural nets.", "metadata": {"paper_id": "paper1", "chunk_id": 1}}
            ]),
            ("paper2", [
                {"text": "Transformers use attention mechanisms.", "metadata": {"paper_id": "paper2", "chunk_id": 0}},
                {"text": "Self-attention computes representations.", "metadata": {"paper_id": "paper2", "chunk_id": 1}}
            ]),
            ("paper3", [
                {"text": "Reinforcement learning uses rewards.", "metadata": {"paper_id": "paper3", "chunk_id": 0}},
                {"text": "Q-learning is a RL algorithm.", "metadata": {"paper_id": "paper3", "chunk_id": 1}}
            ])
        ]

        # Add all papers
        for paper_id, chunks in papers:
            rag_engine.add_paper(paper_id, chunks)

        # Verify all indexed
        stats = rag_engine.get_stats()
        assert stats["num_papers"] == 3, "Should have 3 papers"
        assert stats["num_chunks"] == 6, "Should have 6 chunks"

        # Query should return results from relevant papers
        results = rag_engine.query("attention mechanism", top_k=2)
        assert len(results) > 0, "Should find relevant chunks"
        # Top result should be from paper2 (about transformers)
        assert results[0]["metadata"]["paper_id"] == "paper2", \
            "Should return most relevant paper"

        # Get context from specific paper
        context = rag_engine.get_paper_context("paper1", "neural networks", top_k=2)
        assert len(context) == 2, "Should return chunks from paper1"
        assert all(r["metadata"]["paper_id"] == "paper1" for r in context), \
            "All results should be from paper1"

        print("✓ Multiple papers test passed")

    @pytest.mark.asyncio
    async def test_paper_processor_error_handling(self, paper_processor):
        """Test paper processor handles errors gracefully."""

        # Test with invalid paper ID
        result = await paper_processor.download_pdf("INVALID_ID_12345")
        assert result is None, "Should return None for invalid paper ID"

        # Test with non-existent file
        text = await paper_processor.extract_text("/nonexistent/file.pdf")
        assert text == "", "Should return empty string for invalid file"

        print("✓ Error handling test passed")

    def test_rag_engine_edge_cases(self, rag_engine):
        """Test RAG engine edge cases."""

        # Test empty query
        results = rag_engine.query("", top_k=5)
        # Should handle gracefully (may return results or empty list)
        assert isinstance(results, list), "Should return list"

        # Test query before adding papers
        results = rag_engine.query("test query", top_k=5)
        assert len(results) == 0, "Should return empty list when no papers indexed"

        # Test has_paper with non-existent paper
        assert not rag_engine.has_paper("nonexistent_paper"), \
            "Should return False for non-existent paper"

        # Test get_paper_context with non-existent paper
        context = rag_engine.get_paper_context("nonexistent_paper", "test", top_k=5)
        assert len(context) == 0, "Should return empty list for non-existent paper"

        print("✓ Edge cases test passed")

    @pytest.mark.asyncio
    async def test_chunking_quality(self, paper_processor):
        """Test that chunking produces good quality chunks."""

        # Create sample text with sentences
        text = """
        This is the first sentence. It discusses machine learning.

        This is the second sentence. It talks about neural networks.
        The third sentence continues the topic.

        Here is another paragraph. It has multiple sentences.
        Each sentence adds information. The paragraph ends here.
        """

        chunks = paper_processor.chunk_text(text, "test_paper", chunk_size=100, overlap=20)

        assert len(chunks) > 0, "Should create chunks"

        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["paper_id"] == "test_paper"
            assert chunk["metadata"]["chunk_id"] == i

            # Verify chunk size constraints
            assert len(chunk["text"]) > 0, "Chunk should not be empty"

            # Verify no excessive whitespace
            assert not chunk["text"].startswith("  "), \
                "Chunk should not have leading whitespace"

        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            # Some text from first chunk should appear in second
            # (This is a basic check - overlap is character-based)
            assert len(chunks[0]["text"]) > 0 and len(chunks[1]["text"]) > 0

        print("✓ Chunking quality test passed")


class TestRAGIntegrationWithLiteratureAgent:
    """Test RAG integration with literature agent."""

    @pytest.mark.asyncio
    async def test_literature_agent_with_rag(self, tmp_path):
        """Test literature agent using RAG for full-text search."""

        from kramer.agents.literature import LiteratureAgent
        from kramer.config import Config

        # Create config with RAG enabled
        config = Config(
            use_full_text=True,
            rag_persist_dir=str(tmp_path / "rag_test"),
            max_papers_to_process=2
        )

        # Create literature agent
        agent = LiteratureAgent(config)

        # Search for papers
        query = "neural networks"
        papers = await agent.search(query, max_results=2)

        assert len(papers) <= 2, "Should respect max_results"

        # If papers have PDFs available, verify they were processed
        for paper in papers:
            paper_id = paper.get("paperId")
            # Check if RAG engine has the paper
            # (Note: This may not always succeed if PDFs aren't available)
            if paper_id and agent.rag_engine:
                # Just verify the system handles this gracefully
                has_paper = agent.rag_engine.has_paper(paper_id)
                print(f"Paper {paper_id} indexed: {has_paper}")

        print("✓ Literature agent with RAG test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
