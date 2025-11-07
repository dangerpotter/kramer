"""
Example script demonstrating Full-Text RAG Processing for Papers.

This example shows how to use the PaperProcessor and RAGEngine
to download, process, and query scientific papers.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kramer.paper_processor import PaperProcessor
from kramer.rag_engine import RAGEngine


async def main():
    """Example usage of RAG for paper processing."""

    print("=" * 60)
    print("Full-Text RAG Processing Example")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing RAG Engine...")
    rag_engine = RAGEngine(
        persist_directory=Path("data/rag_example"),
        model_name="all-MiniLM-L6-v2"
    )

    print("\n2. Initializing Paper Processor...")
    paper_processor = PaperProcessor()

    # Example paper ID (Semantic Scholar format)
    # This is a sample - replace with actual paper ID
    paper_id = "649def34f8be52c8b66281af98ae884c09aef38b"

    print(f"\n3. Processing paper: {paper_id}")
    print("   Downloading PDF...")

    # Download PDF
    pdf_path = await paper_processor.download_pdf(
        paper_id=paper_id,
        s2_api_key=None  # Optional: add your API key
    )

    if not pdf_path:
        print("   ⚠ PDF not available for this paper")
        print("\n   Note: Many papers require API keys or may not have PDFs available.")
        print("   This is expected behavior - the system will fall back to abstracts.")
        return

    print(f"   ✓ Downloaded to: {pdf_path}")

    # Extract text
    print("\n4. Extracting text from PDF...")
    text = paper_processor.extract_text(pdf_path)
    print(f"   ✓ Extracted {len(text)} characters")
    print(f"   Preview: {text[:200]}...")

    # Chunk text
    print("\n5. Chunking text...")
    chunks = paper_processor.chunk_text(text, chunk_size=500, overlap=50)
    print(f"   ✓ Created {len(chunks)} chunks")

    # Add to RAG
    print("\n6. Adding chunks to RAG engine...")
    chunk_count = rag_engine.add_paper(paper_id=paper_id, chunks=chunks)
    print(f"   ✓ Added {chunk_count} chunks to vector database")

    # Query RAG
    print("\n7. Querying RAG engine...")
    query = "What are the main findings of this paper?"
    results = rag_engine.query(question=query, top_k=3)

    print(f"\n   Query: '{query}'")
    print(f"   Found {len(results)} relevant chunks:\n")

    for i, result in enumerate(results, 1):
        print(f"   Result {i}:")
        print(f"   - Score: {result['score']:.3f}")
        print(f"   - Text: {result['text'][:150]}...")
        print()

    # Get paper context
    print("\n8. Getting paper context...")
    context = rag_engine.get_paper_context(
        paper_id=paper_id,
        query="methodology and results",
        max_chunks=2
    )
    print(f"   ✓ Retrieved context ({len(context)} chars)")
    print(f"   Preview: {context[:300]}...")

    # Stats
    print("\n9. RAG Engine Statistics:")
    stats = rag_engine.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    # Cleanup
    print("\n10. Cleaning up...")
    paper_processor.cleanup(paper_id=paper_id)
    print("   ✓ Removed temporary PDF")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
