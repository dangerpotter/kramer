# Full-Text Paper Processing with RAG

This feature enables Kramer to read full PDFs and extract detailed information using Retrieval Augmented Generation (RAG), rather than relying solely on paper abstracts.

## Overview

The RAG system consists of three main components:

1. **PaperProcessor**: Downloads PDFs and extracts text
2. **RAGEngine**: Creates embeddings and enables semantic search
3. **Enhanced LiteratureAgent**: Integrates full-text processing into literature search

## Installation

Install the required dependencies:

```bash
pip install pymupdf sentence-transformers chromadb
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Components

### PaperProcessor (`src/kramer/paper_processor.py`)

Handles PDF download and text extraction.

**Key Methods:**
- `download_pdf(paper_id, s2_api_key)`: Downloads PDF from Semantic Scholar
- `extract_text(pdf_path)`: Extracts text using PyMuPDF
- `chunk_text(text, chunk_size, overlap)`: Splits text into overlapping chunks

**Example:**
```python
from kramer.paper_processor import PaperProcessor

processor = PaperProcessor()
pdf_path = await processor.download_pdf("paper_id", api_key)
text = processor.extract_text(pdf_path)
chunks = processor.chunk_text(text, chunk_size=500, overlap=50)
```

### RAGEngine (`src/kramer/rag_engine.py`)

Manages vector embeddings and semantic retrieval using ChromaDB.

**Key Methods:**
- `add_paper(paper_id, chunks)`: Adds paper chunks with embeddings
- `query(question, top_k)`: Searches for relevant chunks
- `get_paper_context(paper_id, query, max_chunks)`: Gets context from specific paper

**Example:**
```python
from kramer.rag_engine import RAGEngine

rag = RAGEngine(persist_directory="data/rag_db")
rag.add_paper(paper_id="123", chunks=chunks)
results = rag.query("What are the findings?", top_k=5)
```

### Enhanced LiteratureAgent (`kramer/agents/literature.py`)

The literature agent now supports full-text processing.

**New Methods:**
- `process_full_paper(paper_id)`: Downloads and processes full PDF
- `extract_claims_from_full_text(paper_id, hypothesis_context, paper_metadata)`: Extracts claims using RAG

**Updated Behavior:**
- Automatically processes top N papers with full-text (configurable)
- Falls back to abstract extraction if PDF unavailable
- Tracks which claims came from full-text vs abstracts

## Configuration

Add to your config:

```python
from kramer.config import Config

config = Config(
    use_full_text=True,              # Enable RAG processing
    max_papers_to_process=20,         # Max papers to process with full-text
    rag_persist_dir="data/rag_db",    # Where to store embeddings
    embedding_model="all-MiniLM-L6-v2",  # Sentence transformer model
    chunk_size=500,                   # Chunk size in characters
    chunk_overlap=50                  # Overlap between chunks
)
```

## Usage Example

### Basic Usage

```python
import asyncio
from kramer.agents.literature import LiteratureAgent
from kramer.paper_processor import PaperProcessor
from kramer.rag_engine import RAGEngine
from kramer.world_model import WorldModel

async def main():
    # Initialize components
    world_model = WorldModel()
    paper_processor = PaperProcessor()
    rag_engine = RAGEngine(persist_directory="data/rag_db")

    # Create literature agent with RAG
    agent = LiteratureAgent(
        world_model=world_model,
        anthropic_api_key="your-api-key",
        paper_processor=paper_processor,
        rag_engine=rag_engine,
        use_full_text=True,
        max_papers_to_process=10
    )

    # Search and extract claims (now with full-text!)
    async with agent:
        claims = await agent.search_and_extract(
            query="machine learning interpretability",
            max_papers=5,
            hypotheses=["Neural networks lack transparency"]
        )

    print(f"Extracted {len(claims)} claims")

asyncio.run(main())
```

### Integration with World Model

The world model now tracks full-text processing:

```python
# Papers have new metadata fields
paper_node = world_model.get_node(paper_id)
has_full_text = paper_node["metadata"]["has_full_text"]
rag_id = paper_node["metadata"]["rag_paper_id"]

# Mark a paper as processed
world_model.mark_paper_processed(
    node_id=paper_id,
    rag_paper_id="semantic_scholar_id"
)
```

## How It Works

1. **Search**: Agent searches Semantic Scholar for relevant papers
2. **Download**: For top N papers, downloads PDFs via Semantic Scholar API
3. **Extract**: Uses PyMuPDF to extract text from all pages
4. **Chunk**: Splits text into overlapping chunks (500 chars by default)
5. **Embed**: Generates vector embeddings using sentence-transformers
6. **Store**: Stores chunks and embeddings in ChromaDB
7. **Query**: When extracting claims, retrieves most relevant chunks for hypothesis
8. **Extract**: Uses Claude to extract claims from relevant sections
9. **Fallback**: Falls back to abstract if PDF unavailable

## Benefits

- **Deeper Analysis**: Access full methodology and results, not just abstracts
- **Context-Aware**: RAG retrieves only relevant sections for current hypotheses
- **Scalable**: ChromaDB efficiently handles thousands of papers
- **Cost-Effective**: Limits full-text processing to top N papers
- **Fault-Tolerant**: Gracefully falls back to abstracts when PDFs unavailable

## Performance Considerations

- **Embedding Model**: `all-MiniLM-L6-v2` is fast and efficient (384 dimensions)
- **Chunk Size**: 500 chars balances context and granularity
- **Overlap**: 50 chars ensures important text isn't split
- **Rate Limits**: Agent includes delays between papers
- **Storage**: ChromaDB is persistent and disk-efficient

## Troubleshooting

### PDFs Not Available
Many papers don't have open-access PDFs. This is normal - the system falls back to abstracts.

### Import Errors
Ensure dependencies are installed:
```bash
pip install pymupdf sentence-transformers chromadb
```

### ChromaDB Errors
If ChromaDB has issues, try clearing the database:
```python
rag_engine.clear()
```

### Memory Usage
If processing many papers, consider:
- Reducing `max_papers_to_process`
- Using a smaller embedding model
- Clearing PDFs after processing (automatic by default)

## Future Enhancements

Potential improvements:
- Support for arXiv and PubMed PDFs
- Multi-modal embeddings (text + figures)
- Hierarchical chunking (section-aware)
- Citation graph integration
- Cross-paper semantic search
- Query-specific chunk sizing

## Files Modified

- `src/kramer/paper_processor.py` (new)
- `src/kramer/rag_engine.py` (new)
- `kramer/agents/literature.py` (modified)
- `src/world_model/graph.py` (modified)
- `kramer/config.py` (modified)
- `requirements.txt` (modified)
