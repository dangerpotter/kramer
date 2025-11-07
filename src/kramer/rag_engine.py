"""RAG (Retrieval Augmented Generation) engine for paper embeddings and retrieval."""

from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class RAGEngine:
    """
    Handles vector embeddings and retrieval for paper chunks using ChromaDB.

    Uses sentence-transformers for embedding generation and ChromaDB for
    efficient similarity search.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "paper_chunks"
    ):
        """
        Initialize the RAG engine.

        Args:
            persist_directory: Directory to persist ChromaDB data.
                              If None, uses in-memory storage.
            model_name: Name of sentence-transformers model to use
            collection_name: Name of ChromaDB collection

        Raises:
            ImportError: If required dependencies are not installed
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. "
                "Install it with: pip install chromadb"
            )

        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Initialize ChromaDB client
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory)
            )
        else:
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"model": model_name}
        )

    def add_paper(
        self,
        paper_id: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Add paper chunks with embeddings to the collection.

        Args:
            paper_id: Unique identifier for the paper
            chunks: List of chunk dictionaries with 'chunk_id' and 'text'

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Create unique ID combining paper_id and chunk_id
            chunk_full_id = f"{paper_id}_{chunk['chunk_id']}"
            ids.append(chunk_full_id)
            documents.append(chunk['text'])

            # Store metadata
            metadatas.append({
                "paper_id": paper_id,
                "chunk_id": chunk['chunk_id'],
                "start_char": chunk.get('start_char', 0)
            })

        # Generate embeddings and add to collection
        embeddings = self.model.encode(documents).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
        paper_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar chunks using semantic search.

        Args:
            question: Query string
            top_k: Number of top results to return
            paper_id: Optional filter to only search within a specific paper

        Returns:
            List of dictionaries with:
                - text: The chunk text
                - paper_id: Paper ID
                - chunk_id: Chunk ID
                - score: Similarity score (higher is more similar)
                - start_char: Starting character position
        """
        # Generate embedding for query
        query_embedding = self.model.encode([question])[0].tolist()

        # Prepare query filters
        where = None
        if paper_id:
            where = {"paper_id": paper_id}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # Format results
        formatted_results = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "paper_id": results['metadatas'][0][i]['paper_id'],
                    "chunk_id": results['metadatas'][0][i]['chunk_id'],
                    "start_char": results['metadatas'][0][i].get('start_char', 0),
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })

        return formatted_results

    def get_paper_context(
        self,
        paper_id: str,
        query: str,
        max_chunks: int = 3
    ) -> str:
        """
        Get the most relevant chunks from a specific paper for a query.

        Args:
            paper_id: Paper ID to search within
            query: Query string to find relevant sections
            max_chunks: Maximum number of chunks to return

        Returns:
            Concatenated text of the most relevant chunks
        """
        results = self.query(question=query, top_k=max_chunks, paper_id=paper_id)

        if not results:
            return ""

        # Concatenate chunks
        chunks_text = []
        for i, result in enumerate(results, 1):
            chunks_text.append(f"[Section {i}]\n{result['text']}")

        return "\n\n".join(chunks_text)

    def has_paper(self, paper_id: str) -> bool:
        """
        Check if a paper has been added to the collection.

        Args:
            paper_id: Paper ID to check

        Returns:
            True if paper exists in collection
        """
        try:
            results = self.collection.get(
                where={"paper_id": paper_id},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception:
            return False

    def get_paper_chunk_count(self, paper_id: str) -> int:
        """
        Get the number of chunks stored for a paper.

        Args:
            paper_id: Paper ID

        Returns:
            Number of chunks
        """
        try:
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            return len(results['ids'])
        except Exception:
            return 0

    def delete_paper(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper from the collection.

        Args:
            paper_id: Paper ID to delete

        Returns:
            Number of chunks deleted
        """
        try:
            # Get all chunk IDs for this paper
            results = self.collection.get(
                where={"paper_id": paper_id}
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])

            return 0
        except Exception as e:
            print(f"Error deleting paper {paper_id}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()

        # Get unique papers
        unique_papers = set()
        if count > 0:
            results = self.collection.get()
            for metadata in results['metadatas']:
                unique_papers.add(metadata['paper_id'])

        return {
            "total_chunks": count,
            "total_papers": len(unique_papers),
            "model": self.model_name,
            "collection_name": self.collection.name
        }

    def clear(self):
        """Clear all data from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"model": self.model_name}
        )

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"RAGEngine(papers={stats['total_papers']}, "
            f"chunks={stats['total_chunks']}, "
            f"model={stats['model']})"
        )
