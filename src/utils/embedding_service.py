"""
Embedding Service - Provides semantic similarity and embedding functionality.

This module provides a centralized service for computing text embeddings
and semantic similarity, used for novelty detection and semantic search.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingService:
    """
    Service for computing text embeddings and semantic similarity.

    Uses sentence-transformers for efficient embedding generation.
    Includes caching for repeated embedding computations.
    """

    # Singleton instance
    _instance: Optional["EmbeddingService"] = None
    _model: Optional[Any] = None

    # Cache for embeddings (text_hash -> embedding)
    _embedding_cache: Dict[str, np.ndarray] = {}
    _cache_max_size: int = 1000

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       "all-MiniLM-L6-v2" is fast and good for general use.
                       "all-mpnet-base-v2" is more accurate but slower.
        """
        if self._initialized:
            return

        self.model_name = model_name
        self._initialized = True

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Using fallback similarity.")
            return

        try:
            print(f"Loading embedding model: {model_name}")
            EmbeddingService._model = SentenceTransformer(model_name)
            print(f"Embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load embedding model: {e}")
            EmbeddingService._model = None

    def _text_hash(self, text: str) -> str:
        """Generate a hash for text to use as cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _manage_cache(self):
        """Remove oldest entries if cache exceeds max size."""
        if len(self._embedding_cache) > self._cache_max_size:
            # Remove oldest 20% of entries
            num_to_remove = int(self._cache_max_size * 0.2)
            keys_to_remove = list(self._embedding_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self._embedding_cache[key]

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Numpy array of embedding, or None if embedding not available
        """
        if not text or not text.strip():
            return None

        if self._model is None:
            return None

        # Check cache first
        text_hash = self._text_hash(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)

            # Cache the result
            self._manage_cache()
            self._embedding_cache[text_hash] = embedding

            return embedding
        except Exception as e:
            print(f"Warning: Failed to compute embedding: {e}")
            return None

    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failed texts)
        """
        if self._model is None:
            return [None] * len(texts)

        # Separate cached and uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            text_hash = self._text_hash(text)
            if text_hash in self._embedding_cache:
                results[i] = self._embedding_cache[text_hash]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached texts
        if uncached_texts:
            try:
                embeddings = self._model.encode(uncached_texts, convert_to_numpy=True)

                # Store results and update cache
                self._manage_cache()
                for idx, embedding, text in zip(uncached_indices, embeddings, uncached_texts):
                    results[idx] = embedding
                    text_hash = self._text_hash(text)
                    self._embedding_cache[text_hash] = embedding
            except Exception as e:
                print(f"Warning: Failed to compute batch embeddings: {e}")

        return results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1 (1 = identical meaning)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        if emb1 is None or emb2 is None:
            # Fall back to Jaccard similarity
            return self._jaccard_similarity(text1, text2)

        return self._cosine_similarity(emb1, emb2)

    def compute_max_similarity(
        self,
        text: str,
        candidates: List[str],
        return_index: bool = False
    ):
        """
        Compute maximum similarity between text and a list of candidates.

        Args:
            text: The text to compare
            candidates: List of candidate texts
            return_index: If True, also return the index of most similar candidate

        Returns:
            If return_index is False: Maximum similarity score (float)
            If return_index is True: Tuple of (max_similarity, index_of_most_similar)
        """
        if not candidates:
            return (0.0, -1) if return_index else 0.0

        text_emb = self.get_embedding(text)

        if text_emb is None:
            # Fall back to Jaccard
            max_sim = 0.0
            max_idx = 0
            for i, candidate in enumerate(candidates):
                sim = self._jaccard_similarity(text, candidate)
                if sim > max_sim:
                    max_sim = sim
                    max_idx = i
            return (max_sim, max_idx) if return_index else max_sim

        # Batch compute candidate embeddings
        candidate_embeddings = self.get_embeddings_batch(candidates)

        max_sim = 0.0
        max_idx = 0
        for i, cand_emb in enumerate(candidate_embeddings):
            if cand_emb is not None:
                sim = self._cosine_similarity(text_emb, cand_emb)
                if sim > max_sim:
                    max_sim = sim
                    max_idx = i

        return (max_sim, max_idx) if return_index else max_sim

    def find_contradictions(
        self,
        findings: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
        min_confidence_diff: float = 0.3
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Find potentially contradictory findings based on semantic similarity
        but opposite confidence or contradictory terms.

        Two findings are potentially contradictory if:
        - They are semantically similar (same topic)
        - One contains negative framing of the other
        - They have significantly different confidence levels suggesting disagreement

        Args:
            findings: List of finding dictionaries with 'text' and 'confidence' keys
            similarity_threshold: Minimum similarity to consider same topic
            min_confidence_diff: Minimum confidence difference to flag

        Returns:
            List of tuples (finding1, finding2, similarity_score)
        """
        if len(findings) < 2:
            return []

        contradictions = []

        # Get all texts and embeddings
        texts = [f.get("text", "") for f in findings]
        embeddings = self.get_embeddings_batch(texts)

        # Check pairs for potential contradictions
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                if embeddings[i] is None or embeddings[j] is None:
                    continue

                similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                # Skip if not similar enough (different topics)
                if similarity < similarity_threshold:
                    continue

                # Check for contradiction indicators
                text1 = texts[i].lower()
                text2 = texts[j].lower()

                # Negation patterns that might indicate contradiction
                negation_pairs = [
                    ("not ", " "), ("no ", " "), ("without ", "with "),
                    ("decrease", "increase"), ("lower", "higher"),
                    ("negative", "positive"), ("fail", "success"),
                    ("reject", "accept"), ("disprove", "prove"),
                    ("insignificant", "significant"), ("weak", "strong"),
                ]

                has_negation = False
                for neg, pos in negation_pairs:
                    if (neg in text1 and pos in text2) or (neg in text2 and pos in text1):
                        has_negation = True
                        break

                # Check confidence difference
                conf1 = findings[i].get("confidence", 0.5)
                conf2 = findings[j].get("confidence", 0.5)
                conf_diff = abs(conf1 - conf2)

                # Flag as potential contradiction if:
                # - High similarity but contains negation patterns, OR
                # - Very high similarity but significant confidence difference
                if has_negation or (similarity > 0.85 and conf_diff > min_confidence_diff):
                    contradictions.append((findings[i], findings[j], similarity))

        # Sort by similarity (most similar = most likely contradiction)
        contradictions.sort(key=lambda x: x[2], reverse=True)

        return contradictions

    def compute_topic_coverage(
        self,
        objective: str,
        findings: List[str],
        num_aspects: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze how well findings cover different aspects of an objective.

        Uses embedding clustering to identify distinct aspects covered.

        Args:
            objective: The original objective text
            findings: List of finding texts
            num_aspects: Number of aspects to analyze

        Returns:
            Dictionary with coverage analysis
        """
        if not findings:
            return {
                "overall_coverage": 0.0,
                "aspects_covered": 0,
                "coverage_distribution": [],
                "gaps": [objective],
            }

        # Get objective embedding
        obj_emb = self.get_embedding(objective)
        if obj_emb is None:
            return {
                "overall_coverage": len(findings) / 10.0,  # Rough estimate
                "aspects_covered": min(len(findings), num_aspects),
                "coverage_distribution": [],
                "gaps": [],
            }

        # Get finding embeddings
        finding_embs = self.get_embeddings_batch(findings)
        valid_embs = [(i, emb) for i, emb in enumerate(finding_embs) if emb is not None]

        if not valid_embs:
            return {
                "overall_coverage": 0.0,
                "aspects_covered": 0,
                "coverage_distribution": [],
                "gaps": [objective],
            }

        # Compute similarity of each finding to objective
        similarities = []
        for i, emb in valid_embs:
            sim = self._cosine_similarity(obj_emb, emb)
            similarities.append((i, findings[i], sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)

        # Compute overall coverage (average of top similarities)
        top_sims = [s[2] for s in similarities[:num_aspects]]
        overall_coverage = sum(top_sims) / len(top_sims) if top_sims else 0.0

        # Estimate aspects covered by looking at diversity of findings
        # (findings that are similar to objective but different from each other)
        covered_aspects = 1
        if len(valid_embs) > 1:
            # Check diversity
            emb_matrix = np.array([emb for _, emb in valid_embs])
            # Compute pairwise similarities
            similarities_matrix = np.dot(emb_matrix, emb_matrix.T)
            # Count "distinct" findings (similarity < 0.8 to others)
            distinct_count = 0
            for i in range(len(valid_embs)):
                is_distinct = True
                for j in range(i):
                    if similarities_matrix[i, j] > 0.8:
                        is_distinct = False
                        break
                if is_distinct:
                    distinct_count += 1
            covered_aspects = min(distinct_count, num_aspects)

        return {
            "overall_coverage": round(overall_coverage, 3),
            "aspects_covered": covered_aspects,
            "num_findings": len(findings),
            "top_relevant_findings": [
                {"text": s[1][:100], "relevance": round(s[2], 3)}
                for s in similarities[:5]
            ],
            "coverage_gap": round(1.0 - overall_coverage, 3),
        }

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Fallback Jaccard similarity for when embeddings aren't available."""
        # Simple tokenization
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "but", "or", "that", "this", "it", "they", "we",
        }

        def tokenize(text):
            import re
            words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
            return words - stopwords

        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self._model is not None

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "max_size": self._cache_max_size,
        }


# Global convenience functions
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts."""
    return get_embedding_service().compute_similarity(text1, text2)


def find_max_similarity(text: str, candidates: List[str]) -> float:
    """Find maximum similarity between text and candidates."""
    return get_embedding_service().compute_max_similarity(text, candidates)
