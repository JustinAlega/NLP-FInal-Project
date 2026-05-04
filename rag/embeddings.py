"""
Embedding Utilities
====================
Text embedding for semantic search using sentence-transformers (local).
"""

import logging
import numpy as np
from typing import Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string."""
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class VectorIndex:
    """Simple in-memory vector index for semantic search."""

    def __init__(self):
        self.embeddings = []
        self.metadata = []

    def add(self, text: str, meta: dict):
        """Add a text with metadata to the index."""
        emb = embed_text(text)
        self.embeddings.append(emb)
        self.metadata.append(meta)

    def add_batch(self, texts: list[str], metas: list[dict]):
        """Add a batch of texts with metadata."""
        embs = embed_texts(texts)
        self.embeddings.extend(embs)
        self.metadata.extend(metas)
        logger.info(f"Indexed {len(texts)} texts (total: {len(self.embeddings)})")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for the most similar texts to a query."""
        if not self.embeddings:
            return []
        query_emb = embed_text(query)
        scores = [cosine_similarity(query_emb, emb) for emb in self.embeddings]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                **self.metadata[idx],
            })
        return results

    def __len__(self):
        return len(self.embeddings)
