from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it (thread-safe via lru_cache)."""
    print("🔄 Loading embedding model...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    """Embed a single string."""
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """
    Embed a list of strings in a single vectorized call.
    Much faster than calling get_embedding() in a loop because
    SentenceTransformer.encode() runs the entire list through the
    model in one batched forward pass on GPU/CPU.
    Returns an empty list when texts is empty.
    """
    if not texts:
        return []
    model = get_model()
    # encode() returns shape (N, dim); split back into a list of 1-D arrays
    matrix = model.encode(texts, normalize_embeddings=True, batch_size=64)
    return [matrix[i] for i in range(len(texts))]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))