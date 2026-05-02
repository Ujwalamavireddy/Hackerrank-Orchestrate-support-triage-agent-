import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Create a sentence-transformers model for embeddings."""
    return SentenceTransformer(model_name)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize embedding vectors so cosine similarity works with FAISS inner product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return vectors / norms


def build_faiss_index(documents: List[Dict[str, str]], embedder: SentenceTransformer) -> Tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Build a FAISS index from corpus documents."""
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = normalize_vectors(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, documents


def search_documents(
    query: str,
    index: faiss.IndexFlatIP,
    embedder: SentenceTransformer,
    documents: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict[str, str]]:
    """Search the indexed support corpus for query-relevant documents."""
    query_embedding = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_embedding = normalize_vectors(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    results: List[Dict[str, str]] = []

    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(documents):
            continue
        document = documents[idx].copy()
        document["score"] = float(score)
        results.append(document)

    return results
