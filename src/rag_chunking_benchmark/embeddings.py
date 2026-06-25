"""Embedding and FAISS indexing functions."""

from .common import *
from .schemas import Chunk

def load_embedding_model(model_name: str):
    if not ST_AVAILABLE:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.warning(f"Could not load embedding model '{model_name}': {e}")
        return None

def embed_texts(model, texts: List[str]) -> np.ndarray:
    if model is None or not texts:
        dim = 384
        return np.random.randn(len(texts), dim).astype(np.float32)
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # L2-normalise for cosine via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (vecs / norms).astype(np.float32)

def embed_chunks(model, chunks: List[Chunk]) -> Tuple[np.ndarray, float]:
    texts = [c.chunk_text for c in chunks]
    t0 = time.perf_counter()
    vecs = embed_texts(model, texts)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return vecs, elapsed_ms

def build_faiss_index(embeddings: np.ndarray) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    if not FAISS_AVAILABLE:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return None, elapsed_ms
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalised = cosine
    index.add(embeddings)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return index, elapsed_ms
