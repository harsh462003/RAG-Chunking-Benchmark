"""Top-k retrieval over per-method FAISS indices."""

from .common import *
from .schemas import RetrievalResult
from .utils import count_tokens

def retrieve_top_k(index, query_vec: np.ndarray, chunks: List[Chunk],
                   top_k: int = 5) -> Tuple[List[Dict], float]:
    t0 = time.perf_counter()
    if index is None or not FAISS_AVAILABLE:
        # fallback: brute-force cosine
        chunk_texts = [c.chunk_text for c in chunks]
        scores = [cosine_sim(query_vec.flatten(),
                             np.array(list(c.chunk_text.encode()[:384]), dtype=np.float32))
                  for c in chunks]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [{"rank": r + 1, "chunk": chunks[i], "score": float(scores[i])}
                   for r, i in enumerate(top_indices)]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return results, elapsed_ms
    qv = query_vec.reshape(1, -1).astype(np.float32)
    k = min(top_k, len(chunks))
    scores, indices = index.search(qv, k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < len(chunks):
            results.append({"rank": rank + 1, "chunk": chunks[idx], "score": float(score)})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms
