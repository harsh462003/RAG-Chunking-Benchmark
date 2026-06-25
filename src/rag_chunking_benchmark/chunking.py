"""All supported chunking strategies and chunk statistics."""

from .common import *
from .schemas import Chunk
from .utils import count_tokens
from .embeddings import embed_texts

def _make_chunk(idx: int, text: str, method: str, doc_name: str) -> Chunk:
    text = text.strip()
    return Chunk(
        chunk_id=f"{method}_{idx}",
        chunk_index=idx,
        chunk_text=text,
        chunk_word_count=count_words(text),
        chunk_char_count=len(text),
        chunk_token_count=count_tokens(text),
        chunking_method=method,
        document_name=doc_name,
    )

def chunk_fixed_size(text: str, doc_name: str, chunk_size_words: int = 300) -> List[Chunk]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size_words]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(_make_chunk(len(chunks), chunk_text, "fixed_size", doc_name))
        i += chunk_size_words
    return chunks

def chunk_fixed_overlap(text: str, doc_name: str, chunk_size_words: int = 300, overlap_words: int = 50) -> List[Chunk]:
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size_words - overlap_words)
    while i < len(words):
        chunk_words = words[i:i + chunk_size_words]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(_make_chunk(len(chunks), chunk_text, "fixed_overlap", doc_name))
        i += step
        if i + chunk_size_words >= len(words) and i < len(words):
            remainder = " ".join(words[i:])
            if remainder.strip() and remainder not in [c.chunk_text for c in chunks[-2:]]:
                chunks.append(_make_chunk(len(chunks), remainder, "fixed_overlap", doc_name))
            break
    return chunks

def chunk_sentence_based(text: str, doc_name: str, sentences_per_chunk: int = 5) -> List[Chunk]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, i = [], 0
    while i < len(sentences):
        group = sentences[i:i + sentences_per_chunk]
        chunk_text = " ".join(group)
        if chunk_text.strip():
            chunks.append(_make_chunk(len(chunks), chunk_text, "sentence_based", doc_name))
        i += sentences_per_chunk
    return chunks

def chunk_paragraph_based(text: str, doc_name: str, merge_short: bool = True, min_words: int = 80) -> List[Chunk]:
    paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return chunk_fixed_size(text, doc_name)
    merged = []
    buffer = ""
    for para in paragraphs:
        if merge_short and count_words(buffer + " " + para) < min_words:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)
    return [_make_chunk(i, t, "paragraph_based", doc_name) for i, t in enumerate(merged) if t.strip()]

def chunk_semantic(text: str, doc_name: str, embed_fn=None,
                   similarity_threshold: float = 0.65,
                   min_words: int = 100, max_words: int = 500) -> List[Chunk]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences or embed_fn is None:
        return chunk_sentence_based(text, doc_name)
    try:
        embeddings = embed_fn(sentences)
    except Exception:
        return chunk_sentence_based(text, doc_name)
    groups = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        current_words = count_words(" ".join(current))
        if sim >= similarity_threshold and current_words < max_words:
            current.append(sentences[i])
        else:
            groups.append(" ".join(current))
            current = [sentences[i]]
    if current:
        groups.append(" ".join(current))
    # merge tiny chunks
    final = []
    buf = ""
    for g in groups:
        if count_words(buf + " " + g) < min_words:
            buf = (buf + " " + g).strip()
        else:
            if buf:
                final.append(buf)
            buf = g
    if buf:
        final.append(buf)
    return [_make_chunk(i, t, "semantic_chunking", doc_name) for i, t in enumerate(final) if t.strip()]

def chunk_sliding_window(text: str, doc_name: str, window_size: int = 300, step_size: int = 150) -> List[Chunk]:
    words = text.split()
    chunks, i = [], 0
    seen = set()
    while i < len(words):
        chunk_words = words[i:i + window_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip() and chunk_text not in seen:
            seen.add(chunk_text)
            chunks.append(_make_chunk(len(chunks), chunk_text, "sliding_window", doc_name))
        i += step_size
        if i >= len(words):
            break
    return chunks

def chunk_recursive(text: str, doc_name: str, chunk_size_chars: int = 1200,
                    overlap_chars: int = 200,
                    separators: Optional[List[str]] = None) -> List[Chunk]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(t: str, sep_idx: int) -> List[str]:
        if len(t) <= chunk_size_chars:
            return [t]
        if sep_idx >= len(separators):
            # hard split
            result = []
            for j in range(0, len(t), chunk_size_chars - overlap_chars):
                result.append(t[j:j + chunk_size_chars])
            return result
        sep = separators[sep_idx]
        parts = t.split(sep) if sep else list(t)
        merged, buf = [], ""
        for part in parts:
            candidate = buf + (sep if buf else "") + part
            if len(candidate) <= chunk_size_chars:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                if len(part) > chunk_size_chars:
                    merged.extend(_split(part, sep_idx + 1))
                    buf = ""
                else:
                    buf = part
        if buf:
            merged.append(buf)
        return merged

    pieces = _split(text, 0)
    return [_make_chunk(i, t, "recursive_character_splitter", doc_name)
            for i, t in enumerate(pieces) if t.strip()]

def chunk_hybrid(text: str, doc_name: str, embed_fn=None,
                 max_words: int = 400, min_words: int = 80) -> List[Chunk]:
    # 1. split by paragraphs
    paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    # 2. split large paragraphs by sentences
    segments = []
    for para in paragraphs:
        if count_words(para) > max_words:
            sents = re.split(r'(?<=[.!?])\s+', para)
            buf = ""
            for sent in sents:
                if count_words(buf + " " + sent) <= max_words:
                    buf = (buf + " " + sent).strip()
                else:
                    if buf:
                        segments.append(buf)
                    buf = sent
            if buf:
                segments.append(buf)
        else:
            segments.append(para)
    # 3. merge tiny segments using semantic sim if available
    if embed_fn is not None and len(segments) > 1:
        try:
            embeddings = embed_fn(segments)
            merged, current_segs = [], [segments[0]]
            for i in range(1, len(segments)):
                sim = cosine_sim(embeddings[i - 1], embeddings[i])
                combined_words = count_words(" ".join(current_segs) + " " + segments[i])
                if sim > 0.55 and combined_words < max_words:
                    current_segs.append(segments[i])
                else:
                    merged.append(" ".join(current_segs))
                    current_segs = [segments[i]]
            if current_segs:
                merged.append(" ".join(current_segs))
            segments = merged
        except Exception:
            pass
    # 4. merge still-short segments
    final, buf = [], ""
    for seg in segments:
        if count_words(buf + " " + seg) < min_words:
            buf = (buf + " " + seg).strip()
        else:
            if buf:
                final.append(buf)
            buf = seg
    if buf:
        final.append(buf)
    return [_make_chunk(i, t, "hybrid_chunking", doc_name) for i, t in enumerate(final) if t.strip()]

def apply_chunking(method: str, text: str, doc_name: str,
                   params: Dict, embed_fn=None) -> Tuple[List[Chunk], float]:
    t0 = time.perf_counter()
    if method == "fixed_size":
        chunks = chunk_fixed_size(text, doc_name, params.get("chunk_size_words", 300))
    elif method == "fixed_overlap":
        chunks = chunk_fixed_overlap(text, doc_name,
                                     params.get("chunk_size_words", 300),
                                     params.get("overlap_words", 50))
    elif method == "sentence_based":
        chunks = chunk_sentence_based(text, doc_name, params.get("sentences_per_chunk", 5))
    elif method == "paragraph_based":
        chunks = chunk_paragraph_based(text, doc_name,
                                       params.get("merge_short", True),
                                       params.get("min_words", 80))
    elif method == "semantic_chunking":
        chunks = chunk_semantic(text, doc_name, embed_fn,
                                params.get("similarity_threshold", 0.65),
                                params.get("min_chunk_words", 100),
                                params.get("max_chunk_words", 500))
    elif method == "sliding_window":
        chunks = chunk_sliding_window(text, doc_name,
                                      params.get("window_size_words", 300),
                                      params.get("step_size_words", 150))
    elif method == "recursive_character_splitter":
        sep_raw = params.get("separators", ["\n\n", "\n", ".", " ", ""])
        chunks = chunk_recursive(text, doc_name,
                                 params.get("chunk_size_chars", 1200),
                                 params.get("overlap_chars", 200),
                                 sep_raw)
    elif method == "hybrid_chunking":
        chunks = chunk_hybrid(text, doc_name, embed_fn,
                              params.get("max_words", 400),
                              params.get("min_words", 80))
    else:
        chunks = chunk_fixed_size(text, doc_name)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return chunks, elapsed_ms

def chunk_stats(chunks: List[Chunk]) -> Dict:
    if not chunks:
        return {}
    sizes = [c.chunk_word_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size_words": float(np.mean(sizes)),
        "std_chunk_size": float(np.std(sizes)),
        "min_chunk_size": int(np.min(sizes)),
        "max_chunk_size": int(np.max(sizes)),
    }
