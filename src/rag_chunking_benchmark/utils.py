"""Small reusable helpers for text statistics and similarity scoring."""

from .common import *

def count_tokens(text: str) -> int:
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text.split()) * 4 // 3)

def count_words(text: str) -> int:
    return len(text.split())

def count_sentences(text: str) -> int:
    return len(re.findall(r'[.!?]+', text)) or 1

def normalize_text(text: str) -> str:
    # Collapse horizontal whitespace only (spaces, tabs) — preserve blank-line
    # paragraph breaks so paragraph_based / hybrid_chunking can still find them.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)      # strip leading spaces on wrapped lines
    text = re.sub(r'[ \t]+\n', '\n', text)      # strip trailing spaces before newline
    text = re.sub(r'\n{3,}', '\n\n', text)      # cap runs of blank lines at one
    return text.strip()

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def overlap_ratio(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)
