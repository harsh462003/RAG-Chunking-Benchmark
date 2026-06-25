"""Typed data containers used throughout the benchmark pipeline."""

from .common import dataclass, field, Dict, List

class Chunk:
    chunk_id: str
    chunk_index: int
    chunk_text: str
    chunk_word_count: int
    chunk_char_count: int
    chunk_token_count: int
    chunking_method: str
    document_name: str

class RetrievalResult:
    question_id: str
    question_text: str
    chunking_method: str
    retrieved_chunks: List[Dict]
    context_text: str
    context_token_count: int
    retrieval_latency_ms: float
    top_similarity: float
    avg_similarity: float

class AnswerResult:
    question_id: str
    question_text: str
    chunking_method: str
    answer_text: str
    answer_status: str  # ANSWERED / INSUFFICIENT_CONTEXT
    answer_token_count: int
    generation_latency_ms: float
    model_name: str

class EvalScores:
    relevance: float = 0.0
    correctness: float = 0.0
    completeness: float = 0.0
    faithfulness: float = 0.0
    context_sufficiency: float = 0.0
    hallucination_risk: float = 0.0
    reason: str = ""

class MethodMetrics:
    method: str
    # chunk stats
    total_chunks: int = 0
    avg_chunk_size_words: float = 0.0
    std_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    # timing
    chunking_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    index_creation_time_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0
    total_pipeline_time_ms: float = 0.0
    # quality
    avg_relevance: float = 0.0
    avg_correctness: float = 0.0
    avg_completeness: float = 0.0
    avg_faithfulness: float = 0.0
    avg_context_sufficiency: float = 0.0
    avg_hallucination_risk: float = 0.0
    avg_similarity: float = 0.0
    # token usage
    avg_context_tokens: float = 0.0
    avg_answer_tokens: float = 0.0
    # ranking
    final_score: float = 0.0
    rank: int = 0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendation: str = ""
