"""Metric aggregation, ranking, and recommendation logic."""

from .common import *
from .schemas import MethodMetrics

def _minmax(arr: List[float]) -> List[float]:
    mn, mx = min(arr), max(arr)
    if mx == mn:
        return [0.5] * len(arr)
    return [(v - mn) / (mx - mn) for v in arr]

def rank_methods(method_metrics: List[MethodMetrics],
                 weights: Dict[str, float]) -> List[MethodMetrics]:
    if not method_metrics:
        return method_metrics

    def extract(attr): return [getattr(m, attr) for m in method_metrics]

    # normalise positive metrics (higher = better) → normalised directly
    rel    = _minmax(extract("avg_relevance"))
    corr   = _minmax(extract("avg_correctness"))
    compl  = _minmax(extract("avg_completeness"))
    faith  = _minmax(extract("avg_faithfulness"))
    ctx_s  = _minmax(extract("avg_context_sufficiency"))
    sim    = _minmax(extract("avg_similarity"))

    # normalise negative metrics (lower = better) → invert
    lat_raw  = extract("total_pipeline_time_ms")
    tok_raw  = extract("avg_context_tokens")
    hall_raw = extract("avg_hallucination_risk")
    chunk_raw= extract("total_chunks")

    lat_n   = [1.0 - v for v in _minmax(lat_raw)]
    tok_n   = [1.0 - v for v in _minmax(tok_raw)]
    hall_n  = [1.0 - v for v in _minmax(hall_raw)]
    chunk_n = [1.0 - v for v in _minmax(chunk_raw)]

    answer_quality   = [(r + co + cp) / 3 for r, co, cp in zip(rel, corr, compl)]
    retrieval_quality= [(s + cs) / 2 for s, cs in zip(sim, ctx_s)]
    faithfulness_n   = [(f + 1 - h) / 2 for f, h in zip(faith, hall_n)]
    speed_n          = lat_n
    token_n          = tok_n
    chunk_q          = chunk_n

    w = weights
    for i, m in enumerate(method_metrics):
        score = (
            w.get("answer_quality", 0.4)    * answer_quality[i]
            + w.get("retrieval_quality", 0.25) * retrieval_quality[i]
            + w.get("faithfulness", 0.15)      * faithfulness_n[i]
            + w.get("speed_efficiency", 0.10)  * speed_n[i]
            + w.get("token_efficiency", 0.05)  * token_n[i]
            + w.get("chunk_quality", 0.05)     * chunk_q[i]
        )
        m.final_score = round(score, 4)

    method_metrics.sort(key=lambda x: x.final_score, reverse=True)
    for rank, m in enumerate(method_metrics, start=1):
        m.rank = rank
        m.strengths, m.weaknesses = _compute_sw(m)
        m.recommendation = _build_recommendation(m, method_metrics)

    return method_metrics

def _compute_sw(m: MethodMetrics) -> Tuple[List[str], List[str]]:
    strengths, weaknesses = [], []
    if m.avg_correctness > 0.65:  strengths.append("High answer correctness")
    if m.avg_relevance   > 0.65:  strengths.append("High answer relevance")
    if m.avg_faithfulness > 0.65: strengths.append("Strong faithfulness")
    if m.total_pipeline_time_ms < 2000: strengths.append("Fast pipeline")
    if m.avg_similarity  > 0.60:  strengths.append("Good retrieval similarity")
    if m.avg_correctness < 0.35:  weaknesses.append("Low answer correctness")
    if m.avg_faithfulness < 0.35: weaknesses.append("Low faithfulness / hallucination risk")
    if m.total_pipeline_time_ms > 10000: weaknesses.append("Slow pipeline")
    if m.total_chunks < 3:        weaknesses.append("Too few chunks (low coverage)")
    if m.avg_hallucination_risk > 0.5: weaknesses.append("High hallucination risk")
    if not strengths: strengths.append("Balanced performance")
    if not weaknesses: weaknesses.append("No major weaknesses detected")
    return strengths, weaknesses

def _build_recommendation(m: MethodMetrics, all_methods: List[MethodMetrics]) -> str:
    if m.rank == 1:
        return (f"{m.method} ranked first with a score of {m.final_score:.3f}. "
                f"Key strengths: {'; '.join(m.strengths[:2])}.")
    best = all_methods[0]
    return (f"{m.method} ranked #{m.rank}. "
            f"Compared to {best.method} (rank 1), it has "
            f"{'better' if m.total_pipeline_time_ms < best.total_pipeline_time_ms else 'slower'} "
            f"pipeline speed but "
            f"{'higher' if m.avg_correctness > best.avg_correctness else 'lower'} correctness.")
