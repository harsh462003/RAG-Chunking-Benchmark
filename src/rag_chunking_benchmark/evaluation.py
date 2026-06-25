"""Heuristic and optional LLM-as-judge evaluation."""

from .common import *
from .schemas import EvalScores
from .utils import overlap_ratio

def heuristic_judge(question: str, context: str, answer: str) -> EvalScores:
    """Lightweight lexical heuristics when LLM judge is unavailable."""
    if not answer or answer.strip() == "":
        return EvalScores(reason="empty answer")

    if "INSUFFICIENT_CONTEXT" in answer or "API_ERROR" in answer:
        return EvalScores(
            relevance=0.1, correctness=0.0, completeness=0.0,
            faithfulness=0.5, context_sufficiency=0.0, hallucination_risk=0.3,
            reason="model reported insufficient context"
        )

    q_words = set(re.findall(r'\w+', question.lower()))
    a_words = set(re.findall(r'\w+', answer.lower()))
    c_words = set(re.findall(r'\w+', context.lower()))

    # relevance: overlap between question keywords and answer
    relevance = safe_divide(len(q_words & a_words), len(q_words), 0.3)
    relevance = min(1.0, relevance * 2.5)

    # faithfulness: answer words that appear in context
    if a_words:
        faithfulness = safe_divide(len(a_words & c_words), len(a_words), 0.5)
        faithfulness = min(1.0, faithfulness * 1.2)
    else:
        faithfulness = 0.0

    # completeness: rough length heuristic (penalise very short answers)
    ans_len = len(answer.split())
    completeness = min(1.0, ans_len / 30.0)

    # context sufficiency: does context cover question keywords?
    ctx_sufficiency = safe_divide(len(q_words & c_words), len(q_words), 0.5)
    ctx_sufficiency = min(1.0, ctx_sufficiency * 1.5)

    # hallucination risk: answer words NOT in context (excluding stopwords)
    stopwords = {"the","a","an","is","are","was","were","in","on","of","to","for","and","or","it"}
    non_stopword_a = a_words - stopwords
    out_of_context = non_stopword_a - c_words - q_words
    if non_stopword_a:
        hallucination_risk = safe_divide(len(out_of_context), len(non_stopword_a), 0.2)
        hallucination_risk = min(1.0, hallucination_risk * 0.7)
    else:
        hallucination_risk = 0.2

    # correctness: approximate by faithfulness + relevance
    correctness = min(1.0, (faithfulness + relevance) / 2.0)

    return EvalScores(
        relevance=round(relevance, 3),
        correctness=round(correctness, 3),
        completeness=round(completeness, 3),
        faithfulness=round(faithfulness, 3),
        context_sufficiency=round(ctx_sufficiency, 3),
        hallucination_risk=round(hallucination_risk, 3),
        reason="heuristic scoring (no LLM judge)"
    )

def llm_judge(api_key: str, model_name: str,
              question: str, context: str, answer: str,
              base_url: Optional[str] = None) -> EvalScores:
    judge_prompt = f"""You are an expert RAG evaluation judge.
Evaluate the following answer based on the context provided.
Return ONLY valid JSON with keys: relevance, correctness, completeness, faithfulness, context_sufficiency, hallucination_risk, reason.
All numeric scores must be floats between 0.0 and 1.0.

Context:
{context[:2000]}

Question:
{question}

Answer:
{answer}

Respond with JSON only:"""
    raw, _ = generate_answer_api(api_key, model_name, judge_prompt, base_url)
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        return EvalScores(
            relevance=float(data.get("relevance", 0.5)),
            correctness=float(data.get("correctness", 0.5)),
            completeness=float(data.get("completeness", 0.5)),
            faithfulness=float(data.get("faithfulness", 0.5)),
            context_sufficiency=float(data.get("context_sufficiency", 0.5)),
            hallucination_risk=float(data.get("hallucination_risk", 0.2)),
            reason=str(data.get("reason", "")),
        )
    except Exception:
        return heuristic_judge(question, context, answer)
