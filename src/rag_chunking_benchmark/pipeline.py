"""End-to-end benchmark orchestration."""

from .common import *
from .schemas import *
from .chunking import apply_chunking, chunk_stats
from .embeddings import embed_chunks, build_faiss_index
from .retrieval import retrieve_top_k
from .generation import generate_answer_local, generate_answer_api
from .evaluation import heuristic_judge, llm_judge
from .ranking import rank_methods

def run_benchmark(
    doc_text: str, doc_name: str,
    questions: List[Dict],
    selected_methods: List[str],
    chunking_params: Dict[str, Dict],
    embed_model,
    gen_tokenizer, gen_model,
    api_key: str, api_model: str, api_base: Optional[str],
    top_k: int,
    weights: Dict[str, float],
    use_llm_judge: bool,
    progress_cb=None,
) -> Tuple[List[MethodMetrics], List[Dict], Dict]:

    all_qa_records: List[Dict] = []
    method_metrics_list: List[MethodMetrics] = []

    total_steps = len(selected_methods) * len(questions)
    step = 0

    def update_progress(msg: str):
        nonlocal step
        step += 1
        if progress_cb:
            progress_cb(step / total_steps, msg)

    full_results: Dict = {
        "document": doc_name,
        "questions": questions,
        "methods": {},
    }

    for method in selected_methods:
        mm = MethodMetrics(method=method)
        params = chunking_params.get(method, {})

        # --- embed_fn for semantic methods ---
        def embed_fn(texts):
            return embed_texts(embed_model, texts)

        # 1. Chunking
        chunks, chunk_time_ms = apply_chunking(method, doc_text, doc_name, params, embed_fn)
        mm.chunking_time_ms = chunk_time_ms
        stats = chunk_stats(chunks)
        mm.total_chunks         = stats.get("total_chunks", 0)
        mm.avg_chunk_size_words = stats.get("avg_chunk_size_words", 0.0)
        mm.std_chunk_size       = stats.get("std_chunk_size", 0.0)
        mm.min_chunk_size       = stats.get("min_chunk_size", 0)
        mm.max_chunk_size       = stats.get("max_chunk_size", 0)

        if not chunks:
            method_metrics_list.append(mm)
            continue

        # 2. Embeddings
        chunk_embeddings, embed_time_ms = embed_chunks(embed_model, chunks)
        mm.embedding_time_ms = embed_time_ms

        # 3. FAISS index
        faiss_index, index_time_ms = build_faiss_index(chunk_embeddings)
        mm.index_creation_time_ms = index_time_ms

        # 4. Per-question pipeline
        method_qa: List[Dict] = []
        retrieval_latencies, gen_latencies = [], []
        context_tokens_all, answer_tokens_all = [], []
        sims_all = []
        eval_scores_all: List[EvalScores] = []

        for q in questions:
            qid = q["id"]
            qtext = q["text"]

            # embed question
            q_vec = embed_texts(embed_model, [qtext])

            # retrieve
            retrieved, ret_lat = retrieve_top_k(faiss_index, q_vec, chunks, top_k)
            retrieval_latencies.append(ret_lat)
            context_text = "\n\n".join([r["chunk"].chunk_text for r in retrieved])
            ctx_tokens = count_tokens(context_text)
            context_tokens_all.append(ctx_tokens)
            top_sim = retrieved[0]["score"] if retrieved else 0.0
            avg_sim = float(np.mean([r["score"] for r in retrieved])) if retrieved else 0.0
            sims_all.append(avg_sim)

            # generate answer
            prompt = ANSWER_PROMPT.format(context=context_text, question=qtext)
            if api_key and api_key.strip():
                answer_text, gen_lat = generate_answer_api(api_key, api_model, prompt, api_base)
                model_used = api_model
            elif gen_model is not None:
                answer_text, gen_lat = generate_answer_local(gen_tokenizer, gen_model, prompt)
                model_used = api_model  # flan-t5
            else:
                answer_text = "INSUFFICIENT_CONTEXT [no generation model available]"
                gen_lat = 0.0
                model_used = "none"
            gen_latencies.append(gen_lat)
            ans_tokens = count_tokens(answer_text)
            answer_tokens_all.append(ans_tokens)
            answer_status = "INSUFFICIENT_CONTEXT" if "INSUFFICIENT_CONTEXT" in answer_text else "ANSWERED"

            # evaluate
            if use_llm_judge and api_key and api_key.strip():
                scores = llm_judge(api_key, api_model, qtext, context_text, answer_text, api_base)
            else:
                scores = heuristic_judge(qtext, context_text, answer_text)
            eval_scores_all.append(scores)

            rec = {
                "question_id": qid,
                "question_text": qtext,
                "method": method,
                "context_text": context_text,
                "context_token_count": ctx_tokens,
                "answer": answer_text,
                "answer_status": answer_status,
                "answer_token_count": ans_tokens,
                "model_used": model_used,
                "retrieval_latency_ms": round(ret_lat, 2),
                "generation_latency_ms": round(gen_lat, 2),
                "top_similarity": round(top_sim, 4),
                "avg_similarity": round(avg_sim, 4),
                "retrieved_chunks": [
                    {
                        "rank": r["rank"],
                        "score": round(r["score"], 4),
                        "chunk_id": r["chunk"].chunk_id,
                        "chunk_text": r["chunk"].chunk_text[:500],
                        "chunk_words": r["chunk"].chunk_word_count,
                    }
                    for r in retrieved
                ],
                "relevance": scores.relevance,
                "correctness": scores.correctness,
                "completeness": scores.completeness,
                "faithfulness": scores.faithfulness,
                "context_sufficiency": scores.context_sufficiency,
                "hallucination_risk": scores.hallucination_risk,
                "judge_reason": scores.reason,
            }
            method_qa.append(rec)
            all_qa_records.append(rec)
            update_progress(f"{method} | {qid}")

        # aggregate metrics
        if eval_scores_all:
            mm.avg_relevance          = round(float(np.mean([s.relevance for s in eval_scores_all])), 3)
            mm.avg_correctness        = round(float(np.mean([s.correctness for s in eval_scores_all])), 3)
            mm.avg_completeness       = round(float(np.mean([s.completeness for s in eval_scores_all])), 3)
            mm.avg_faithfulness       = round(float(np.mean([s.faithfulness for s in eval_scores_all])), 3)
            mm.avg_context_sufficiency= round(float(np.mean([s.context_sufficiency for s in eval_scores_all])), 3)
            mm.avg_hallucination_risk = round(float(np.mean([s.hallucination_risk for s in eval_scores_all])), 3)
        mm.avg_similarity         = round(float(np.mean(sims_all)) if sims_all else 0.0, 3)
        mm.avg_retrieval_latency_ms   = round(float(np.mean(retrieval_latencies)) if retrieval_latencies else 0.0, 2)
        mm.avg_generation_latency_ms  = round(float(np.mean(gen_latencies)) if gen_latencies else 0.0, 2)
        mm.avg_context_tokens         = round(float(np.mean(context_tokens_all)) if context_tokens_all else 0.0, 1)
        mm.avg_answer_tokens          = round(float(np.mean(answer_tokens_all)) if answer_tokens_all else 0.0, 1)
        mm.total_pipeline_time_ms     = round(
            mm.chunking_time_ms + mm.embedding_time_ms + mm.index_creation_time_ms
            + sum(retrieval_latencies) + sum(gen_latencies), 2
        )

        method_metrics_list.append(mm)
        full_results["methods"][method] = {
            "metrics": asdict(mm),
            "chunks_sample": [asdict(c) for c in chunks[:5]],
            "total_chunks": len(chunks),
            "qa": method_qa,
        }

    # ranking
    method_metrics_list = rank_methods(method_metrics_list, weights)

    return method_metrics_list, all_qa_records, full_results
