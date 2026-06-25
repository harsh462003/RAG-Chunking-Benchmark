"""Streamlit entry point for the RAG Chunking Benchmark application."""

from rag_chunking_benchmark import *

st.set_page_config(
    page_title="RAG Chunking Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0d14;
    color: #d4dae8;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #131929, #1a2440);
    border: 1px solid #2a3655;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
}
.rank-badge {
    background: linear-gradient(90deg, #4F8EF7, #8F4FF7);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9em;
}
.method-tag {
    background: #1e2d4a;
    border: 1px solid #3a5080;
    color: #7eb8f7;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.85em;
    font-family: 'JetBrains Mono', monospace;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #0e1520;
    border-bottom: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    color: #6a80a8;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #4F8EF7 !important;
    border-bottom: 2px solid #4F8EF7 !important;
}
.stButton>button {
    background: linear-gradient(90deg, #4F8EF7, #7F4FF7);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 0.5rem 2rem;
}
.stButton>button:hover { opacity: 0.85; }
.stAlert { border-radius: 8px; }
hr { border-color: #1e2d4a; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ RAG Benchmark")
    st.caption("Chunking Strategy Evaluator")
    st.divider()

    st.markdown("### 🔧 Models")
    embed_model_name = st.text_input(
        "Embedding Model", DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers model name or HuggingFace path"
    )
    gen_mode = st.radio("Generation Mode", ["Local (Flan-T5)", "API (OpenAI-compatible)"], index=0)
    gen_model_name = DEFAULT_LOCAL_GEN_MODEL
    api_key, api_model, api_base = "", "gpt-4o-mini", ""
    if gen_mode == "API (OpenAI-compatible)":
        api_key   = st.text_input("API Key", type="password")
        api_model = st.text_input("API Model", "gpt-4o-mini")
        api_base  = st.text_input("API Base URL (optional)", "")
        use_llm_judge = st.checkbox("Use LLM as Judge", value=True)
    else:
        gen_model_name = st.text_input("Local Gen Model", DEFAULT_LOCAL_GEN_MODEL)
        use_llm_judge  = False

    st.divider()
    st.markdown("### 🎯 Retrieval")
    top_k = st.slider("Top-K chunks", 1, 15, DEFAULT_TOP_K)

    st.divider()
    st.markdown("### ⚖️ Ranking Weights")
    wt_answer   = st.slider("Answer Quality",    0.0, 1.0, 0.40, 0.05)
    wt_retrieval= st.slider("Retrieval Quality", 0.0, 1.0, 0.25, 0.05)
    wt_faith    = st.slider("Faithfulness",      0.0, 1.0, 0.15, 0.05)
    wt_speed    = st.slider("Speed Efficiency",  0.0, 1.0, 0.10, 0.05)
    wt_token    = st.slider("Token Efficiency",  0.0, 1.0, 0.05, 0.05)
    wt_chunk    = st.slider("Chunk Quality",     0.0, 1.0, 0.05, 0.05)
    total_w = wt_answer + wt_retrieval + wt_faith + wt_speed + wt_token + wt_chunk
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum = {total_w:.2f} (should be 1.0). Normalising.")
    weights = {
        "answer_quality":    wt_answer / total_w,
        "retrieval_quality": wt_retrieval / total_w,
        "faithfulness":      wt_faith / total_w,
        "speed_efficiency":  wt_speed / total_w,
        "token_efficiency":  wt_token / total_w,
        "chunk_quality":     wt_chunk / total_w,
    }

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_setup, tab_run, tab_leaderboard, tab_method, tab_question, tab_chunks, tab_metrics, tab_charts, tab_export = st.tabs([
    "📖 Overview", "⚙️ Setup", "🚀 Run", "🏆 Leaderboard",
    "🔍 Method Details", "❓ Questions", "📄 Chunks",
    "📊 Metrics", "📈 Charts", "💾 Export",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB: OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("# ⚡ RAG Chunking Benchmark System")
    st.markdown("""
> **Goal:** Help you decide *which chunking strategy* is best for your document, question set, and application requirements.

This system applies **8 chunking strategies** to the same document, runs a full RAG pipeline for each, judges every answer, and produces a ranked leaderboard with full diagnostics.

---

### Workflow
1. **Upload** a PDF / DOCX / TXT document
2. **Add test questions** (manual, CSV, TXT, or JSON)
3. **Select chunking methods** and adjust parameters
4. **Hit Run** — the pipeline handles chunking → embedding → indexing → retrieval → generation → evaluation
5. **Inspect results** across Leaderboard, Method Details, Questions, Charts, and Export tabs

---

### Chunking Methods
| Method | Best For |
|--------|----------|
| `fixed_size` | Simple, predictable baseline |
| `fixed_overlap` | When context continuity matters |
| `sentence_based` | Dialogue, QA corpora |
| `paragraph_based` | Well-structured documents |
| `semantic_chunking` | Thematically diverse docs |
| `sliding_window` | Dense technical text |
| `recursive_character_splitter` | Mixed-format documents |
| `hybrid_chunking` | General-purpose best-effort |

---

### Evaluation Dimensions
- **Answer Quality** — relevance, correctness, completeness
- **Faithfulness** — grounded in retrieved context
- **Retrieval Quality** — similarity scores, context sufficiency
- **Efficiency** — latency, token usage
- **Chunk Quality** — distribution, coverage

---
    """)
    cols = st.columns(4)
    cols[0].metric("Chunking Methods", "8")
    cols[1].metric("Eval Dimensions", "6")
    cols[2].metric("Export Formats", "6")
    cols[3].metric("Chart Types", "6")

# ────────────────────────────────────────────────────────────────────────────
# TAB: SETUP
# ────────────────────────────────────────────────────────────────────────────
with tab_setup:
    st.markdown("## ⚙️ Setup: Document & Questions")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 📄 Upload Document")
        uploaded_doc = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"],
                                         key="doc_upload")
        if uploaded_doc:
            if "doc_text" not in st.session_state or st.session_state.get("doc_name") != uploaded_doc.name:
                with st.spinner("Extracting text…"):
                    text, pages, dname = extract_document(uploaded_doc)
                    st.session_state["doc_text"] = text
                    st.session_state["doc_name"] = dname
                    st.session_state["doc_pages"] = pages
                    st.session_state["doc_stats"] = doc_statistics(text, pages)

            stats = st.session_state.get("doc_stats", {})
            st.success(f"✅ {st.session_state['doc_name']}")
            sc = st.columns(5)
            sc[0].metric("Characters",  f"{stats.get('characters',0):,}")
            sc[1].metric("Words",       f"{stats.get('words',0):,}")
            sc[2].metric("Sentences",   f"{stats.get('sentences',0):,}")
            sc[3].metric("Est. Tokens", f"{stats.get('estimated_tokens',0):,}")
            sc[4].metric("Pages",       stats.get("pages", "N/A"))
            with st.expander("Preview first 1000 chars"):
                st.text(st.session_state["doc_text"][:1000])

    with c2:
        st.markdown("### ❓ Questions")
        q_mode = st.radio("Input mode", ["Manual text", "Upload file"], horizontal=True)
        manual_q = ""
        q_file = None
        if q_mode == "Manual text":
            manual_q = st.text_area("One question per line", height=200,
                                    placeholder="What is the main topic?\nWhen was this published?")
        else:
            q_file = st.file_uploader("CSV / TXT / JSON", type=["csv", "txt", "json"], key="q_upload")

        if st.button("Parse Questions"):
            questions = parse_questions(manual_q, q_file)
            st.session_state["questions"] = questions
            st.success(f"✅ {len(questions)} question(s) parsed")

        if "questions" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["questions"]), width='stretch')

    st.divider()
    st.markdown("### 🔩 Chunking Method Selection & Parameters")
    selected_methods = st.multiselect(
        "Select methods to benchmark",
        CHUNKING_METHODS,
        default=["fixed_overlap", "sentence_based", "paragraph_based", "semantic_chunking"],
    )
    st.session_state["selected_methods"] = selected_methods

    if selected_methods:
        st.markdown("#### Per-Method Parameters")
        chunking_params = {}
        cols_per_row = 2
        method_groups = [selected_methods[i:i+cols_per_row]
                         for i in range(0, len(selected_methods), cols_per_row)]
        for group in method_groups:
            cols = st.columns(cols_per_row)
            for ci, method in enumerate(group):
                with cols[ci]:
                    with st.expander(f"`{method}`", expanded=False):
                        p = {}
                        if method == "fixed_size":
                            p["chunk_size_words"] = st.number_input("chunk_size_words", 50, 1000, 300, key=f"{method}_csw")
                        elif method == "fixed_overlap":
                            p["chunk_size_words"] = st.number_input("chunk_size_words", 50, 1000, 300, key=f"{method}_csw")
                            p["overlap_words"]    = st.number_input("overlap_words", 0, 200, 50, key=f"{method}_ow")
                        elif method == "sentence_based":
                            p["sentences_per_chunk"] = st.number_input("sentences_per_chunk", 1, 20, 5, key=f"{method}_spc")
                        elif method == "paragraph_based":
                            p["merge_short"] = st.checkbox("merge_short_paragraphs", True, key=f"{method}_ms")
                            p["min_words"]   = st.number_input("min_words", 10, 300, 80, key=f"{method}_mw")
                        elif method == "semantic_chunking":
                            p["similarity_threshold"] = st.slider("similarity_threshold", 0.0, 1.0, 0.65, 0.05, key=f"{method}_st")
                            p["min_chunk_words"]      = st.number_input("min_chunk_words", 20, 300, 100, key=f"{method}_minw")
                            p["max_chunk_words"]      = st.number_input("max_chunk_words", 100, 1000, 500, key=f"{method}_maxw")
                        elif method == "sliding_window":
                            p["window_size_words"] = st.number_input("window_size_words", 50, 1000, 300, key=f"{method}_ws")
                            p["step_size_words"]   = st.number_input("step_size_words", 10, 500, 150, key=f"{method}_ss")
                        elif method == "recursive_character_splitter":
                            p["chunk_size_chars"] = st.number_input("chunk_size_chars", 200, 5000, 1200, key=f"{method}_csc")
                            p["overlap_chars"]    = st.number_input("overlap_chars", 0, 500, 200, key=f"{method}_oc")
                        elif method == "hybrid_chunking":
                            p["max_words"] = st.number_input("max_words", 100, 1000, 400, key=f"{method}_mxw")
                            p["min_words"] = st.number_input("min_words", 10, 300, 80, key=f"{method}_mnw")
                        chunking_params[method] = p
        st.session_state["chunking_params"] = chunking_params

# ────────────────────────────────────────────────────────────────────────────
# TAB: RUN
# ────────────────────────────────────────────────────────────────────────────
with tab_run:
    st.markdown("## 🚀 Run Benchmark")

    prereqs_ok = True
    if "doc_text" not in st.session_state or not st.session_state["doc_text"]:
        st.warning("⚠️ No document loaded. Go to **Setup** tab first.")
        prereqs_ok = False
    if "questions" not in st.session_state or not st.session_state["questions"]:
        st.warning("⚠️ No questions parsed. Go to **Setup** tab first.")
        prereqs_ok = False
    if "selected_methods" not in st.session_state or not st.session_state["selected_methods"]:
        st.warning("⚠️ No chunking methods selected. Go to **Setup** tab first.")
        prereqs_ok = False

    if prereqs_ok:
        n_methods  = len(st.session_state.get("selected_methods", []))
        n_questions= len(st.session_state.get("questions", []))
        st.info(f"Ready to benchmark **{n_methods}** method(s) × **{n_questions}** question(s) = **{n_methods * n_questions}** pipeline runs")

        if st.button("▶ Run Full Benchmark", use_container_width=True):
            # load models
            progress_bar = st.progress(0.0, text="Loading models…")
            log_area     = st.empty()
            log_lines    = []

            def log(msg):
                log_lines.append(msg)
                log_area.text("\n".join(log_lines[-20:]))

            log("Loading embedding model…")
            embed_model = load_embedding_model(embed_model_name)
            log(f"Embedding model ready: {embed_model_name}")

            gen_tok, gen_mdl = None, None
            if gen_mode == "Local (Flan-T5)":
                log("Loading local generation model…")
                gen_tok, gen_mdl = load_local_gen_model(gen_model_name)
                if gen_mdl:
                    log(f"Generation model ready: {gen_model_name}")
                else:
                    log("Generation model failed to load; answers may be unavailable.")

            def progress_cb(frac: float, msg: str):
                progress_bar.progress(min(frac, 1.0), text=f"Processing: {msg}")
                log(f"  {msg}")

            log("Starting benchmark…")
            method_metrics, qa_records, full_results = run_benchmark(
                doc_text=st.session_state["doc_text"],
                doc_name=st.session_state["doc_name"],
                questions=st.session_state["questions"],
                selected_methods=st.session_state["selected_methods"],
                chunking_params=st.session_state.get("chunking_params", {}),
                embed_model=embed_model,
                gen_tokenizer=gen_tok,
                gen_model=gen_mdl,
                api_key=api_key,
                api_model=api_model,
                api_base=api_base if api_base else None,
                top_k=top_k,
                weights=weights,
                use_llm_judge=use_llm_judge,
                progress_cb=progress_cb,
            )
            progress_bar.progress(1.0, text="✅ Benchmark complete!")
            st.session_state["method_metrics"] = method_metrics
            st.session_state["qa_records"]     = qa_records
            st.session_state["full_results"]   = full_results
            log("✅ Benchmark complete! Switch to the Leaderboard tab.")
            st.success("✅ Done! Navigate to the **Leaderboard** and other tabs to explore results.")

# ────────────────────────────────────────────────────────────────────────────
# TAB: LEADERBOARD
# ────────────────────────────────────────────────────────────────────────────
with tab_leaderboard:
    st.markdown("## 🏆 Leaderboard")
    if "method_metrics" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        mm_list = st.session_state["method_metrics"]
        best = mm_list[0]
        st.success(f"🥇 **Best method: `{best.method}`** — Score {best.final_score:.3f}  |  {best.recommendation}")

        rows = []
        for m in mm_list:
            rows.append({
                "Rank": f"#{m.rank}",
                "Method": m.method,
                "Score": f"{m.final_score:.3f}",
                "Relevance": f"{m.avg_relevance:.3f}",
                "Correctness": f"{m.avg_correctness:.3f}",
                "Faithfulness": f"{m.avg_faithfulness:.3f}",
                "Avg Sim": f"{m.avg_similarity:.3f}",
                "Latency(ms)": f"{m.total_pipeline_time_ms:.0f}",
                "Ctx Tokens": f"{m.avg_context_tokens:.0f}",
                "Chunks": m.total_chunks,
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        if PLOTLY_AVAILABLE:
            st.plotly_chart(chart_final_scores(mm_list), key='lb_scores')

# ────────────────────────────────────────────────────────────────────────────
# TAB: METHOD DETAILS
# ────────────────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown("## 🔍 Method Details")
    if "method_metrics" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        mm_list = st.session_state["method_metrics"]
        selected_m = st.selectbox("Select method", [m.method for m in mm_list])
        m = next(x for x in mm_list if x.method == selected_m)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rank", f"#{m.rank}")
        c2.metric("Final Score", f"{m.final_score:.3f}")
        c3.metric("Total Chunks", m.total_chunks)
        c4.metric("Avg Chunk Size", f"{m.avg_chunk_size_words:.0f} words")

        st.markdown("#### ⏱ Timing")
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Chunking",   f"{m.chunking_time_ms:.0f} ms")
        tc2.metric("Embedding",  f"{m.embedding_time_ms:.0f} ms")
        tc3.metric("Indexing",   f"{m.index_creation_time_ms:.0f} ms")
        tc4.metric("Retrieval",  f"{m.avg_retrieval_latency_ms:.0f} ms")
        tc5.metric("Generation", f"{m.avg_generation_latency_ms:.0f} ms")

        st.markdown("#### 📈 Quality Scores")
        qc = st.columns(6)
        qc[0].metric("Relevance",    m.avg_relevance)
        qc[1].metric("Correctness",  m.avg_correctness)
        qc[2].metric("Completeness", m.avg_completeness)
        qc[3].metric("Faithfulness", m.avg_faithfulness)
        qc[4].metric("Ctx Suff.",    m.avg_context_sufficiency)
        qc[5].metric("Hall. Risk",   m.avg_hallucination_risk)

        with st.expander("✅ Strengths / ⚠️ Weaknesses"):
            st.markdown("**Strengths:**")
            for s in m.strengths: st.markdown(f"  - {s}")
            st.markdown("**Weaknesses:**")
            for w in m.weaknesses: st.markdown(f"  - {w}")
            st.markdown(f"**Recommendation:** {m.recommendation}")

        # sample chunks from full_results
        if "full_results" in st.session_state:
            method_data = st.session_state["full_results"]["methods"].get(selected_m, {})
            sample_chunks = method_data.get("chunks_sample", [])
            if sample_chunks:
                with st.expander(f"Sample Chunks (first {len(sample_chunks)})"):
                    for c_dict in sample_chunks:
                        st.markdown(f"**Chunk {c_dict['chunk_index']}** — "
                                    f"{c_dict['chunk_word_count']} words / "
                                    f"{c_dict['chunk_token_count']} tokens")
                        st.text(c_dict["chunk_text"][:400])
                        st.divider()

# ────────────────────────────────────────────────────────────────────────────
# TAB: QUESTION ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tab_question:
    st.markdown("## ❓ Question-wise Analysis")
    if "qa_records" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        qa_records = st.session_state["qa_records"]
        questions  = st.session_state.get("questions", [])
        q_ids = list(dict.fromkeys([r["question_id"] for r in qa_records]))

        selected_q = st.selectbox("Select question", q_ids)
        q_recs = [r for r in qa_records if r["question_id"] == selected_q]

        if q_recs:
            st.markdown(f"**{q_recs[0]['question_text']}**")
            for rec in q_recs:
                with st.expander(f"`{rec['method']}` — Correctness: {rec['correctness']:.2f}  |  "
                                 f"Faithfulness: {rec['faithfulness']:.2f}"):
                    ac1, ac2, ac3 = st.columns(3)
                    ac1.metric("Relevance",   f"{rec['relevance']:.3f}")
                    ac2.metric("Correctness", f"{rec['correctness']:.3f}")
                    ac3.metric("Faithfulness",f"{rec['faithfulness']:.3f}")
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Completeness",    f"{rec['completeness']:.3f}")
                    bc2.metric("Context Suff.",   f"{rec['context_sufficiency']:.3f}")
                    bc3.metric("Hall. Risk",      f"{rec['hallucination_risk']:.3f}")
                    st.markdown("**Answer:**")
                    st.info(rec["answer"] or "*(no answer)*")
                    st.caption(f"Generation: {rec['generation_latency_ms']:.0f} ms | "
                               f"Answer tokens: {rec['answer_token_count']} | "
                               f"Status: {rec['answer_status']}")
                    if rec.get("judge_reason"):
                        st.caption(f"Judge note: {rec['judge_reason']}")

# ────────────────────────────────────────────────────────────────────────────
# TAB: RETRIEVED CHUNKS
# ────────────────────────────────────────────────────────────────────────────
with tab_chunks:
    st.markdown("## 📄 Retrieved Chunk Viewer")
    if "qa_records" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        qa_records = st.session_state["qa_records"]
        q_ids = list(dict.fromkeys([r["question_id"] for r in qa_records]))
        methods = list(dict.fromkeys([r["method"] for r in qa_records]))

        cc1, cc2 = st.columns(2)
        sel_q = cc1.selectbox("Question", q_ids, key="chunk_q")
        sel_m = cc2.selectbox("Method", methods, key="chunk_m")

        match = next((r for r in qa_records
                      if r["question_id"] == sel_q and r["method"] == sel_m), None)
        if match:
            st.markdown(f"**Q:** {match['question_text']}")
            st.caption(f"Context tokens: {match['context_token_count']}  |  "
                       f"Top sim: {match['top_similarity']:.4f}  |  "
                       f"Avg sim: {match['avg_similarity']:.4f}  |  "
                       f"Retrieval: {match['retrieval_latency_ms']:.1f} ms")
            for rc in match.get("retrieved_chunks", []):
                st.markdown(f"**Rank {rc['rank']}** — Score: `{rc['score']:.4f}` — "
                            f"{rc['chunk_words']} words — `{rc['chunk_id']}`")
                st.text(rc["chunk_text"])
                st.divider()

# ────────────────────────────────────────────────────────────────────────────
# TAB: METRICS
# ────────────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("## 📊 Detailed Metric Tables")
    if "method_metrics" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        mm_list = st.session_state["method_metrics"]

        st.markdown("### Chunk Statistics")
        chunk_rows = [{
            "Method": m.method, "Chunks": m.total_chunks,
            "Avg Words": f"{m.avg_chunk_size_words:.1f}",
            "Std Dev": f"{m.std_chunk_size:.1f}",
            "Min": m.min_chunk_size, "Max": m.max_chunk_size,
        } for m in mm_list]
        st.dataframe(pd.DataFrame(chunk_rows), width='stretch', hide_index=True)

        st.markdown("### Timing (ms)")
        timing_rows = [{
            "Method": m.method,
            "Chunking": f"{m.chunking_time_ms:.1f}",
            "Embedding": f"{m.embedding_time_ms:.1f}",
            "Indexing": f"{m.index_creation_time_ms:.1f}",
            "Avg Retrieval": f"{m.avg_retrieval_latency_ms:.1f}",
            "Avg Generation": f"{m.avg_generation_latency_ms:.1f}",
            "Total": f"{m.total_pipeline_time_ms:.1f}",
        } for m in mm_list]
        st.dataframe(pd.DataFrame(timing_rows), width='stretch', hide_index=True)

        st.markdown("### Quality Scores")
        quality_rows = [{
            "Method": m.method,
            "Relevance": m.avg_relevance,
            "Correctness": m.avg_correctness,
            "Completeness": m.avg_completeness,
            "Faithfulness": m.avg_faithfulness,
            "Ctx Suff.": m.avg_context_sufficiency,
            "Hall. Risk": m.avg_hallucination_risk,
            "Avg Sim": m.avg_similarity,
        } for m in mm_list]
        st.dataframe(pd.DataFrame(quality_rows), width='stretch', hide_index=True)

        st.markdown("### Token Usage")
        token_rows = [{
            "Method": m.method,
            "Avg Context Tokens": m.avg_context_tokens,
            "Avg Answer Tokens": m.avg_answer_tokens,
        } for m in mm_list]
        st.dataframe(pd.DataFrame(token_rows), width='stretch', hide_index=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB: CHARTS
# ────────────────────────────────────────────────────────────────────────────
with tab_charts:
    st.markdown("## 📈 Visual Charts")
    if not PLOTLY_AVAILABLE:
        st.warning("plotly not installed. Install with `pip install plotly`.")
    elif "method_metrics" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        mm_list = st.session_state["method_metrics"]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_final_scores(mm_list), key='ch_scores')
        with c2:
            st.plotly_chart(chart_answer_quality(mm_list), key='ch_quality')
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_latency(mm_list), key='ch_latency')
        with c4:
            st.plotly_chart(chart_tokens(mm_list), key='ch_tokens')
        st.plotly_chart(chart_scatter(mm_list), key='ch_scatter')
        st.plotly_chart(chart_radar(mm_list), key='ch_radar')

# ────────────────────────────────────────────────────────────────────────────
# TAB: EXPORT
# ────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("## 💾 Export Results")
    if "method_metrics" not in st.session_state:
        st.info("Run the benchmark first.")
    else:
        mm_list     = st.session_state["method_metrics"]
        qa_records  = st.session_state["qa_records"]
        full_results= st.session_state["full_results"]
        questions   = st.session_state.get("questions", [])
        doc_name    = st.session_state.get("doc_name", "document")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "⬇ full_results.json",
                data=export_json(full_results),
                file_name="full_results.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "⬇ leaderboard.csv",
                data=export_leaderboard_csv(mm_list),
                file_name="leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇ per_question_results.csv",
                data=export_per_question_csv(qa_records),
                file_name="per_question_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            # retrieval CSV
            ret_rows = []
            for rec in qa_records:
                for rc in rec.get("retrieved_chunks", []):
                    ret_rows.append({
                        "question_id": rec["question_id"],
                        "method": rec["method"],
                        "chunk_rank": rc["rank"],
                        "chunk_id": rc["chunk_id"],
                        "similarity_score": rc["score"],
                        "chunk_words": rc["chunk_words"],
                        "chunk_text_preview": rc["chunk_text"][:200],
                    })
            st.download_button(
                "⬇ retrieval_results.csv",
                data=pd.DataFrame(ret_rows).to_csv(index=False),
                file_name="retrieval_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col3:
            method_summary_rows = [asdict(m) for m in mm_list]
            st.download_button(
                "⬇ method_summary.csv",
                data=pd.DataFrame(method_summary_rows).to_csv(index=False),
                file_name="method_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            report_md = export_markdown_report(doc_name, questions, mm_list, qa_records)
            st.download_button(
                "⬇ final_report.md",
                data=report_md,
                file_name="final_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("Preview: final_report.md"):
            st.markdown(report_md)
