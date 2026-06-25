# RAG Chunking Strategy Benchmarking and Recommendation System

A professional Streamlit project that benchmarks multiple text chunking strategies for Retrieval-Augmented Generation (RAG). The app accepts a document and a set of test questions, runs the same RAG pipeline across different chunking methods, evaluates the results, and produces a ranked recommendation.

## Project structure

```text
rag-chunking-benchmark/
├── app.py                         # Streamlit UI entry point
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Package metadata
├── README.md                      # Project overview and run steps
├── .gitignore                     # Files Git should ignore
├── data/
│   ├── sample_documents/          # Example PDF documents for testing
│   └── sample_questions/          # Example question sets
├── docs/
│   └── VERSION1.docx              # Full project documentation/report
├── outputs/                       # Keep exported benchmark outputs here
├── src/
│   └── rag_chunking_benchmark/
│       ├── __init__.py
│       ├── common.py              # Shared constants/imports/dependency flags
│       ├── schemas.py             # Dataclasses for chunks, retrieval, answers, metrics
│       ├── utils.py               # Text statistics and similarity helpers
│       ├── document_loaders.py    # PDF/DOCX/TXT parsing
│       ├── questions.py           # Question parsing from text/CSV/JSON
│       ├── chunking.py            # 8 chunking strategies
│       ├── embeddings.py          # SentenceTransformer embeddings + FAISS index build
│       ├── retrieval.py           # Top-k chunk retrieval
│       ├── generation.py          # Local Flan-T5/API answer generation
│       ├── evaluation.py          # Heuristic judge + optional LLM judge
│       ├── ranking.py             # Weighted scoring and recommendation logic
│       ├── exports.py             # JSON/CSV/Markdown exports
│       ├── pipeline.py            # End-to-end benchmark orchestration
│       └── visualization.py       # Plotly charts
└── tests/
    └── test_smoke.py              # Basic import smoke test
```

## Why this version is cleaner

The original project logic has been separated by responsibility. The Streamlit file now focuses only on UI, while the actual RAG logic lives inside reusable modules under `src/rag_chunking_benchmark/`. This makes the repository easier to read, debug, extend, and present on GitHub.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scriptsctivate          # Windows

pip install -r requirements.txt
streamlit run app.py
```

The app should open at `http://localhost:8501`.

## Sample data

Use the files inside `data/sample_documents/` and `data/sample_questions/` to reproduce the benchmark runs described in `docs/VERSION1.docx`.

## Notes

- Local generation uses `google/flan-t5-base`, which can be slow on CPU.
- API mode is optional and requires an OpenAI-compatible API key.
- Exported results can be saved under `outputs/`.
