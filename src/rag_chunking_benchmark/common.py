"""Shared imports, optional dependency flags, and project-wide constants."""

import io
import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# macOS / Apple Silicon OpenMP safety. Must be set before heavy ML imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

HF_AVAILABLE = True

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_GEN_MODEL = "google/flan-t5-base"
DEFAULT_TOP_K = 5

DEFAULT_RANKING_WEIGHTS = {
    "answer_quality": 0.40,
    "retrieval_quality": 0.25,
    "faithfulness": 0.15,
    "speed_efficiency": 0.10,
    "token_efficiency": 0.05,
    "chunk_quality": 0.05,
}

CHUNKING_METHODS = [
    "fixed_size",
    "fixed_overlap",
    "sentence_based",
    "paragraph_based",
    "semantic_chunking",
    "sliding_window",
    "recursive_character_splitter",
    "hybrid_chunking",
]

ANSWER_PROMPT = (
    "Use the following context to answer the question.\n"
    "If the answer is not present in the context, say \"INSUFFICIENT_CONTEXT\".\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

CHART_COLORS = [
    "#4F8EF7", "#8F4FF7", "#F74F8F", "#F7B84F",
    "#4FF7B8", "#4FF7F7", "#B8F74F", "#F74F4F",
]
