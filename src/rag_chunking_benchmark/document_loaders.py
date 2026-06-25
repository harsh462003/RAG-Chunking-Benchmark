"""Document parsing helpers for PDF, DOCX, and TXT uploads."""

from .common import *
from .utils import normalize_text, count_words, count_sentences, count_tokens

def read_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """Returns (text, num_pages)."""
    if not PYPDF_AVAILABLE:
        st.error("pypdf not installed. Cannot read PDF.")
        return "", 0
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages), len(reader.pages)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return "", 0

def read_docx(file_bytes: bytes) -> Tuple[str, int]:
    if not DOCX_AVAILABLE:
        st.error("python-docx not installed. Cannot read DOCX.")
        return "", 0
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs), 1
    except Exception as e:
        st.error(f"DOCX read error: {e}")
        return "", 0

def read_txt(file_bytes: bytes) -> Tuple[str, int]:
    try:
        return file_bytes.decode("utf-8", errors="replace"), 1
    except Exception as e:
        st.error(f"TXT read error: {e}")
        return "", 0

def extract_document(uploaded_file) -> Tuple[str, int, str]:
    """Returns (text, num_pages, doc_name)."""
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()
    if ext == "pdf":
        text, pages = read_pdf(raw)
    elif ext in ("docx", "doc"):
        text, pages = read_docx(raw)
    elif ext == "txt":
        text, pages = read_txt(raw)
    else:
        st.error(f"Unsupported file type: .{ext}")
        return "", 0, name
    return normalize_text(text), pages, name

def doc_statistics(text: str, pages: int) -> Dict:
    return {
        "characters": len(text),
        "words": count_words(text),
        "sentences": count_sentences(text),
        "estimated_tokens": count_tokens(text),
        "pages": pages,
    }
