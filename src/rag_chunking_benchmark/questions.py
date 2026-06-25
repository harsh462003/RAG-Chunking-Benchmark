"""Question parsing utilities for manual text, CSV, TXT, and JSON inputs."""

from .common import *

def parse_questions(raw_text: str = "", uploaded_file=None) -> List[Dict]:
    questions = []
    if uploaded_file is not None:
        name = uploaded_file.name
        ext = name.rsplit(".", 1)[-1].lower()
        raw_bytes = uploaded_file.read()
        if ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes))
                col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])
                id_col = next((c for c in df.columns if "id" in c.lower()), None)
                for i, row in df.iterrows():
                    qid = str(row[id_col]) if id_col else f"Q{i+1}"
                    questions.append({"id": qid, "text": str(row[col]).strip()})
            except Exception as e:
                st.warning(f"CSV parse error: {e}")
        elif ext == "json":
            try:
                data = json.loads(raw_bytes)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, str):
                            questions.append({"id": f"Q{i+1}", "text": item.strip()})
                        elif isinstance(item, dict):
                            qid = item.get("id", item.get("question_id", f"Q{i+1}"))
                            qt  = item.get("text", item.get("question_text", item.get("question", "")))
                            questions.append({"id": str(qid), "text": qt.strip()})
            except Exception as e:
                st.warning(f"JSON parse error: {e}")
        else:  # txt
            lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    questions.append({"id": f"Q{i+1}", "text": line.strip()})
    if raw_text.strip():
        lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
        offset = len(questions)
        for i, line in enumerate(lines):
            questions.append({"id": f"Q{offset+i+1}", "text": line})
    # deduplicate by text
    seen, unique = set(), []
    for q in questions:
        if q["text"] not in seen:
            seen.add(q["text"])
            unique.append(q)
    return unique
