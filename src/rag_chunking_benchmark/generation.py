"""Answer generation using local Flan-T5 or an OpenAI-compatible API."""

from .common import *
from .schemas import AnswerResult

def load_local_gen_model(model_name: str):
    """Lazy-load to avoid OMP segfault on macOS Apple Silicon."""
    import os
    # Prevent multiple OMP runtime conflicts (PyTorch + FAISS + sklearn each bundle libomp)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"Could not load generation model '{model_name}': {e}")
        return None, None

def generate_answer_local(tokenizer, model, prompt: str) -> Tuple[str, float]:
    if tokenizer is None or model is None:
        return "INSUFFICIENT_CONTEXT [model not loaded]", 0.0
    t0 = time.perf_counter()
    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        answer = f"GENERATION_ERROR: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return answer, elapsed_ms

def generate_answer_api(api_key: str, model_name: str, prompt: str,
                        base_url: Optional[str] = None) -> Tuple[str, float]:
    t0 = time.perf_counter()
    try:
        if OPENAI_AVAILABLE:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip()
        else:
            import urllib.request
            payload = json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }).encode()
            req = urllib.request.Request(
                (base_url or "https://api.openai.com") + "/v1/chat/completions",
                data=payload,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            answer = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"API_ERROR: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return answer, elapsed_ms
