"""
seo_common.py - Shared contract for the English SEO Copy QLoRA project.

This module is intentionally dependency-light: importing it only needs the
Python standard library, so it runs on the local machine (Python 3.13, no
torch / transformers) for dataset generation AND on Colab for training / eval.
Heavy dependencies (`datasets`, a transformers tokenizer) are imported lazily
inside the functions that need them, or passed in as arguments.

It is the single source of truth for:
  * the data record schema            -> {topic, keywords, title, meta}
  * the prompt / chat formatting       -> identical for train / eval / infer
  * the completion-only response template (per base-model family)
  * length / keyword quality metrics   -> shared by make_dataset and eval
  * robust JSON extraction             -> shared by make_dataset / eval / infer
  * a deterministic train/test split   -> shared by train and eval
"""

from __future__ import annotations

import json
import os
import re

# ---------------------------------------------------------------------------
# Hard SEO constraints, in characters. These are the limits we train toward.
# ---------------------------------------------------------------------------
TITLE_MAX = 60
META_MAX = 155

# Default artifact locations (overridable via env vars). Paths are relative to
# the repository root; every script is run from there (Colab does `%cd repo`).
ADAPTER_DIR = os.environ.get("SEO_ADAPTER_DIR", "seo-qlora-adapter")
DATA_PATH = os.environ.get("SEO_DATA_PATH", "data/seo_dataset_full.jsonl")

# ---------------------------------------------------------------------------
# Base-model registry. Default = Qwen (ungated, Apache-2.0, T4-friendly).
# The completion-only response template differs per chat format; for Qwen the
# token ids of "<|im_start|>assistant\n" are known and asserted, for Phi they
# are derived at runtime and asserted (see scripts/check_masking.py).
# ---------------------------------------------------------------------------
MODELS = {
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "response_template": "<|im_start|>assistant\n",
        "expected_template_ids": [151644, 77091, 198],
    },
    "phi": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "response_template": "<|assistant|>\n",
        "expected_template_ids": None,  # derived + asserted at runtime
    },
}


def get_model_key() -> str:
    return os.environ.get("SEO_BASE", "qwen").strip().lower()


def get_model_config() -> dict:
    key = get_model_key()
    if key not in MODELS:
        raise ValueError(f"Unknown SEO_BASE={key!r}; choose from {list(MODELS)}")
    cfg = dict(MODELS[key])
    cfg["key"] = key
    # Allow overriding the HF id (e.g. a local path) without touching code.
    cfg["hf_id"] = os.environ.get("SEO_BASE_ID", cfg["hf_id"])
    return cfg


# ---------------------------------------------------------------------------
# Prompt / chat formatting - the SAME structure for training, eval and infer.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert SEO copywriter. Given a product or business and its "
    "target keywords, write an SEO title tag of at most 60 characters and a "
    "meta description of at most 155 characters. The meta description must read "
    "naturally and end with a call to action. Respond with a single JSON object "
    'with exactly two keys: {"title": "...", "meta": "..."}.'
)


def _kw_list(keywords) -> list:
    """Normalize keywords (list or comma-separated string) into a clean list."""
    if isinstance(keywords, str):
        items = keywords.split(",")
    else:
        items = list(keywords or [])
    return [str(k).strip() for k in items if str(k).strip()]


def build_user_prompt(topic: str, keywords) -> str:
    kws = ", ".join(_kw_list(keywords))
    return f"Product or business: {topic}\nTarget keywords: {kws}"


def build_assistant_json(title: str, meta: str) -> str:
    """Canonical assistant target. Deterministic key order (title then meta).
    This exact string is what the model is trained to emit."""
    return json.dumps({"title": title, "meta": meta}, ensure_ascii=False)


def build_messages(record: dict, include_target: bool = True) -> list:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": build_user_prompt(record["topic"], record["keywords"])},
    ]
    if include_target:
        msgs.append({"role": "assistant",
                     "content": build_assistant_json(record["title"], record["meta"])})
    return msgs


def render_chat_text(tokenizer, record: dict) -> str:
    """Full training example (system+user+assistant) rendered with the model's
    own chat template, so the special tokens are exactly right."""
    return tokenizer.apply_chat_template(
        build_messages(record, include_target=True),
        tokenize=False, add_generation_prompt=False,
    )


def render_inference_prompt(tokenizer, topic: str, keywords) -> str:
    """Prompt for generation: system+user, ending with the assistant header."""
    return tokenizer.apply_chat_template(
        build_messages({"topic": topic, "keywords": keywords}, include_target=False),
        tokenize=False, add_generation_prompt=True,
    )


def response_template_ids(tokenizer, cfg: dict | None = None) -> list:
    """Token ids of the assistant-response header for completion-only loss.
    For Qwen the known ids are asserted; for others they are derived."""
    cfg = cfg or get_model_config()
    ids = tokenizer.encode(cfg["response_template"], add_special_tokens=False)
    expected = cfg.get("expected_template_ids")
    if expected is not None:
        assert ids == expected, (
            f"Response-template ids {ids} != expected {expected} for "
            f"{cfg['hf_id']}. Tokenizer/template mismatch - refusing to train.")
    assert len(ids) >= 1, "Empty response-template ids."
    return ids


# ---------------------------------------------------------------------------
# Quality metrics (shared by make_dataset.py and eval.py).
# ---------------------------------------------------------------------------
def title_len(title: str) -> int:
    return len(title or "")


def meta_len(meta: str) -> int:
    return len(meta or "")


def title_ok(title: str) -> bool:
    return 0 < title_len(title) <= TITLE_MAX


def meta_ok(meta: str) -> bool:
    return 0 < meta_len(meta) <= META_MAX


def _kw_present(kw: str, blob: str) -> bool:
    """A keyword counts as present if the exact phrase appears, or if every word
    of the keyword appears (order-independent). This matches how SEO relevance
    actually works and rewards natural copy that reorders keyword words, rather
    than demanding a brittle contiguous phrase match."""
    kw = kw.lower().strip()
    if not kw:
        return False
    if kw in blob:
        return True
    toks = [t for t in re.split(r"\s+", kw) if t]
    return bool(toks) and all(t in blob for t in toks)


def keyword_coverage(keywords, *texts) -> float:
    """Fraction of keywords that appear (case-insensitive) in the given texts."""
    kws = _kw_list(keywords)
    if not kws:
        return 0.0
    blob = " ".join(t for t in texts if t).lower()
    hit = sum(1 for k in kws if _kw_present(k, blob))
    return hit / len(kws)


def has_any_keyword(keywords, *texts) -> bool:
    return keyword_coverage(keywords, *texts) > 0.0


def normalize_topic(topic: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace - for dedup keys."""
    t = re.sub(r"[^a-z0-9 ]", " ", (topic or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def clean_record(obj: dict) -> dict | None:
    """Validate + normalize a generated record. Returns a clean dict or None.
    Does NOT enforce length limits - callers apply the length filter so the
    quality report can measure raw compliance first."""
    if not isinstance(obj, dict):
        return None
    topic = str(obj.get("topic", "")).strip()
    title = str(obj.get("title", "")).strip()
    meta = str(obj.get("meta", "")).strip()
    keywords = _kw_list(obj.get("keywords", []))
    if not topic or not title or not meta or len(keywords) < 2:
        return None
    return {"topic": topic, "keywords": keywords, "title": title, "meta": meta}


# ---------------------------------------------------------------------------
# Robust JSON extraction (shared by make_dataset / eval / infer).
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)  # flat objects only (records have none nested)


def _strip_fences(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_json_objects(text: str) -> list:
    """Best-effort extraction of a list of record dicts from arbitrary model
    output: a JSON array, a {"examples": [...]} wrapper, a single object, or
    several brace-delimited objects scattered in prose."""
    if not text:
        return []
    raw = _strip_fences(text)
    # 1) Whole thing parses cleanly.
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in ("examples", "data", "items", "records", "results"):
                if isinstance(data.get(key), list):
                    return [d for d in data[key] if isinstance(d, dict)]
            return [data]
    except Exception:
        pass
    # 2) Fall back to scanning for individual flat objects.
    out = []
    for m in _OBJ.finditer(raw):
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                out.append(d)
        except Exception:
            continue
    return out


def parse_model_output(text: str) -> dict:
    """Extract {title, meta} from a single model generation. Robust to extra
    prose or code fences; returns empty strings on failure."""
    for obj in extract_json_objects(text):
        if "title" in obj or "meta" in obj:
            return {"title": str(obj.get("title", "")).strip(),
                    "meta": str(obj.get("meta", "")).strip()}
    return {"title": "", "meta": ""}


# ---------------------------------------------------------------------------
# Deterministic train/test split (shared by train.py and eval.py).
# Imported lazily so seo_common stays usable without `datasets` installed.
# ---------------------------------------------------------------------------
def load_and_split(path: str = None, test_size: float = 0.1, seed: int = 42):
    """Load the JSONL dataset and produce a deterministic train/test split.
    `load_dataset` does not shuffle; the seeded split is identical in both
    train.py (uses ['train']) and eval.py (uses ['test']) -> a true hold-out."""
    from datasets import load_dataset  # lazy import (Colab only)
    path = path or DATA_PATH
    ds = load_dataset("json", data_files=path, split="train")
    return ds.train_test_split(test_size=test_size, seed=seed)
