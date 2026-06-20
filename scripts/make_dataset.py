#!/usr/bin/env python
"""
make_dataset.py - Generate an English SEO copy dataset locally with Ollama.

This is the core "distillation" step: a small local model (qwen2.5:7b) writes
fluent English SEO title tags + meta descriptions, which become the training
data for a tiny QLoRA adapter. Because English copy is well within a 7B model's
ability, the data is genuinely produced locally for $0 with no manual curation.

Pipeline:
  preflight (server up, model pulled, <=8B, 100% GPU)
    -> seed records (data/seo_seed.jsonl, the quality anchor)
    -> generate diverse batches via /api/chat
    -> dedup on normalized topic + DOUBLE length filter (title<=60, meta<=155)
    -> atomic write to data/seo_dataset_full.jsonl
    -> quality report (raw compliance %, keyword inclusion, diversity)

Guardrails: refuses models > 8B params and refuses if the model is offloaded to
CPU/RAM (would risk freezing an 8 GB-VRAM machine). Stops early on low free RAM.

Usage:
  python scripts/make_dataset.py --n 30      # validation pass first
  python scripts/make_dataset.py --n 300     # full build
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seo_common as seo  # noqa: E402

KEEP_ALIVE = "5m"
MAX_PARAM_B = 8.05  # refuse anything strictly larger than ~8B params


def _ollama_base() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


OLLAMA = _ollama_base()


def ollama_get(path: str, timeout: int = 15):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_post(path: str, payload: dict, timeout: int = 240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_billions(s: str):
    m = re.search(r"([\d.]+)\s*B", str(s), re.IGNORECASE)
    return float(m.group(1)) if m else None


def free_ram_gb():
    """Best-effort available-RAM reading without third-party deps."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullAvailPhys / (1024 ** 3), 1)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Preflight: server, model, size guard, GPU residency.
# ---------------------------------------------------------------------------
def preflight(model: str):
    if not shutil.which("ollama"):
        sys.exit("[preflight] 'ollama' not found on PATH.")
    try:
        tags = ollama_get("/api/tags")
    except Exception as e:
        sys.exit(f"[preflight] Ollama server not reachable at {OLLAMA}: {e}\n"
                 "Start it with:  ollama serve")
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"[preflight] Model {model!r} not pulled. Run:  ollama pull {model}")

    info = ollama_post("/api/show", {"name": model}, timeout=30)
    psize = (info.get("details") or {}).get("parameter_size", "")
    b = parse_billions(psize)
    if b is not None and b > MAX_PARAM_B:
        sys.exit(f"[preflight] Model {model} is {psize} (> 8B). Refusing: it would "
                 "offload to CPU/RAM on 8 GB VRAM and risk a freeze.")
    print(f"[preflight] model={model} params={psize or '?'} -> size guard OK")

    # Warmup, then confirm 100% GPU residency.
    ollama_post("/api/generate", {
        "model": model, "prompt": "ok", "stream": False,
        "keep_alive": KEEP_ALIVE, "options": {"num_predict": 1}}, timeout=180)
    ps = ollama_get("/api/ps")
    running = {m.get("name", ""): m for m in ps.get("models", [])}
    m = running.get(model) or running.get(f"{model}:latest")
    if not m:
        sys.exit("[preflight] model not resident after warmup.")
    size = m.get("size", 0) or 0
    vram = m.get("size_vram", 0) or 0
    frac = (vram / size) if size else 0.0
    print(f"[preflight] GPU residency: size_vram/size = {frac:.1%}")
    if frac < 0.99:
        sys.exit(f"[preflight] CPU offload detected ({frac:.0%} on GPU). "
                 "Stopping per the 8 GB-VRAM guardrail.")


# ---------------------------------------------------------------------------
# Generation prompts.
# ---------------------------------------------------------------------------
INDUSTRIES = [
    "SaaS analytics", "e-commerce fashion", "local restaurant", "dental clinic",
    "fitness and gym", "real estate agency", "B2B manufacturing", "online course",
    "pet grooming", "specialty coffee roaster", "plumbing service", "accounting firm",
    "travel agency", "skincare brand", "mobile app", "fintech app", "gardening service",
    "auto repair shop", "wedding photography", "yoga studio", "cybersecurity",
    "HR software", "solar installer", "artisan bakery", "independent bookstore",
    "meal kit delivery", "language tutoring", "interior design", "moving company",
    "electric bikes", "board game cafe", "podcast production", "drone services",
    "3D printing", "supplement brand", "VPN service", "project management tool",
    "veterinary clinic", "landscaping", "tax preparation", "car detailing",
    "wedding planning", "pottery studio", "craft brewery", "music lessons",
    "house cleaning", "personal training", "graphic design studio", "florist",
    "electric scooter rental",
]


def build_gen_messages(seed_examples, industries, avoid_topics, n):
    shots = "\n".join(json.dumps(e, ensure_ascii=False) for e in seed_examples)
    system = (
        "You are an expert SEO copywriter generating high-quality training data. "
        "Each example is a JSON object with EXACTLY these keys: "
        '"topic", "keywords", "title", "meta".\n'
        f"Rules:\n"
        f"- topic: one concrete product, service, or business.\n"
        f"- keywords: 3 to 5 short English keyword phrases.\n"
        f"- title: an SEO title tag, at most {seo.TITLE_MAX} characters.\n"
        f"- meta: a meta description, at most {seo.META_MAX} characters, natural, "
        "click-worthy, ending with a call to action.\n"
        "- Every keyword MUST appear (use the exact words) in the title or the "
        "meta; put the most important keyword in the title.\n"
        "Return ONLY a JSON array of objects. No prose, no code fences.\n\n"
        "Format examples:\n" + shots
    )
    avoid = "; ".join(list(avoid_topics)[-14:])
    user = (
        f"Generate {n} NEW and DIVERSE SEO examples. "
        f"Focus on these industries: {', '.join(industries)}. "
        "Each example must be a different concrete product, service, or business "
        "with a distinct angle, audience, and tone. "
        + (f"Do NOT repeat or paraphrase these already-used topics: {avoid}. "
           if avoid else "")
        + "Return ONLY a JSON array."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def load_seed(path):
    recs = []
    if not os.path.exists(path):
        return recs
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = seo.clean_record(json.loads(line))
        if r:
            recs.append(r)
    return recs


def atomic_write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="target number of generated records")
    ap.add_argument("--batch", type=int, default=6, help="records requested per call")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seed", type=int, default=42, help="base RNG/sampling seed")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed-file", default="data/seo_seed.jsonl")
    ap.add_argument("--out", default=seo.DATA_PATH)
    ap.add_argument("--max-calls", type=int, default=0, help="0 = auto (3x ceil(n/batch))")
    ap.add_argument("--min-free-ram", type=float, default=1.5, help="GB; stop below this")
    ap.add_argument("--min-kw-coverage", type=float, default=0.5,
                    help="drop records whose keyword coverage is below this")
    args = ap.parse_args()

    preflight(args.model)

    import random
    rng = random.Random(args.seed)

    seed_records = load_seed(args.seed_file)
    print(f"[gen] loaded {len(seed_records)} seed records from {args.seed_file}")

    # Seed first - the quality anchor. Length-filter the seed too (defensive).
    kept = {}
    for r in seed_records:
        nt = seo.normalize_topic(r["topic"])
        if (nt and nt not in kept and seo.title_ok(r["title"]) and seo.meta_ok(r["meta"])
                and seo.keyword_coverage(r["keywords"], r["title"], r["meta"])
                >= args.min_kw_coverage):
            kept[nt] = r
    seed_count = len(kept)

    shots = seed_records[:3]
    avoid = deque(maxlen=80)
    for r in seed_records:
        avoid.append(r["topic"])

    raw_parsed = raw_title_ok = raw_meta_ok = raw_kw = raw_cov_ok = 0
    calls = 0
    max_calls = args.max_calls or (3 * ((args.n + args.batch - 1) // args.batch) + 2)

    while (len(kept) - seed_count) < args.n and calls < max_calls:
        calls += 1
        industries = rng.sample(INDUSTRIES, k=min(4, len(INDUSTRIES)))
        messages = build_gen_messages(shots, industries, avoid, args.batch)
        payload = {
            "model": args.model, "messages": messages, "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": args.temperature, "top_p": args.top_p,
                        "seed": args.seed + calls, "num_predict": 1400},
        }
        try:
            resp = ollama_post("/api/chat", payload)
            content = resp.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[gen] call {calls} failed: {e}")
            continue

        for o in seo.extract_json_objects(content):
            title = str(o.get("title", "")).strip()
            meta = str(o.get("meta", "")).strip()
            raw_parsed += 1
            if seo.title_ok(title):
                raw_title_ok += 1
            if seo.meta_ok(meta):
                raw_meta_ok += 1
            if seo.has_any_keyword(o.get("keywords", []), title, meta):
                raw_kw += 1
            if seo.keyword_coverage(o.get("keywords", []), title, meta) >= args.min_kw_coverage:
                raw_cov_ok += 1
            rec = seo.clean_record(o)
            if not rec:
                continue
            if not (seo.title_ok(rec["title"]) and seo.meta_ok(rec["meta"])):
                continue  # DOUBLE length filter
            if seo.keyword_coverage(rec["keywords"], rec["title"], rec["meta"]) \
                    < args.min_kw_coverage:
                continue  # keyword-coverage filter
            nt = seo.normalize_topic(rec["topic"])
            if not nt or nt in kept:
                continue
            kept[nt] = rec
            avoid.append(rec["topic"])

        free = free_ram_gb()
        gen = len(kept) - seed_count
        print(f"[gen] call {calls}/{max_calls}: kept={len(kept)} "
              f"(generated {gen}/{args.n}) free_ram={free} GB")
        if free is not None and free < args.min_free_ram:
            print(f"[gen] WARNING: free RAM {free} GB < {args.min_free_ram} GB. "
                  "Stopping early to avoid a freeze.")
            break

    records = list(kept.values())
    atomic_write_jsonl(args.out, records)
    report(records, seed_count, raw_parsed, raw_title_ok, raw_meta_ok, raw_kw,
           raw_cov_ok, args.min_kw_coverage, calls, args.out)


def report(records, seed_count, raw_parsed, raw_title_ok, raw_meta_ok, raw_kw,
           raw_cov_ok, min_cov, calls, out):
    def pct(a, b):
        return (100.0 * a / b) if b else 0.0

    kw_cov = (sum(seo.keyword_coverage(r["keywords"], r["title"], r["meta"])
                  for r in records) / len(records)) if records else 0.0
    vocab = {k.lower() for r in records for k in r["keywords"]}
    avg_kw = (sum(len(r["keywords"]) for r in records) / len(records)) if records else 0.0

    print("\n==================== DATASET QUALITY REPORT ====================")
    print(f"output file              : {out}")
    print(f"seed records kept        : {seed_count}")
    print(f"generated records kept   : {len(records) - seed_count}")
    print(f"TOTAL records            : {len(records)}")
    print(f"ollama calls             : {calls}")
    print("--- raw model output (before dedup/length filter) ---")
    print(f"raw objects parsed       : {raw_parsed}")
    print(f"  title <= 60 compliance : {pct(raw_title_ok, raw_parsed):.1f}%")
    print(f"  meta  <= 155 compliance: {pct(raw_meta_ok, raw_parsed):.1f}%")
    print(f"  keyword inclusion (any): {pct(raw_kw, raw_parsed):.1f}%")
    print(f"  keyword coverage >={min_cov:.0%}: {pct(raw_cov_ok, raw_parsed):.1f}%")
    print("--- final dataset (post-filter, length-compliant by construction) ---")
    print(f"avg keyword coverage     : {100 * kw_cov:.1f}% (fraction of kw present)")
    print(f"distinct keyword vocab   : {len(vocab)}")
    print(f"avg keywords / record    : {avg_kw:.1f}")
    print(f"unique topics            : {len(records)}")
    print("--- samples (generated) ---")
    for r in (records[seed_count:seed_count + 3] or records[:3]):
        print(f"  title({len(r['title'])}c): {r['title']}")
        print(f"  meta ({len(r['meta'])}c): {r['meta']}")
    print("================================================================")


if __name__ == "__main__":
    main()
