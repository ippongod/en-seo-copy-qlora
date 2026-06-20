#!/usr/bin/env python
"""
eval.py - Multi-metric base-vs-fine-tuned evaluation ($0, deterministic).

Loads the base model in 4-bit once, attaches the trained adapter, and compares:
  * base       -> adapter disabled  (model.disable_adapter())
  * fine-tuned -> adapter enabled
on the held-out 10% split (seed=42, identical to train.py's hold-out).

Metrics (greedy decoding, fully deterministic):
  * valid-JSON output rate
  * title  <= 60 char compliance
  * meta   <= 155 char compliance
  * keyword inclusion rate
Optionally, a local qwen2.5:7b LLM judge scores quality 1-5 (skipped when
Ollama is unreachable, e.g. on Colab). Writes results.json + results.md.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seo_common as seo  # noqa: E402

import torch  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig)
from peft import PeftModel  # noqa: E402

MAX_NEW = 96


def generate(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def score(records, model, tok, label):
    n = len(records)
    parse_ok = title_ok = meta_ok = kw_ok = 0
    samples = []
    for i, ex in enumerate(records):
        prompt = seo.render_inference_prompt(tok, ex["topic"], ex["keywords"])
        raw = generate(model, tok, prompt)
        pred = seo.parse_model_output(raw)
        if pred["title"] and pred["meta"]:
            parse_ok += 1
        if seo.title_ok(pred["title"]):
            title_ok += 1
        if seo.meta_ok(pred["meta"]):
            meta_ok += 1
        if seo.has_any_keyword(ex["keywords"], pred["title"], pred["meta"]):
            kw_ok += 1
        if i < 5:
            samples.append({"topic": ex["topic"], "pred": pred,
                            "reference": {"title": ex["title"], "meta": ex["meta"]}})
    pc = lambda a: round(100.0 * a / n, 1) if n else 0.0
    return {"label": label, "n": n,
            "parse_success_pct": pc(parse_ok),
            "title_le60_pct": pc(title_ok),
            "meta_le155_pct": pc(meta_ok),
            "keyword_inclusion_pct": pc(kw_ok),
            "samples": samples}


def try_llm_judge(records, model, tok, host="http://127.0.0.1:11434",
                  judge_model="qwen2.5:7b", limit=10):
    """Optional local judge. Returns None if Ollama is unreachable (e.g. Colab)."""
    try:
        urllib.request.urlopen(host + "/api/tags", timeout=3).read()
    except Exception:
        print("[eval] LLM judge skipped (Ollama not reachable).")
        return None

    def ask(topic, keywords, pred):
        prompt = (
            "Rate the SEO quality of this title+meta for the given product on a "
            "scale of 1 (poor) to 5 (excellent). Consider relevance, keyword use, "
            "clarity, and call to action. Reply with ONLY the integer.\n"
            f"Product: {topic}\nKeywords: {', '.join(keywords)}\n"
            f"Title: {pred['title']}\nMeta: {pred['meta']}")
        body = json.dumps({"model": judge_model, "prompt": prompt, "stream": False,
                           "keep_alive": "5m",
                           "options": {"temperature": 0, "num_predict": 4}}).encode()
        req = urllib.request.Request(host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["response"]
        import re
        m = re.search(r"[1-5]", txt)
        return int(m.group(0)) if m else None

    subset = records[:limit]
    out = {}
    for label, use_adapter in (("base", False), ("fine_tuned", True)):
        scores = []
        ctx = model.disable_adapter() if not use_adapter else _noop()
        with ctx:
            for ex in subset:
                prompt = seo.render_inference_prompt(tok, ex["topic"], ex["keywords"])
                pred = seo.parse_model_output(generate(model, tok, prompt))
                s = ask(ex["topic"], ex["keywords"], pred)
                if s is not None:
                    scores.append(s)
        out[label] = round(sum(scores) / len(scores), 2) if scores else None
    return {"judge_model": judge_model, "n": len(subset),
            "base_avg": out.get("base"), "fine_tuned_avg": out.get("fine_tuned")}


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def write_md(results):
    b, f = results["base"], results["fine_tuned"]
    rows = [("Valid JSON output", "parse_success_pct"),
            ("Title <= 60 chars", "title_le60_pct"),
            ("Meta <= 155 chars", "meta_le155_pct"),
            ("Keyword inclusion", "keyword_inclusion_pct")]
    lines = [f"# Evaluation results",
             "",
             f"- Base model: `{results['base_model']}`",
             f"- Held-out test set: {results['n_test']} examples (seed=42)",
             f"- Decoding: greedy (deterministic)",
             "",
             "| Metric | Base (no adapter) | Fine-tuned (QLoRA) |",
             "|---|---|---|"]
    for name, key in rows:
        lines.append(f"| {name} | {b[key]:.1f}% | {f[key]:.1f}% |")
    if results.get("llm_judge"):
        j = results["llm_judge"]
        lines += ["", f"Local LLM judge (`{j['judge_model']}`, 1-5, n={j['n']}): "
                  f"base {j['base_avg']} -> fine-tuned {j['fine_tuned_avg']}."]
    if f.get("samples"):
        lines += ["", "## Sample (fine-tuned)", ""]
        s = f["samples"][0]
        lines += [f"- Topic: {s['topic']}",
                  f"- Title ({len(s['pred']['title'])} chars): {s['pred']['title']}",
                  f"- Meta ({len(s['pred']['meta'])} chars): {s['pred']['meta']}"]
    with open("results.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    cfg = seo.get_model_config()
    base_id = cfg["hf_id"]
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cdt = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdt)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb, device_map="auto", torch_dtype=cdt)
    model = PeftModel.from_pretrained(base, seo.ADAPTER_DIR)
    model.eval()

    test = seo.load_and_split(seo.DATA_PATH)["test"]
    records = [dict(r) for r in test]
    print(f"[eval] held-out test records: {len(records)}")

    with model.disable_adapter():
        base_res = score(records, model, tok, "base")
    ft_res = score(records, model, tok, "fine-tuned")

    results = {"base_model": base_id, "n_test": len(records),
               "base": base_res, "fine_tuned": ft_res}
    judge = try_llm_judge(records, model, tok)
    if judge:
        results["llm_judge"] = judge

    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    write_md(results)
    print("[eval] wrote results.json + results.md")

    print("\n==================== EVAL SUMMARY ====================")
    for key in ("parse_success_pct", "title_le60_pct", "meta_le155_pct",
                "keyword_inclusion_pct"):
        print(f"{key:24s}: base {base_res[key]:5.1f}%  ->  "
              f"fine-tuned {ft_res[key]:5.1f}%")
    print("=====================================================")


if __name__ == "__main__":
    main()
