#!/usr/bin/env python
"""
fill_results.py - Write the eval results into the README Results block.

Reads results.json (produced by eval.py) and replaces the content between
<!--RESULTS_START--> and <!--RESULTS_END--> in README.md. Idempotent and
BOM-tolerant (reads as utf-8-sig, writes plain utf-8).

Usage:  python scripts/fill_results.py [README.md] [results.json]
"""

from __future__ import annotations

import json
import os
import sys

START = "<!--RESULTS_START-->"
END = "<!--RESULTS_END-->"


def md_table(results):
    b, f = results["base"], results["fine_tuned"]
    rows = [("Valid JSON output", "parse_success_pct"),
            ("Title <= 60 chars", "title_le60_pct"),
            ("Meta <= 155 chars", "meta_le155_pct"),
            ("Keyword inclusion", "keyword_inclusion_pct")]
    lines = [
        f"_Held-out test set: {results['n_test']} examples (seed=42). "
        f"Base model: `{results['base_model']}`. Greedy decoding._",
        "",
        "| Metric | Base (no adapter) | Fine-tuned (QLoRA) |",
        "|---|---|---|",
    ]
    for name, key in rows:
        lines.append(f"| {name} | {b[key]:.1f}% | {f[key]:.1f}% |")
    if results.get("llm_judge"):
        j = results["llm_judge"]
        lines += ["",
                  f"Local LLM judge (`{j['judge_model']}`, 1-5, n={j['n']}): "
                  f"base **{j['base_avg']}** -> fine-tuned **{j['fine_tuned_avg']}**."]
    f_samples = f.get("samples") or []
    if f_samples:
        s = f_samples[0]
        lines += ["", "**Example (fine-tuned output):**", "",
                  f"- Topic: {s['topic']}",
                  f"- Title ({len(s['pred']['title'])} chars): {s['pred']['title']}",
                  f"- Meta ({len(s['pred']['meta'])} chars): {s['pred']['meta']}"]
    return "\n".join(lines)


def main():
    readme = sys.argv[1] if len(sys.argv) > 1 else "README.md"
    rj = sys.argv[2] if len(sys.argv) > 2 else "results.json"

    if not os.path.exists(rj):
        sys.exit(f"[fill_results] {rj} not found; run eval.py first.")
    with open(rj, encoding="utf-8-sig") as fh:
        results = json.load(fh)
    with open(readme, encoding="utf-8-sig") as fh:
        text = fh.read()

    block = f"{START}\n{md_table(results)}\n{END}"
    if START in text and END in text:
        pre = text.split(START, 1)[0]
        post = text.split(END, 1)[1]
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"

    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[fill_results] {readme} updated from {rj}.")


if __name__ == "__main__":
    main()
