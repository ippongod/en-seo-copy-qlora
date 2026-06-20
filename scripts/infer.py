#!/usr/bin/env python
"""
infer.py - Local inference: topic + keywords -> {title, meta} JSON.

Loads the base model (4-bit on GPU, or fp32 on CPU with --cpu) and the trained
adapter if present, then generates one SEO title+meta pair.

Examples:
  python scripts/infer.py --topic "Cold brew coffee subscription" \
      --keywords "cold brew, coffee subscription, organic"
  python scripts/infer.py --topic "..." --keywords "..." --cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seo_common as seo  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--keywords", required=True, help="comma-separated keywords")
    ap.add_argument("--adapter", default=seo.ADAPTER_DIR)
    ap.add_argument("--cpu", action="store_true", help="force CPU fp32 (no bitsandbytes)")
    ap.add_argument("--max-new", type=int, default=96)
    args = ap.parse_args()

    cfg = seo.get_model_config()
    base_id = cfg["hf_id"]

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_cpu = args.cpu or not torch.cuda.is_available()
    if use_cpu:
        model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
        model.to("cpu")
    else:
        from transformers import BitsAndBytesConfig
        bf16 = torch.cuda.is_bf16_supported()
        cdt = torch.bfloat16 if bf16 else torch.float16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdt)
        model = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, device_map="auto", torch_dtype=cdt)

    if os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print(f"[infer] note: adapter '{args.adapter}' not found; using base model.",
              file=sys.stderr)
    model.eval()

    prompt = seo.render_inference_prompt(tok, args.topic, args.keywords)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = seo.parse_model_output(raw)
    pred["title_len"] = len(pred["title"])
    pred["meta_len"] = len(pred["meta"])
    print(json.dumps(pred, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
