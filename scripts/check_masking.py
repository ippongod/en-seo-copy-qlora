#!/usr/bin/env python
"""
check_masking.py - Prove the completion-only collator is correct.

It builds labels for one example exactly as training does, then ASSERTS that:
  1. the response template actually occurs in the tokenized example
     (otherwise DataCollatorForCompletionOnlyLM silently masks EVERYTHING and
     the model learns nothing);
  2. at least one label is unmasked;
  3. the unmasked labels form a single contiguous span at the END of the
     sequence;
  4. that span begins immediately after the response template;
  5. the decoded span equals the JSON answer (prompt is fully masked out).

Runs on CPU and is also the first cell of the Colab notebook, so a broken
response template (e.g. when switching to the Phi base) fails loudly BEFORE any
GPU time is spent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seo_common as seo  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from trl import DataCollatorForCompletionOnlyLM  # noqa: E402

SAMPLE = {
    "topic": "Wireless noise-cancelling headphones for frequent travelers",
    "keywords": ["noise-cancelling headphones", "travel headphones", "wireless"],
    "title": "Noise-Cancelling Travel Headphones - Fly in Silence",
    "meta": ("Block engine roar with wireless noise-cancelling headphones. "
             "40-hour battery, foldable design. Shop today and fly in calm."),
}


def _find_sub(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def _norm(s):
    return s.replace(" ", "")


def main():
    cfg = seo.get_model_config()
    print(f"[check_masking] base={cfg['hf_id']}")
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    text = seo.render_chat_text(tok, SAMPLE)
    enc = tok(text, add_special_tokens=False)
    input_ids = enc["input_ids"]

    resp_ids = seo.response_template_ids(tok, cfg)  # asserts known ids for Qwen
    pos = _find_sub(input_ids, resp_ids)
    assert pos != -1, (
        f"Response template {resp_ids} not found in the tokenized example. "
        "Completion-only masking would drop the whole sequence.")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=resp_ids, tokenizer=tok)
    batch = collator([{"input_ids": input_ids,
                       "attention_mask": enc["attention_mask"]}])
    labels = batch["labels"][0].tolist()

    unmasked = [i for i, l in enumerate(labels) if l != -100]
    assert unmasked, "All labels masked (-100) - no tokens to learn from."
    assert unmasked == list(range(unmasked[0], unmasked[-1] + 1)), \
        "Unmasked label span is not contiguous."
    assert unmasked[-1] >= len(labels) - 2, \
        "Unmasked span is not at the end of the sequence."
    assert unmasked[0] == pos + len(resp_ids), \
        "Unmasked span does not start right after the response template."

    span_ids = [labels[i] for i in unmasked]
    decoded = tok.decode(span_ids)
    target = seo.build_assistant_json(SAMPLE["title"], SAMPLE["meta"])
    assert _norm(target) in _norm(decoded), (
        "Decoded answer span does not contain the target JSON.\n"
        f"target={target!r}\nspan  ={decoded!r}")
    assert SAMPLE["topic"] not in decoded, "Prompt leaked into the learned span."

    print("[check_masking] PASS")
    print(f"  response template ids : {resp_ids}")
    print(f"  sequence length       : {len(input_ids)} tokens")
    print(f"  learned tokens        : {len(unmasked)} "
          f"(positions {unmasked[0]}..{unmasked[-1]})")
    print(f"  learned span          : {decoded!r}")


if __name__ == "__main__":
    main()
