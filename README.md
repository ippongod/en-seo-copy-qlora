# English SEO Copy Generator — QLoRA distillation ($0)

Fine-tune a tiny instruct model to turn a **product/business + keywords** into a
ready-to-ship **SEO title tag (≤60 chars)** and **meta description (≤155 chars)**,
returned as one JSON object:

```json
{"title": "Organic Cold Brew Coffee Subscription, Delivered",
 "meta":  "Fresh organic cold brew delivered weekly. Smooth, low-acid, ready to pour. Start your subscription and save 20% today."}
```

Trained with **4-bit QLoRA + completion-only loss** on a **free Colab T4**. The
whole pipeline — dataset generation, training, and evaluation — costs **$0** and
uses **no paid APIs and no API keys**.

---

## Why this project is interesting

- **Real local distillation.** The training data is written by a small local
  model (`qwen2.5:7b` via [Ollama](https://ollama.com)). English SEO copy is well
  within a 7B model's ability, so the data is genuinely generated locally for $0
  **with no manual curation** — the local model *is* the teacher.
- **Structured, two-field output** with two independent length constraints
  (title ≤ 60, meta ≤ 155), not a single free-text blob.
- **Completion-only training, proven.** Loss is computed only on the JSON answer;
  the prompt is masked. A standalone script (`scripts/check_masking.py`) *asserts*
  the masking is correct before any GPU time is spent.
- **Multi-metric evaluation.** Base vs. fine-tuned on a true held-out split,
  measuring valid-JSON rate, title/meta length compliance, and keyword inclusion
  — plus an optional local LLM judge.
- **Reproducible.** Pinned dependency versions, a deterministic seeded split, and
  a one-click Colab notebook.

> This is the English counterpart to an earlier Hungarian SEO experiment. The key
> difference: in a low-resource language a ≤7B model produced fragmented copy and
> the data had to be hand-curated, whereas **in English the local model is good
> enough to be the actual data source.** That is what makes this a real, honest
> local-distillation pipeline rather than a curation exercise.

---

## How it works

```
qwen2.5:7b (local, Ollama)                 Colab T4 (free)
        │                                         │
  make_dataset.py  ──►  data/seo_dataset_full.jsonl
        │  (dedup + double length filter)         │
        │                                  load_and_split (seed=42)
        │                                   ├── train ─► train.py  ─► seo-qlora-adapter/
        │                                   └── test  ─► eval.py   ─► results.json / .md
        └──────────────► seed: data/seo_seed.jsonl (20 hand-checked anchors)
```

1. **Generate** (`scripts/make_dataset.py`, runs locally): preflight checks the
   Ollama server, refuses models > 8B, and verifies the model is **100% on the
   GPU** (no CPU offload). It then prompts `qwen2.5:7b` for diverse `{topic,
   keywords, title, meta}` records, **dedups** on the normalized topic, applies a
   **double length filter** (drop if title > 60 *or* meta > 155), and writes
   `data/seo_dataset_full.jsonl` atomically. A quality report prints the raw
   length-compliance and keyword-inclusion rates of the local model.
2. **Train** (`scripts/train.py`, Colab T4): 4-bit NF4 base + LoRA adapters
   (`all-linear`), completion-only loss via `DataCollatorForCompletionOnlyLM`.
3. **Evaluate** (`scripts/eval.py`, Colab T4): base (adapter disabled) vs.
   fine-tuned on the held-out 10%, greedy/deterministic.

---

## Quickstart

### A) Generate the dataset locally ($0, needs Ollama + a GPU)

```bash
ollama serve                 # if not already running
ollama pull qwen2.5:7b       # ~4.7 GB, fits 8 GB VRAM
python scripts/make_dataset.py --n 30     # quick validation pass
python scripts/make_dataset.py --n 300    # full build
```

### B) Train + evaluate on a free Colab T4

Open `notebooks/colab_train.ipynb` in Colab
(**File → Open notebook → GitHub →** `ippongod/en-seo-copy-qlora`), set the
runtime to **T4 GPU**, and **Run all**. It clones the repo, installs the pinned
deps, asserts the masking, trains, evaluates, fills the results below, and
downloads `seo-qlora-adapter.zip`.

### C) Inference

```bash
python scripts/infer.py \
  --topic "Cold brew coffee subscription" \
  --keywords "cold brew, coffee subscription, organic"
# add --cpu to run without a GPU (fp32, no bitsandbytes)
```

---

## Results

<!--RESULTS_START-->
_Held-out test set: 33 examples (seed=42). Base model: `Qwen/Qwen2.5-1.5B-Instruct`. Greedy decoding._

| Metric | Base (no adapter) | Fine-tuned (QLoRA) |
|---|---|---|
| Valid JSON output | 100.0% | 100.0% |
| Title <= 60 chars | 60.6% | 93.9% |
| Meta <= 155 chars | 60.6% | 97.0% |
| Keyword inclusion | 97.0% | 97.0% |
<!--RESULTS_END-->

---

## Repository layout

```
en-seo-copy-qlora/
├── data/
│   ├── seo_seed.jsonl            # 20 hand-checked English anchor examples
│   └── seo_dataset_full.jsonl    # generated locally by make_dataset.py
├── scripts/
│   ├── seo_common.py             # single source of truth (schema, prompts, metrics)
│   ├── make_dataset.py           # local $0 data generation via Ollama
│   ├── train.py                  # QLoRA completion-only fine-tune (Colab T4)
│   ├── check_masking.py          # asserts completion-only masking is correct
│   ├── eval.py                   # multi-metric base-vs-fine-tuned evaluation
│   ├── infer.py                  # local inference (GPU 4-bit or CPU fp32)
│   └── fill_results.py           # writes eval numbers into this README
├── notebooks/
│   └── colab_train.ipynb         # one-click train + eval
├── requirements.txt              # pinned versions
└── LICENSE                       # MIT
```

## Models & versions

- **Base model (default):** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, ungated,
  T4-friendly). A `microsoft/Phi-3.5-mini-instruct` path is also supported via
  `SEO_BASE=phi` — its response template is derived and asserted by
  `check_masking.py`.
- **Pinned:** transformers 4.46.3 · trl 0.12.2 · peft 0.13.2 · bitsandbytes
  0.44.1 · accelerate 1.1.1 · datasets 3.1.0. (torch comes from Colab.)

## Honest limitations

- The base is a 1.5B model; outputs are good for SEO drafts, not a replacement
  for a human editor on high-stakes pages.
- The dataset is fully synthetic (distilled from `qwen2.5:7b`); it inherits that
  model's stylistic tendencies.
- Character-count limits are a useful proxy; real SERP truncation is pixel-based.

## License

MIT (code). The base model and any uploaded adapter follow their own licenses
(Qwen2.5 is Apache-2.0).
