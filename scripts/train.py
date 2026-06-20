#!/usr/bin/env python
"""
train.py - QLoRA fine-tune of an instruct model for SEO title+meta generation.

* 4-bit NF4 quantization (bitsandbytes) + LoRA adapters.
* Completion-only loss: only the JSON answer contributes to the loss; the
  system+user prompt is masked. See scripts/check_masking.py for the proof.
* Runs on a free Colab T4 (no bf16 -> fp16) or any bf16-capable GPU.

DO NOT run this locally on the fragile 8 GB machine: it is designed for Colab.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seo_common as seo  # noqa: E402

import torch  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM  # noqa: E402


def main():
    cfg = seo.get_model_config()
    base_id = cfg["hf_id"]

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"[train] base={base_id}  bf16={bf16}  compute_dtype={compute_dtype} "
          f"(T4 has no bf16 -> fp16)")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb, device_map="auto",
        torch_dtype=compute_dtype)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Train on the SAME deterministic 90% split that eval.py holds out from.
    train_ds = seo.load_and_split(seo.DATA_PATH)["train"]

    def to_text(ex):
        return {"text": seo.render_chat_text(tokenizer, ex)}

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    print(f"[train] training examples: {len(train_ds)}")

    response_ids = seo.response_template_ids(tokenizer, cfg)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids, tokenizer=tokenizer)

    args = SFTConfig(
        output_dir="outputs",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.model.save_pretrained(seo.ADAPTER_DIR)
    tokenizer.save_pretrained(seo.ADAPTER_DIR)
    print(f"[train] adapter saved to ./{seo.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
