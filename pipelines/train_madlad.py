"""
train_madlad.py — LoRA fine-tuning of google/madlad400-10b-mt.

Memory budget (33 GB available after other process):
  Full fine-tune needs ~60 GB (20 GB bf16 weights + 20 GB grads + 20 GB 8-bit Adam).
  LoRA + 8-bit base loading needs ~14 GB:
    8-bit base weights  : ~10 GB
    LoRA trainable params: ~0.3 GB (r=16, T5 attn+FFN, ~150 M params)
    LoRA gradients       : ~0.3 GB
    fp32 Adam for LoRA   : ~1.2 GB
    Activations (grad ckpt, batch=4): ~2 GB

Training protocol:
  Base model   : 8-bit (bitsandbytes load_in_8bit) -- frozen
  LoRA adapters: r=16, a=32, T5 attention + FFN projections
  Trainer      : Seq2SeqTrainer with predict_with_generate
  Eval metric  : COMET-22
  Saved artefact: LoRA adapter weights under outputs/finetuned/madlad/{dir}/

Standalone usage:
    python train_madlad.py [--direction ar-en] [--all]
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utlis.gpu_utils import require_cuda
from utlis.config import (
    MODELS,
    MADLAD_LANG_CODES,
    MADLAD_TRAINING_ARGS,
    MADLAD_LORA_CONFIG,
    TRANSLATION_DIRECTIONS,
    MAX_INPUT_LENGTH,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SEED,
)
from utlis.data_loader import load_dataset_for_direction, get_src_tgt_lists
from pipelines.evaluate import compute_comet22

logger = logging.getLogger(__name__)

MODEL_NAME = MODELS["madlad"]
MODEL_KEY  = "madlad"


def tokenize_batch(examples, tokenizer, tgt_lang, max_length):
    tgt_code = MADLAD_LANG_CODES[tgt_lang]
    prefixed = [f"{tgt_code} {s}" for s in examples["src"]]
    return tokenizer(
        prefixed,
        text_target=examples["tgt"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )


def make_compute_metrics(tokenizer, dev_sources, dev_references):
    def compute_metrics(eval_preds):
        preds, _ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        hypotheses = tokenizer.batch_decode(preds, skip_special_tokens=True)
        n = min(len(hypotheses), len(dev_sources), len(dev_references))
        return {"comet22": compute_comet22(dev_sources[:n], hypotheses[:n], dev_references[:n])}
    return compute_metrics


def train_madlad_direction(src_lang: str, tgt_lang: str) -> str:
    direction = f"{src_lang}-{tgt_lang}"
    logger.info("=" * 64 + "  LoRA fine-tuning MADLAD | %s  " + "=" * 64, direction)

    ckpt_dir   = os.path.join(CHECKPOINT_DIR, "madlad", direction)
    output_dir = os.path.join(OUTPUT_DIR, "finetuned", "madlad", direction)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load base model in 8-bit: 10B x 1 byte = ~10 GB vs 20 GB in bf16
    logger.info("Loading MADLAD in 8-bit ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    # Casts layer norms to fp32, enables gradient checkpointing for quantised model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Attach LoRA
    model = get_peft_model(model, LoraConfig(**MADLAD_LORA_CONFIG))
    model.print_trainable_parameters()

    datasets = load_dataset_for_direction(src_lang, tgt_lang)
    max_len   = MAX_INPUT_LENGTH[MODEL_KEY]

    def tokenize(batch):
        return tokenize_batch(batch, tokenizer, tgt_lang, max_len)

    train_ds = datasets["train"].map(tokenize, batched=True, remove_columns=["src", "tgt"],
                                     desc=f"Tokenising train [{direction}]")
    dev_ds   = datasets["dev"].map(tokenize,   batched=True, remove_columns=["src", "tgt"],
                                   desc=f"Tokenising dev [{direction}]")

    dev_sources, dev_references = get_src_tgt_lists(src_lang, tgt_lang, "dev")

    # Prevent GradScaler from being created for bf16
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    training_args = Seq2SeqTrainingArguments(
        output_dir=ckpt_dir,
        **MADLAD_TRAINING_ARGS,
        predict_with_generate=True,
        generation_max_length=512,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8),
        compute_metrics=make_compute_metrics(tokenizer, dev_sources, dev_references),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting LoRA training ...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA adapter saved -> %s", output_dir)
    return output_dir


def train_all_madlad():
    for src_lang, tgt_lang in TRANSLATION_DIRECTIONS:
        train_madlad_direction(src_lang, tgt_lang)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="LoRA fine-tune MADLAD-400-10B")
    parser.add_argument("--direction", default=None, help="e.g. ar-hi. Omit for all 6.")
    args = parser.parse_args()

    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if args.direction:
        src, tgt = args.direction.split("-")
        train_madlad_direction(src, tgt)
    else:
        train_all_madlad()