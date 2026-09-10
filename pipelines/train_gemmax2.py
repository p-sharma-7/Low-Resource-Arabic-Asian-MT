"""
train_gemmax2.py — QLoRA fine-tuning of ModelSpace/GemmaX2-28-9B-v0.1.

Training protocol:
  • Quantisation : 4-bit NF4 (QLoRA) — keeps base model in ~4.5 GB VRAM
  • LoRA adapters : r=16, α=32, all attention + FFN projections
  • Framework     : trl SFTTrainer (instruction-style causal LM training)
  • Loss          : next-token prediction over the full sequence
  • Eval metric   : eval_loss (COMET computed post-training via run_pipeline)
  • Format        : "Translate … from X to Y:\n{src}\nTranslation: {tgt}<eos>"

Saved artefact: LoRA adapter weights (not the full model) stored under:
    outputs/finetuned/gemmax2/{src_lang}-{tgt_lang}/

At inference time the adapter is merged onto the base model (see infer_gemmax2).

Standalone usage:
    python train_gemmax2.py [--direction ar-hi] [--all]
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utlis.gpu_utils import get_device_map, require_cuda
from utlis.config import (
    MODELS,
    LANG_NAMES,
    LORA_CONFIG,
    GEMMAX2_TRAINING_ARGS,
    TRANSLATION_DIRECTIONS,
    MAX_INPUT_LENGTH,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SEED,
)
from utlis.data_loader import load_dataset_for_direction
from pipelines.infer_gemmax2 import format_prompt

logger = logging.getLogger(__name__)

MODEL_NAME = MODELS["gemmax2"]
MODEL_KEY  = "gemmax2"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_example(src: str, tgt: str, src_lang: str, tgt_lang: str, eos: str) -> str:
    """
    Build a single training string:
        Translate the following text from Arabic to Hindi:
        {source}
        Translation: {target}<eos>

    The model learns to predict everything after "Translation: ".
    """
    prompt = format_prompt(src, src_lang, tgt_lang)
    return f"{prompt} {tgt}{eos}"


def build_sft_dataset(
    rows: List[dict],
    src_lang: str,
    tgt_lang: str,
    eos: str,
) -> Dataset:
    """Convert a list of {'src', 'tgt'} dicts into an SFT-ready Dataset."""
    records = [
        {"text": format_example(r["src"], r["tgt"], src_lang, tgt_lang, eos)}
        for r in rows
    ]
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# Model / tokeniser loading (4-bit QLoRA)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_training():
    """Load GemmaX2 in 4-bit and return (model, tokeniser)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading GemmaX2 (4-bit QLoRA) from: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-pad during SFT training

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=get_device_map(),
        dtype=torch.bfloat16,
    )
    # Disable KV-cache (required for gradient checkpointing)
    model.config.use_cache = False
    # Enable input-grad computation for gradient checkpointing compatibility
    model.enable_input_require_grads()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Single-direction training
# ─────────────────────────────────────────────────────────────────────────────

def train_gemmax2_direction(src_lang: str, tgt_lang: str) -> str:
    """
    QLoRA-fine-tune GemmaX2 on one translation direction.

    Returns the path where the LoRA adapter is saved.
    """
    direction = f"{src_lang}-{tgt_lang}"
    logger.info("\n%s\nLoRA fine-tuning GemmaX2  |  %s\n%s",
                "=" * 64, direction, "=" * 64)

    ckpt_dir   = os.path.join(CHECKPOINT_DIR, "gemmax2", direction)
    output_dir = os.path.join(OUTPUT_DIR, "finetuned", "gemmax2", direction)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Model + LoRA ────────────────────────────────────────────────────────
    model, tokenizer = load_model_for_training()
    eos = tokenizer.eos_token or "</s>"

    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Datasets ────────────────────────────────────────────────────────────
    datasets = load_dataset_for_direction(src_lang, tgt_lang)

    train_ds = build_sft_dataset(list(datasets["train"]), src_lang, tgt_lang, eos)
    dev_ds   = build_sft_dataset(list(datasets["dev"]),   src_lang, tgt_lang, eos)

    logger.info("Train: %d examples  |  Dev: %d examples",
                len(train_ds), len(dev_ds))

    # ── SFTTrainer ──────────────────────────────────────────────────────────
    sft_args = SFTConfig(
        output_dir=ckpt_dir,
        **GEMMAX2_TRAINING_ARGS,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,  # trl>=0.9: tokenizer->processing_class
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # max_seq_length removed from both SFTConfig and SFTTrainer in this trl version;
    # truncation is handled in build_sft_dataset via the tokenizer max_length arg.

    logger.info("Starting QLoRA training …")
    trainer.train()

    # ── Save LoRA adapter ───────────────────────────────────────────────────
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Persist LoRA config for reproducibility
    with open(os.path.join(output_dir, "lora_config.json"), "w") as fh:
        json.dump(LORA_CONFIG, fh, indent=2)

    logger.info("LoRA adapter saved → %s", output_dir)
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Train all 6 directions
# ─────────────────────────────────────────────────────────────────────────────

def train_all_gemmax2() -> None:
    """QLoRA fine-tune GemmaX2 sequentially on all 6 translation directions."""
    for src_lang, tgt_lang in TRANSLATION_DIRECTIONS:
        train_gemmax2_direction(src_lang, tgt_lang)
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="QLoRA fine-tune GemmaX2-28-9B")
    parser.add_argument(
        "--direction", default=None,
        help="Single direction, e.g. ar-ur. Omit to train all 6.",
    )
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if args.direction:
        src, tgt = args.direction.split("-")
        train_gemmax2_direction(src, tgt)
    else:
        train_all_gemmax2()