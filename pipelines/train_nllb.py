"""
train_nllb.py — Fine-tuning facebook/nllb-200-3.3B with Seq2SeqTrainer.

Training protocol:
  • Optimizer  : AdamW (fp16)
  • Scheduler  : linear warmup + decay
  • Eval metric: COMET-22 (primary) — best checkpoint kept per direction
  • Early stop : patience=2 epochs (on COMET-22)
  • Splits     : train_* for training, dev_* for validation

Each of the 6 translation directions is fine-tuned as a separate run,
saving its best checkpoint under:
    checkpoints/nllb/{src_lang}-{tgt_lang}/

Standalone usage:
    python train_nllb.py [--direction ar-en] [--all]
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utlis.gpu_utils import get_device_map, require_cuda
from utlis.config import (
    MODELS,
    NLLB_LANG_CODES,
    NLLB_TRAINING_ARGS,
    TRANSLATION_DIRECTIONS,
    MAX_INPUT_LENGTH,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SEED,
)
from utlis.data_loader import load_dataset_for_direction, get_src_tgt_lists
from pipelines.evaluate import compute_comet22

logger = logging.getLogger(__name__)

MODEL_NAME = MODELS["nllb"]

def _get_lang_token_id(tokenizer, lang_code):
    """Robust NLLB language token ID lookup for both slow and fast tokenizers."""
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id[lang_code]
    if lang_code in tokenizer.additional_special_tokens:
        idx = tokenizer.additional_special_tokens.index(lang_code)
        return tokenizer.additional_special_tokens_ids[idx]
    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if token_id != tokenizer.unk_token_id:
        return token_id
    raise ValueError("Language code " + repr(lang_code) + " not found in tokenizer vocabulary.")


MODEL_KEY  = "nllb"


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_batch(
    examples: Dict,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
) -> Dict:
    """
    Tokenise source and target sentences for NLLB Seq2Seq training.

    Setting tokenizer.src_lang before calling ensures the correct BOS/EOS
    language tokens are prepended to the encoder input.
    Using `text_target` (instead of the deprecated `as_target_tokenizer()`)
    encodes the target side with the proper target-language handling.
    """
    src_code     = NLLB_LANG_CODES[src_lang]
    tgt_code_tok = NLLB_LANG_CODES[tgt_lang]
    tokenizer.src_lang = src_code
    tokenizer.tgt_lang = tgt_code_tok

    model_inputs = tokenizer(
        examples["src"],
        text_target=examples["tgt"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    return model_inputs


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics callback  (COMET-22)
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(
    tokenizer: AutoTokenizer,
    dev_sources: List[str],
    dev_references: List[str],
):
    """
    Return a compute_metrics function closed over dev sources + references.

    The trainer calls this once per epoch with EvalPrediction(predictions,
    label_ids).  With predict_with_generate=True the predictions are already
    token-id arrays (not logits), so we just decode and score with COMET-22.
    """
    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 padding sentinels before decoding
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        hypotheses = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Ensure equal lengths (safety guard for partial batches)
        n = min(len(hypotheses), len(dev_sources), len(dev_references))
        comet = compute_comet22(
            dev_sources[:n], hypotheses[:n], dev_references[:n]
        )
        return {"comet22": comet}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Single-direction training
# ─────────────────────────────────────────────────────────────────────────────

def train_nllb_direction(src_lang: str, tgt_lang: str) -> str:
    """
    Fine-tune NLLB on one translation direction.

    Returns the path to the best saved checkpoint.
    """
    direction = f"{src_lang}-{tgt_lang}"
    logger.info("\n%s\nFine-tuning NLLB  |  %s\n%s",
                "=" * 64, direction, "=" * 64)

    ckpt_dir   = os.path.join(CHECKPOINT_DIR, "nllb", direction)
    output_dir = os.path.join(OUTPUT_DIR, "finetuned", "nllb", direction)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load tokeniser & model ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False   # required for gradient_checkpointing

    # Set forced_bos for generation (target language start token)
    tgt_code      = NLLB_LANG_CODES[tgt_lang]
    forced_bos_id = _get_lang_token_id(tokenizer, tgt_code)
    model.config.forced_bos_token_id = forced_bos_id

    # ── Datasets ────────────────────────────────────────────────────────────
    datasets = load_dataset_for_direction(src_lang, tgt_lang)
    max_len  = MAX_INPUT_LENGTH[MODEL_KEY]

    def tokenize(batch):
        return tokenize_batch(batch, tokenizer, src_lang, tgt_lang, max_len)

    train_ds = datasets["train"].map(
        tokenize, batched=True,
        remove_columns=["src", "tgt"],
        desc=f"Tokenising train [{direction}]",
    )
    dev_ds = datasets["dev"].map(
        tokenize, batched=True,
        remove_columns=["src", "tgt"],
        desc=f"Tokenising dev [{direction}]",
    )

    # Raw dev text for COMET computation inside compute_metrics
    dev_sources, dev_references = get_src_tgt_lists(src_lang, tgt_lang, "dev")
    compute_metrics = make_compute_metrics(tokenizer, dev_sources, dev_references)

    # ── Trainer ─────────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    training_args = Seq2SeqTrainingArguments(
        output_dir=ckpt_dir,
        **NLLB_TRAINING_ARGS,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training …")
    trainer.train()

    # ── Save best model ──────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Best checkpoint saved → %s", output_dir)

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Train all 6 directions
# ─────────────────────────────────────────────────────────────────────────────

def train_all_nllb() -> None:
    """Fine-tune NLLB-200-3.3B sequentially on all 6 translation directions."""
    for src_lang, tgt_lang in TRANSLATION_DIRECTIONS:
        train_nllb_direction(src_lang, tgt_lang)
        # Free GPU memory between directions
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200-3.3B")
    parser.add_argument(
        "--direction", default=None,
        help="Single direction to train, e.g. ar-en. Omit to train all.",
    )
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if args.direction:
        src, tgt = args.direction.split("-")
        train_nllb_direction(src, tgt)
    else:
        train_all_nllb()