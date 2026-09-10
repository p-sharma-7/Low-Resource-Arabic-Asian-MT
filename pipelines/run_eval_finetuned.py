#!/usr/bin/env python3
"""
run_eval_finetuned.py — Evaluate all fine-tuned checkpoints.

Finds best checkpoint for each model/direction (checks both
outputs/finetuned/ and checkpoints/ directories), runs inference,
and reports all 4 metrics (COMET-22, ChrF2++, BLEU, TER).

Usage:
    python run_eval_finetuned.py                        # all models, dev split
    python run_eval_finetuned.py --split devtest        # held-out test
    python run_eval_finetuned.py --model nllb           # one model only
    python run_eval_finetuned.py --model nllb --direction ar-en
"""

import argparse
import json
import logging
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

from utlis.config import (
    TRANSLATION_DIRECTIONS, OUTPUT_DIR, CHECKPOINT_DIR,
    MODELS, SEED,
)
from utlis.data_loader import get_src_tgt_lists
from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

MODEL_WEIGHT_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint(model: str, src: str, tgt: str):
    """
    Return the best available checkpoint directory for a model/direction.
    Search order:
      1. outputs/finetuned/{model}/{dir}/          (completed training)
      2. checkpoints/{model}/{dir}/checkpoint-*/   (best epoch checkpoint)
      3. checkpoints/{model}/{dir}/                (flat save)
    Returns None if nothing found.
    """
    direction = f"{src}-{tgt}"

    def has_weights(path):
        if not os.path.isdir(path):
            return False
        for fname in os.listdir(path):
            if any(fname == wf or
                   (fname.endswith(".safetensors") and "model" in fname) or
                   (fname.endswith(".bin") and "model" in fname)
                   for wf in MODEL_WEIGHT_FILES):
                return True
        return False

    # 1. Final output dir
    out = os.path.join(OUTPUT_DIR, "finetuned", model, direction)
    if has_weights(out):
        return out

    # 2. Best epoch checkpoint subdir (pick highest step number)
    ckpt_base = os.path.join(CHECKPOINT_DIR, model, direction)
    if os.path.isdir(ckpt_base):
        subdirs = [
            e.path for e in os.scandir(ckpt_base)
            if e.is_dir() and e.name.startswith("checkpoint-")
            and has_weights(e.path)
        ]
        if subdirs:
            # Sort by step number descending, pick highest (best saved)
            subdirs.sort(key=lambda p: int(p.rsplit("-", 1)[-1]), reverse=True)
            return subdirs[0]

        # 3. Flat save directly in ckpt_base
        if has_weights(ckpt_base):
            return ckpt_base

    return None



def is_eval_done(model: str, src: str, tgt: str, split: str, out_dir: str) -> bool:
    """
    Returns True if hypothesis and metrics files already exist for this
    model/direction/split — meaning eval was completed in a prior run.
    """
    direction  = f"{src}-{tgt}"
    tag_map    = {"nllb": "nllb_ft", "madlad": "madlad_ft", "gemmax2": "gemmax2_ft"}
    tag        = tag_map[model]
    hyp_file   = os.path.join(out_dir, f"hyp_{tag}_{direction}_{split}.txt")
    metric_tag = f"{model}_finetuned"
    met_file   = os.path.join(out_dir, f"metrics_{metric_tag}_{direction}_{split}.txt")
    return os.path.exists(hyp_file) and os.path.exists(met_file)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model eval helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_nllb(directions, split, out_dir):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from infer_nllb import translate_nllb
    results = {}
    for src, tgt in directions:
        direction = f"{src}-{tgt}"
        if is_eval_done("nllb", src, tgt, split, out_dir):
            logger.info("NLLB  %s  [SKIP — already evaluated]", direction)
            continue
        ckpt = find_checkpoint("nllb", src, tgt)
        if not ckpt:
            logger.warning("NLLB — no checkpoint for %s, skipping.", direction)
            continue
        logger.info("NLLB  %s  ← %s", direction, ckpt)
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        sources, references = get_src_tgt_lists(src, tgt, split)
        hyps = translate_nllb(sources, model, tokenizer, src, tgt)
        save_hypotheses(hyps, src, tgt, "nllb_ft", split, out_dir)
        metrics = evaluate_all(sources, hyps, references, src, tgt)
        log_metrics(metrics, src, tgt, "nllb_finetuned", split, out_dir)
        results[direction] = metrics
        del model; torch.cuda.empty_cache()
    return results


def eval_madlad(directions, split, out_dir):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from infer_madlad import translate_madlad
    results = {}
    for src, tgt in directions:
        direction = f"{src}-{tgt}"
        if is_eval_done("madlad", src, tgt, split, out_dir):
            logger.info("MADLAD  %s  [SKIP — already evaluated]", direction)
            continue
        ckpt = find_checkpoint("madlad", src, tgt)
        if not ckpt:
            logger.warning("MADLAD — no checkpoint for %s, skipping.", direction)
            continue
        logger.info("MADLAD  %s  ← %s", direction, ckpt)

        # Detect LoRA adapter vs full model
        is_lora = any(
            f.startswith("adapter_") for f in os.listdir(ckpt)
        ) or os.path.exists(os.path.join(ckpt, "adapter_config.json"))

        if is_lora:
            logger.info("  Loading as 8-bit base + LoRA adapter")
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            base = AutoModelForSeq2SeqLM.from_pretrained(
                MODELS["madlad"],
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base, ckpt)
        else:
            logger.info("  Loading as full fine-tuned model")
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map="auto"
            )

        model.eval()
        sources, references = get_src_tgt_lists(src, tgt, split)
        hyps = translate_madlad(sources, model, tokenizer, src, tgt)
        save_hypotheses(hyps, src, tgt, "madlad_ft", split, out_dir)
        metrics = evaluate_all(sources, hyps, references, src, tgt)
        log_metrics(metrics, src, tgt, "madlad_finetuned", split, out_dir)
        results[direction] = metrics
        del model; torch.cuda.empty_cache()
    return results


def eval_gemmax2(directions, split, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from infer_gemmax2 import translate_gemmax2
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    results = {}
    for src, tgt in directions:
        direction = f"{src}-{tgt}"
        if is_eval_done("gemmax2", src, tgt, split, out_dir):
            logger.info("GemmaX2  %s  [SKIP — already evaluated]", direction)
            continue
        ckpt = find_checkpoint("gemmax2", src, tgt)
        if not ckpt:
            logger.warning("GemmaX2 — no checkpoint for %s, skipping.", direction)
            continue
        logger.info("GemmaX2  %s  ← %s", direction, ckpt)
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            MODELS["gemmax2"], quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, ckpt)
        model.eval()
        sources, references = get_src_tgt_lists(src, tgt, split)
        hyps = translate_gemmax2(sources, model, tokenizer, src, tgt)
        save_hypotheses(hyps, src, tgt, "gemmax2_ft", split, out_dir)
        metrics = evaluate_all(sources, hyps, references, src, tgt)
        log_metrics(metrics, src, tgt, "gemmax2_finetuned", split, out_dir)
        results[direction] = metrics
        del model, base; torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all fine-tuned checkpoints")
    parser.add_argument("--model", default="all",
                        choices=["nllb", "madlad", "gemmax2", "all"])
    parser.add_argument("--split", default="dev",
                        choices=["dev", "devtest"],
                        help="dev for validation, devtest for final held-out scoring")
    parser.add_argument("--direction", default=None, metavar="SRC-TGT",
                        help="Single direction e.g. ar-en. Default: all 6.")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    directions = (
        [tuple(args.direction.split("-"))]
        if args.direction else list(TRANSLATION_DIRECTIONS)
    )

    all_results = {}
    out_base = os.path.join(OUTPUT_DIR, "eval_finetuned")

    if args.model in ("all", "nllb"):
        logger.info("=" * 64 + "  NLLB")
        out = os.path.join(out_base, "nllb")
        all_results["nllb"] = eval_nllb(directions, args.split, out)

    if args.model in ("all", "madlad"):
        logger.info("=" * 64 + "  MADLAD")
        out = os.path.join(out_base, "madlad")
        all_results["madlad"] = eval_madlad(directions, args.split, out)

    if args.model in ("all", "gemmax2"):
        logger.info("=" * 64 + "  GemmaX2")
        out = os.path.join(out_base, "gemmax2")
        all_results["gemmax2"] = eval_gemmax2(directions, args.split, out)

    # Save combined results
    result_path = os.path.join(out_base, f"all_results_{args.split}.json")
    os.makedirs(out_base, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("All results saved → %s", result_path)


if __name__ == "__main__":
    main()