"""
run_pipeline.py — Master orchestrator for WMT26 Arabic-Asian MT Challenge.

Pipeline stages
───────────────
  zero_shot   Run all 3 models on devtest with no fine-tuning (baseline).
  finetune    Fine-tune all 3 models on train/dev using the best protocol
              for each architecture (Seq2SeqTrainer or SFTTrainer+LoRA).
  eval        Load saved fine-tuned checkpoints, translate devtest, report
              all 5 metrics per direction.
  all         Run zero_shot → finetune → eval in sequence.

Examples
────────
  # Full pipeline, all models
  python run_pipeline.py --stage all

  # Zero-shot baselines only
  python run_pipeline.py --stage zero_shot

  # Fine-tune only NLLB
  python run_pipeline.py --stage finetune --model nllb

  # Evaluate fine-tuned GemmaX2 on dev split
  python run_pipeline.py --stage eval --model gemmax2 --split dev

  # Single direction fine-tune for quick iteration
  python run_pipeline.py --stage finetune --model madlad --direction ar-hi
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from utlis.gpu_utils import get_device_map, require_cuda
from utlis.config import (
    SEED,
    LOG_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    TRANSLATION_DIRECTIONS,
    MODELS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(stage: str, model_tag: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{stage}_{model_tag}_{ts}.log")

    fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    root_logger = logging.getLogger()
    root_logger.info("Log file: %s", log_file)
    return root_logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

BANNER = "█" * 64

def _banner(msg: str) -> None:
    logger.info("\n%s\n  %s\n%s", BANNER, msg, BANNER)


def _save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Results saved → %s", path)


def _directions_for(direction_filter: Optional[str]):
    """Return list of (src, tgt) tuples, optionally filtered to one direction."""
    if direction_filter:
        src, tgt = direction_filter.split("-")
        return [(src, tgt)]
    return list(TRANSLATION_DIRECTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Zero-shot inference
# ─────────────────────────────────────────────────────────────────────────────

def run_zero_shot(
    model_filter: str = "all",
    split: str = "devtest",
    direction_filter: Optional[str] = None,
) -> dict:
    """Run zero-shot inference for the requested model(s) and directions."""
    all_results: dict = {}

    # ── NLLB ────────────────────────────────────────────────────────────────
    if model_filter in ("all", "nllb"):
        _banner("ZERO-SHOT | NLLB-200-3.3B")
        from pipelines.infer_nllb import load_nllb, translate_nllb
        from utlis.data_loader import get_src_tgt_lists
        from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

        out_dir = os.path.join(OUTPUT_DIR, "zero_shot", "nllb")
        model, tokenizer = load_nllb()
        nllb_results: dict = {}

        for src_lang, tgt_lang in _directions_for(direction_filter):
            d = f"{src_lang}-{tgt_lang}"
            logger.info("  NLLB zero-shot  %s  [%s]", d, split)
            sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)
            hyps = translate_nllb(sources, model, tokenizer, src_lang, tgt_lang)
            save_hypotheses(hyps, src_lang, tgt_lang, "nllb", split, out_dir)
            metrics = evaluate_all(sources, hyps, references, src_lang, tgt_lang)
            log_metrics(metrics, src_lang, tgt_lang, "nllb", split, out_dir)
            nllb_results[d] = metrics

        all_results["nllb"] = nllb_results
        del model; torch.cuda.empty_cache()

    # ── MADLAD ──────────────────────────────────────────────────────────────
    if model_filter in ("all", "madlad"):
        _banner("ZERO-SHOT | MADLAD-400-10B")
        from pipelines.infer_madlad import load_madlad, translate_madlad
        from utlis.data_loader import get_src_tgt_lists
        from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

        out_dir = os.path.join(OUTPUT_DIR, "zero_shot", "madlad")
        model, tokenizer = load_madlad()
        madlad_results: dict = {}

        for src_lang, tgt_lang in _directions_for(direction_filter):
            d = f"{src_lang}-{tgt_lang}"
            logger.info("  MADLAD zero-shot  %s  [%s]", d, split)
            sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)
            hyps = translate_madlad(sources, model, tokenizer, src_lang, tgt_lang)
            save_hypotheses(hyps, src_lang, tgt_lang, "madlad", split, out_dir)
            metrics = evaluate_all(sources, hyps, references, src_lang, tgt_lang)
            log_metrics(metrics, src_lang, tgt_lang, "madlad", split, out_dir)
            madlad_results[d] = metrics

        all_results["madlad"] = madlad_results
        del model; torch.cuda.empty_cache()

    # ── GemmaX2 ─────────────────────────────────────────────────────────────
    if model_filter in ("all", "gemmax2"):
        _banner("ZERO-SHOT | GemmaX2-28-9B")
        from pipelines.infer_gemmax2 import load_gemmax2, translate_gemmax2
        from utlis.data_loader import get_src_tgt_lists
        from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

        out_dir = os.path.join(OUTPUT_DIR, "zero_shot", "gemmax2")
        model, tokenizer = load_gemmax2()
        gemmax2_results: dict = {}

        for src_lang, tgt_lang in _directions_for(direction_filter):
            d = f"{src_lang}-{tgt_lang}"
            logger.info("  GemmaX2 zero-shot  %s  [%s]", d, split)
            sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)
            hyps = translate_gemmax2(sources, model, tokenizer, src_lang, tgt_lang)
            save_hypotheses(hyps, src_lang, tgt_lang, "gemmax2", split, out_dir)
            metrics = evaluate_all(sources, hyps, references, src_lang, tgt_lang)
            log_metrics(metrics, src_lang, tgt_lang, "gemmax2", split, out_dir)
            gemmax2_results[d] = metrics

        all_results["gemmax2"] = gemmax2_results
        del model; torch.cuda.empty_cache()

    _save_json(all_results,
               os.path.join(OUTPUT_DIR, "zero_shot", "all_results.json"))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(
    model_filter: str = "all",
    direction_filter: Optional[str] = None,
) -> None:
    """Fine-tune the requested model(s) on the requested direction(s)."""

    if model_filter in ("all", "nllb"):
        _banner("FINE-TUNE | NLLB-200-3.3B")
        from pipelines.train_nllb import train_nllb_direction
        for src_lang, tgt_lang in _directions_for(direction_filter):
            train_nllb_direction(src_lang, tgt_lang)
            torch.cuda.empty_cache()

    if model_filter in ("all", "madlad"):
        _banner("FINE-TUNE | MADLAD-400-10B")
        from pipelines.train_madlad import train_madlad_direction
        for src_lang, tgt_lang in _directions_for(direction_filter):
            train_madlad_direction(src_lang, tgt_lang)
            torch.cuda.empty_cache()

    if model_filter in ("all", "gemmax2"):
        _banner("FINE-TUNE | GemmaX2-28-9B (QLoRA)")
        from pipelines.train_gemmax2 import train_gemmax2_direction
        for src_lang, tgt_lang in _directions_for(direction_filter):
            train_gemmax2_direction(src_lang, tgt_lang)
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Evaluate fine-tuned checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    model_filter: str = "all",
    split: str = "devtest",
    direction_filter: Optional[str] = None,
) -> dict:
    """
    Load saved fine-tuned checkpoints, run inference, compute all 5 metrics.
    """
    from utlis.data_loader import get_src_tgt_lists
    from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

    all_results: dict = {}

    # ── NLLB ────────────────────────────────────────────────────────────────
    if model_filter in ("all", "nllb"):
        _banner("EVAL (fine-tuned) | NLLB-200-3.3B")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from pipelines.infer_nllb import translate_nllb

        nllb_results: dict = {}
        for src_lang, tgt_lang in _directions_for(direction_filter):
            d      = f"{src_lang}-{tgt_lang}"
            ft_dir = os.path.join(OUTPUT_DIR, "finetuned_checkpoints", "nllb", d)
            if not os.path.isdir(ft_dir):
                logger.warning("No NLLB checkpoint for %s at %s – skipping.", d, ft_dir)
                continue

            out_dir = os.path.join(OUTPUT_DIR, "eval_finetuned", "nllb")
            tokenizer = AutoTokenizer.from_pretrained(ft_dir)
            model     = AutoModelForSeq2SeqLM.from_pretrained(
                ft_dir, dtype=torch.float16, device_map=get_device_map()
            )
            sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)
            hyps = translate_nllb(sources, model, tokenizer, src_lang, tgt_lang)
            save_hypotheses(hyps, src_lang, tgt_lang, "nllb_ft", split, out_dir)
            metrics = evaluate_all(sources, hyps, references, src_lang, tgt_lang)
            log_metrics(metrics, src_lang, tgt_lang, "nllb_finetuned", split, out_dir)
            nllb_results[d] = metrics
            del model; torch.cuda.empty_cache()

        all_results["nllb"] = nllb_results

    # ── MADLAD ──────────────────────────────────────────────────────────────
    if model_filter in ("all", "madlad"):
        _banner("EVAL (fine-tuned) | MADLAD-400-10B")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from pipelines.infer_madlad import translate_madlad

        madlad_results: dict = {}
        for src_lang, tgt_lang in _directions_for(direction_filter):
            d      = f"{src_lang}-{tgt_lang}"
            ft_dir = os.path.join(OUTPUT_DIR, "finetuned_checkpoints", "madlad", d)
            if not os.path.isdir(ft_dir):
                logger.warning("No MADLAD checkpoint for %s at %s – skipping.", d, ft_dir)
                continue

            out_dir = os.path.join(OUTPUT_DIR, "eval_finetuned", "madlad")
            tokenizer = AutoTokenizer.from_pretrained(ft_dir)
            model     = AutoModelForSeq2SeqLM.from_pretrained(
                ft_dir, dtype=torch.float16, device_map=get_device_map()
            )
            sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)
            hyps = translate_madlad(sources, model, tokenizer, src_lang, tgt_lang)
            save_hypotheses(hyps, src_lang, tgt_lang, "madlad_ft", split, out_dir)
            metrics = evaluate_all(sources, hyps, references, src_lang, tgt_lang)
            log_metrics(metrics, src_lang, tgt_lang, "madlad_finetuned", split, out_dir)
            madlad_results[d] = metrics
            del model; torch.cuda.empty_cache()

        all_results["madlad"] = madlad_results

    # ── GemmaX2 ─────────────────────────────────────────────────────────────
    if model_filter in ("all", "gemmax2"):
        _banner("EVAL (fine-tuned) | GemmaX2-28-9B + LoRA")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from pipelines.infer_gemmax2 import translate_gemmax2

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        gemmax2_results: dict = {}
        for src_lang, tgt_lang in _directions_for(direction_filter):
            d           = f"{src_lang}-{tgt_lang}"
            adapter_dir = os.path.join(OUTPUT_DIR, "finetuned_checkpoints", "gemmax2", d)
            if not os.path.isdir(adapter_dir):
                logger.warning("No GemmaX2 adapter for %s at %s – skipping.", d, adapter_dir)
                continue

            out_dir = os.path.join(OUTPUT_DIR, "eval_finetuned", "gemmax2")
            logger.info("Loading GemmaX2 + LoRA adapter for %s …", d)
            base = AutoModelForCausalLM.from_pretrained(
                MODELS["gemmax2"],
                quantization_config=bnb_config,
                device_map=get_device_map(),
                dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(base, adapter_dir)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)
            hyps = translate_gemmax2(sources, model, tokenizer, src_lang, tgt_lang)
            save_hypotheses(hyps, src_lang, tgt_lang, "gemmax2_ft", split, out_dir)
            metrics = evaluate_all(sources, hyps, references, src_lang, tgt_lang)
            log_metrics(metrics, src_lang, tgt_lang, "gemmax2_finetuned", split, out_dir)
            gemmax2_results[d] = metrics
            del model, base; torch.cuda.empty_cache()

        all_results["gemmax2"] = gemmax2_results

    _save_json(all_results,
               os.path.join(OUTPUT_DIR, "eval_finetuned", "all_results.json"))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="WMT26 Arabic-Asian MT Challenge — full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["zero_shot", "finetune", "eval", "all"],
        help="Pipeline stage(s) to execute.",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["nllb", "madlad", "gemmax2", "all"],
        help="Model to run (default: all).",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "devtest"],
        help="Evaluation split (default: dev for zero-shot/eval; use devtest for final held-out scoring).",
    )
    parser.add_argument(
        "--direction",
        default=None,
        metavar="SRC-TGT",
        help="Restrict to a single direction, e.g. ar-hi.  Default: all 6.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    logger = setup_logging(args.stage, args.model)
    set_seed(SEED)
    require_cuda()   # abort early if no GPU visible

    logger.info("Stage=%s | Model=%s | Split=%s | Direction=%s | Seed=%d",
                args.stage, args.model, args.split,
                args.direction or "all", SEED)

    if args.stage in ("zero_shot", "all"):
        run_zero_shot(
            model_filter=args.model,
            split=args.split,
            direction_filter=args.direction,
        )

    if args.stage in ("finetune", "all"):
        run_finetune(
            model_filter=args.model,
            direction_filter=args.direction,
        )

    if args.stage in ("eval", "all"):
        run_eval(
            model_filter=args.model,
            split=args.split,
            direction_filter=args.direction,
        )

    _banner("PIPELINE COMPLETE")