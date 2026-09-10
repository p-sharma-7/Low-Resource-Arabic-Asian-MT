#!/usr/bin/env python3
"""
resume_finetune.py — Resume fine-tuning from where it stopped.

Checks finetuned_checkpoints/{model}/{direction}/ for a saved model.
Skips directions that already have a completed checkpoint.
Runs remaining directions sequentially.

Usage:
    python resume_finetune.py                   # all 3 models, all 6 directions
    python resume_finetune.py --model nllb      # only NLLB remaining directions
    python resume_finetune.py --dry_run         # print what would run, don't train
"""
import argparse
import logging
import os
import sys
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get("WMT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utlis.config import TRANSLATION_DIRECTIONS, OUTPUT_DIR, CHECKPOINT_DIR, SEED

MODELS_ORDER = ["nllb", "madlad", "gemmax2"]


def _dir_has_model(path: str) -> bool:
    """
    Return True if a directory contains any model weight file.
    Handles single-file (model.safetensors) and sharded
    (model-00001-of-00003.safetensors) formats, plus LoRA adapters.
    """
    if not os.path.isdir(path):
        return False
    for fname in os.listdir(path):
        if (fname.endswith(".safetensors") or
                fname.endswith(".bin") and "model" in fname):
            return True
    return False


def is_done(model: str, src: str, tgt: str) -> bool:
    """
    A direction is complete if EITHER:
      1. finetuned_checkpoints/{model}/{dir}/ contains any model weight file.
      2. checkpoints/{model}/{dir}/ contains any subdirectory (epoch
         checkpoint) that has model weight files inside it.

    Handles single-file, sharded safetensors, and LoRA adapter formats.
    Handles interrupted training where final save_model() was never called.
    """
    direction = f"{src}-{tgt}"

    # Case 1: final output dir
    out_dir = os.path.join(OUTPUT_DIR, "finetuned_checkpoints", model, direction)
    if _dir_has_model(out_dir):
        return True

    # Case 2: any checkpoint subdirectory
    ckpt_dir = os.path.join(CHECKPOINT_DIR, model, direction)
    if os.path.isdir(ckpt_dir):
        # Check immediate children (could be checkpoint-N dirs or flat save)
        for entry in os.scandir(ckpt_dir):
            if entry.is_dir():
                if _dir_has_model(entry.path):
                    return True
            # Also handle flat saves directly in ckpt_dir
        if _dir_has_model(ckpt_dir):
            return True

    return False


def run(model_filter: str = "all", dry_run: bool = False) -> None:
    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    models = MODELS_ORDER if model_filter == "all" else [model_filter]

    for model in models:
        logger.info("=" * 64)
        logger.info("Model: %s", model.upper())
        logger.info("=" * 64)

        pending = []
        for src, tgt in TRANSLATION_DIRECTIONS:
            direction = f"{src}-{tgt}"
            if is_done(model, src, tgt):
                logger.info("  [SKIP — already done] %s", direction)
            else:
                logger.info("  [PENDING]             %s", direction)
                pending.append((src, tgt))

        if not pending:
            logger.info("  All directions complete for %s.", model)
            continue

        if dry_run:
            logger.info("  Dry run — would train: %s",
                        [f"{s}-{t}" for s, t in pending])
            continue

        # ── Load trainer for this model ───────────────────────────────────
        if model == "nllb":
            from pipelines.train_nllb import train_nllb_direction as train_fn
        elif model == "madlad":
            from pipelines.train_madlad import train_madlad_direction as train_fn
        elif model == "gemmax2":
            from pipelines.train_gemmax2 import train_gemmax2_direction as train_fn

        for src, tgt in pending:
            direction = f"{src}-{tgt}"
            logger.info("\n  >>> Training %s  %s <<<", model.upper(), direction)
            try:
                train_fn(src, tgt)
                torch.cuda.empty_cache()
                logger.info("  [DONE] %s  %s", model.upper(), direction)
            except Exception as e:
                logger.error("  [FAILED] %s  %s: %s", model.upper(), direction, e)
                logger.error("  Stopping — fix the error and re-run to resume.")
                raise


def _debug_checkpoint_scan(model_filter: str = "all") -> None:
    """Print exactly what files is_done() sees in checkpoint and output dirs."""
    models = MODELS_ORDER if model_filter == "all" else [model_filter]
    for model in models:
        print(f"\n{'='*60}  {model.upper()}")
        for src, tgt in TRANSLATION_DIRECTIONS:
            direction = f"{src}-{tgt}"
            done = is_done(model, src, tgt)
            ckpt = os.path.join(CHECKPOINT_DIR, model, direction)
            out  = os.path.join(OUTPUT_DIR, "finetuned_checkpoints", model, direction)
            print(f"  {direction}  done={done}")
            for label, path in [("  ckpt", ckpt), ("  out ", out)]:
                if os.path.isdir(path):
                    entries = []
                    for root, dirs, files in os.walk(path):
                        for f in files:
                            if f.endswith(".safetensors") or \
                               (f.endswith(".bin") and "model" in f):
                                rel = os.path.relpath(os.path.join(root, f), path)
                                entries.append(rel)
                    print(f"    {label}: {path}")
                    if entries:
                        for e in entries[:4]: print(f"          {e}")
                        if len(entries) > 4: print(f"          ... +{len(entries)-4} more")
                    else:
                        print(f"          (no model files found)")
                else:
                    print(f"    {label}: NOT FOUND  {path}")


def main():
    parser = argparse.ArgumentParser(description="Resume WMT26 fine-tuning")
    parser.add_argument("--model", default="all",
                        choices=["nllb", "madlad", "gemmax2", "all"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Print pending directions without training")
    parser.add_argument("--debug", action="store_true",
                        help="Show exactly what files is_done() finds")
    args = parser.parse_args()

    # GPU check
    if not torch.cuda.is_available():
        logger.error("No CUDA GPU found — aborting.")
        sys.exit(1)

    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    total_gb = torch.cuda.mem_get_info(0)[1] / 1024**3
    logger.info("GPU: %s  |  Free: %.1f GB / %.1f GB",
                torch.cuda.get_device_name(0), free_gb, total_gb)

    if free_gb < 25 and not args.dry_run:
        logger.warning(
            "Only %.1f GB free. NLLB needs ~28 GB, MADLAD ~38 GB. "
            "Kill other GPU processes first:\n"
            "  nvidia-smi   # find PID\n"
            "  kill -9 PID",
            free_gb,
        )
        if free_gb < 15:
            logger.error("Less than 15 GB free — too risky to start. Aborting.")
            sys.exit(1)

    if args.debug:
        _debug_checkpoint_scan(args.model)
        return
    run(model_filter=args.model, dry_run=args.dry_run)
    logger.info("Resume script complete.")


if __name__ == "__main__":
    main()