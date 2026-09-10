#!/usr/bin/env python3
"""
run_eval_finetuned.py — Evaluate fine-tuned checkpoints on the FULL
                       NLP-IIT Patna Challenge-Test set.

Scope:
  • 6 translation directions (ar↔ur, ar→hi, ar→en, en→ar, hi→ar)
  • 3 fine-tuned models: NLLB, MADLAD, GemmaX2
  • Reads the COMPLETE challenge test file for each direction (no sampling,
    no train/dev/test split — every line is translated).
  • Inputs longer than 512 tokens are split at sentence boundaries
    (multi-script aware), translated in chunks, then reassembled — no truncation.
  • Saves translated hypotheses always; computes COMET-22 / ChrF2++ / BLEU / TER
    only if gold references exist next to the source file.

Usage:
    python run_eval_finetuned.py                        # all 3 models, all 6 directions
    python run_eval_finetuned.py --model nllb           # one model only
    python run_eval_finetuned.py --direction ar-en      # one direction only
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Callable, List, Optional, Tuple

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

from utlis.config import OUTPUT_DIR, CHECKPOINT_DIR, MODELS, SEED
from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

CHALLENGE_ROOT = "/mnt/storage/Ahtisham_Aziz/NLP-IIT Patna/Challenge-Test"

# 6 directions × (source_file, optional_reference_file) — paths relative to CHALLENGE_ROOT.
# Reference files don't ship today; they'll be auto-picked-up if you add them later.
CHALLENGE_FILES = {
    ("ar", "ur"): (
        "Sub-Task 2E_Arabic-to-Urdu/Ar-to-Ur/test_challenge_ar_ar-to-ur.txt",
        "Sub-Task 2E_Arabic-to-Urdu/Ar-to-Ur/test_challenge_ur_ar-to-ur.txt",
    ),
    ("ur", "ar"): (
        "Sub-Task 1E_Urdu-to-Arabic/Ur-to-Ar/test_challenge_ur_ur-to-ar.txt",
        "Sub-Task 1E_Urdu-to-Arabic/Ur-to-Ar/test_challenge_ar_ur-to-ar.txt",
    ),
    ("ar", "hi"): (
        "Sub-Task 2B_Arabic-to-Hindi/Ar-to-Hi/test_challenge_ar_ar-to-hi.txt",
        "Sub-Task 2B_Arabic-to-Hindi/Ar-to-Hi/test_challenge_hi_ar-to-hi.txt",
    ),
    ("ar", "en"): (
        "Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt",
        "Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_en_ar-to-en.txt",
    ),
    ("en", "ar"): (
        "Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt",
        "Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_ar_en-to-ar.txt",
    ),
    ("hi", "ar"): (
        "Sub-Task 1B_Hindi-to-Arabic/Hi-to-Ar/test_challenge_hi_hi-to-ar.txt",
        "Sub-Task 1B_Hindi-to-Arabic/Hi-to-Ar/test_challenge_ar_hi-to-ar.txt",
    ),
}

# Output naming uses this tag instead of a split name.
EVAL_TAG = "challenge"

MODEL_WEIGHT_FILES = [
    "model.safetensors", "pytorch_model.bin",
    "adapter_model.safetensors", "adapter_model.bin",
]

# Token budget per chunk — 480 leaves headroom for special tokens (lang IDs, BOS/EOS)
# on top of the model's 512-token limit.
CHUNK_MAX_TOKENS = 480


# ═════════════════════════════════════════════════════════════════════════════
# Sliding-window chunking (sentence-aware, multi-script)
# ═════════════════════════════════════════════════════════════════════════════

# Sentence boundary punctuation for the 4 scripts:
#   .  !  ?   → Latin / English
#   ؟        → Arabic question mark
#   ۔        → Urdu / Arabic full stop
#   ।        → Devanagari (Hindi) danda
_SENT_BOUNDARY = re.compile(r"(?<=[.!?؟۔।])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _n_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _word_split(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Fallback splitter for single sentences longer than the token budget."""
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if _n_tokens(" ".join(buf), tokenizer) > max_tokens:
            buf.pop()
            if buf:
                chunks.append(" ".join(buf))
            buf = [w]
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_by_tokens(text: str, tokenizer, max_tokens: int = CHUNK_MAX_TOKENS) -> List[str]:
    """
    Split text into chunks each ≤ max_tokens.

    • Short input fits in one chunk → returned as-is (no overhead).
    • Long input → split at sentence boundaries, greedily packed.
    • A single overlong sentence falls back to word-level splits.
    """
    if not text or not text.strip():
        return [text]

    if _n_tokens(text, tokenizer) <= max_tokens:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return _word_split(text, tokenizer, max_tokens)

    chunks, buf = [], []
    for sent in sentences:
        candidate = " ".join(buf + [sent])
        if _n_tokens(candidate, tokenizer) > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf = [sent]
        else:
            buf.append(sent)
    if buf:
        chunks.append(" ".join(buf))

    # Any chunk still too long? (single giant sentence) → word-level split.
    final = []
    for c in chunks:
        if _n_tokens(c, tokenizer) > max_tokens:
            final.extend(_word_split(c, tokenizer, max_tokens))
        else:
            final.append(c)
    return final


def translate_with_chunking(
    sources: List[str],
    translate_fn: Callable[[List[str]], List[str]],
    tokenizer,
    max_tokens: int = CHUNK_MAX_TOKENS,
    join_with: str = " ",
) -> List[str]:
    """
    Wrap any translate_fn(batch) -> list[str] with chunking.

      1. Each source is split into ≤max_tokens chunks (sentence-aware).
      2. All chunks are flattened and sent through translate_fn in ONE call
         so the underlying function can batch internally — no efficiency hit.
      3. Translated chunks are stitched back to one output per original source.
    """
    flat_chunks: List[str] = []
    owner: List[int] = []
    for i, src_text in enumerate(sources):
        for chunk in chunk_by_tokens(src_text, tokenizer, max_tokens=max_tokens):
            flat_chunks.append(chunk)
            owner.append(i)

    translated = translate_fn(flat_chunks)

    if len(translated) != len(flat_chunks):
        raise RuntimeError(
            f"translate_fn returned {len(translated)} outputs for "
            f"{len(flat_chunks)} chunks — mismatch."
        )

    assembled: List[List[str]] = [[] for _ in range(len(sources))]
    for chunk_idx, src_idx in enumerate(owner):
        assembled[src_idx].append(translated[chunk_idx])
    return [join_with.join(parts) for parts in assembled]


# ═════════════════════════════════════════════════════════════════════════════
# File I/O — reads the COMPLETE challenge file (no sampling, no truncation)
# ═════════════════════════════════════════════════════════════════════════════

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def load_full_test(src: str, tgt: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Return (sources, references) — full file contents. references may be None."""
    paths = CHALLENGE_FILES.get((src, tgt))
    if paths is None:
        return None, None
    src_rel, ref_rel = paths
    src_path = os.path.join(CHALLENGE_ROOT, src_rel)
    if not os.path.exists(src_path):
        return None, None
    sources = _read_lines(src_path)

    ref_path = os.path.join(CHALLENGE_ROOT, ref_rel)
    references = _read_lines(ref_path) if os.path.exists(ref_path) else None
    return sources, references


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint discovery
# ═════════════════════════════════════════════════════════════════════════════

def find_checkpoint(model: str, src: str, tgt: str) -> Optional[str]:
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

    out = os.path.join(OUTPUT_DIR, "finetuned", model, direction)
    if has_weights(out):
        return out

    ckpt_base = os.path.join(CHECKPOINT_DIR, model, direction)
    if os.path.isdir(ckpt_base):
        subdirs = [
            e.path for e in os.scandir(ckpt_base)
            if e.is_dir() and e.name.startswith("checkpoint-") and has_weights(e.path)
        ]
        if subdirs:
            subdirs.sort(key=lambda p: int(p.rsplit("-", 1)[-1]), reverse=True)
            return subdirs[0]
        if has_weights(ckpt_base):
            return ckpt_base
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Save hypotheses + (optionally) score
# ═════════════════════════════════════════════════════════════════════════════

def _score_and_save(
    sources, references, hyps,
    src, tgt, model_tag, file_tag, out_dir, results_dict,
):
    direction = f"{src}-{tgt}"
    save_hypotheses(hyps, src, tgt, file_tag, EVAL_TAG, out_dir)

    if references is None:
        logger.info(
            "[%s | %s] no references found — hypotheses saved, metrics skipped.",
            model_tag, direction,
        )
        results_dict[direction] = {"note": "hypotheses only; references unavailable"}
        return

    if len(references) != len(sources):
        logger.warning(
            "[%s | %s] source/reference length mismatch (%d vs %d) — metrics skipped.",
            model_tag, direction, len(sources), len(references),
        )
        results_dict[direction] = {
            "note": f"length mismatch src={len(sources)} ref={len(references)}"
        }
        return

    metrics = evaluate_all(sources, hyps, references, src, tgt)
    log_metrics(metrics, src, tgt, model_tag, EVAL_TAG, out_dir)
    results_dict[direction] = metrics


# ═════════════════════════════════════════════════════════════════════════════
# Per-model evaluation — all three wrap their translate_* with chunking
# ═════════════════════════════════════════════════════════════════════════════

def eval_nllb(directions, out_dir):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from pipelines.infer_nllb import translate_nllb

    results = {}
    for src, tgt in directions:
        direction = f"{src}-{tgt}"
        ckpt = find_checkpoint("nllb", src, tgt)
        if not ckpt:
            logger.warning("NLLB — no checkpoint for %s, skipping.", direction)
            continue

        sources, references = load_full_test(src, tgt)
        if sources is None:
            logger.warning("NLLB — no test data for %s, skipping.", direction)
            continue

        logger.info("NLLB  %s  ← %s  (full set: %d lines)", direction, ckpt, len(sources))
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()

        hyps = translate_with_chunking(
            sources,
            translate_fn=lambda batch: translate_nllb(batch, model, tokenizer, src, tgt),
            tokenizer=tokenizer,
        )

        _score_and_save(sources, references, hyps, src, tgt,
                        "nllb_finetuned", "nllb_ft", out_dir, results)
        del model
        torch.cuda.empty_cache()
    return results


def eval_madlad(directions, out_dir):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from pipelines.infer_madlad import translate_madlad

    results = {}
    for src, tgt in directions:
        direction = f"{src}-{tgt}"
        ckpt = find_checkpoint("madlad", src, tgt)
        if not ckpt:
            logger.warning("MADLAD — no checkpoint for %s, skipping.", direction)
            continue

        sources, references = load_full_test(src, tgt)
        if sources is None:
            logger.warning("MADLAD — no test data for %s, skipping.", direction)
            continue

        logger.info("MADLAD  %s  ← %s  (full set: %d lines)", direction, ckpt, len(sources))

        is_lora = any(f.startswith("adapter_") for f in os.listdir(ckpt)) \
                  or os.path.exists(os.path.join(ckpt, "adapter_config.json"))

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
                ckpt, torch_dtype=torch.bfloat16, device_map="auto",
            )
        model.eval()

        hyps = translate_with_chunking(
            sources,
            translate_fn=lambda batch: translate_madlad(batch, model, tokenizer, src, tgt),
            tokenizer=tokenizer,
        )

        _score_and_save(sources, references, hyps, src, tgt,
                        "madlad_finetuned", "madlad_ft", out_dir, results)
        del model
        torch.cuda.empty_cache()
    return results


def eval_gemmax2(directions, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from pipelines.infer_gemmax2 import translate_gemmax2

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    results = {}
    for src, tgt in directions:
        direction = f"{src}-{tgt}"
        ckpt = find_checkpoint("gemmax2", src, tgt)
        if not ckpt:
            logger.warning("GemmaX2 — no checkpoint for %s, skipping.", direction)
            continue

        sources, references = load_full_test(src, tgt)
        if sources is None:
            logger.warning("GemmaX2 — no test data for %s, skipping.", direction)
            continue

        logger.info("GemmaX2  %s  ← %s  (full set: %d lines)", direction, ckpt, len(sources))
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

        hyps = translate_with_chunking(
            sources,
            translate_fn=lambda batch: translate_gemmax2(batch, model, tokenizer, src, tgt),
            tokenizer=tokenizer,
        )

        _score_and_save(sources, references, hyps, src, tgt,
                        "gemmax2_finetuned", "gemmax2_ft", out_dir, results)
        del model, base
        torch.cuda.empty_cache()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned NLLB / MADLAD / GemmaX2 on the FULL "
                    "NLP-IIT Patna Challenge-Test set (6 directions, no splits)."
    )
    parser.add_argument("--model", default="all",
                        choices=["nllb", "madlad", "gemmax2", "all"])
    parser.add_argument("--direction", default=None, metavar="SRC-TGT",
                        help="Run a single direction (e.g. ar-en). Default: all 6.")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if args.direction:
        directions = [tuple(args.direction.split("-"))]
        if tuple(directions[0]) not in CHALLENGE_FILES:
            logger.error("Direction %s is not in the challenge set. Valid: %s",
                         args.direction,
                         ", ".join(f"{a}-{b}" for a, b in CHALLENGE_FILES.keys()))
            sys.exit(1)
    else:
        directions = list(CHALLENGE_FILES.keys())

    logger.info("Evaluating %d direction(s) on full challenge test set:",
                len(directions))
    for s, t in directions:
        n = len(_read_lines(os.path.join(CHALLENGE_ROOT, CHALLENGE_FILES[(s, t)][0])))
        logger.info("   %s-%s  → %d lines", s, t, n)

    all_results = {}
    out_base = os.path.join(OUTPUT_DIR, f"eval_finetuned_{EVAL_TAG}")

    if args.model in ("all", "nllb"):
        logger.info("=" * 64 + "  NLLB")
        all_results["nllb"] = eval_nllb(directions, os.path.join(out_base, "nllb"))

    if args.model in ("all", "madlad"):
        logger.info("=" * 64 + "  MADLAD")
        all_results["madlad"] = eval_madlad(directions, os.path.join(out_base, "madlad"))

    if args.model in ("all", "gemmax2"):
        logger.info("=" * 64 + "  GemmaX2")
        all_results["gemmax2"] = eval_gemmax2(directions, os.path.join(out_base, "gemmax2"))

    os.makedirs(out_base, exist_ok=True)
    result_path = os.path.join(out_base, f"all_results_{EVAL_TAG}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("All results saved → %s", result_path)


if __name__ == "__main__":
    main()