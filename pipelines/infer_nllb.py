"""
infer_nllb.py — Zero-shot inference with facebook/nllb-200-3.3B.

Runs all 6 translation directions on the specified split (default: devtest),
saves hypotheses, and reports all five evaluation metrics.

Standalone usage:
    python infer_nllb.py [--split devtest] [--batch_size 16]
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utlis.gpu_utils import get_device_map, require_cuda
from utlis.config import (
    MODELS,
    NLLB_LANG_CODES,
    TRANSLATION_DIRECTIONS,
    MAX_INPUT_LENGTH,
    OUTPUT_DIR,
    SEED,
)
from utlis.data_loader import get_src_tgt_lists
from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

logger = logging.getLogger(__name__)

MODEL_NAME = MODELS["nllb"]

# ---------------------------------------------------------------------------
# Language-token ID helper
# ---------------------------------------------------------------------------

def _get_lang_token_id(tokenizer, lang_code):
    # type: (object, str) -> int
    """
    Retrieve the NLLB language token ID robustly.

    NllbTokenizer (slow)  exposes tokenizer.lang_code_to_id.
    NllbTokenizerFast     does NOT, but language codes are always in
                          tokenizer.additional_special_tokens.
    Falls back to convert_tokens_to_ids and raises a clear error on failure.
    """
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id[lang_code]
    if lang_code in tokenizer.additional_special_tokens:
        idx = tokenizer.additional_special_tokens.index(lang_code)
        return tokenizer.additional_special_tokens_ids[idx]
    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if token_id != tokenizer.unk_token_id:
        return token_id
    sample = tokenizer.additional_special_tokens[:10]
    raise ValueError(
        "Language code " + repr(lang_code) + " not found in tokenizer. "
        "First 10 special tokens: " + repr(sample)
    )


MODEL_KEY  = "nllb"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_nllb(
    model_path: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load NLLB tokeniser + model.

    Parameters
    ----------
    model_path : HF Hub identifier or local path to a fine-tuned checkpoint.
                 Defaults to the pre-trained NLLB-200-3.3B weights.
    dtype      : torch dtype (float16 by default to fit within 48 GB).
    """
    path = model_path or MODEL_NAME
    logger.info("Loading NLLB model from: %s", path)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        dtype=dtype,
        device_map=get_device_map(),
    )
    model.eval()
    logger.info(
        "NLLB loaded  |  device_map=auto  |  dtype=%s", dtype
    )
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_nllb(
    texts: List[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 16,
    max_length: int = 512,
    num_beams: int = 4,
) -> List[str]:
    """
    Translate a list of source sentences using NLLB-200.

    The NLLB tokeniser uses BCP-47-like codes (ara_Arab, eng_Latn, …).
    We set `src_lang` on the tokeniser and pass `forced_bos_token_id`
    so the decoder always starts with the correct target language token.
    """
    src_code = NLLB_LANG_CODES[src_lang]
    tgt_code = NLLB_LANG_CODES[tgt_lang]

    tokenizer.src_lang = src_code
    forced_bos_id = _get_lang_token_id(tokenizer, tgt_code)

    translations: List[str] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH[MODEL_KEY],
        ).to(model.device)

        src_len = inputs["input_ids"].shape[1]
        max_new = min(max_length, int(src_len * 2.5))

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=max_new,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                length_penalty=0.8,
            )

        import unicodedata
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        decoded = [unicodedata.normalize("NFC", t) for t in decoded]
        translations.extend(decoded)

        done = min(start + batch_size, total)
        if (start // batch_size) % 20 == 0:
            logger.info("  Translated %d / %d", done, total)

    return translations


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot runner
# ─────────────────────────────────────────────────────────────────────────────

def run_zero_shot_nllb(
    split: str = "devtest",
    batch_size: int = 16,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """
    Run NLLB-200-3.3B zero-shot across all 6 translation directions.

    Returns a dict  {direction_str: metrics_dict}.
    """
    if output_dir is None:
        suffix = "zero_shot" if model_path is None else "finetuned_eval"
        output_dir = os.path.join(OUTPUT_DIR, suffix, "nllb")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_nllb(model_path)
    all_results: dict = {}

    for src_lang, tgt_lang in TRANSLATION_DIRECTIONS:
        direction = f"{src_lang}-{tgt_lang}"
        logger.info("\n%s\nNLLB  |  %s  |  split=%s\n%s",
                    "─" * 56, direction, split, "─" * 56)

        sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)

        hypotheses = translate_nllb(
            sources, model, tokenizer, src_lang, tgt_lang,
            batch_size=batch_size,
        )

        save_hypotheses(hypotheses, src_lang, tgt_lang, "nllb", split, output_dir)

        metrics = evaluate_all(sources, hypotheses, references, src_lang, tgt_lang)
        log_metrics(metrics, src_lang, tgt_lang, "nllb", split, output_dir)
        all_results[direction] = metrics

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="NLLB-200 zero-shot inference")
    parser.add_argument("--split",      default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model_path", default=None,
                        help="Path to fine-tuned checkpoint (optional)")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_zero_shot_nllb(
        split=args.split,
        batch_size=args.batch_size,
        model_path=args.model_path,
    )