"""
infer_madlad.py — Zero-shot inference with google/madlad400-10b-mt.

MADLAD-400 is a T5-based seq2seq model.  Translation is triggered by
prepending a language tag ("<2hi>", "<2ar>", …) to the source sentence.
No src_lang needs to be set on the tokeniser.

Standalone usage:
    python infer_madlad.py [--split devtest] [--batch_size 8]
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
    MADLAD_LANG_CODES,
    TRANSLATION_DIRECTIONS,
    MAX_INPUT_LENGTH,
    OUTPUT_DIR,
    SEED,
)
from utlis.data_loader import get_src_tgt_lists
from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

logger = logging.getLogger(__name__)

MODEL_NAME = MODELS["madlad"]
MODEL_KEY  = "madlad"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_madlad(
    model_path: Optional[str] = None,
    lora_adapter_path: Optional[str] = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load MADLAD for inference.
    If lora_adapter_path is given, loads the base model in 8-bit and
    attaches the LoRA adapter (matches how train_madlad saves checkpoints).
    Otherwise loads the base model in bf16 for zero-shot inference.
    """
    from transformers import BitsAndBytesConfig
    path = model_path or MODEL_NAME

    if lora_adapter_path:
        logger.info("Loading MADLAD (8-bit) + LoRA from: %s", lora_adapter_path)
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    else:
        logger.info("Loading MADLAD (bf16) from: %s", path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=get_device_map(),
        )

    model.eval()
    logger.info("MADLAD ready")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_madlad(
    texts: List[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str,       # kept for API symmetry; MADLAD only needs tgt_lang
    tgt_lang: str,
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4,
) -> List[str]:
    """
    Translate using MADLAD-400.

    MADLAD's translation protocol: prepend the target-language tag to each
    source sentence, e.g.  "<2hi> مرحبا بالعالم".
    No forced_bos_token_id is required — the model infers direction from the
    prefix.
    """
    tgt_code = MADLAD_LANG_CODES[tgt_lang]
    prefixed = [f"{tgt_code} {t}" for t in texts]

    translations: List[str] = []
    total = len(prefixed)

    for start in range(0, total, batch_size):
        batch = prefixed[start : start + batch_size]

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

def run_zero_shot_madlad(
    split: str = "devtest",
    batch_size: int = 8,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """
    Run MADLAD-400-10B zero-shot across all 6 translation directions.

    Returns a dict  {direction_str: metrics_dict}.
    """
    if output_dir is None:
        suffix = "zero_shot" if model_path is None else "finetuned_eval"
        output_dir = os.path.join(OUTPUT_DIR, suffix, "madlad")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_madlad(model_path)
    all_results: dict = {}

    for src_lang, tgt_lang in TRANSLATION_DIRECTIONS:
        direction = f"{src_lang}-{tgt_lang}"
        logger.info("\n%s\nMADLAD  |  %s  |  split=%s\n%s",
                    "─" * 56, direction, split, "─" * 56)

        sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)

        hypotheses = translate_madlad(
            sources, model, tokenizer, src_lang, tgt_lang,
            batch_size=batch_size,
        )

        save_hypotheses(hypotheses, src_lang, tgt_lang, "madlad", split, output_dir)

        metrics = evaluate_all(sources, hypotheses, references, src_lang, tgt_lang)
        log_metrics(metrics, src_lang, tgt_lang, "madlad", split, output_dir)
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
    parser = argparse.ArgumentParser(description="MADLAD-400 zero-shot inference")
    parser.add_argument("--split",      default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--model_path", default=None,
                        help="Path to fine-tuned checkpoint (optional)")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_zero_shot_madlad(
        split=args.split,
        batch_size=args.batch_size,
        model_path=args.model_path,
    )