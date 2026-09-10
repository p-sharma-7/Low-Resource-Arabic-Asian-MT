"""
infer_gemmax2.py — Zero-shot inference with ModelSpace/GemmaX2-28-9B-v0.1.

GemmaX2 is a Gemma-2-based decoder-only LLM trained for multilingual MT.
We use instruction-formatted prompts and decode only the newly generated
tokens (everything after "Translation:").

Standalone usage:
    python infer_gemmax2.py [--split devtest] [--batch_size 4]
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utlis.gpu_utils import get_device_map, require_cuda
from utlis.config import (
    MODELS,
    LANG_NAMES,
    TRANSLATION_DIRECTIONS,
    MAX_INPUT_LENGTH,
    OUTPUT_DIR,
    SEED,
)
from utlis.data_loader import get_src_tgt_lists
from pipelines.evaluate import evaluate_all, log_metrics, save_hypotheses

logger = logging.getLogger(__name__)

MODEL_NAME = MODELS["gemmax2"]
MODEL_KEY  = "gemmax2"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(source: str, src_lang: str, tgt_lang: str) -> str:
    """
    Build the instruction prompt for GemmaX2.

    Example output:
        Translate the following text from Arabic to Hindi:
        مرحبا بالعالم
        Translation:
    """
    src_name = LANG_NAMES[src_lang]
    tgt_name = LANG_NAMES[tgt_lang]
    return (
        f"Translate the following text from {src_name} to {tgt_name}:\n"
        f"{source}\n"
        f"Translation:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gemmax2(
    model_path: Optional[str] = None,
    quantize_4bit: bool = False,
    lora_adapter_path: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load GemmaX2 for inference.

    Parameters
    ----------
    model_path         : HF Hub id or local base model path.
    quantize_4bit      : Load in 4-bit (NF4 QLoRA) to save VRAM during eval.
    lora_adapter_path  : If given, merge a LoRA adapter on top of the base.
    """
    path = model_path or MODEL_NAME

    bnb_config = None
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info("Loading GemmaX2 from: %s  (4-bit=%s)", path, quantize_4bit)
    tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter_path if lora_adapter_path else path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for decoder-only batch inference

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map=get_device_map(),
        quantization_config=bnb_config,
    )

    if lora_adapter_path:
        from peft import PeftModel
        logger.info("Attaching LoRA adapter from: %s", lora_adapter_path)
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    model.eval()
    logger.info("GemmaX2 ready  |  device_map=auto  |  dtype=bfloat16")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_gemmax2(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> List[str]:
    """
    Translate using GemmaX2 via instruction prompting.

    Only the tokens generated *after* the prompt are returned;
    multi-line outputs are trimmed to the first line only to avoid
    including repeated prompts or hallucinated follow-up text.
    """
    prompts = [format_prompt(t, src_lang, tgt_lang) for t in texts]

    translations: List[str] = []
    total = len(prompts)

    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH[MODEL_KEY],
        ).to(model.device)

        prompt_len = inputs["input_ids"].shape[1]

        # Cap at 2.5x prompt length to kill proper-noun variant loops
        actual_max_new = min(max_new_tokens, int(prompt_len * 2.5))

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=actual_max_new,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                length_penalty=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens (exclude prompt)
        import unicodedata
        new_tokens = output_ids[:, prompt_len:]
        decoded    = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # Keep only the first non-empty line (trim artefacts)
        cleaned = []
        for text in decoded:
            first_line = text.strip().split("\n")[0].strip()
            cleaned.append(unicodedata.normalize("NFC", first_line))
        translations.extend(cleaned)

        done = min(start + batch_size, total)
        if (start // batch_size) % 20 == 0:
            logger.info("  Translated %d / %d", done, total)

    return translations


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot runner
# ─────────────────────────────────────────────────────────────────────────────

def run_zero_shot_gemmax2(
    split: str = "devtest",
    batch_size: int = 4,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    lora_adapter_path: Optional[str] = None,
    quantize_4bit: bool = False,
) -> dict:
    """
    Run GemmaX2-28-9B zero-shot across all 6 translation directions.

    Returns a dict  {direction_str: metrics_dict}.
    """
    if output_dir is None:
        suffix = "zero_shot" if (model_path is None and lora_adapter_path is None) \
                 else "finetuned_eval"
        output_dir = os.path.join(OUTPUT_DIR, suffix, "gemmax2")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_gemmax2(
        model_path=model_path,
        quantize_4bit=quantize_4bit,
        lora_adapter_path=lora_adapter_path,
    )
    all_results: dict = {}

    for src_lang, tgt_lang in TRANSLATION_DIRECTIONS:
        direction = f"{src_lang}-{tgt_lang}"
        logger.info("\n%s\nGemmaX2  |  %s  |  split=%s\n%s",
                    "─" * 56, direction, split, "─" * 56)

        sources, references = get_src_tgt_lists(src_lang, tgt_lang, split)

        hypotheses = translate_gemmax2(
            sources, model, tokenizer, src_lang, tgt_lang,
            batch_size=batch_size,
        )

        tag = "gemmax2_lora" if lora_adapter_path else "gemmax2"
        save_hypotheses(hypotheses, src_lang, tgt_lang, tag, split, output_dir)

        metrics = evaluate_all(sources, hypotheses, references, src_lang, tgt_lang)
        log_metrics(metrics, src_lang, tgt_lang, tag, split, output_dir)
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
    parser = argparse.ArgumentParser(description="GemmaX2 zero-shot inference")
    parser.add_argument("--split",            default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--batch_size",       default=4, type=int)
    parser.add_argument("--model_path",       default=None)
    parser.add_argument("--lora_adapter",     default=None,
                        help="Path to a trained LoRA adapter directory")
    parser.add_argument("--quantize_4bit",    action="store_true",
                        help="Load in 4-bit (QLoRA) to save VRAM")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_zero_shot_gemmax2(
        split=args.split,
        batch_size=args.batch_size,
        model_path=args.model_path,
        lora_adapter_path=args.lora_adapter,
        quantize_4bit=args.quantize_4bit,
    )