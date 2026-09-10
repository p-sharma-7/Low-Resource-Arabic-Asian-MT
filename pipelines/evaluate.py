"""
evaluate.py — Evaluation pipeline for WMT26 Arabic-Asian MT Challenge.

Metrics (in priority order per spec):
  1. COMET-22      (Unbabel/wmt22-comet-da)          — primary / model selection
  2. ChrF2++       (sacrebleu, word_order=2)          — secondary
  3. BLEU          (SacreBLEU, tokeniser per language)
  4. TER           (SacreBLEU)
  5. COMET-Kiwi    (Unbabel/wmt23-cometkiwi-da)       — reference-free QE

COMET models are loaded lazily and cached globally to avoid redundant I/O.
"""

import logging
import os
from typing import Dict, List, Optional

import sacrebleu

from utlis.config import (
    COMET_MODEL,
    COMET_KIWI_MODEL,
    SACREBLEU_TOKENIZER,
    LANG_NAMES,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Robust unbabel-comet importer
#
# Problem: both `comet-ml` (experiment tracking) and `unbabel-comet`
# (MT evaluation) install under the `comet` namespace.  Whichever was
# pip-installed last wins, so a plain `from comet import download_model`
# fails when comet-ml is on top.
#
# Strategy:
#   1. Try the standard unbabel-comet 2.x / 3.x import path.
#   2. Try the unbabel-comet 1.x path (comet.models).
#   3. Walk sys.path looking for the unbabel-comet dist directly so we
#      can import it even when comet-ml has hijacked the `comet` name.
#   4. Raise a helpful message telling the user exactly how to fix it.
# ─────────────────────────────────────────────────────────────────────────────

def _load_unbabel_comet():
    """
    Return (download_model, load_from_checkpoint) from unbabel-comet,
    regardless of whether comet-ml is also installed.
    """
    import importlib
    import importlib.metadata
    import importlib.util
    import sys

    # ── Strategy 1: standard path (works when unbabel-comet is on top) ──────
    try:
        from comet import download_model, load_from_checkpoint  # noqa: F401
        # Verify these symbols actually belong to unbabel-comet by checking
        # that load_from_checkpoint accepts a path string (comet-ml does not
        # export this function at all, so its presence is the proof).
        import inspect
        sig = inspect.signature(load_from_checkpoint)
        return download_model, load_from_checkpoint
    except (ImportError, Exception):
        pass

    # ── Strategy 2: unbabel-comet 1.x path ───────────────────────────────────
    try:
        from comet.models import download_model, load_from_checkpoint  # noqa: F401
        return download_model, load_from_checkpoint
    except (ImportError, Exception):
        pass

    # ── Strategy 3: locate unbabel-comet on disk and import it directly ──────
    # This works even if comet-ml has registered itself as `comet` in
    # site-packages, because we bypass the normal import machinery.
    try:
        dist = importlib.metadata.distribution("unbabel-comet")
        # The distribution's top-level package is still named `comet`; find
        # its actual filesystem location from the recorded files.
        comet_init = None
        for f in dist.files or []:
            if str(f).endswith("comet/__init__.py"):
                comet_init = str(dist.locate_file(f))
                break
        if comet_init:
            spec = importlib.util.spec_from_file_location(
                "_unbabel_comet", comet_init
            )
            mod = importlib.util.module_from_spec(spec)
            # Register under a private name so we don't clobber `comet`
            sys.modules["_unbabel_comet"] = mod
            spec.loader.exec_module(mod)
            return mod.download_model, mod.load_from_checkpoint
    except Exception:
        pass

    # ── Strategy 4: clear error ──────────────────────────────────────────────
    raise ImportError(
        "\n\n"
        "  Could not import `download_model` / `load_from_checkpoint` from\n"
        "  unbabel-comet.  This usually means comet-ml is shadowing it.\n\n"
        "  Fix (run in your conda env):\n"
        "    pip install unbabel-comet --force-reinstall\n\n"
        "  If comet-ml must stay installed too:\n"
        "    pip uninstall comet-ml -y && pip install unbabel-comet\n"
        "    (or use separate conda environments for each)\n"
    )


_download_model, _load_from_checkpoint = _load_unbabel_comet()

# ─────────────────────────────────────────────────────────────────────────────
# Lazy COMET model cache
# ─────────────────────────────────────────────────────────────────────────────
_COMET22_MODEL    = None
_COMET_KIWI_MODEL = None


def _download_comet_model(model_id: str) -> str:
    """
    Download a COMET checkpoint robustly.

    Strategy 1: comet's own registry (fast; works for wmt22-comet-da).
    Strategy 2: huggingface_hub.snapshot_download() as a fallback for any
    model not yet listed in the installed comet version's legacy registry
    (e.g. wmt23-cometkiwi-da on older comet installs).
    """
    try:
        path = _download_model(model_id)
        logger.info("COMET registry download OK: %s", model_id)
        return path
    except Exception as exc:
        logger.warning(
            "comet download_model failed for '%s' (%s). "
            "Falling back to huggingface_hub.snapshot_download ...",
            model_id, exc,
        )
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=model_id)
        logger.info("HF Hub snapshot download OK: %s -> %s", model_id, local_dir)
        return local_dir
    except Exception as exc2:
        raise RuntimeError(
            f"Could not download COMET model '{model_id}'."
            f"  Fix: pip install unbabel-comet --upgrade"
        ) from exc2


def _get_comet22():
    global _COMET22_MODEL
    if _COMET22_MODEL is None:
        logger.info("Downloading / loading COMET-22: %s", COMET_MODEL)
        path = _download_comet_model(COMET_MODEL)
        _COMET22_MODEL = _load_from_checkpoint(path)
    return _COMET22_MODEL


def _get_comet_kiwi():
    global _COMET_KIWI_MODEL
    if _COMET_KIWI_MODEL is None:
        logger.info("Downloading / loading COMET-Kiwi: %s", COMET_KIWI_MODEL)
        path = _download_comet_model(COMET_KIWI_MODEL)
        _COMET_KIWI_MODEL = _load_from_checkpoint(path)
    return _COMET_KIWI_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    tgt_lang: str,
) -> float:
    tokenizer = SACREBLEU_TOKENIZER.get(tgt_lang, "13a")
    result = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenizer)
    return round(result.score, 4)


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """ChrF2++ (word_order=2)."""
    result = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return round(result.score, 4)


def compute_ter(
    hypotheses: List[str],
    references: List[str],
) -> float:
    result = sacrebleu.corpus_ter(hypotheses, [references])
    return round(result.score, 4)


def compute_comet22(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    batch_size: int = 16,
    gpus: int = 1,
) -> float:
    """COMET-22 (reference-based, DA)."""
    model = _get_comet22()
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return round(float(output.system_score), 4)


def compute_comet_kiwi(
    sources: List[str],
    hypotheses: List[str],
    batch_size: int = 16,
    gpus: int = 1,
) -> float:
    """COMET-Kiwi (reference-free quality estimation)."""
    model = _get_comet_kiwi()
    data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return round(float(output.system_score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    src_lang: str,
    tgt_lang: str,
    compute_kiwi: bool = False,   # disabled by default; enable only when needed
) -> Dict[str, float]:
    """
    Compute all five metrics and return as an ordered dict.

    Parameters
    ----------
    sources     : source-language sentences (for COMET)
    hypotheses  : model translations
    references  : gold references
    src_lang    : ISO-639-1 code of source language
    tgt_lang    : ISO-639-1 code of target language
    compute_kiwi: whether to also compute COMET-Kiwi (slower)
    """
    metrics: Dict[str, float] = {}

    logger.info("  ► COMET-22 ...")
    metrics["COMET-22"]  = compute_comet22(sources, hypotheses, references)

    logger.info("  ► ChrF2++ ...")
    metrics["ChrF2++"]   = compute_chrf(hypotheses, references)

    logger.info("  ► BLEU ...")
    metrics["BLEU"]      = compute_bleu(hypotheses, references, tgt_lang)

    logger.info("  ► TER ...")
    metrics["TER"]       = compute_ter(hypotheses, references)

    if compute_kiwi:
        logger.info("  ► COMET-Kiwi ...")
        metrics["COMET-Kiwi"] = compute_comet_kiwi(sources, hypotheses)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Logging / persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_metrics(
    metrics: Dict[str, float],
    src_lang: str,
    tgt_lang: str,
    model_tag: str,
    split: str,
    output_dir: str,
) -> None:
    """Pretty-print metrics and append them to a text file."""
    direction = f"{LANG_NAMES[src_lang]} → {LANG_NAMES[tgt_lang]}"
    sep = "=" * 64
    lines = [
        "",
        sep,
        f"  Model  : {model_tag}",
        f"  Task   : {direction}",
        f"  Split  : {split}",
        sep,
    ]
    for name, value in metrics.items():
        lines.append(f"  {name:<15}: {value}")
    lines.append(sep)
    text = "\n".join(lines)

    print(text)

    os.makedirs(output_dir, exist_ok=True)
    fname = f"metrics_{model_tag}_{src_lang}-{tgt_lang}_{split}.txt"
    with open(os.path.join(output_dir, fname), "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def save_hypotheses(
    hypotheses: List[str],
    src_lang: str,
    tgt_lang: str,
    model_tag: str,
    split: str,
    output_dir: str,
) -> str:
    """Write translation hypotheses to disk; return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"hyp_{model_tag}_{src_lang}-{tgt_lang}_{split}.txt"
    path  = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(hypotheses))
    logger.info("Hypotheses saved → %s", path)
    return path