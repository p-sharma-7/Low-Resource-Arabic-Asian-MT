"""
data_loader.py — Parallel corpus loading for WMT26 Arabic-Asian MT Challenge.

Each dataset folder uses the naming convention:
    {split}_{lang}_{pair}.txt

Line N in the source file is always the translation of line N in the target
file.  This module resolves correct file paths for both forward and reverse
directions and returns HuggingFace Dataset objects.
"""

import logging
import os
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict

from utlis.config import DATASET_DIR, PAIR_FOLDER_MAP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File-path resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_file_paths(src_lang: str, tgt_lang: str, split: str) -> Tuple[str, str]:
    """
    Return (src_file_path, tgt_file_path) for a translation direction + split.

    The folder always has Arabic as the "first" language in the name, so:
      - Forward  (ar → X): Arabic file is source, X file is target.
      - Reverse  (X → ar): X file is source, Arabic file is target.
    """
    folder = PAIR_FOLDER_MAP[(src_lang, tgt_lang)]          # e.g. "Ar-En"
    lang1, lang2 = folder.lower().split("-")                # e.g. "ar", "en"
    pair_suffix  = f"{lang1}-{lang2}"                       # e.g. "ar-en"

    file_lang1 = os.path.join(
        DATASET_DIR, folder, f"{split}_{lang1}_{pair_suffix}.txt"
    )
    file_lang2 = os.path.join(
        DATASET_DIR, folder, f"{split}_{lang2}_{pair_suffix}.txt"
    )

    # lang1 is always Arabic; decide which is source and which is target
    if src_lang == lang1:          # forward direction  (ar → X)
        return file_lang1, file_lang2
    else:                          # reverse direction  (X → ar)
        return file_lang2, file_lang1


# ─────────────────────────────────────────────────────────────────────────────
# Core loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_parallel_data(
    src_lang: str,
    tgt_lang: str,
    split: str,
) -> List[Dict[str, str]]:
    """
    Load a parallel corpus as a list of {'src': ..., 'tgt': ...} dicts.

    Skips blank lines (both sides must be non-empty after stripping).
    Raises FileNotFoundError if the expected files are absent.
    Raises AssertionError if the two files have different line counts.
    """
    src_path, tgt_path = get_file_paths(src_lang, tgt_lang, split)
    direction = f"{src_lang}→{tgt_lang}"

    logger.info(
        "Loading %-10s %-8s  src=%s", direction, split,
        os.path.basename(src_path)
    )

    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Check that the dataset is placed under: {DATASET_DIR}"
            )

    with open(src_path, encoding="utf-8") as f:
        src_lines = [line.rstrip("\n") for line in f]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [line.rstrip("\n") for line in f]

    assert len(src_lines) == len(tgt_lines), (
        f"Line-count mismatch for {direction} [{split}]: "
        f"{src_path} has {len(src_lines)} lines, "
        f"{tgt_path} has {len(tgt_lines)} lines."
    )

    data = [
        {"src": s.strip(), "tgt": t.strip()}
        for s, t in zip(src_lines, tgt_lines)
        if s.strip() and t.strip()
    ]

    n_dropped = len(src_lines) - len(data)
    logger.info(
        "  → %d sentence pairs loaded  (dropped %d empty lines)",
        len(data), n_dropped,
    )
    return data


def load_dataset_for_direction(
    src_lang: str,
    tgt_lang: str,
    splits: Tuple[str, ...] = ("train", "dev", "devtest"),
) -> DatasetDict:
    """
    Load train / dev / devtest splits for one translation direction.

    Returns a HuggingFace DatasetDict.  Splits whose files are missing on disk
    are silently skipped with a warning.
    """
    dataset_splits: Dict[str, Dataset] = {}

    for split in splits:
        try:
            rows = load_parallel_data(src_lang, tgt_lang, split)
            dataset_splits[split] = Dataset.from_list(rows)
        except FileNotFoundError as exc:
            logger.warning("Skipping split '%s' for %s→%s: %s", split, src_lang, tgt_lang, exc)

    return DatasetDict(dataset_splits)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: raw text lists
# ─────────────────────────────────────────────────────────────────────────────

def get_src_tgt_lists(
    src_lang: str,
    tgt_lang: str,
    split: str,
) -> Tuple[List[str], List[str]]:
    """Return (sources, references) as plain Python lists of strings."""
    data = load_parallel_data(src_lang, tgt_lang, split)
    return [d["src"] for d in data], [d["tgt"] for d in data]
