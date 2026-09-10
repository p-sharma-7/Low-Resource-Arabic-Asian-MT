#!/usr/bin/env python3
"""
eda_train.py — Phase 1 EDA · Multilingual MT Training Corpora
==============================================================
Analyses Ar-En · Ar-Hi · Ar-Ur parallel corpora.

Output tree
-----------
eda_train/
├── per_file/
│   ├── train_ar_ar-en/   stats.json  top100.txt  4× .png
│   ├── train_en_ar-en/   ...
│   ├── train_ar_ar-hi/   ...
│   ├── train_hi_ar-hi/   ...
│   ├── train_ar_ar-ur/   ...
│   └── train_ur_ar-ur/   ...
├── per_language/
│   ├── arabic/           stats.json  top100.txt  4× .png
│   ├── english/          ...
│   ├── hindi/            ...
│   └── urdu/             ...
├── alignment_check.json
└── summary.csv

Usage
-----
    python eda_train.py
    python eda_train.py --dataset-root /path/to/dataset --output-root /path/to/out
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                    # non-interactive; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset layout
#   FILE_MAP  :  relative path under DATASET_ROOT → (language_tag, pair_tag)
#   ALIGN_PAIRS: triples (src_rel, tgt_rel, pair_label) for alignment checks
# ──────────────────────────────────────────────────────────────────────────────
FILE_MAP: Dict[str, Tuple[str, str]] = {
    "Ar-En/train_ar_ar-en.txt": ("arabic",  "Ar-En"),
    "Ar-En/train_en_ar-en.txt": ("english", "Ar-En"),
    "Ar-Hi/train_ar_ar-hi.txt": ("arabic",  "Ar-Hi"),
    "Ar-Hi/train_hi_ar-hi.txt": ("hindi",   "Ar-Hi"),
    "Ar-Ur/train_ar_ar-ur.txt": ("arabic",  "Ar-Ur"),
    "Ar-Ur/train_ur_ar-ur.txt": ("urdu",    "Ar-Ur"),
}

ALIGN_PAIRS: List[Tuple[str, str, str]] = [
    ("Ar-En/train_ar_ar-en.txt", "Ar-En/train_en_ar-en.txt", "Ar-En"),
    ("Ar-Hi/train_ar_ar-hi.txt", "Ar-Hi/train_hi_ar-hi.txt", "Ar-Hi"),
    ("Ar-Ur/train_ar_ar-ur.txt", "Ar-Ur/train_ur_ar-ur.txt", "Ar-Ur"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_lines(path: Path) -> List[str]:
    """Read every line, strip trailing newline only (preserve leading spaces)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def normalize(text: str) -> str:
    """Lowercase · strip leading/trailing whitespace · collapse internal runs."""
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenisation — consistent with the spec."""
    return text.split()


# ──────────────────────────────────────────────────────────────────────────────
# Core statistics
# ──────────────────────────────────────────────────────────────────────────────

def sentence_dataset_stats(
    lines: List[str],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Compute dataset-level counts and sentence-length statistics.

    Returns
    -------
    stats   : flat dict (total / unique / duplicate / empty / length stats)
    lengths : int32 array of per-sentence token counts (whitespace split)
    """
    total  = len(lines)
    empty  = sum(1 for l in lines if l.strip() == "")
    counts = Counter(lines)
    unique = len(counts)
    dupes  = total - unique

    lengths = np.array([len(tokenize(l)) for l in lines], dtype=np.int32)

    stats: Dict[str, Any] = {
        "total_sentences":     total,
        "unique_sentences":    unique,
        "duplicate_sentences": dupes,
        "duplicate_pct":       round(100.0 * dupes / total, 4) if total else 0.0,
        "empty_sentences":     empty,
    }

    if total:
        stats.update({
            "mean_len":   round(float(np.mean(lengths)),           4),
            "median_len": round(float(np.median(lengths)),         4),
            "min_len":    int(np.min(lengths)),
            "max_len":    int(np.max(lengths)),
            "std_len":    round(float(np.std(lengths)),            4),
            "p95_len":    round(float(np.percentile(lengths, 95)), 4),
            "p99_len":    round(float(np.percentile(lengths, 99)), 4),
        })
    else:
        stats.update(
            {k: 0 for k in ("mean_len", "median_len", "min_len", "max_len",
                             "std_len",  "p95_len",    "p99_len")}
        )

    return stats, lengths


def vocabulary_stats(tokens: List[str]) -> Tuple[Dict[str, Any], Counter]:
    """Total tokens, unique tokens, and TTR from a flat token list."""
    counter    = Counter(tokens)
    total_tok  = len(tokens)
    unique_tok = len(counter)
    return {
        "total_tokens":  total_tok,
        "unique_tokens": unique_tok,
        "ttr":           round(unique_tok / total_tok, 6) if total_tok else 0.0,
    }, counter


def frequency_bands(counter: Counter) -> Dict[str, int]:
    """
    Classify vocabulary by occurrence count.
    Reveals long-tail distribution structure.
    """
    vals = counter.values()
    return {
        "freq_exactly_1":  sum(1 for v in vals if v == 1),
        "freq_exactly_2":  sum(1 for v in vals if v == 2),
        "freq_3_to_5":     sum(1 for v in vals if 3 <= v <= 5),
        "freq_over_100":   sum(1 for v in vals if v > 100),
    }


def token_char_length_stats(
    counter: Counter,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Character-length statistics weighted by token frequency.
    (Weighted = reflects what the model actually sees in the corpus.)

    Returns
    -------
    stats : mean / median / min / max character lengths
    arr   : full weighted array (for histogram)
    """
    parts = [
        np.full(cnt, len(tok), dtype=np.int32)
        for tok, cnt in counter.items()
    ]
    arr = np.concatenate(parts) if parts else np.array([], dtype=np.int32)

    if arr.size:
        stats: Dict[str, Any] = {
            "mean_token_char_len":   round(float(np.mean(arr)),   4),
            "median_token_char_len": float(np.median(arr)),
            "min_token_char_len":    int(np.min(arr)),
            "max_token_char_len":    int(np.max(arr)),
        }
    else:
        stats = {k: 0 for k in (
            "mean_token_char_len", "median_token_char_len",
            "min_token_char_len",  "max_token_char_len",
        )}

    return stats, arr


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────

_BLUE   = "#4C72B0"
_ORANGE = "#DD8452"
_GREEN  = "#55A868"
_RED    = "#C44E52"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _comma_fmt(x: float, _pos: Any) -> str:
    return f"{int(x):,}"


# ── 1. Sentence-length histogram ──────────────────────────────────────────────

def plot_sent_len_hist(
    lengths: np.ndarray, title: str, path: Path
) -> None:
    """Histogram of sentence lengths, clipped to P99 for readability."""
    if lengths.size == 0:
        return
    clip = int(np.percentile(lengths, 99))
    data = lengths[lengths <= clip]
    bins = max(10, min(80, clip))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data, bins=bins, color=_BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sentence length (whitespace tokens)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Sentence Length Distribution — {title}",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_fmt))
    ax.annotate(
        f"clipped at P99 = {clip} tokens  ({len(lengths) - len(data):,} beyond)",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, color="grey",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── 2. Token-frequency histogram ─────────────────────────────────────────────

def plot_token_freq_hist(
    counter: Counter, title: str, path: Path
) -> None:
    """
    Log-log histogram of per-type frequency counts.
    x = how often a word type appears, y = how many word types have that frequency.
    """
    if not counter:
        return
    freqs  = np.array(list(counter.values()), dtype=np.float64)
    max_f  = freqs.max()
    bins   = np.logspace(0, np.log10(max_f + 1), num=60)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(freqs, bins=bins, color=_ORANGE, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token frequency  (log scale)", fontsize=11)
    ax.set_ylabel("Number of word types  (log scale)", fontsize=11)
    ax.set_title(f"Token Frequency Distribution — {title}",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_fmt))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── 3. Zipf curve ────────────────────────────────────────────────────────────

def plot_zipf(counter: Counter, title: str, path: Path) -> None:
    """
    Rank vs frequency on log-log axes.
    Includes ideal Zipf reference line (f ∝ 1/rank).
    """
    if not counter:
        return
    sorted_f = np.array(sorted(counter.values(), reverse=True), dtype=np.float64)
    ranks    = np.arange(1, len(sorted_f) + 1, dtype=np.float64)
    ideal    = sorted_f[0] / ranks           # f(1) / rank

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(ranks, sorted_f, color=_BLUE, lw=1.5, label="Observed")
    ax.loglog(ranks, ideal,    color=_RED,  lw=1.2, ls="--", alpha=0.75,
              label="Ideal Zipf  (C / rank)")
    ax.set_xlabel("Rank  (log scale)", fontsize=11)
    ax.set_ylabel("Frequency  (log scale)", fontsize=11)
    ax.set_title(f"Zipf Curve — {title}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── 4. Token character-length histogram ──────────────────────────────────────

def plot_token_char_len_hist(
    char_len_arr: np.ndarray, title: str, path: Path
) -> None:
    """
    Distribution of token character lengths (weighted by frequency).
    Especially informative for Arabic / Hindi / Urdu morphology.
    """
    if char_len_arr.size == 0:
        return
    clip = int(np.percentile(char_len_arr, 99))
    data = char_len_arr[char_len_arr <= clip]
    bins = max(5, min(60, clip))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data, bins=bins, color=_GREEN, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Token character length", fontsize=11)
    ax.set_ylabel("Frequency (weighted occurrences)", fontsize=11)
    ax.set_title(f"Token Character Length Distribution — {title}",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_fmt))
    ax.annotate(
        f"clipped at P99 = {clip} chars",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, color="grey",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file analysis
# ──────────────────────────────────────────────────────────────────────────────

def write_top100(counter: Counter, label: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Top 100 words — {label}\n")
        fh.write("─" * 60 + "\n")
        fh.write(f"{'Rank':>5}  {'Token':<45}  {'Count':>10}\n")
        fh.write("─" * 60 + "\n")
        for rank, (tok, cnt) in enumerate(counter.most_common(100), 1):
            fh.write(f"{rank:>5}.  {tok:<45}  {cnt:>10,}\n")


def analyze_file(filepath: Path, out_dir: Path) -> Dict[str, Any]:
    """Full EDA for one training file. Writes all artefacts to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem
    log.info("► file      %s", filepath)

    # ── load once ─────────────────────────────────────────────────────────────
    lines = load_lines(filepath)

    # ── sentence-level ────────────────────────────────────────────────────────
    s_stats, lengths = sentence_dataset_stats(lines)

    # ── raw vocabulary ────────────────────────────────────────────────────────
    raw_tokens         = [tok for line in lines for tok in tokenize(line)]
    rv_stats, r_ctr    = vocabulary_stats(raw_tokens)
    r_bands            = frequency_bands(r_ctr)
    r_charlen, r_arr   = token_char_length_stats(r_ctr)

    # ── normalised vocabulary (lowercase · strip · collapse spaces) ───────────
    norm_lines         = [normalize(l) for l in lines]
    norm_tokens        = [tok for l in norm_lines for tok in tokenize(l)]
    nv_stats, n_ctr    = vocabulary_stats(norm_tokens)
    n_bands            = frequency_bands(n_ctr)
    n_charlen, _       = token_char_length_stats(n_ctr)

    result: Dict[str, Any] = {
        "file": str(filepath),
        **s_stats,
        "raw_vocab": {
            **rv_stats,
            **r_bands,
            **r_charlen,
        },
        "norm_vocab": {
            **nv_stats,
            **n_bands,
            **n_charlen,
        },
    }

    # ── persist ───────────────────────────────────────────────────────────────
    (out_dir / "stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_top100(r_ctr, stem, out_dir / "top100_words.txt")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_sent_len_hist      (lengths, stem, out_dir / "sentence_length_hist.png")
    plot_token_freq_hist    (r_ctr,   stem, out_dir / "token_freq_hist.png")
    plot_zipf               (r_ctr,   stem, out_dir / "zipf_curve.png")
    plot_token_char_len_hist(r_arr,   stem, out_dir / "token_char_len_hist.png")

    log.info(
        "  ✓  sentences=%s  raw_vocab=%s  norm_vocab=%s  dupes=%.2f%%",
        f"{s_stats['total_sentences']:,}",
        f"{rv_stats['unique_tokens']:,}",
        f"{nv_stats['unique_tokens']:,}",
        s_stats["duplicate_pct"],
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Language aggregation
# ──────────────────────────────────────────────────────────────────────────────

def analyze_language(
    lang: str, paths: List[Path], out_dir: Path
) -> Dict[str, Any]:
    """
    Merge all files for one language into a single corpus and compute
    combined statistics.  The merged corpus is *not* deduplicated before
    analysis so counts reflect the full training signal seen by a model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("► language  %s  ← %d file(s)", lang, len(paths))

    all_lines: List[str] = []
    for fp in paths:
        all_lines.extend(load_lines(fp))

    s_stats, lengths   = sentence_dataset_stats(all_lines)

    raw_tokens         = [tok for l in all_lines for tok in tokenize(l)]
    rv_stats, r_ctr    = vocabulary_stats(raw_tokens)
    r_bands            = frequency_bands(r_ctr)
    r_charlen, r_arr   = token_char_length_stats(r_ctr)

    norm_lines         = [normalize(l) for l in all_lines]
    norm_tokens        = [tok for l in norm_lines for tok in tokenize(l)]
    nv_stats, n_ctr    = vocabulary_stats(norm_tokens)
    n_bands            = frequency_bands(n_ctr)
    n_charlen, _       = token_char_length_stats(n_ctr)

    result: Dict[str, Any] = {
        "language":     lang,
        "source_files": [str(p) for p in paths],
        **s_stats,
        "raw_vocab": {
            **rv_stats,
            **r_bands,
            **r_charlen,
        },
        "norm_vocab": {
            **nv_stats,
            **n_bands,
            **n_charlen,
        },
    }

    (out_dir / "stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_top100(r_ctr, lang, out_dir / "top100_words.txt")

    plot_sent_len_hist      (lengths, lang, out_dir / "sentence_length_hist.png")
    plot_token_freq_hist    (r_ctr,   lang, out_dir / "token_freq_hist.png")
    plot_zipf               (r_ctr,   lang, out_dir / "zipf_curve.png")
    plot_token_char_len_hist(r_arr,   lang, out_dir / "token_char_len_hist.png")

    log.info(
        "  ✓  sentences=%s  raw_vocab=%s  norm_vocab=%s",
        f"{s_stats['total_sentences']:,}",
        f"{rv_stats['unique_tokens']:,}",
        f"{nv_stats['unique_tokens']:,}",
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Alignment check
# ──────────────────────────────────────────────────────────────────────────────

def alignment_check(
    pairs: List[Tuple[str, str, str]], dataset_root: Path
) -> List[Dict[str, Any]]:
    """
    Verify that each parallel file pair has the same number of lines.
    Prints a human-readable report and returns structured results.
    """
    bar = "═" * 62
    print(f"\n{bar}")
    print("  PARALLEL CORPUS ALIGNMENT CHECK")
    print(bar)

    results: List[Dict[str, Any]] = []
    all_pass = True

    for src_rel, tgt_rel, pair in pairs:
        src_path = dataset_root / src_rel
        tgt_path = dataset_root / tgt_rel

        if not src_path.exists():
            log.error("  Missing: %s", src_path)
            continue
        if not tgt_path.exists():
            log.error("  Missing: %s", tgt_path)
            continue

        with open(src_path, encoding="utf-8", errors="replace") as fh:
            src_n = sum(1 for _ in fh)
        with open(tgt_path, encoding="utf-8", errors="replace") as fh:
            tgt_n = sum(1 for _ in fh)

        ok    = src_n == tgt_n
        delta = abs(src_n - tgt_n)
        if not ok:
            all_pass = False

        status = "✓  PASS" if ok else f"✗  FAIL  (Δ = {delta:,} lines)"
        print(f"\n  {pair}")
        print(f"    {src_path.name:<45}  {src_n:>10,}")
        print(f"    {tgt_path.name:<45}  {tgt_n:>10,}")
        print(f"    Status : {status}")

        results.append({
            "pair":          pair,
            "src":           src_rel,
            "tgt":           tgt_rel,
            "src_sentences": src_n,
            "tgt_sentences": tgt_n,
            "aligned":       ok,
            "delta":         delta,
        })

    overall = "ALL PASS ✓" if all_pass else "FAILURES DETECTED ✗"
    print(f"\n  Overall: {overall}")
    print(f"{bar}\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ──────────────────────────────────────────────────────────────────────────────

def build_summary_csv(
    per_file: List[Dict[str, Any]], out_path: Path
) -> None:
    """Flatten all per-file stat dicts into a single wide-format CSV."""
    rows = []
    for r in per_file:
        rv = r["raw_vocab"]
        nv = r["norm_vocab"]
        rows.append({
            # identity
            "file":                    Path(r["file"]).name,
            # dataset
            "total_sentences":         r["total_sentences"],
            "unique_sentences":        r["unique_sentences"],
            "duplicate_sentences":     r["duplicate_sentences"],
            "duplicate_pct":           r["duplicate_pct"],
            "empty_sentences":         r["empty_sentences"],
            # sentence length
            "mean_len":                r["mean_len"],
            "median_len":              r["median_len"],
            "min_len":                 r["min_len"],
            "max_len":                 r["max_len"],
            "std_len":                 r["std_len"],
            "p95_len":                 r["p95_len"],
            "p99_len":                 r["p99_len"],
            # raw vocab
            "raw_total_tokens":        rv["total_tokens"],
            "raw_unique_tokens":       rv["unique_tokens"],
            "raw_ttr":                 rv["ttr"],
            "raw_freq_1":              rv["freq_exactly_1"],
            "raw_freq_2":              rv["freq_exactly_2"],
            "raw_freq_3_5":            rv["freq_3_to_5"],
            "raw_freq_gt100":          rv["freq_over_100"],
            "raw_mean_token_char_len": rv["mean_token_char_len"],
            # norm vocab
            "norm_total_tokens":       nv["total_tokens"],
            "norm_unique_tokens":      nv["unique_tokens"],
            "norm_ttr":                nv["ttr"],
            "norm_freq_1":             nv["freq_exactly_1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info("Summary CSV  →  %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 EDA — multilingual MT training corpora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", default="dataset", type=Path, metavar="DIR",
        help="Root directory that contains Ar-En/, Ar-Hi/, Ar-Ur/ sub-folders",
    )
    p.add_argument(
        "--output-root", default="eda_train", type=Path, metavar="DIR",
        help="Destination for all EDA outputs",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args         = parse_args()
    dataset_root = args.dataset_root
    output_root  = args.output_root

    if not dataset_root.exists():
        log.error("Dataset root not found: %s", dataset_root)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    log.info("Dataset root : %s", dataset_root.resolve())
    log.info("Output root  : %s", output_root.resolve())

    # ── 1. Alignment check ────────────────────────────────────────────────────
    align_results = alignment_check(ALIGN_PAIRS, dataset_root)
    (output_root / "alignment_check.json").write_text(
        json.dumps(align_results, indent=2), encoding="utf-8"
    )

    # ── 2. Per-file analysis ──────────────────────────────────────────────────
    per_file_results: List[Dict[str, Any]] = []
    lang_to_files:    Dict[str, List[Path]] = {}

    for rel_path, (lang, _pair) in FILE_MAP.items():
        fp = dataset_root / rel_path
        if not fp.exists():
            log.warning("Skipping missing file: %s", fp)
            continue

        out_dir = output_root / "per_file" / fp.stem
        result  = analyze_file(fp, out_dir)
        per_file_results.append(result)
        lang_to_files.setdefault(lang, []).append(fp)

    # ── 3. Language-level aggregation ────────────────────────────────────────
    for lang, files in sorted(lang_to_files.items()):
        out_dir = output_root / "per_language" / lang
        analyze_language(lang, files, out_dir)

    # ── 4. Summary CSV ────────────────────────────────────────────────────────
    if per_file_results:
        build_summary_csv(per_file_results, output_root / "summary.csv")

    log.info("Phase 1 EDA complete.  All outputs → %s", output_root.resolve())


if __name__ == "__main__":
    main()