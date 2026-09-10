#!/usr/bin/env python3
"""
cross_split_eda.py — Phase 2: Train ↔ Dev ↔ DevTest Split Analysis
===================================================================
Analyses every translation direction across TRAIN / DEV / DEVTEST splits.

Covers 11 analyses:
  1.  Dataset statistics          7.  Vocabulary frequency shift
  2.  Vocabulary coverage         8.  Semantic similarity (TF-IDF / SBERT)
  3.  OOV analysis                9.  Distribution divergence (JSD / KLD)
  4.  Exact duplicate detection  10.  Leakage detection report
  5.  N-gram overlap             11.  Cross-language summary
  6.  Sentence length shift

Plus 6 visualisations and 2 Markdown reports.

Usage
-----
    python cross_split_eda.py
    python cross_split_eda.py --dataset-root /data --output-root /out
    python cross_split_eda.py --skip-semantic          # skip embedding analysis
    python cross_split_eda.py --semantic-method tfidf   # or sbert
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from tqdm import tqdm

# Optional heavy dependency --------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

warnings.filterwarnings("ignore", category=FutureWarning)

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
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SEED                   = 42
MAX_EMBED_SAMPLES      = 50_000
EMBED_BATCH            = 512
TFIDF_MAX_FEATURES     = 50_000
SIM_BATCH              = 500
TOP_K_OOV              = 200
TOP_K_FREQ_SHIFT       = 100
NGRAM_ORDERS           = (1, 2, 3, 4)

SPLIT_COLOURS: Dict[str, str] = {
    "train":   "#4C72B0",
    "dev":     "#DD8452",
    "devtest": "#55A868",
}

_FNAME_RE = re.compile(
    r"^(train|dev|devtest)_([a-z]{2})_([a-z]{2}-[a-z]{2})\.txt$"
)

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitCorpus:
    """Loaded + pre-indexed data for a single file."""
    name: str                   # e.g. "train"
    path: Path
    lines: List[str]            # stripped sentences
    counter: Counter = field(repr=False, default_factory=Counter)
    lengths: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        self.counter = Counter(
            tok for line in self.lines for tok in line.split()
        )
        self.lengths = np.array(
            [len(line.split()) for line in self.lines], dtype=np.int32
        )

    @property
    def vocab(self) -> Set[str]:
        return set(self.counter)

    @property
    def total_tokens(self) -> int:
        return sum(self.counter.values())

    @property
    def total_sentences(self) -> int:
        return len(self.lines)

    @property
    def unique_sentences(self) -> int:
        return len(set(self.lines))


@dataclass
class AnalysisUnit:
    """One (pair, lang) with its three splits."""
    pair: str       # e.g. "ar-en"
    lang: str       # e.g. "ar"
    train: SplitCorpus
    dev: SplitCorpus
    devtest: SplitCorpus

    @property
    def direction(self) -> str:
        return f"{self.pair}/{self.lang}"


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def _load_lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def discover_units(root: Path) -> List[AnalysisUnit]:
    """
    Walk *root* for files matching {split}_{lang}_{pair}.txt,
    group by (pair, lang), and build AnalysisUnit objects for triples
    where all three splits exist.
    """
    registry: Dict[Tuple[str, str], Dict[str, Path]] = {}

    for txt in sorted(root.rglob("*.txt")):
        m = _FNAME_RE.match(txt.name)
        if not m:
            continue
        split, lang, pair = m.groups()
        registry.setdefault((pair, lang), {})[split] = txt

    units: List[AnalysisUnit] = []
    for (pair, lang), splits in sorted(registry.items()):
        missing = {"train", "dev", "devtest"} - set(splits)
        if missing:
            log.warning("Skipping %s/%s — missing splits: %s", pair, lang, missing)
            continue
        log.info("Discovered  %s / %s  (train=%s)", pair, lang, splits["train"].name)
        units.append(AnalysisUnit(
            pair=pair, lang=lang,
            train=SplitCorpus("train",   splits["train"],   _load_lines(splits["train"])),
            dev=SplitCorpus("dev",       splits["dev"],     _load_lines(splits["dev"])),
            devtest=SplitCorpus("devtest", splits["devtest"], _load_lines(splits["devtest"])),
        ))
    return units


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 4) if den else 0.0


def _ngram_set(lines: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Unique n-gram types from a list of sentences."""
    s: Set[Tuple[str, ...]] = set()
    for line in lines:
        toks = line.split()
        for i in range(len(toks) - n + 1):
            s.add(tuple(toks[i : i + n]))
    return s


def _ngram_token_overlap(
    train_lines: List[str], eval_lines: List[str], n: int
) -> float:
    """
    Token-level n-gram overlap:
    fraction of eval n-gram *tokens* whose type also exists in train.
    """
    train_types = _ngram_set(train_lines, n)
    covered = 0
    total   = 0
    for line in eval_lines:
        toks = line.split()
        for i in range(len(toks) - n + 1):
            total += 1
            if tuple(toks[i : i + n]) in train_types:
                covered += 1
    return _pct(covered, total)


def _aligned_distributions(
    ca: Counter, cb: Counter, eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """Create smoothed, normalised probability vectors over the union vocab."""
    keys = sorted(set(ca) | set(cb))
    ta = sum(ca.values())
    tb = sum(cb.values())
    p = np.array([ca.get(k, 0) / ta + eps for k in keys])
    q = np.array([cb.get(k, 0) / tb + eps for k in keys])
    p /= p.sum()
    q /= q.sum()
    return p, q


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Dataset statistics
# ──────────────────────────────────────────────────────────────────────────────

def analysis_1(units: List[AnalysisUnit]) -> pd.DataFrame:
    """Per-split sentence / token counts."""
    rows: List[Dict[str, Any]] = []
    for u in units:
        for sc in (u.train, u.dev, u.devtest):
            dupes = sc.total_sentences - sc.unique_sentences
            rows.append({
                "direction":           u.direction,
                "split":               sc.name,
                "total_sentences":     sc.total_sentences,
                "unique_sentences":    sc.unique_sentences,
                "duplicate_sentences": dupes,
                "duplicate_pct":       _pct(dupes, sc.total_sentences),
                "total_tokens":        sc.total_tokens,
                "unique_tokens":       len(sc.vocab),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Vocabulary coverage
# ──────────────────────────────────────────────────────────────────────────────

def analysis_2(units: List[AnalysisUnit]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for u in units:
        train_vocab = u.train.vocab
        for sc in (u.dev, u.devtest):
            # token-level coverage
            covered_tok = sum(
                cnt for tok, cnt in sc.counter.items() if tok in train_vocab
            )
            tok_cov = _pct(covered_tok, sc.total_tokens)

            # type-level coverage
            covered_types = len(sc.vocab & train_vocab)
            type_cov = _pct(covered_types, len(sc.vocab))

            rows.append({
                "direction":             u.direction,
                "eval_split":            sc.name,
                "token_coverage_pct":    tok_cov,
                "unique_token_cov_pct":  type_cov,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 3 — OOV analysis
# ──────────────────────────────────────────────────────────────────────────────

def analysis_3(
    units: List[AnalysisUnit],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows: List[Dict[str, Any]] = []
    oov_rows:   List[Dict[str, Any]] = []

    for u in units:
        train_vocab = u.train.vocab
        for sc in (u.dev, u.devtest):
            oov_counter: Counter = Counter()
            for tok, cnt in sc.counter.items():
                if tok not in train_vocab:
                    oov_counter[tok] = cnt

            oov_tok_count  = sum(oov_counter.values())
            oov_type_count = len(oov_counter)

            stats_rows.append({
                "direction":       u.direction,
                "eval_split":      sc.name,
                "oov_token_count": oov_tok_count,
                "oov_token_pct":   _pct(oov_tok_count, sc.total_tokens),
                "oov_unique_count": oov_type_count,
                "oov_unique_pct":  _pct(oov_type_count, len(sc.vocab)),
            })

            for tok, freq in oov_counter.most_common(TOP_K_OOV):
                oov_rows.append({
                    "direction": u.direction,
                    "split":     sc.name,
                    "language":  u.lang,
                    "token":     tok,
                    "frequency": freq,
                })

    return pd.DataFrame(stats_rows), pd.DataFrame(oov_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 4 — Exact duplicate sentences
# ──────────────────────────────────────────────────────────────────────────────

def analysis_4(
    units: List[AnalysisUnit],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary_df, duplicates_df)."""
    summary_rows: List[Dict[str, Any]] = []
    dup_rows:     List[Dict[str, Any]] = []

    for u in units:
        train_set = set(u.train.lines)
        for sc in (u.dev, u.devtest):
            eval_set    = set(sc.lines)
            overlap     = train_set & eval_set
            overlap_n   = len(overlap)
            overlap_pct = _pct(overlap_n, sc.unique_sentences)

            summary_rows.append({
                "direction":        u.direction,
                "comparison":       f"train↔{sc.name}",
                "identical_sents":  overlap_n,
                "duplicate_pct":    overlap_pct,
            })

            # Record each duplicate
            eval_counter = Counter(sc.lines)
            for sent in sorted(overlap):
                dup_rows.append({
                    "direction": u.direction,
                    "split":     sc.name,
                    "sentence":  sent,
                    "frequency": eval_counter[sent],
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(dup_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 5 — N-gram overlap
# ──────────────────────────────────────────────────────────────────────────────

def analysis_5(units: List[AnalysisUnit]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for u in tqdm(units, desc="N-gram overlap", leave=False):
        for sc in (u.dev, u.devtest):
            row: Dict[str, Any] = {
                "direction":  u.direction,
                "eval_split": sc.name,
            }
            for n in NGRAM_ORDERS:
                label = {1: "unigram", 2: "bigram", 3: "trigram", 4: "fourgram"}[n]
                row[label] = _ngram_token_overlap(u.train.lines, sc.lines, n)
            rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 6 — Sentence-length distribution shift
# ──────────────────────────────────────────────────────────────────────────────

def analysis_6(units: List[AnalysisUnit]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for u in units:
        for sc in (u.train, u.dev, u.devtest):
            if sc.lengths.size == 0:
                continue
            rows.append({
                "direction": u.direction,
                "split":     sc.name,
                "mean":      round(float(np.mean(sc.lengths)),           4),
                "median":    round(float(np.median(sc.lengths)),         4),
                "std":       round(float(np.std(sc.lengths)),            4),
                "p95":       round(float(np.percentile(sc.lengths, 95)), 4),
                "p99":       round(float(np.percentile(sc.lengths, 99)), 4),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 7 — Vocabulary frequency shift
# ──────────────────────────────────────────────────────────────────────────────

def analysis_7(units: List[AnalysisUnit]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for u in units:
        train_total = u.train.total_tokens
        dev_total   = u.dev.total_tokens
        dt_total    = u.devtest.total_tokens

        for tok, cnt in u.train.counter.most_common(TOP_K_FREQ_SHIFT):
            rows.append({
                "direction":       u.direction,
                "token":           tok,
                "train_freq":      cnt,
                "train_pct":       _pct(cnt, train_total),
                "dev_freq":        u.dev.counter.get(tok, 0),
                "dev_pct":         _pct(u.dev.counter.get(tok, 0), dev_total),
                "devtest_freq":    u.devtest.counter.get(tok, 0),
                "devtest_pct":     _pct(u.devtest.counter.get(tok, 0), dt_total),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 8 — Semantic similarity
# ──────────────────────────────────────────────────────────────────────────────

def _sample(lines: List[str], n: int) -> List[str]:
    rng = np.random.RandomState(SEED)
    if len(lines) <= n:
        return lines
    idx = rng.choice(len(lines), n, replace=False)
    return [lines[i] for i in sorted(idx)]


def _semantic_tfidf(
    train_sents: List[str],
    eval_sents: List[str],
) -> np.ndarray:
    """
    TF-IDF cosine similarity fallback.
    Returns array of max-similarity scores for each eval sentence.
    """
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
    )
    train_mat = vec.fit_transform(train_sents)
    eval_mat  = vec.transform(eval_sents)

    max_sims: List[float] = []
    for start in tqdm(
        range(0, eval_mat.shape[0], SIM_BATCH),
        desc="  TF-IDF similarity", leave=False,
    ):
        end   = min(start + SIM_BATCH, eval_mat.shape[0])
        batch = eval_mat[start:end]
        sims  = sklearn_cosine(batch, train_mat)     # (batch, train)
        max_sims.extend(sims.max(axis=1).tolist())
    return np.array(max_sims, dtype=np.float32)


def _semantic_sbert(
    train_sents: List[str],
    eval_sents: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """SBERT cosine similarity — preferred if available."""
    model = SentenceTransformer(model_name)
    log.info("  Encoding train (%d sentences) …", len(train_sents))
    train_emb = model.encode(
        train_sents, batch_size=EMBED_BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    log.info("  Encoding eval  (%d sentences) …", len(eval_sents))
    eval_emb = model.encode(
        eval_sents, batch_size=EMBED_BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    max_sims: List[float] = []
    for start in range(0, len(eval_emb), SIM_BATCH):
        end   = min(start + SIM_BATCH, len(eval_emb))
        batch = eval_emb[start:end]
        sims  = batch @ train_emb.T                  # cosine (pre-normalised)
        max_sims.extend(sims.max(axis=1).tolist())
    return np.array(max_sims, dtype=np.float32)


def analysis_8(
    units: List[AnalysisUnit],
    method: str = "tfidf",
    skip: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Returns (summary_df, {direction+split: sims_array}).
    The sims dict is used for the histogram visualisation.
    """
    rows:     List[Dict[str, Any]]     = []
    all_sims: Dict[str, np.ndarray]    = {}

    if skip:
        log.info("Semantic similarity skipped (--skip-semantic)")
        return pd.DataFrame(rows), all_sims

    use_sbert = method == "sbert" and HAS_SBERT
    if method == "sbert" and not HAS_SBERT:
        log.warning("sentence-transformers not installed — falling back to TF-IDF")
        use_sbert = False

    actual_method = "sbert" if use_sbert else "tfidf"
    log.info("Semantic similarity method: %s", actual_method)

    for u in units:
        train_samp = _sample(u.train.lines, MAX_EMBED_SAMPLES)
        for sc in (u.dev, u.devtest):
            log.info("  %s  train→%s", u.direction, sc.name)
            eval_samp = _sample(sc.lines, MAX_EMBED_SAMPLES)
            if not train_samp or not eval_samp:
                continue

            try:
                if use_sbert:
                    sims = _semantic_sbert(train_samp, eval_samp)
                else:
                    sims = _semantic_tfidf(train_samp, eval_samp)
            except Exception as exc:
                log.error("  Semantic similarity failed: %s — skipping", exc)
                continue

            key = f"{u.direction}|{sc.name}"
            all_sims[key] = sims
            rows.append({
                "direction":       u.direction,
                "eval_split":      sc.name,
                "method":          actual_method,
                "mean_similarity": round(float(np.mean(sims)),           4),
                "median_similarity": round(float(np.median(sims)),       4),
                "p95_similarity":  round(float(np.percentile(sims, 95)), 4),
            })

    return pd.DataFrame(rows), all_sims


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 9 — Distribution divergence
# ──────────────────────────────────────────────────────────────────────────────

def analysis_9(units: List[AnalysisUnit]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for u in units:
        for sc in (u.dev, u.devtest):
            p, q = _aligned_distributions(u.train.counter, sc.counter)
            jsd_val = float(jensenshannon(p, q) ** 2)   # distance² = divergence
            kld_val = float(np.sum(rel_entr(p, q)))
            rows.append({
                "direction":  u.direction,
                "eval_split": sc.name,
                "jsd":        round(jsd_val, 6),
                "kld":        round(kld_val, 6),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 10 — Leakage detection report
# ──────────────────────────────────────────────────────────────────────────────

def analysis_10(
    dup_summary: pd.DataFrame, ngram_df: pd.DataFrame
) -> str:
    """Generate a Markdown leakage report from Analyses 4 and 5."""
    lines = [
        "# Leakage Detection Report\n",
        f"*Auto-generated by `cross_split_eda.py`*\n",
        "---\n",
    ]

    if dup_summary.empty and ngram_df.empty:
        lines.append("No data available for leakage analysis.\n")
        return "\n".join(lines)

    # Merge trigram overlap into dup summary by direction
    merged = dup_summary.copy()
    tri_map: Dict[Tuple[str, str], float] = {}
    if not ngram_df.empty:
        for _, row in ngram_df.iterrows():
            tri_map[(row["direction"], row["eval_split"])] = row.get("trigram", 0.0)

    for idx, row in merged.iterrows():
        split_name = row["comparison"].split("↔")[1]
        merged.loc[idx, "trigram_overlap"] = tri_map.get(
            (row["direction"], split_name), 0.0
        )

    for direction in sorted(merged["direction"].unique()):
        sub = merged[merged["direction"] == direction]
        lines.append(f"## {direction}\n")

        for _, row in sub.iterrows():
            dup_pct = row["duplicate_pct"]
            tri_ov  = row.get("trigram_overlap", 0.0)

            if dup_pct > 5 or tri_ov > 95:
                risk = "🔴 **HIGH RISK**"
            elif dup_pct > 1:
                risk = "🟡 **MEDIUM RISK**"
            else:
                risk = "🟢 **LOW RISK**"

            evidence: List[str] = []
            if dup_pct > 0:
                evidence.append(f"Exact duplicate sentences: {dup_pct:.2f}%")
            if tri_ov > 0:
                evidence.append(f"Trigram overlap: {tri_ov:.2f}%")

            lines.append(f"**{row['comparison']}**  — {risk}\n")
            for e in evidence:
                lines.append(f"- {e}")
            lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 11 — Cross-language summary
# ──────────────────────────────────────────────────────────────────────────────

def analysis_11(
    units: List[AnalysisUnit],
    cov_df: pd.DataFrame,
    oov_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    sem_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate key metrics per language (Arabic / English / Hindi / Urdu)."""
    lang_data: Dict[str, Dict[str, List[float]]] = {}

    for u in units:
        lang = u.lang
        ld = lang_data.setdefault(lang, {
            "token_coverage": [], "oov_pct": [],
            "duplicate_pct": [], "mean_similarity": [],
        })

        sub_cov = cov_df[cov_df["direction"] == u.direction]
        for _, r in sub_cov.iterrows():
            ld["token_coverage"].append(r["token_coverage_pct"])

        sub_oov = oov_df[oov_df["direction"] == u.direction]
        for _, r in sub_oov.iterrows():
            ld["oov_pct"].append(r["oov_token_pct"])

        sub_dup = dup_df[dup_df["direction"] == u.direction]
        for _, r in sub_dup.iterrows():
            ld["duplicate_pct"].append(r["duplicate_pct"])

        if not sem_df.empty:
            sub_sem = sem_df[sem_df["direction"] == u.direction]
            for _, r in sub_sem.iterrows():
                ld["mean_similarity"].append(r["mean_similarity"])

    rows: List[Dict[str, Any]] = []
    for lang, metrics in sorted(lang_data.items()):
        rows.append({
            "language":            lang,
            "avg_token_coverage":  round(np.mean(metrics["token_coverage"]), 2)
                                   if metrics["token_coverage"] else None,
            "avg_oov_pct":         round(np.mean(metrics["oov_pct"]), 2)
                                   if metrics["oov_pct"] else None,
            "avg_duplicate_pct":   round(np.mean(metrics["duplicate_pct"]), 2)
                                   if metrics["duplicate_pct"] else None,
            "avg_semantic_sim":    round(np.mean(metrics["mean_similarity"]), 4)
                                   if metrics["mean_similarity"] else None,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _comma(x: float, _: Any) -> str:
    return f"{int(x):,}"


def plot_coverage_bar(cov_df: pd.DataFrame, path: Path) -> None:
    if cov_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    df = cov_df.copy()
    df["label"] = df["direction"] + " / " + df["eval_split"]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["token_coverage_pct"],    w, label="Token coverage %",
           color="#4C72B0")
    ax.bar(x + w / 2, df["unique_token_cov_pct"],  w, label="Unique token coverage %",
           color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coverage %", fontsize=11)
    ax.set_title("Vocabulary Coverage: Train → Eval", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_oov_bar(oov_df: pd.DataFrame, path: Path) -> None:
    if oov_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    df = oov_df.copy()
    df["label"] = df["direction"] + " / " + df["eval_split"]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["oov_token_pct"],  w, label="OOV token %",  color="#DD8452")
    ax.bar(x + w / 2, df["oov_unique_pct"], w, label="OOV unique %", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("OOV Rate %", fontsize=11)
    ax.set_title("Out-of-Vocabulary Rate: Eval vs Train", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ngram_heatmap(ngram_df: pd.DataFrame, path: Path) -> None:
    if ngram_df.empty:
        return
    df = ngram_df.copy()
    df["label"] = df["direction"] + " / " + df["eval_split"]
    cols = ["unigram", "bigram", "trigram", "fourgram"]
    matrix = df[cols].values

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(df))))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Uni", "Bi", "Tri", "4-gram"], fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=8)
    # annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 70 else "black"
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("N-gram Overlap %  (Eval tokens found in Train)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="%", shrink=0.8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_semantic_hist(
    sims_dict: Dict[str, np.ndarray], path: Path
) -> None:
    if not sims_dict:
        return
    n = len(sims_dict)
    cols = min(3, n)
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 4 * rows_), squeeze=False)
    for idx, (key, sims) in enumerate(sorted(sims_dict.items())):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.hist(sims, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        direction, split = key.split("|")
        ax.set_title(f"{direction}  train→{split}", fontsize=10, fontweight="bold")
        ax.axvline(np.mean(sims), color="#C44E52", ls="--", lw=1,
                   label=f"mean={np.mean(sims):.3f}")
        ax.legend(fontsize=7)
    # hide unused axes
    for idx in range(len(sims_dict), rows_ * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    plt.suptitle("Semantic Similarity Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_divergence_bar(div_df: pd.DataFrame, path: Path) -> None:
    if div_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df = div_df.copy()
    df["label"] = df["direction"] + " / " + df["eval_split"]
    x = np.arange(len(df))

    for ax, metric, colour, title in [
        (axes[0], "jsd", "#DD8452", "Jensen-Shannon Divergence"),
        (axes[1], "kld", "#4C72B0", "KL Divergence (Train → Eval)"),
    ]:
        ax.bar(x, df[metric], color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sentence_length(units: List[AnalysisUnit], path: Path) -> None:
    """Grid of sentence-length histograms (one subplot per direction)."""
    n = len(units)
    cols = min(3, n)
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 4 * rows_), squeeze=False)

    for idx, u in enumerate(units):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for sc in (u.train, u.dev, u.devtest):
            if sc.lengths.size == 0:
                continue
            clip = int(np.percentile(sc.lengths, 99))
            data = sc.lengths[sc.lengths <= clip]
            ax.hist(
                data, bins=max(10, min(60, clip)),
                alpha=0.55, label=sc.name, color=SPLIT_COLOURS[sc.name],
                edgecolor="white", linewidth=0.3,
            )
        ax.set_xlabel("Tokens", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(u.direction, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    for idx in range(n, rows_ * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.suptitle("Sentence Length Distribution by Split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────────────────────────

def _assessment(
    u: AnalysisUnit,
    cov_df: pd.DataFrame,
    oov_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    ngram_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    div_df: pd.DataFrame,
) -> str:
    """Per-direction assessment block for the final report."""
    d = u.direction
    parts = [f"### {d}\n"]

    for split in ("dev", "devtest"):
        parts.append(f"**Train → {split}**\n")

        # coverage
        row = cov_df[(cov_df["direction"] == d) & (cov_df["eval_split"] == split)]
        if not row.empty:
            tc = row.iloc[0]["token_coverage_pct"]
            parts.append(f"- Token coverage: {tc:.2f}%")

        # OOV
        row = oov_df[(oov_df["direction"] == d) & (oov_df["eval_split"] == split)]
        if not row.empty:
            ov = row.iloc[0]["oov_token_pct"]
            parts.append(f"- OOV rate: {ov:.2f}%")

        # duplicates
        comp = f"train↔{split}"
        row = dup_df[(dup_df["direction"] == d) & (dup_df["comparison"] == comp)]
        if not row.empty:
            dp = row.iloc[0]["duplicate_pct"]
            parts.append(f"- Exact duplicates: {dp:.2f}%")

        # n-gram
        row = ngram_df[(ngram_df["direction"] == d) & (ngram_df["eval_split"] == split)]
        if not row.empty:
            r = row.iloc[0]
            parts.append(
                f"- N-gram overlap: uni={r['unigram']:.1f}% "
                f"bi={r['bigram']:.1f}% "
                f"tri={r['trigram']:.1f}% "
                f"4g={r['fourgram']:.1f}%"
            )

        # semantic
        if not sem_df.empty:
            row = sem_df[(sem_df["direction"] == d) & (sem_df["eval_split"] == split)]
            if not row.empty:
                ms = row.iloc[0]["mean_similarity"]
                parts.append(f"- Mean semantic similarity: {ms:.4f}")

        # divergence
        row = div_df[(div_df["direction"] == d) & (div_df["eval_split"] == split)]
        if not row.empty:
            j = row.iloc[0]["jsd"]
            k = row.iloc[0]["kld"]
            parts.append(f"- JSD: {j:.6f}  |  KLD: {k:.6f}")

        # verdict
        tc_val = float(
            cov_df.loc[
                (cov_df["direction"] == d) & (cov_df["eval_split"] == split),
                "token_coverage_pct",
            ].iloc[0]
        ) if not cov_df[
            (cov_df["direction"] == d) & (cov_df["eval_split"] == split)
        ].empty else 0

        dup_val = float(
            dup_df.loc[
                (dup_df["direction"] == d) & (dup_df["comparison"] == comp),
                "duplicate_pct",
            ].iloc[0]
        ) if not dup_df[
            (dup_df["direction"] == d) & (dup_df["comparison"] == comp)
        ].empty else 0

        representative = tc_val > 90
        leakage = dup_val > 1

        parts.append("")
        parts.append(
            f"  → Representative: {'Yes' if representative else 'Needs review'}"
        )
        if leakage:
            parts.append(f"  → ⚠️  Possible leakage ({dup_val:.2f}% duplicates)")
        else:
            parts.append("  → No leakage detected")
        parts.append("")

    return "\n".join(parts)


def build_final_report(
    units: List[AnalysisUnit],
    stats_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    oov_df: pd.DataFrame,
    dup_summary: pd.DataFrame,
    ngram_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    div_df: pd.DataFrame,
    lang_df: pd.DataFrame,
) -> str:
    sections: List[str] = []

    # header
    sections.append("# Train ↔ Dev ↔ DevTest Analysis Report\n")
    sections.append(f"*Auto-generated by `cross_split_eda.py`*\n")
    sections.append("---\n")

    # 1 dataset statistics
    sections.append("## 1. Dataset Statistics\n")
    sections.append(stats_df.to_markdown(index=False))
    sections.append("")

    # 2 coverage
    sections.append("## 2. Vocabulary Coverage\n")
    sections.append(cov_df.to_markdown(index=False))
    sections.append("")

    # 3 OOV
    sections.append("## 3. OOV Analysis\n")
    sections.append(oov_df.to_markdown(index=False))
    sections.append("")

    # 4 leakage / duplicates
    sections.append("## 4. Leakage — Exact Duplicates\n")
    sections.append(dup_summary.to_markdown(index=False))
    sections.append("")

    # 5 n-gram
    sections.append("## 5. N-gram Overlap\n")
    sections.append(ngram_df.to_markdown(index=False))
    sections.append("")

    # 6 semantic
    if not sem_df.empty:
        sections.append("## 6. Semantic Similarity\n")
        sections.append(sem_df.to_markdown(index=False))
        sections.append("")

    # 7 divergence
    sections.append("## 7. Distribution Divergence\n")
    sections.append(div_df.to_markdown(index=False))
    sections.append("")

    # 8 language summary
    sections.append("## 8. Cross-Language Summary\n")
    sections.append(lang_df.to_markdown(index=False))
    sections.append("")

    # 9 per-direction assessment
    sections.append("## 9. Per-Direction Assessment\n")
    for u in units:
        sections.append(_assessment(
            u, cov_df, oov_df, dup_summary, ngram_df, sem_df, div_df,
        ))

    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 EDA — Train ↔ Dev ↔ DevTest split analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default="dataset",  type=Path, metavar="DIR")
    p.add_argument("--output-root",  default="eda_cross_split", type=Path, metavar="DIR")
    p.add_argument("--skip-semantic", action="store_true",
                   help="Skip semantic similarity analysis (Analysis 8)")
    p.add_argument("--semantic-method", choices=["tfidf", "sbert"], default="tfidf",
                   help="Embedding method for semantic similarity")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    root: Path = args.dataset_root
    out:  Path = args.output_root

    if not root.exists():
        log.error("Dataset root not found: %s", root)
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    csv_dir  = out / "csv";   csv_dir.mkdir(exist_ok=True)
    plot_dir = out / "plots"; plot_dir.mkdir(exist_ok=True)

    log.info("Dataset root : %s", root.resolve())
    log.info("Output root  : %s", out.resolve())

    # ── discover ──────────────────────────────────────────────────────────────
    units = discover_units(root)
    if not units:
        log.error("No complete (train+dev+devtest) triples found. Exiting.")
        sys.exit(1)
    log.info("Found %d analysis units", len(units))

    # ── analysis 1 ────────────────────────────────────────────────────────────
    log.info("Analysis  1/11 — Dataset statistics")
    stats_df = analysis_1(units)
    stats_df.to_csv(csv_dir / "split_statistics.csv", index=False)

    # ── analysis 2 ────────────────────────────────────────────────────────────
    log.info("Analysis  2/11 — Vocabulary coverage")
    cov_df = analysis_2(units)
    cov_df.to_csv(csv_dir / "vocabulary_coverage.csv", index=False)

    # ── analysis 3 ────────────────────────────────────────────────────────────
    log.info("Analysis  3/11 — OOV analysis")
    oov_stats_df, top_oov_df = analysis_3(units)
    oov_stats_df.to_csv(csv_dir / "oov_statistics.csv",  index=False)
    top_oov_df.to_csv(csv_dir / "top_oov_tokens.csv",    index=False)

    # ── analysis 4 ────────────────────────────────────────────────────────────
    log.info("Analysis  4/11 — Exact duplicate detection")
    dup_summary_df, dup_sents_df = analysis_4(units)
    dup_summary_df.to_csv(csv_dir / "exact_duplicate_summary.csv", index=False)
    dup_sents_df.to_csv(csv_dir / "exact_duplicate_sentences.csv", index=False)

    # ── analysis 5 ────────────────────────────────────────────────────────────
    log.info("Analysis  5/11 — N-gram overlap")
    ngram_df = analysis_5(units)
    ngram_df.to_csv(csv_dir / "ngram_overlap.csv", index=False)

    # ── analysis 6 ────────────────────────────────────────────────────────────
    log.info("Analysis  6/11 — Sentence length distribution")
    len_df = analysis_6(units)
    len_df.to_csv(csv_dir / "sentence_length_distribution.csv", index=False)

    # ── analysis 7 ────────────────────────────────────────────────────────────
    log.info("Analysis  7/11 — Vocabulary frequency shift")
    freq_df = analysis_7(units)
    freq_df.to_csv(csv_dir / "frequency_shift.csv", index=False)

    # ── analysis 8 ────────────────────────────────────────────────────────────
    log.info("Analysis  8/11 — Semantic similarity")
    sem_df, sims_dict = analysis_8(
        units, method=args.semantic_method, skip=args.skip_semantic,
    )
    if not sem_df.empty:
        sem_df.to_csv(csv_dir / "semantic_similarity.csv", index=False)

    # ── analysis 9 ────────────────────────────────────────────────────────────
    log.info("Analysis  9/11 — Distribution divergence")
    div_df = analysis_9(units)
    div_df.to_csv(csv_dir / "distribution_divergence.csv", index=False)

    # ── analysis 10 ───────────────────────────────────────────────────────────
    log.info("Analysis 10/11 — Leakage report")
    leakage_md = analysis_10(dup_summary_df, ngram_df)
    (out / "LEAKAGE_REPORT.md").write_text(leakage_md, encoding="utf-8")

    # ── analysis 11 ───────────────────────────────────────────────────────────
    log.info("Analysis 11/11 — Cross-language summary")
    lang_df = analysis_11(units, cov_df, oov_stats_df, dup_summary_df, sem_df)
    lang_df.to_csv(csv_dir / "language_summary.csv", index=False)

    # ── visualisations ────────────────────────────────────────────────────────
    log.info("Generating visualisations …")
    plot_coverage_bar(cov_df,               plot_dir / "vocabulary_coverage_barplot.png")
    plot_oov_bar(oov_stats_df,              plot_dir / "oov_rate_barplot.png")
    plot_ngram_heatmap(ngram_df,            plot_dir / "ngram_overlap_heatmap.png")
    plot_semantic_hist(sims_dict,           plot_dir / "semantic_similarity_histogram.png")
    plot_divergence_bar(div_df,            plot_dir / "distribution_divergence_barplot.png")
    plot_sentence_length(units,            plot_dir / "sentence_length_distribution_comparison.png")

    # ── final report ──────────────────────────────────────────────────────────
    log.info("Writing final report …")
    try:
        report_md = build_final_report(
            units, stats_df, cov_df, oov_stats_df,
            dup_summary_df, ngram_df, sem_df, div_df, lang_df,
        )
    except Exception as exc:
        log.warning("to_markdown() failed (%s) — writing plain-text tables", exc)
        report_md = "# Report\n\nPlease install `tabulate` for Markdown tables.\n"
    (out / "TRAIN_DEV_ANALYSIS_REPORT.md").write_text(report_md, encoding="utf-8")

    log.info("Phase 2 EDA complete.  All outputs → %s", out.resolve())


if __name__ == "__main__":
    main()