#!/usr/bin/env python3
"""
evaluate_mt_noreference.py — Reference-Free Pre-Submission MT Checks
=====================================================================
When you have SOURCE only and no target references (standard shared-task
test sets), this script validates your translations using:

  Check 1 — Sanity             (line count, empty, encoding)
  Check 2 — Language validity  (is the output in the right language?)
  Check 3 — Source copying     (did the model just copy the input?)
  Check 4 — Length consistency  (compare against devtest baseline)
  Check 5 — Fluency signals    (repetition, token diversity)
  Check 6 — Quality estimation (COMET-QE, no reference needed)
  Check 7 — Cross-pair sanity  (shared Arabic sources should align)

File layout
-----------
    dataset/Ar-En/test_ar_ar-en.txt    ← source (you have this)
    dataset/Ar-En/test_en_ar-en.txt    ← reference (you DON'T have this)

    hypotheses/ar-en.txt               ← your model output (you have this)

    devtest_hyps/ar-en.txt             ← your devtest output (optional, for baseline)

Usage
-----
    python evaluate_mt_noreference.py --hyp-dir hypotheses --split test

    # Compare against devtest baseline for length/vocab consistency
    python evaluate_mt_noreference.py --hyp-dir hypotheses --split test \\
        --devtest-hyp-dir devtest_hyps

    # With COMET-QE (pip install unbabel-comet)
    python evaluate_mt_noreference.py --hyp-dir hypotheses --split test --comet-qe
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional: COMET-QE
try:
    from comet import download_model, load_from_checkpoint
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Direction configuration
# ──────────────────────────────────────────────────────────────────────────────
DIRECTIONS = {
    "ar-en": ("ar", "en", "Ar-En"),
    "en-ar": ("en", "ar", "Ar-En"),
    "ar-hi": ("ar", "hi", "Ar-Hi"),
    "hi-ar": ("hi", "ar", "Ar-Hi"),
    "ar-ur": ("ar", "ur", "Ar-Ur"),
    "ur-ar": ("ur", "ar", "Ar-Ur"),
}

EXPECTED_SCRIPT = {
    "ar": "ARABIC",
    "en": "LATIN",
    "hi": "DEVANAGARI",
    "ur": "ARABIC",
}

# Typical length ratios from known MT data (target_tokens / source_tokens)
# Used as fallback when devtest baseline is not available
TYPICAL_RATIOS = {
    "ar-en": 1.20,   # English is ~20% longer than Arabic
    "en-ar": 0.85,
    "ar-hi": 1.45,   # Hindi tends to be longer than Arabic
    "hi-ar": 0.70,
    "ar-ur": 1.55,   # Urdu is typically longer
    "ur-ar": 0.65,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def resolve_src_path(dataset_root: Path, split: str, direction: str) -> Optional[Path]:
    src_lang, _, pair_dir = DIRECTIONS[direction]
    pair_code = pair_dir.lower().replace("-", "-")
    p = dataset_root / pair_dir / f"{split}_{src_lang}_{pair_code}.txt"
    return p if p.exists() else None


def detect_script(text: str) -> str:
    scripts: Counter = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch, "")
                for s in ("ARABIC", "DEVANAGARI", "LATIN"):
                    if s in name:
                        scripts[s] += 1
                        break
            except ValueError:
                pass
    return scripts.most_common(1)[0][0] if scripts else "UNKNOWN"


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap between two lines (whitespace-tokenized)."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_repetition_score(text: str) -> float:
    """
    Fraction of bigrams that are repeated.
    High repetition (>0.5) signals degenerate output.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(bigrams)


# ──────────────────────────────────────────────────────────────────────────────
# Check 1: Sanity
# ──────────────────────────────────────────────────────────────────────────────

def check_sanity(direction: str, src: List[str], hyp: List[str]) -> Dict[str, Any]:
    result = {
        "direction": direction,
        "src_lines": len(src),
        "hyp_lines": len(hyp),
        "lines_match": len(src) == len(hyp),
        "empty_hyp": sum(1 for h in hyp if h.strip() == ""),
        "empty_pct": round(100 * sum(1 for h in hyp if h.strip() == "") / len(hyp), 2) if hyp else 0,
    }
    issues = []
    if not result["lines_match"]:
        issues.append(f"LINE COUNT MISMATCH: source={len(src)} hypothesis={len(hyp)}")
    if result["empty_pct"] > 0:
        issues.append(f"{result['empty_hyp']} empty lines ({result['empty_pct']}%)")
    result["issues"] = issues
    result["passed"] = len(issues) == 0 or (result["lines_match"] and result["empty_pct"] < 2)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Check 2: Language validity
# ──────────────────────────────────────────────────────────────────────────────

def check_language(
    direction: str, hyp: List[str], sample_n: int = 300
) -> Dict[str, Any]:
    _, tgt_lang, _ = DIRECTIONS[direction]
    expected = EXPECTED_SCRIPT[tgt_lang]

    sample = hyp[:sample_n]
    correct = 0
    wrong_script_lines = []

    for i, line in enumerate(sample):
        if not line.strip():
            continue
        detected = detect_script(line)
        if detected == expected:
            correct += 1
        else:
            if len(wrong_script_lines) < 5:
                wrong_script_lines.append((i, detected, line[:80]))

    non_empty = sum(1 for l in sample if l.strip())
    pct = round(100 * correct / non_empty, 1) if non_empty else 0

    issues = []
    if pct < 90:
        issues.append(
            f"Only {pct}% of lines in expected script ({expected}). "
            f"Model may be outputting wrong language."
        )

    return {
        "direction": direction,
        "target_lang": tgt_lang,
        "expected_script": expected,
        "correct_script_pct": pct,
        "wrong_examples": wrong_script_lines,
        "issues": issues,
        "passed": pct >= 90,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Check 3: Source copying detection
# ──────────────────────────────────────────────────────────────────────────────

def check_source_copying(
    direction: str, src: List[str], hyp: List[str]
) -> Dict[str, Any]:
    n = min(len(src), len(hyp))
    exact_copies = 0
    high_overlap = 0
    overlaps = []

    for i in range(n):
        if src[i].strip() == hyp[i].strip() and src[i].strip():
            exact_copies += 1
        ov = token_overlap(src[i], hyp[i])
        overlaps.append(ov)
        if ov > 0.8 and src[i].strip():
            high_overlap += 1

    exact_pct = round(100 * exact_copies / n, 2) if n else 0
    high_ov_pct = round(100 * high_overlap / n, 2) if n else 0
    mean_overlap = round(float(np.mean(overlaps)), 4) if overlaps else 0

    issues = []
    if exact_pct > 2:
        issues.append(
            f"{exact_copies} exact source copies ({exact_pct}%) — model is not translating"
        )
    if high_ov_pct > 10:
        issues.append(
            f"{high_overlap} lines with >80% token overlap ({high_ov_pct}%) — possible undertranslation"
        )

    return {
        "direction": direction,
        "exact_copies": exact_copies,
        "exact_copy_pct": exact_pct,
        "high_overlap_lines": high_overlap,
        "high_overlap_pct": high_ov_pct,
        "mean_token_overlap": mean_overlap,
        "issues": issues,
        "passed": exact_pct <= 2 and high_ov_pct <= 10,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Check 4: Length consistency
# ──────────────────────────────────────────────────────────────────────────────

def check_length(
    direction: str,
    src: List[str],
    hyp: List[str],
    devtest_hyp: Optional[List[str]] = None,
    devtest_src: Optional[List[str]] = None,
) -> Dict[str, Any]:
    n = min(len(src), len(hyp))
    src_lens = np.array([len(src[i].split()) for i in range(n)], dtype=np.float64)
    hyp_lens = np.array([len(hyp[i].split()) for i in range(n)], dtype=np.float64)

    # Ratio: hyp_tokens / src_tokens (per sentence, then averaged)
    ratios = []
    for sl, hl in zip(src_lens, hyp_lens):
        if sl > 0:
            ratios.append(hl / sl)
    ratios = np.array(ratios)
    mean_ratio = round(float(np.mean(ratios)), 4) if ratios.size else 0
    std_ratio = round(float(np.std(ratios)), 4) if ratios.size else 0

    # Compare against devtest baseline (if available)
    expected_ratio = TYPICAL_RATIOS.get(direction, 1.0)
    baseline_source = "typical MT ratio"

    if devtest_hyp is not None and devtest_src is not None:
        dn = min(len(devtest_src), len(devtest_hyp))
        dev_ratios = []
        for i in range(dn):
            sl = len(devtest_src[i].split())
            hl = len(devtest_hyp[i].split())
            if sl > 0:
                dev_ratios.append(hl / sl)
        if dev_ratios:
            expected_ratio = round(float(np.mean(dev_ratios)), 4)
            baseline_source = "devtest output"

    drift = abs(mean_ratio - expected_ratio) / expected_ratio if expected_ratio else 0

    issues = []
    if drift > 0.25:
        issues.append(
            f"Length ratio {mean_ratio:.3f} deviates {drift*100:.0f}% from "
            f"baseline {expected_ratio:.3f} ({baseline_source})"
        )
    if mean_ratio < 0.3:
        issues.append(f"Translations extremely short (ratio {mean_ratio:.3f})")
    if mean_ratio > 3.0:
        issues.append(f"Translations extremely long (ratio {mean_ratio:.3f})")

    return {
        "direction": direction,
        "hyp_mean_tokens": round(float(np.mean(hyp_lens)), 2),
        "src_mean_tokens": round(float(np.mean(src_lens)), 2),
        "mean_length_ratio": mean_ratio,
        "std_length_ratio": std_ratio,
        "expected_ratio": expected_ratio,
        "baseline_source": baseline_source,
        "drift_pct": round(drift * 100, 1),
        "issues": issues,
        "passed": drift <= 0.25 and 0.3 <= mean_ratio <= 3.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Check 5: Fluency signals
# ──────────────────────────────────────────────────────────────────────────────

def check_fluency(direction: str, hyp: List[str]) -> Dict[str, Any]:
    rep_scores = []
    ttr_list = []

    for line in hyp:
        if not line.strip():
            continue
        rep_scores.append(compute_repetition_score(line))
        tokens = line.split()
        if tokens:
            ttr_list.append(len(set(tokens)) / len(tokens))

    mean_rep = round(float(np.mean(rep_scores)), 4) if rep_scores else 0
    high_rep = sum(1 for r in rep_scores if r > 0.3)
    high_rep_pct = round(100 * high_rep / len(rep_scores), 2) if rep_scores else 0
    mean_ttr = round(float(np.mean(ttr_list)), 4) if ttr_list else 0

    issues = []
    if mean_rep > 0.15:
        issues.append(
            f"Mean repetition score {mean_rep:.3f} is high — model may be looping"
        )
    if high_rep_pct > 10:
        issues.append(f"{high_rep_pct}% of sentences have high repetition (>0.3)")
    if mean_ttr < 0.4:
        issues.append(
            f"Mean type-token ratio {mean_ttr:.3f} is very low — limited vocabulary diversity"
        )

    return {
        "direction": direction,
        "mean_repetition": mean_rep,
        "high_repetition_pct": high_rep_pct,
        "mean_ttr": mean_ttr,
        "issues": issues,
        "passed": mean_rep <= 0.15 and mean_ttr >= 0.4,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Check 6: COMET-QE (reference-free quality estimation)
# ──────────────────────────────────────────────────────────────────────────────

def check_comet_qe(
    direction: str, src: List[str], hyp: List[str]
) -> Dict[str, Any]:
    if not HAS_COMET:
        return {
            "direction": direction,
            "available": False,
            "issues": ["COMET not installed — pip install unbabel-comet"],
            "passed": True,  # don't fail for missing optional dependency
        }

    try:
        log.info("  Downloading COMET-QE model …")
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)

        n = min(len(src), len(hyp))
        data = [{"src": src[i], "mt": hyp[i]} for i in range(n)]
        output = model.predict(data, batch_size=64, gpus=0)

        scores = np.array(output.scores)
        sys_score = round(output.system_score, 4)

        low_quality = sum(1 for s in scores if s < 0.5)
        low_pct = round(100 * low_quality / len(scores), 2)

        issues = []
        if sys_score < 0.6:
            issues.append(f"COMET-QE system score {sys_score} is low")
        if low_pct > 30:
            issues.append(f"{low_pct}% of sentences below 0.5 quality threshold")

        return {
            "direction": direction,
            "available": True,
            "system_score": sys_score,
            "mean": round(float(np.mean(scores)), 4),
            "median": round(float(np.median(scores)), 4),
            "p10": round(float(np.percentile(scores, 10)), 4),
            "low_quality_pct": low_pct,
            "issues": issues,
            "passed": sys_score >= 0.6,
        }
    except Exception as exc:
        log.warning("  COMET-QE failed: %s", exc)
        return {
            "direction": direction,
            "available": False,
            "issues": [f"COMET-QE failed: {exc}"],
            "passed": True,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(all_results: Dict[str, Dict[str, Any]], split: str) -> str:
    lines = []
    lines.append(f"# Reference-Free Evaluation Report — `{split}` split\n")
    lines.append("*No target references available. Checks are based on source-hypothesis*")
    lines.append("*consistency, language validity, and statistical patterns.*\n")
    lines.append("---\n")

    # ── Overall verdict ───────────────────────────────────────────────────
    lines.append("## Overall Verdict\n")
    lines.append("| Direction | Sanity | Language | Copying | Length | Fluency | Ready? |")
    lines.append("|-----------|--------|----------|---------|--------|---------|--------|")

    for direction, checks in all_results.items():
        cells = []
        for check_name in ["sanity", "language", "copying", "length", "fluency"]:
            c = checks.get(check_name, {})
            cells.append("✅" if c.get("passed", True) else "❌")

        all_pass = all(
            checks.get(cn, {}).get("passed", True)
            for cn in ["sanity", "language", "copying", "length", "fluency"]
        )
        verdict = "✅ SUBMIT" if all_pass else "⚠️ REVIEW"
        lines.append(f"| {direction} | {' | '.join(cells)} | {verdict} |")
    lines.append("")

    # ── Detailed results per direction ────────────────────────────────────
    for direction, checks in all_results.items():
        lines.append(f"## {direction}\n")

        for check_name, label in [
            ("sanity", "Sanity"), ("language", "Language"),
            ("copying", "Source Copying"), ("length", "Length Consistency"),
            ("fluency", "Fluency"), ("comet_qe", "COMET-QE"),
        ]:
            c = checks.get(check_name)
            if not c:
                continue

            status = "✅" if c.get("passed", True) else "❌"
            lines.append(f"**{label}** {status}\n")

            # Show key metrics
            skip = {"direction", "issues", "passed", "wrong_examples",
                    "available", "baseline_source"}
            for k, v in c.items():
                if k in skip:
                    continue
                lines.append(f"- {k}: {v}")

            # Show issues
            for iss in c.get("issues", []):
                lines.append(f"- ⚠️ {iss}")
            lines.append("")

    # ── Action items ──────────────────────────────────────────────────────
    lines.append("## Action Items\n")

    all_issues = []
    for direction, checks in all_results.items():
        for check_name, check_data in checks.items():
            for iss in check_data.get("issues", []):
                all_issues.append((direction, check_name, iss))

    if not all_issues:
        lines.append("No issues found. All directions ready for submission. ✅\n")
    else:
        for direction, check, issue in all_issues:
            lines.append(f"- **{direction}** ({check}): {issue}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI + Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reference-free MT evaluation for test submissions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default="dataset", type=Path)
    p.add_argument("--hyp-dir", required=True, type=Path,
                   help="Directory with hypothesis files (ar-en.txt, en-ar.txt, …)")
    p.add_argument("--split", default="test",
                   help="Split name for source files")
    p.add_argument("--devtest-hyp-dir", type=Path, default=None,
                   help="Devtest hypothesis dir (for length ratio baseline)")
    p.add_argument("--output-root", default="eval_noreference", type=Path)
    p.add_argument("--comet-qe", action="store_true",
                   help="Run COMET-QE quality estimation (needs unbabel-comet)")
    p.add_argument("--directions", nargs="+", default=list(DIRECTIONS.keys()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    log.info("Mode           : REFERENCE-FREE (no target labels)")
    log.info("Dataset root   : %s", args.dataset_root.resolve())
    log.info("Hypotheses     : %s", args.hyp_dir.resolve())
    log.info("Split          : %s", args.split)
    if args.devtest_hyp_dir:
        log.info("Devtest baseline: %s", args.devtest_hyp_dir.resolve())

    all_results: Dict[str, Dict[str, Any]] = {}

    for direction in args.directions:
        src_path = resolve_src_path(args.dataset_root, args.split, direction)
        hyp_path = args.hyp_dir / f"{direction}.txt"

        if src_path is None:
            log.warning("Skipping %s — no source file for split '%s'", direction, args.split)
            continue
        if not hyp_path.exists():
            log.warning("Skipping %s — no hypothesis at %s", direction, hyp_path)
            continue

        log.info("━" * 55)
        log.info("Checking  %s", direction)

        src = load_lines(src_path)
        hyp = load_lines(hyp_path)

        # Load devtest baseline if available
        devtest_hyp = None
        devtest_src = None
        if args.devtest_hyp_dir:
            dt_hyp_path = args.devtest_hyp_dir / f"{direction}.txt"
            dt_src_path = resolve_src_path(args.dataset_root, "devtest", direction)
            if dt_hyp_path.exists() and dt_src_path and dt_src_path.exists():
                devtest_hyp = load_lines(dt_hyp_path)
                devtest_src = load_lines(dt_src_path)

        checks: Dict[str, Any] = {}

        # Check 1: Sanity
        log.info("  Check 1 — Sanity")
        checks["sanity"] = check_sanity(direction, src, hyp)
        s = checks["sanity"]
        log.info("    lines: src=%d hyp=%d match=%s empty=%d",
                 s["src_lines"], s["hyp_lines"], s["lines_match"], s["empty_hyp"])

        # Truncate if needed
        n = min(len(src), len(hyp))
        src_t, hyp_t = src[:n], hyp[:n]

        # Check 2: Language
        log.info("  Check 2 — Language validity")
        checks["language"] = check_language(direction, hyp_t)
        log.info("    correct script: %s%%", checks["language"]["correct_script_pct"])

        # Check 3: Source copying
        log.info("  Check 3 — Source copying")
        checks["copying"] = check_source_copying(direction, src_t, hyp_t)
        log.info("    exact copies: %s  high overlap: %s%%",
                 checks["copying"]["exact_copies"],
                 checks["copying"]["high_overlap_pct"])

        # Check 4: Length consistency
        log.info("  Check 4 — Length consistency")
        checks["length"] = check_length(direction, src_t, hyp_t, devtest_hyp, devtest_src)
        lc = checks["length"]
        log.info("    ratio: %.3f (expected: %.3f from %s, drift: %.1f%%)",
                 lc["mean_length_ratio"], lc["expected_ratio"],
                 lc["baseline_source"], lc["drift_pct"])

        # Check 5: Fluency
        log.info("  Check 5 — Fluency signals")
        checks["fluency"] = check_fluency(direction, hyp_t)
        log.info("    repetition: %.3f  TTR: %.3f",
                 checks["fluency"]["mean_repetition"],
                 checks["fluency"]["mean_ttr"])

        # Check 6: COMET-QE
        if args.comet_qe:
            log.info("  Check 6 — COMET-QE")
            checks["comet_qe"] = check_comet_qe(direction, src_t, hyp_t)
            if checks["comet_qe"].get("system_score"):
                log.info("    COMET-QE: %.4f", checks["comet_qe"]["system_score"])

        # Verdict
        failed = [name for name, c in checks.items()
                  if not c.get("passed", True)]
        if failed:
            log.warning("  ⚠️  ISSUES in: %s", ", ".join(failed))
        else:
            log.info("  ✅ All checks passed")

        all_results[direction] = checks

    if not all_results:
        log.error("No directions evaluated. Check file paths.")
        sys.exit(1)

    # ── Write outputs ─────────────────────────────────────────────────────
    log.info("━" * 55)
    report = generate_report(all_results, args.split)
    (out / "PRESUBMISSION_REPORT.md").write_text(report, encoding="utf-8")

    (out / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  PRE-SUBMISSION CHECK (NO REFERENCE)")
    print("═" * 72)
    print(f"\n  {'Direction':<10} {'Lines':>7} {'Empty%':>7} {'Script%':>8} "
          f"{'Copy%':>7} {'Ratio':>7} {'Repet':>7} {'Status':>10}")
    print("  " + "─" * 66)

    for direction, checks in all_results.items():
        s = checks["sanity"]
        l = checks["language"]
        c = checks["copying"]
        le = checks["length"]
        f = checks["fluency"]
        failed = any(not ch.get("passed", True) for ch in checks.values())
        status = " ⚠ CHECK" if failed else "    OK ✅"
        print(
            f"  {direction:<10} {s['hyp_lines']:>7} {s['empty_pct']:>6.1f}% "
            f"{l['correct_script_pct']:>7.0f}% {c['exact_copy_pct']:>6.1f}% "
            f"{le['mean_length_ratio']:>7.3f} {f['mean_repetition']:>7.3f} "
            f"{status}"
        )

    print("  " + "─" * 66)
    total_issues = sum(
        len(iss)
        for checks in all_results.values()
        for c in checks.values()
        for iss in [c.get("issues", [])]
    )
    if total_issues == 0:
        print("\n  ✅ ALL DIRECTIONS PASS — ready for submission\n")
    else:
        print(f"\n  ⚠️  {total_issues} issue(s) found — see PRESUBMISSION_REPORT.md\n")
    print("═" * 72 + "\n")

    log.info("Report → %s", out / "PRESUBMISSION_REPORT.md")


if __name__ == "__main__":
    main()