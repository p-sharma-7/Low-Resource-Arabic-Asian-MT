#!/usr/bin/env python3
"""
run_noref_eval.py — Reference-free pre-submission checks for WMT26 challenge outputs.

Evaluates all 18 hypothesis files (3 models × 6 directions) against the challenge
source-only test sets using 5 deterministic checks + optional COMET-Kiwi QE.

Fixes 5 bugs in the original EDA/evaluate_mt_noreference.py:
  1. Correct source path resolution (challenge Sub-Task directory tree)
  2. Correct hypothesis filename pattern (hyp_{model}_ft_{dir}_challenge.txt)
  3. COMET-Kiwi model loaded ONCE and reused (not reloaded per direction)
  4. COMET-Kiwi runs on GPU (gpus=1, not gpus=0)
  5. Uses wmt23-cometkiwi-da (not wmt22-cometkiwi-da)

Usage
-----
    cd /mnt/storage/Pushkar/challenge/wmt

    # Fast checks only (~30 sec):
    /mnt/storage/abhay/envs/wmt/bin/python run_noref_eval.py

    # Include COMET-Kiwi QE scores (~10 min, GPU):
    /mnt/storage/abhay/envs/wmt/bin/python run_noref_eval.py --comet-qe

    # Single model:
    /mnt/storage/abhay/envs/wmt/bin/python run_noref_eval.py --models nllb

Outputs → outputs/eval_noreference/
    REPORT_nllb.md, REPORT_madlad.md, REPORT_gemmax2.md
    all_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Import shared check functions from EDA script ────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "EDA"))
from evaluate_mt_noreference import (  # noqa: E402
    DIRECTIONS,
    load_lines,
    check_sanity,
    check_language,
    check_source_copying,
    check_length,
    check_fluency,
    generate_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Challenge test source files (source-only, no references) ─────────────────
_CHALLENGE_ROOT = Path(
    "/mnt/storage/Ahtisham_Aziz/NLP-IIT Patna/Challenge-Test"
)
CHALLENGE_SRCS: Dict[str, Path] = {
    "ar-en": _CHALLENGE_ROOT / "Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt",
    "en-ar": _CHALLENGE_ROOT / "Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt",
    "ar-hi": _CHALLENGE_ROOT / "Sub-Task 2B_Arabic-to-Hindi/Ar-to-Hi/test_challenge_ar_ar-to-hi.txt",
    "hi-ar": _CHALLENGE_ROOT / "Sub-Task 1B_Hindi-to-Arabic/Hi-to-Ar/test_challenge_hi_hi-to-ar.txt",
    "ar-ur": _CHALLENGE_ROOT / "Sub-Task 2E_Arabic-to-Urdu/Ar-to-Ur/test_challenge_ar_ar-to-ur.txt",
    "ur-ar": _CHALLENGE_ROOT / "Sub-Task 1E_Urdu-to-Arabic/Ur-to-Ar/test_challenge_ur_ur-to-ar.txt",
}

HYP_ROOT = Path("outputs/eval_finetuned_challenge")
ALL_MODELS = ["nllb", "madlad", "gemmax2"]


def hyp_path(model: str, direction: str) -> Path:
    return HYP_ROOT / model / f"hyp_{model}_ft_{direction}_challenge.txt"


# ── Fixed COMET-Kiwi QE (load once, run on GPU) ──────────────────────────────

def load_comet_kiwi():
    """Download wmt23-cometkiwi-da and load it once. Returns model object."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        log.error("unbabel-comet not installed. Run: pip install unbabel-comet")
        return None
    log.info("Downloading / loading Unbabel/wmt23-cometkiwi-da …")
    model_path = download_model("Unbabel/wmt23-cometkiwi-da")
    return load_from_checkpoint(model_path)


def check_comet_qe_gpu(
    direction: str,
    src: List[str],
    hyp: List[str],
    comet_model: Any,
) -> Dict[str, Any]:
    """Reference-free QE using a pre-loaded COMET-Kiwi model on GPU."""
    if comet_model is None:
        return {
            "direction": direction, "available": False,
            "issues": ["COMET-Kiwi model failed to load"],
            "passed": True,
        }
    try:
        n = min(len(src), len(hyp))
        data = [{"src": src[i], "mt": hyp[i]} for i in range(n)]
        output = comet_model.predict(data, batch_size=64, gpus=1)
        scores = np.array(output.scores)
        sys_score = round(float(output.system_score), 4)
        low_pct = round(100 * sum(1 for s in scores if s < 0.5) / len(scores), 2)

        issues = []
        if sys_score < 0.6:
            issues.append(f"COMET-Kiwi system score {sys_score} is low (<0.6)")
        if low_pct > 30:
            issues.append(f"{low_pct}% of sentences score below 0.5 quality threshold")

        return {
            "direction": direction, "available": True,
            "system_score": sys_score,
            "mean":   round(float(np.mean(scores)), 4),
            "median": round(float(np.median(scores)), 4),
            "p10":    round(float(np.percentile(scores, 10)), 4),
            "low_quality_pct": low_pct,
            "issues": issues,
            "passed": sys_score >= 0.6,
        }
    except Exception as exc:
        log.warning("COMET-Kiwi failed for %s: %s", direction, exc)
        return {
            "direction": direction, "available": False,
            "issues": [f"COMET-Kiwi error: {exc}"],
            "passed": True,
        }


# ── Summary printer ───────────────────────────────────────────────────────────

def print_model_summary(model: str, model_results: Dict[str, Dict]) -> None:
    print(f"\n{'═' * 72}")
    print(f"  MODEL: {model.upper()}")
    print(f"{'═' * 72}")
    print(f"  {'Direction':<10} {'Lines':>6} {'Empty%':>7} {'Script%':>8} "
          f"{'Copy%':>7} {'Ratio':>7} {'Repet':>7}", end="")
    if any("comet_qe" in v for v in model_results.values()):
        print(f" {'COMET-QE':>9}", end="")
    print(f" {'Status':>10}")
    print("  " + "─" * 70)

    for direction, checks in model_results.items():
        s  = checks["sanity"]
        la = checks["language"]
        co = checks["copying"]
        le = checks["length"]
        fl = checks["fluency"]
        failed = any(not ch.get("passed", True) for ch in checks.values())
        status = " ⚠ REVIEW" if failed else "    OK ✅"
        row = (
            f"  {direction:<10} {s['hyp_lines']:>6} {s['empty_pct']:>6.1f}% "
            f"{la['correct_script_pct']:>7.0f}% {co['exact_copy_pct']:>6.1f}% "
            f"{le['mean_length_ratio']:>7.3f} {fl['mean_repetition']:>7.3f}"
        )
        if "comet_qe" in checks and checks["comet_qe"].get("available"):
            row += f" {checks['comet_qe']['system_score']:>9.4f}"
        elif any("comet_qe" in v for v in model_results.values()):
            row += f" {'N/A':>9}"
        row += f" {status}"
        print(row)

    print("  " + "─" * 70)
    total_issues = sum(
        len(c.get("issues", []))
        for checks in model_results.values()
        for c in checks.values()
    )
    if total_issues == 0:
        print(f"  ✅ All directions PASS — {model.upper()} ready for submission\n")
    else:
        print(f"  ⚠️  {total_issues} issue(s) found — see REPORT_{model}.md\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reference-free pre-submission checks for WMT26 challenge outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, metavar="MODEL",
                   help="Which models to evaluate")
    p.add_argument("--comet-qe", action="store_true",
                   help="Run COMET-Kiwi QE (wmt23-cometkiwi-da, GPU, ~10 min for all models)")
    p.add_argument("--out", type=Path,
                   default=Path("eval_noreference"),
                   help="Output directory for reports and JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log.info("Reference-free evaluation — WMT26 challenge test set")
    log.info("Models     : %s", args.models)
    log.info("COMET-QE   : %s", args.comet_qe)
    log.info("Output dir : %s", args.out.resolve())

    # Verify source files exist before starting
    for direction, src_path in CHALLENGE_SRCS.items():
        if not src_path.exists():
            log.error("Missing source file for %s: %s", direction, src_path)
            sys.exit(1)

    # Load COMET-Kiwi once if requested
    comet_model = load_comet_kiwi() if args.comet_qe else None

    all_results: Dict[str, Dict] = {}

    for model in args.models:
        log.info("━" * 55)
        log.info("Evaluating model: %s", model.upper())

        model_results: Dict[str, Dict] = {}

        for direction in DIRECTIONS:
            hp = hyp_path(model, direction)
            if not hp.exists():
                log.warning("Missing hypothesis: %s — skipping", hp)
                continue

            src_path = CHALLENGE_SRCS[direction]
            log.info("  %s  →  %s", direction, hp.name)

            src = load_lines(src_path)
            hyp = load_lines(hp)

            n = min(len(src), len(hyp))
            src_t, hyp_t = src[:n], hyp[:n]

            checks: Dict[str, Any] = {}

            checks["sanity"]   = check_sanity(direction, src, hyp)
            checks["language"] = check_language(direction, hyp_t)
            checks["copying"]  = check_source_copying(direction, src_t, hyp_t)
            checks["length"]   = check_length(direction, src_t, hyp_t)
            checks["fluency"]  = check_fluency(direction, hyp_t)

            if comet_model is not None:
                log.info("    Running COMET-Kiwi …")
                checks["comet_qe"] = check_comet_qe_gpu(
                    direction, src_t, hyp_t, comet_model
                )

            failed_checks = [k for k, v in checks.items() if not v.get("passed", True)]
            if failed_checks:
                log.warning("    ⚠️  Issues in: %s", ", ".join(failed_checks))
            else:
                log.info("    ✅ All checks passed")

            model_results[direction] = checks

        # Per-model markdown report
        report = generate_report(model_results, f"challenge_{model}")
        report_path = args.out / f"REPORT_{model}.md"
        report_path.write_text(report, encoding="utf-8")
        log.info("  Report → %s", report_path)

        print_model_summary(model, model_results)
        all_results[model] = model_results

    # Aggregated JSON
    json_path = args.out / "all_results.json"
    json_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Aggregated JSON → %s", json_path)

    # Final verdict across all models
    print(f"\n{'═' * 72}")
    print("  FINAL SUBMISSION READINESS")
    print(f"{'═' * 72}")
    for model in args.models:
        if model not in all_results:
            continue
        model_issues = sum(
            len(c.get("issues", []))
            for checks in all_results[model].values()
            for c in checks.values()
        )
        dirs_done = len(all_results[model])
        status = "✅ READY" if model_issues == 0 else f"⚠️  {model_issues} issues"
        print(f"  {model.upper():<10}  {dirs_done}/6 directions   {status}")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
