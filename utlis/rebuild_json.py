"""
rebuild_json.py — Reconstruct all_results_dev.json from existing metric .txt files.

Run from the wmt directory:
    python rebuild_json.py
"""
import glob
import json
import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE     = os.path.join(_REPO_ROOT, "outputs", "eval_finetuned")
OUT_JSON     = os.path.join(BASE, "all_results_dev.json")
OUT_JSON_TMP = os.path.join(_REPO_ROOT, "all_results_dev_rebuilt.json")

MODEL_DIRS = {
    "nllb":    "nllb",
    "madlad":  "madlad",
    "gemmax2": "gemmax2",
}

# Metric line pattern:  "  COMET-22       : 0.8423"
_METRIC_RE = re.compile(r"^\s+([\w\-\+]+)\s*:\s*([\d.]+)\s*$")


def parse_metrics(path: str) -> dict:
    metrics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _METRIC_RE.match(line)
            if m:
                metrics[m.group(1)] = float(m.group(2))
    return metrics


def direction_from_filename(fname: str, model: str) -> str:
    # e.g. metrics_nllb_finetuned_ar-en_dev.txt  →  ar-en
    prefix = f"metrics_{model}_finetuned_"
    suffix = "_dev.txt"
    name = os.path.basename(fname)
    return name[len(prefix):-len(suffix)]


# Load existing JSON so any already-correct entries are preserved
if os.path.exists(OUT_JSON):
    with open(OUT_JSON, encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

for model_key, subdir in MODEL_DIRS.items():
    model_dir = os.path.join(BASE, subdir)
    pattern   = os.path.join(model_dir, f"metrics_{model_key}_finetuned_*_dev.txt")
    files     = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARN] No metric files found for {model_key} in {model_dir}")
        continue

    parsed = {}
    for fpath in files:
        direction = direction_from_filename(fpath, model_key)
        metrics   = parse_metrics(fpath)
        if metrics:
            parsed[direction] = metrics
            print(f"  {model_key}/{direction}: {metrics}")
        else:
            print(f"[WARN] No metrics parsed from {fpath}")

    existing = results.get(model_key, {})
    existing.update(parsed)
    results[model_key] = existing

try:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUT_JSON}")
except PermissionError:
    with open(OUT_JSON_TMP, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Permission denied on outputs dir] Saved to fallback → {OUT_JSON_TMP}")
    print(f"Copy it manually:  cp {OUT_JSON_TMP} {OUT_JSON}")
for model, dirs in results.items():
    print(f"  {model}: {list(dirs.keys())}")
