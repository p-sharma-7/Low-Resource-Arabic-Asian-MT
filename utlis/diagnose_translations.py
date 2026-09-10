"""
diagnose_translations.py — Print sample translations to spot obvious errors.

Usage (run from your wmt/ directory):
    python diagnose_translations.py

Reads the saved hypothesis files from outputs/zero_shot/nllb/ and prints
5 samples side-by-side with the reference for visual inspection.
"""
import os
import unicodedata

OUTPUT_DIR = os.environ.get(
    "WMT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
HYP_DIR    = os.path.join(OUTPUT_DIR, "outputs", "zero_shot", "nllb")
DATA_DIR   = os.path.join(OUTPUT_DIR, "dataset")

# (src_lang, tgt_lang, dataset_folder, pair_suffix)
# pair_suffix is always the folder name lowercased e.g. "ar-en" — never "en-ar"
DIRECTIONS = [
    ("ar", "en", "Ar-En", "ar-en"),
    ("en", "ar", "Ar-En", "ar-en"),   # same folder/suffix, just source/target swap
    ("ar", "hi", "Ar-Hi", "ar-hi"),
    ("hi", "ar", "Ar-Hi", "ar-hi"),
    ("ar", "ur", "Ar-Ur", "ar-ur"),
    ("ur", "ar", "Ar-Ur", "ar-ur"),
]

N_SAMPLES = 3

def read(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]

for src_lang, tgt_lang, folder, pair_suffix in DIRECTIONS:
    direction = f"{src_lang}-{tgt_lang}"
    hyp_file  = os.path.join(HYP_DIR, f"hyp_nllb_{direction}_dev.txt")

    # File names use the fixed pair_suffix (always ar-xx), not the direction
    src_file  = os.path.join(DATA_DIR, folder, f"dev_{src_lang}_{pair_suffix}.txt")
    ref_file  = os.path.join(DATA_DIR, folder, f"dev_{tgt_lang}_{pair_suffix}.txt")

    if not os.path.exists(hyp_file):
        print(f"\n[SKIP — no hypothesis file yet] {hyp_file}")
        continue

    hyps = read(hyp_file)
    refs = read(ref_file)
    srcs = read(src_file)

    print(f"\n{'='*72}")
    print(f"  {direction.upper()}  |  {len(hyps)} hypotheses  |  hyp: {os.path.basename(hyp_file)}")
    print(f"{'='*72}")

    for i in range(min(N_SAMPLES, len(hyps))):
        h = hyps[i]; r = refs[i]; s = srcs[i]
        h_words = len(h.split()); r_words = len(r.split())
        ratio   = h_words / max(r_words, 1)

        print(f"\n  [{i}] SRC ({len(s.split())}w) : {s[:100]}")
        print(f"  [{i}] HYP ({h_words}w)  : {h[:100]}")
        print(f"  [{i}] REF ({r_words}w)  : {r[:100]}")
        print(f"       len ratio HYP/REF = {ratio:.2f}x"
              + ("  ⚠ LOOP?" if ratio > 2.0 else "  ✓"))

        # Detect which Unicode scripts appear in the hypothesis
        scripts = set()
        for ch in h:
            name = unicodedata.name(ch, "")
            if "ARABIC" in name:    scripts.add("Arabic")
            elif "LATIN" in name:   scripts.add("Latin")
            elif "DEVANA" in name:  scripts.add("Devanagari")
        print(f"       scripts in HYP     : {scripts if scripts else '(none detected)'}")

print()