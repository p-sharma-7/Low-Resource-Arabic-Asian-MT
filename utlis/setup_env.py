#!/usr/bin/env python3
"""
setup_env.py — Environment validation and conflict-resolution for WMT26 pipeline.

Run this ONCE before running run_pipeline.py to verify every dependency is
installed correctly.

Usage:
    python setup_env.py          # check only
    python setup_env.py --fix    # auto-fix detected problems
"""

import importlib
import importlib.metadata
import subprocess
import sys
import argparse

OK   = "\033[32m✓\033[0m"
WARN = "\033[33m⚠\033[0m"
FAIL = "\033[31m✗\033[0m"


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])


# ─────────────────────────────────────────────────────────────────────────────
# Check helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_python_version():
    v = sys.version_info
    ok = v >= (3, 10)
    mark = OK if ok else FAIL
    print(f"  {mark}  Python {v.major}.{v.minor}.{v.micro}  (need ≥ 3.10)")
    return ok


def check_package(import_name: str, dist_name: str, min_version: str = None):
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                ver = importlib.metadata.version(dist_name)
            except Exception:
                ver = "unknown"
        mark = OK
        note = f"v{ver}"
        if min_version and ver != "unknown":
            from packaging.version import Version
            if Version(ver) < Version(min_version):
                mark = WARN
                note += f"  (need ≥ {min_version})"
        print(f"  {mark}  {dist_name:<30} {note}")
        return True
    except ImportError:
        print(f"  {FAIL}  {dist_name:<30} NOT INSTALLED")
        return False


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            n  = torch.cuda.device_count()
            nm = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  {OK}  CUDA  {n} GPU(s) — {nm}  ({gb:.0f} GB)")
        else:
            print(f"  {WARN}  CUDA not available — pipeline will run on CPU (very slow)")
        return True
    except ImportError:
        print(f"  {FAIL}  torch not installed — cannot check CUDA")
        return False


def check_unbabel_comet(fix: bool = False):
    """
    Detect the comet-ml vs unbabel-comet namespace conflict.
    Returns True if unbabel-comet is importable correctly.
    """
    print("\n── COMET conflict check ─────────────────────────────────────────")

    # 1. Is unbabel-comet installed at all?
    try:
        dist = importlib.metadata.distribution("unbabel-comet")
        print(f"  {OK}  unbabel-comet dist  v{dist.version}  (installed)")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {FAIL}  unbabel-comet dist  NOT INSTALLED")
        if fix:
            print("       → pip install unbabel-comet")
            pip("install", "unbabel-comet")
        return False

    # 2. Is comet-ml also installed? (creates the namespace conflict)
    comet_ml_installed = False
    try:
        importlib.metadata.distribution("comet-ml")
        comet_ml_installed = True
        print(f"  {WARN}  comet-ml is also installed — namespace conflict possible")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {OK}  comet-ml  NOT installed (no conflict)")

    # 3. Can we actually import download_model?
    try:
        from comet import download_model, load_from_checkpoint  # noqa: F401
        import inspect
        inspect.signature(load_from_checkpoint)   # comet-ml won't have this
        print(f"  {OK}  `from comet import download_model` — OK")
        return True
    except (ImportError, Exception):
        pass

    try:
        from comet.models import download_model, load_from_checkpoint  # noqa: F401
        print(f"  {OK}  `from comet.models import download_model` — OK (v1.x path)")
        return True
    except (ImportError, Exception):
        pass

    print(f"  {FAIL}  unbabel-comet import BROKEN (shadowed by comet-ml)")
    if fix:
        print("       → force-reinstalling unbabel-comet on top of comet-ml …")
        pip("install", "unbabel-comet", "--force-reinstall")
        # verify
        try:
            importlib.invalidate_caches()
            import importlib as _il
            import comet as _c
            _il.reload(_c)
            from comet import download_model  # noqa: F401
            print(f"  {OK}  Fixed — unbabel-comet now on top")
            return True
        except Exception:
            print(
                f"  {FAIL}  Still broken after reinstall.\n"
                "       The safest fix is a clean conda env:\n"
                "         conda create -n wmt python=3.10 -y\n"
                "         conda activate wmt\n"
                "         pip install -r requirements.txt"
            )
            return False
    else:
        print(
            "\n       To fix, run ONE of:\n"
            "         python setup_env.py --fix\n"
            "         pip install unbabel-comet --force-reinstall\n"
            "         pip uninstall comet-ml -y && pip install unbabel-comet"
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="Auto-install / reinstall broken packages")
    args = parser.parse_args()

    print("\n═══════════════════════════════════════════════════════════")
    print("  WMT26 Pipeline — environment check")
    print("═══════════════════════════════════════════════════════════\n")

    print("── Python & CUDA ────────────────────────────────────────────")
    check_python_version()
    check_cuda()

    print("\n── Core packages ────────────────────────────────────────────")
    results = []
    pkgs = [
        ("torch",           "torch",           "2.2.0"),
        ("transformers",    "transformers",    "4.40.0"),
        ("accelerate",      "accelerate",      "0.27.0"),
        ("datasets",        "datasets",        "2.18.0"),
        ("peft",            "peft",            "0.10.0"),
        ("trl",             "trl",             "0.8.0"),
        ("bitsandbytes",    "bitsandbytes",    "0.43.0"),
        ("sacrebleu",       "sacrebleu",       "2.4.0"),
        ("sentencepiece",   "sentencepiece",   None),
    ]
    for imp, dist, minv in pkgs:
        ok = check_package(imp, dist, minv)
        if not ok and args.fix:
            print(f"       → pip install {dist}")
            pip("install", dist)
        results.append(ok)

    comet_ok = check_unbabel_comet(fix=args.fix)
    results.append(comet_ok)

    print("\n═══════════════════════════════════════════════════════════")
    n_fail = results.count(False)
    if n_fail == 0:
        print(f"  {OK}  All checks passed — ready to run run_pipeline.py")
    else:
        print(f"  {FAIL}  {n_fail} issue(s) detected.")
        if not args.fix:
            print("       Re-run with  --fix  to auto-resolve them.")
    print("═══════════════════════════════════════════════════════════\n")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
