#!/usr/bin/env python3
"""
generate_report_pdf.py — Generate a comprehensive PDF EDA report for the
WMT26 multilingual MT dataset (Ar-En, Ar-Hi, Ar-Ur).

Usage:
    python generate_report_pdf.py \
        --phase1-dir eda_train \
        --phase2-dir eda_split \
        --output MT_Dataset_EDA_Report.pdf
"""

# ── Bootstrap missing packages ─────────────────────────────────────────────
import subprocess, sys

_DEPS = [
    ("fpdf2",           "fpdf"),
    ("Pillow",          "PIL"),
    ("arabic-reshaper", "arabic_reshaper"),
    ("python-bidi",     "bidi"),
]
for _pkg, _imp in _DEPS:
    try:
        __import__(_imp)
    except ImportError:
        print(f"Installing {_pkg} …", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

# ── Standard imports ───────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import os
import re
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Layout constants ───────────────────────────────────────────────────────
MARGIN    = 20          # mm
CONTENT_W = 210 - 2 * MARGIN   # 170 mm

COL_GREY   = (242, 242, 242)
COL_WHITE  = (255, 255, 255)
COL_HDR_BG = (30,  80, 140)
COL_HDR_FG = (255, 255, 255)
COL_RULE   = (190, 190, 190)
COL_MISS   = (160, 160, 160)
COL_SECBG  = (210, 225, 245)

FONT_SIZE = {1: 16, 2: 13, 3: 11}
BODY_SZ   = 9
LINE_H    = 5   # mm

# ── Font paths & download URLs ─────────────────────────────────────────────
FONT_CACHE  = Path.home() / ".local" / "share" / "fonts"
DEJAVU_REG    = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEJAVU_BOLD   = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DEJAVU_ITALIC = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")

_NOTO = {
    "NotoArabic": (
        FONT_CACHE / "NotoSansArabic-Regular.ttf",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansArabic/NotoSansArabic-Regular.ttf",
    ),
    "NotoDevanagari": (
        FONT_CACHE / "NotoSansDevanagari-Regular.ttf",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    ),
}

_FONTS_OK: set = set()   # populated by setup_fonts()


# ── Font helpers ───────────────────────────────────────────────────────────

def setup_fonts(pdf: "EDAReport") -> None:
    """Register DejaVu + try to download/register Noto fonts."""
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    pdf.add_font("DejaVu", style="",  fname=str(DEJAVU_REG))
    pdf.add_font("DejaVu", style="B", fname=str(DEJAVU_BOLD))
    if DEJAVU_ITALIC.exists():
        pdf.add_font("DejaVu", style="I", fname=str(DEJAVU_ITALIC))
    _FONTS_OK.add("DejaVu")

    for family, (local, url) in _NOTO.items():
        if not local.exists():
            log.info("Downloading %s …", local.name)
            try:
                urllib.request.urlretrieve(url, str(local))
            except Exception as exc:
                log.warning("Cannot download %s: %s — using DejaVu fallback", family, exc)
                continue
        try:
            pdf.add_font(family, style="", fname=str(local))
            _FONTS_OK.add(family)
            log.info("Font registered: %s", family)
        except Exception as exc:
            log.warning("Cannot register %s: %s", family, exc)


def pick_font(text: str) -> str:
    """Return best available font for the Unicode content of text."""
    for ch in text:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:   # Arabic / Urdu
            return "NotoArabic" if "NotoArabic" in _FONTS_OK else "DejaVu"
        if 0x0900 <= cp <= 0x097F:                               # Devanagari / Hindi
            return "NotoDevanagari" if "NotoDevanagari" in _FONTS_OK else "DejaVu"
    return "DejaVu"


def _shape(text: str) -> str:
    """Apply Arabic bidi reshaping when needed; return text unchanged otherwise."""
    for ch in text:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            try:
                import arabic_reshaper
                from bidi.algorithm import get_display
                return get_display(arabic_reshaper.reshape(text))
            except Exception:
                pass
            break
    return text


# ── I/O helpers ───────────────────────────────────────────────────────────

def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Cannot read JSON %s: %s", path, exc)
        return None


def read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        log.warning("Cannot read CSV %s: %s", path, exc)
        return []


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return ""


def csv_to_table(rows: List[Dict]) -> Tuple[List[str], List[List[str]]]:
    if not rows:
        return [], []
    headers = list(rows[0].keys())
    data    = [[r.get(h, "") for h in headers] for r in rows]
    return headers, data


def flat_stats(data: dict) -> List[Tuple[str, str]]:
    """Flatten stats.json into (label, value) pairs for kv_table."""
    SKIP = {"file", "language", "source_files", "raw_vocab", "norm_vocab"}

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows: List[Tuple[str, str]] = []
    for k, v in data.items():
        if k in SKIP:
            continue
        rows.append((k.replace("_", " ").title(), fmt(v)))

    for section in ("raw_vocab", "norm_vocab"):
        block = data.get(section)
        if not block:
            continue
        rows.append((f"── {section.replace('_', ' ').title()} ──", ""))
        for k, v in block.items():
            rows.append(("  " + k.replace("_", " ").title(), fmt(v)))

    return rows


def extract_leakage_table(md: str) -> List[Dict[str, str]]:
    """Parse LEAKAGE_REPORT.md into row dicts; fall back to pipe-table parsing."""
    rows: List[Dict[str, str]] = []

    # Try markdown pipe table rows (skip header/divider)
    for line in md.splitlines():
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if len(cells) >= 4 and not re.fullmatch(r"[-: |]+", line):
            rows.append({
                "Direction":   cells[0] if len(cells) > 0 else "",
                "Comparison":  cells[1] if len(cells) > 1 else "",
                "Duplicate %": cells[2] if len(cells) > 2 else "",
                "Trigram %":   cells[3] if len(cells) > 3 else "",
                "Risk":        cells[4] if len(cells) > 4 else "",
            })

    # Remove the header row if it crept in
    rows = [r for r in rows if r.get("Direction", "").lower() not in ("direction", "pair")]
    return rows


def extract_per_direction_assessment(md: str) -> List[Tuple[str, str]]:
    """Return [(direction, body_text), …] from the Per-Direction Assessment section."""
    # Find the section
    sec = re.search(
        r"Per.Direction Assessment(.*?)(?=\n##\s|\Z)", md,
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    if not sec:
        return []

    text = sec.group(1)
    block_re = re.compile(r"#{2,5}\s+([\w\-/ ]+)\n(.*?)(?=#{2,5}|\Z)", re.DOTALL)
    results = []
    for m in block_re.finditer(text):
        direction = m.group(1).strip()
        body      = m.group(2).strip()
        if body:
            results.append((direction, body))
    return results


# ── PDF class ──────────────────────────────────────────────────────────────

class EDAReport(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self._doc_title = "Multilingual MT Dataset — EDA Report"

    # ── Header / footer ──────────────────────────────────────────────────

    def header(self) -> None:
        if self.page_no() <= 3:   # cover + TOC pages
            return
        self.set_font("DejaVu", size=7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, self._doc_title, align="R")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("DejaVu", size=7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    # ── Section title ────────────────────────────────────────────────────

    def section_title(self, text: str, level: int = 1) -> None:
        self.ln(4 if level > 1 else 8)
        size = FONT_SIZE.get(level, BODY_SZ)

        if level == 1:
            self.set_font("DejaVu", style="B", size=size)
            self.set_text_color(*COL_HDR_BG)
            self.multi_cell(CONTENT_W, 9, text, align="L")
            self.set_draw_color(*COL_RULE)
            y = self.get_y()
            self.line(MARGIN, y, MARGIN + CONTENT_W, y)
            self.ln(3)
        elif level == 2:
            self.set_font("DejaVu", style="B", size=size)
            self.set_text_color(50, 50, 50)
            self.multi_cell(CONTENT_W, 7, text, align="L")
            self.ln(2)
        else:
            self.set_font("DejaVu", style="B", size=size)
            self.set_text_color(70, 70, 70)
            self.multi_cell(CONTENT_W, 6, text, align="L")
            self.ln(1)

        self.set_text_color(0, 0, 0)
        self.start_section(text, level=level - 1)

    # ── Key-value table ──────────────────────────────────────────────────

    def kv_table(self, rows: List[Tuple[str, str]]) -> None:
        col_k = CONTENT_W * 0.45
        col_v = CONTENT_W * 0.55

        for i, (k, v) in enumerate(rows):
            if k.startswith("──"):          # sub-section divider
                self.set_font("DejaVu", style="B", size=BODY_SZ)
                self.set_fill_color(*COL_SECBG)
                self.cell(CONTENT_W, LINE_H, k, border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
                continue
            fill = (i % 2 == 0)
            self.set_fill_color(*(COL_GREY if fill else COL_WHITE))
            self.set_font("DejaVu", style="B", size=BODY_SZ)
            self.cell(col_k, LINE_H, k, border=0, new_x="RIGHT", new_y="TOP", fill=fill)
            self.set_font("DejaVu", size=BODY_SZ)
            self.cell(col_v, LINE_H, str(v), border=0, new_x="LMARGIN", new_y="NEXT", fill=fill)
        self.ln(2)

    # ── Data table ───────────────────────────────────────────────────────

    def data_table(
        self,
        headers: List[str],
        rows:    List[List[str]],
        col_widths: Optional[List[float]] = None,
        max_rows:   Optional[int]         = None,
        zebra:      bool                  = True,
    ) -> None:
        if not headers:
            return

        n = len(headers)
        if col_widths is None:
            col_widths = [1.0] * n
        total = sum(col_widths)
        col_widths = [w * CONTENT_W / total for w in col_widths]

        # Header row
        self.set_font("DejaVu", style="B", size=BODY_SZ)
        self.set_fill_color(*COL_HDR_BG)
        self.set_text_color(*COL_HDR_FG)
        for i, hdr in enumerate(headers):
            ln_mode = "LMARGIN" if i == n - 1 else "RIGHT"
            ny_mode = "NEXT"    if i == n - 1 else "TOP"
            self.cell(col_widths[i], LINE_H + 1, str(hdr)[:30],
                      border=0, new_x=ln_mode, new_y=ny_mode, fill=True)
        self.set_text_color(0, 0, 0)

        # Data rows
        display = rows[:max_rows] if max_rows else rows
        for ri, row in enumerate(display):
            fill = zebra and (ri % 2 == 0)
            self.set_fill_color(*(COL_GREY if fill else COL_WHITE))
            for ci in range(n):
                val  = str(row[ci]) if ci < len(row) else ""
                txt  = _shape(val)
                font = pick_font(txt)
                self.set_font(font if font in _FONTS_OK else "DejaVu", size=BODY_SZ)
                ln_mode = "LMARGIN" if ci == n - 1 else "RIGHT"
                ny_mode = "NEXT"    if ci == n - 1 else "TOP"
                # Truncate very long cells to avoid overflow
                self.cell(col_widths[ci], LINE_H, txt[:40],
                          border=0, new_x=ln_mode, new_y=ny_mode, fill=fill)

        if max_rows and len(rows) > max_rows:
            italic = "I" if DEJAVU_ITALIC.exists() else ""
            self.set_font("DejaVu", style=italic, size=BODY_SZ - 1)
            self.set_text_color(*COL_MISS)
            self.cell(0, LINE_H, f"  … {len(rows) - max_rows} more rows not shown",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
        self.ln(2)

    # ── Image ────────────────────────────────────────────────────────────

    def embed_image(self, path: str, caption: Optional[str] = None) -> None:
        if not os.path.exists(path):
            self.miss(path)
            return
        try:
            with Image.open(path) as img:
                w_px, h_px = img.size
            aspect = h_px / w_px
            w_mm   = CONTENT_W
            h_mm   = w_mm * aspect
            max_h  = (297 - 2 * MARGIN) * 0.72
            if h_mm > max_h:
                h_mm = max_h
                w_mm = h_mm / aspect
            x = MARGIN + (CONTENT_W - w_mm) / 2
            # Ensure enough space; add page if not
            if self.get_y() + h_mm + 10 > 297 - MARGIN:
                self.add_page()
            self.image(path, x=x, w=w_mm, h=h_mm)
            if caption:
                italic = "I" if "DejaVuI" in (self.fonts or {}) or DEJAVU_ITALIC.exists() else ""
                self.set_font("DejaVu", style=italic, size=BODY_SZ - 1)
                self.set_text_color(100, 100, 100)
                self.cell(0, 4, caption, align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0, 0, 0)
            self.ln(4)
        except Exception as exc:
            log.warning("Cannot embed image %s: %s", path, exc)
            self.miss(path)

    # ── Missing-file note ────────────────────────────────────────────────

    def miss(self, path: str) -> None:
        log.warning("Missing: %s", path)
        italic = "I" if DEJAVU_ITALIC.exists() else ""
        self.set_font("DejaVu", style=italic, size=BODY_SZ - 1)
        self.set_text_color(*COL_MISS)
        self.cell(0, LINE_H, f"[File not found: {os.path.basename(path)}]",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text: str) -> None:
        self.set_font("DejaVu", size=BODY_SZ)
        self.multi_cell(CONTENT_W, 4, text, align="L")


# ── TOC renderer ──────────────────────────────────────────────────────────

def _render_toc(pdf: EDAReport, outline: list) -> None:
    pdf.set_font("DejaVu", style="B", size=14)
    pdf.set_text_color(*COL_HDR_BG)
    pdf.cell(0, 10, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*COL_RULE)
    y = pdf.get_y()
    pdf.line(MARGIN, y, MARGIN + CONTENT_W, y)
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    for entry in outline:
        lvl    = entry.level
        indent = lvl * 5
        sz     = 10 if lvl == 0 else 8
        pdf.set_font("DejaVu", style="B" if lvl == 0 else "", size=sz)
        label  = entry.name
        pg     = str(entry.page_number)
        avail  = CONTENT_W - indent - 12
        pdf.set_x(MARGIN + indent)
        pdf.cell(avail, 5, label, new_x="RIGHT", new_y="TOP")
        pdf.cell(12, 5, pg, align="R", new_x="LMARGIN", new_y="NEXT")


# ── Section builders ──────────────────────────────────────────────────────

def build_cover(pdf: EDAReport, p1: str, p2: str) -> None:
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("DejaVu", style="B", size=22)
    pdf.set_text_color(*COL_HDR_BG)
    pdf.multi_cell(CONTENT_W, 13,
                   "Multilingual MT Dataset\nExploratory Data Analysis Report",
                   align="C")
    pdf.ln(6)
    pdf.set_font("DejaVu", size=15)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(CONTENT_W, 9, "Ar-En  ·  Ar-Hi  ·  Ar-Ur", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_draw_color(*COL_RULE)
    y = pdf.get_y()
    pdf.line(MARGIN + 15, y, MARGIN + CONTENT_W - 15, y)
    pdf.ln(12)
    pdf.set_font("DejaVu", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(CONTENT_W, 6, f"Date: {date.today().strftime('%d %B %Y')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.cell(CONTENT_W, 6, f"Phase 1 — Training Analysis:   {os.path.basename(p1)}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(CONTENT_W, 6, f"Phase 2 — Cross-Split Analysis: {os.path.basename(p2)}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(14)
    pdf.set_font("DejaVu", size=9)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(CONTENT_W, 5,
        "Sections covered:\n"
        "  Part 1 — Training Set Analysis (alignment, per-file stats, language aggregation)\n"
        "  Part 2 — Cross-Split Analysis (OOV, leakage, n-gram overlap, semantic similarity)\n"
        "  Part 3 — Overall Assessment",
        align="C")
    pdf.set_text_color(0, 0, 0)


# ── Part 1 ────────────────────────────────────────────────────────────────

def build_alignment_check(pdf: EDAReport, p1: str) -> None:
    pdf.add_page()
    pdf.section_title("Part 1: Training Set Analysis", level=1)
    pdf.section_title("1.1  Alignment Check", level=2)

    path = os.path.join(p1, "alignment_check.json")
    data = read_json(path)
    if data is None:
        pdf.miss(path)
        return

    items = data if isinstance(data, list) else [data]
    headers = ["Pair", "Source Lines", "Target Lines", "Delta", "Status"]
    rows = []
    for e in items:
        rows.append([
            e.get("pair", ""),
            str(e.get("src_sentences", "")),
            str(e.get("tgt_sentences", "")),
            str(e.get("delta", "")),
            "PASS ✓" if e.get("aligned") else "FAIL ✗",
        ])
    pdf.data_table(headers, rows, col_widths=[3, 3, 3, 2, 2])


def build_summary_table(pdf: EDAReport, p1: str) -> None:
    pdf.section_title("1.2  Per-File Summary", level=2)
    path = os.path.join(p1, "summary.csv")
    rows = read_csv(path)
    if not rows:
        pdf.miss(path)
        return
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)


def build_per_file(pdf: EDAReport, p1: str) -> None:
    pdf.section_title("1.3  Per-File Detailed Statistics", level=2)
    per_file_dir = os.path.join(p1, "per_file")
    if not os.path.isdir(per_file_dir):
        pdf.miss(per_file_dir)
        return

    PLOTS = [
        ("sentence_length_hist.png",  "Sentence Length Distribution"),
        ("token_freq_hist.png",       "Token Frequency Distribution"),
        ("zipf_curve.png",            "Zipf Curve"),
        ("token_char_len_hist.png",   "Token Character Length Distribution"),
    ]

    for subdir in sorted(os.listdir(per_file_dir)):
        folder = os.path.join(per_file_dir, subdir)
        if not os.path.isdir(folder):
            continue
        pdf.add_page()
        pdf.section_title(f"File: {subdir}", level=3)
        stats = read_json(os.path.join(folder, "stats.json"))
        if stats:
            pdf.kv_table(flat_stats(stats))
        else:
            pdf.miss(os.path.join(folder, "stats.json"))
        for fname, caption in PLOTS:
            pdf.embed_image(os.path.join(folder, fname), caption=caption)


def build_per_language(pdf: EDAReport, p1: str) -> None:
    pdf.section_title("1.4  Language-Level Aggregation", level=2)
    lang_dir = os.path.join(p1, "per_language")
    if not os.path.isdir(lang_dir):
        pdf.miss(lang_dir)
        return

    PLOTS = [
        ("sentence_length_hist.png",  "Sentence Length Distribution"),
        ("token_freq_hist.png",       "Token Frequency Distribution"),
        ("zipf_curve.png",            "Zipf Curve"),
        ("token_char_len_hist.png",   "Token Character Length Distribution"),
    ]

    for lang in sorted(os.listdir(lang_dir)):
        folder = os.path.join(lang_dir, lang)
        if not os.path.isdir(folder):
            continue
        pdf.add_page()
        pdf.section_title(f"Language: {lang.capitalize()}", level=3)
        stats = read_json(os.path.join(folder, "stats.json"))
        if stats:
            pdf.kv_table(flat_stats(stats))
        else:
            pdf.miss(os.path.join(folder, "stats.json"))
        for fname, caption in PLOTS:
            pdf.embed_image(os.path.join(folder, fname), caption=caption)


# ── Part 2 helpers ────────────────────────────────────────────────────────

def _p2csv(p2: str, name: str) -> str:
    return os.path.join(p2, "csv", name)

def _p2plot(p2: str, name: str) -> str:
    return os.path.join(p2, "plots", name)


# ── Part 2 section builders ───────────────────────────────────────────────

def build_split_statistics(pdf: EDAReport, p2: str) -> None:
    pdf.add_page()
    pdf.section_title("Part 2: Cross-Split Analysis", level=1)
    pdf.section_title("2.1  Dataset Statistics", level=2)
    rows = read_csv(_p2csv(p2, "split_statistics.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)


def build_vocab_coverage(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.2  Vocabulary Coverage", level=2)
    rows = read_csv(_p2csv(p2, "vocabulary_coverage.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)
    pdf.embed_image(_p2plot(p2, "vocabulary_coverage_barplot.png"),
                    caption="Vocabulary Coverage by Direction and Split")


def build_oov(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.3  Out-of-Vocabulary (OOV) Analysis", level=2)

    rows = read_csv(_p2csv(p2, "oov_statistics.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)

    pdf.section_title("Top 50 OOV Tokens", level=3)
    oov = read_csv(_p2csv(p2, "top_oov_tokens.csv"))
    headers, data = csv_to_table(oov)
    pdf.data_table(headers, data, max_rows=50)

    pdf.embed_image(_p2plot(p2, "oov_rate_barplot.png"),
                    caption="OOV Rate by Direction and Split")


def build_duplicates(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.4  Exact Duplicate / Leakage Detection", level=2)

    rows = read_csv(_p2csv(p2, "exact_duplicate_summary.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)

    pdf.section_title("Duplicate Sentences (first 50 rows)", level=3)
    dup = read_csv(_p2csv(p2, "exact_duplicate_sentences.csv"))
    if dup:
        headers, data = csv_to_table(dup)
        pdf.data_table(headers, data, max_rows=50)
    else:
        pdf.miss(_p2csv(p2, "exact_duplicate_sentences.csv"))

    pdf.section_title("Leakage Risk Summary", level=3)
    md = read_text(os.path.join(p2, "LEAKAGE_REPORT.md"))
    if md:
        leak_rows = extract_leakage_table(md)
        if leak_rows:
            headers, data = csv_to_table(leak_rows)
            pdf.data_table(headers, data)
        else:
            # Fallback: plain text
            pdf.set_font("DejaVu", size=BODY_SZ - 1)
            for line in md.splitlines()[:80]:
                if line.strip():
                    pdf.multi_cell(CONTENT_W, 4, line.strip(), align="L")
    else:
        pdf.miss(os.path.join(p2, "LEAKAGE_REPORT.md"))


def build_ngram(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.5  N-Gram Overlap", level=2)
    rows = read_csv(_p2csv(p2, "ngram_overlap.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)
    pdf.embed_image(_p2plot(p2, "ngram_overlap_heatmap.png"),
                    caption="N-Gram Overlap Heatmap (Train vs Eval)")


def build_length_dist(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.6  Sentence Length Distribution", level=2)
    rows = read_csv(_p2csv(p2, "sentence_length_distribution.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)
    pdf.embed_image(_p2plot(p2, "sentence_length_distribution_comparison.png"),
                    caption="Sentence Length: Train vs Dev vs DevTest")


def build_freq_shift(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.7  Vocabulary Frequency Shift", level=2)
    all_rows = read_csv(_p2csv(p2, "frequency_shift.csv"))
    if not all_rows:
        pdf.miss(_p2csv(p2, "frequency_shift.csv"))
        return

    headers = list(all_rows[0].keys())
    by_dir: Dict[str, List] = defaultdict(list)
    for r in all_rows:
        by_dir[r.get("direction", "unknown")].append(r)

    for direction in sorted(by_dir):
        pdf.section_title(f"Direction: {direction}", level=3)
        data = [[r.get(h, "") for h in headers] for r in by_dir[direction][:20]]
        pdf.data_table(headers, data)


def build_semantic(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.8  Semantic Similarity (TF-IDF)", level=2)
    rows = read_csv(_p2csv(p2, "semantic_similarity.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)
    pdf.embed_image(_p2plot(p2, "semantic_similarity_histogram.png"),
                    caption="TF-IDF Semantic Similarity: Train vs Eval")


def build_divergence(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.9  Distribution Divergence (JSD / KLD)", level=2)
    rows = read_csv(_p2csv(p2, "distribution_divergence.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)
    pdf.embed_image(_p2plot(p2, "distribution_divergence_barplot.png"),
                    caption="Jensen-Shannon & KL Divergence by Direction")


def build_lang_summary(pdf: EDAReport, p2: str) -> None:
    pdf.section_title("2.10  Cross-Language Summary", level=2)
    rows = read_csv(_p2csv(p2, "language_summary.csv"))
    headers, data = csv_to_table(rows)
    pdf.data_table(headers, data)


# ── Part 3 ────────────────────────────────────────────────────────────────

def build_assessment(pdf: EDAReport, p2: str) -> None:
    pdf.add_page()
    pdf.section_title("Part 3: Overall Assessment", level=1)

    path = os.path.join(p2, "TRAIN_DEV_ANALYSIS_REPORT.md")
    md = read_text(path)
    if not md:
        pdf.miss(path)
        return

    blocks = extract_per_direction_assessment(md)
    if blocks:
        for direction, body in blocks:
            pdf.section_title(f"Direction: {direction}", level=2)
            pdf.set_font("DejaVu", size=BODY_SZ)
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped:
                    pdf.ln(2)
                elif stripped.startswith("|"):
                    # Markdown table row — render as plain text
                    pdf.multi_cell(CONTENT_W, 4, stripped, align="L")
                else:
                    pdf.multi_cell(CONTENT_W, 4, stripped, align="L")
            pdf.ln(3)
    else:
        # Fallback: render first 150 lines as plain text
        pdf.set_font("DejaVu", size=BODY_SZ)
        for line in md.splitlines()[:150]:
            stripped = line.strip()
            if not stripped:
                pdf.ln(2)
                continue
            if stripped.startswith("##"):
                pdf.set_font("DejaVu", style="B", size=BODY_SZ + 1)
                pdf.multi_cell(CONTENT_W, 5, stripped.lstrip("#").strip(), align="L")
                pdf.set_font("DejaVu", size=BODY_SZ)
            else:
                pdf.multi_cell(CONTENT_W, 4, stripped, align="L")


# ── CLI + orchestration ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a comprehensive PDF EDA report for the WMT multilingual MT dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase1-dir", default="eda_train",
                   help="Path to Phase 1 EDA output directory")
    p.add_argument("--phase2-dir", default="eda_split",
                   help="Path to Phase 2 EDA output directory")
    p.add_argument("--output", default="MT_Dataset_EDA_Report.pdf",
                   help="Output PDF file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    p1  = os.path.abspath(args.phase1_dir)
    p2  = os.path.abspath(args.phase2_dir)
    out = os.path.abspath(args.output)

    log.info("Phase 1 dir : %s", p1)
    log.info("Phase 2 dir : %s", p2)
    log.info("Output      : %s", out)

    if not os.path.isdir(p1):
        log.error("Phase 1 directory not found: %s", p1)
        sys.exit(1)
    if not os.path.isdir(p2):
        log.error("Phase 2 directory not found: %s", p2)
        sys.exit(1)

    pdf = EDAReport()
    setup_fonts(pdf)
    pdf.set_title("Multilingual MT Dataset — EDA Report")
    pdf.set_author("WMT26 Arabic-Asian MT Challenge")

    # ── Cover (page 1) ────────────────────────────────────────────────────
    build_cover(pdf, p1, p2)

    # ── TOC placeholder (pages 2-3) ───────────────────────────────────────
    pdf.add_page()
    pdf.insert_toc_placeholder(_render_toc, pages=1)

    # ── Part 1 ────────────────────────────────────────────────────────────
    build_alignment_check(pdf, p1)
    build_summary_table(pdf, p1)
    build_per_file(pdf, p1)
    build_per_language(pdf, p1)

    # ── Part 2 ────────────────────────────────────────────────────────────
    build_split_statistics(pdf, p2)
    build_vocab_coverage(pdf, p2)
    build_oov(pdf, p2)
    build_duplicates(pdf, p2)
    build_ngram(pdf, p2)
    build_length_dist(pdf, p2)
    build_freq_shift(pdf, p2)
    build_semantic(pdf, p2)
    build_divergence(pdf, p2)
    build_lang_summary(pdf, p2)

    # ── Part 3 ────────────────────────────────────────────────────────────
    build_assessment(pdf, p2)

    # ── Save ──────────────────────────────────────────────────────────────
    pdf.output(out)
    size_mb = round(os.path.getsize(out) / 1024 / 1024, 1)
    log.info("Report saved → %s  (%s MB)", out, size_mb)


if __name__ == "__main__":
    main()
