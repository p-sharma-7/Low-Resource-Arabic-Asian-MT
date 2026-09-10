"""
config.py — Central configuration for WMT26 Arabic-Asian MT Challenge.

All paths, model names, language codes, and hyper-parameters live here so
every other module imports from a single source of truth.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
#
# _REPO_ROOT = the directory that contains config.py (same dir as the scripts).
# Works whether scripts live in  challenge/wmt/  OR directly in  challenge/.
# Override with env var WMT_ROOT for non-standard layouts:
#   export WMT_ROOT=/mnt/storage/Pushkar/challenge
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT    = os.environ.get(
    "WMT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATASET_DIR   = os.path.join(_REPO_ROOT, "dataset")
OUTPUT_DIR    = os.path.join(_REPO_ROOT, "outputs")
LOG_DIR       = os.path.join(_REPO_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "checkpoints")

for _d in (OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Model identifiers (HuggingFace Hub)
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "nllb":    "facebook/nllb-200-3.3B",
    "madlad":  "google/madlad400-10b-mt",
    "gemmax2": "ModelSpace/GemmaX2-28-9B-v0.1",
}

# ─────────────────────────────────────────────────────────────────────────────
# Language codes per model family
# ─────────────────────────────────────────────────────────────────────────────
NLLB_LANG_CODES = {
    "ar": "arb_Arab",   # Modern Standard Arabic ISO 639-3
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ur": "urd_Arab",
}

MADLAD_LANG_CODES = {
    "ar": "<2ar>",
    "en": "<2en>",
    "hi": "<2hi>",
    "ur": "<2ur>",
}

LANG_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
}

# ─────────────────────────────────────────────────────────────────────────────
# Six translation directions
# Tuple: (src_lang, tgt_lang)
# Both forward and reverse directions share the same dataset folder.
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATION_DIRECTIONS = [
    ("ar", "en"),   # Ar → En
    ("en", "ar"),   # En → Ar
    ("ar", "hi"),   # Ar → Hi
    ("hi", "ar"),   # Hi → Ar
    ("ar", "ur"),   # Ar → Ur
    ("ur", "ar"),   # Ur → Ar
]

# Maps any direction to its dataset folder (Arabic is always the "first" lang)
PAIR_FOLDER_MAP = {
    ("ar", "en"): "Ar-En",
    ("en", "ar"): "Ar-En",
    ("ar", "hi"): "Ar-Hi",
    ("hi", "ar"): "Ar-Hi",
    ("ar", "ur"): "Ar-Ur",
    ("ur", "ar"): "Ar-Ur",
}

# ─────────────────────────────────────────────────────────────────────────────
# Max token lengths
# ─────────────────────────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = {
    "nllb":    512,
    "madlad":  512,
    "gemmax2": 512,   # kept at 512 for batch efficiency (model supports 4096)
}

# ─────────────────────────────────────────────────────────────────────────────
# Training hyper-parameters — NLLB-200-3.3B  (Seq2SeqTrainer)
# ─────────────────────────────────────────────────────────────────────────────
NLLB_TRAINING_ARGS = dict(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,       # effective batch = 32
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    predict_with_generate=True,
    generation_max_length=512,
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="comet22",
    greater_is_better=True,
    logging_steps=100,
    dataloader_num_workers=4,
    seed=SEED,
    report_to="none",
)

# ─────────────────────────────────────────────────────────────────────────────
# Training hyper-parameters — MADLAD-400-10B  (Seq2SeqTrainer)
# Uses gradient-checkpointing + 8-bit Adam to fit within 48 GB VRAM.
# ─────────────────────────────────────────────────────────────────────────────
MADLAD_TRAINING_ARGS = dict(
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,       # effective batch = 32
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",   # fast: no generate/COMET during training
    greater_is_better=False,
    logging_steps=50,
    dataloader_num_workers=4,
    seed=SEED,
    report_to="none",
)

# LoRA config for MADLAD (T5 encoder-decoder attention + FFN projections)
MADLAD_LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Training hyper-parameters — GemmaX2-28-9B  (SFTTrainer / LoRA)
# Quantised to 4-bit (QLoRA) so the adapter fits easily within 48 GB.
# ─────────────────────────────────────────────────────────────────────────────
GEMMAX2_TRAINING_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,       # effective batch = 32
    learning_rate=2e-4,
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",   # COMET too slow for causal-LM mid-training
    greater_is_better=False,
    seed=SEED,
    report_to="none",
    # max_seq_length removed from SFTConfig in trl>=0.9 — passed to SFTTrainer directly
)

# LoRA adapter config (applied to all attention + FFN projections)
LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────
COMET_MODEL       = "Unbabel/wmt22-comet-da"
COMET_KIWI_MODEL  = "Unbabel/wmt23-cometkiwi-da"

# SacreBLEU tokenisers per target language
SACREBLEU_TOKENIZER = {
    "ar": "intl",
    "en": "13a",
    "hi": "flores101",
    "ur": "flores101",
}