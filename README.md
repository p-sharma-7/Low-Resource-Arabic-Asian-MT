# Arabic–Asian Machine Translation — NLP-IIT-Patna @ WMT 2026

Code and artefacts for our submission to the WMT 2026 shared task on Low-Resource
Arabic–Asian Machine Translation. We entered six directions — Arabic↔English,
Arabic↔Hindi and Arabic↔Urdu (Sub-Tasks 1A, 1B, 1E, 2A, 2B, 2E) — and fine-tuned
three pretrained multilingual models, one run per direction.

| System | Checkpoint | Adaptation | Submitted as |
|---|---|---|---|
| MADLAD-400 10B | `google/madlad400-10b-mt` | LoRA, frozen 8-bit base | primary |
| NLLB-200 3.3B | `facebook/nllb-200-3.3B` | full fine-tuning, bf16 | contrastive-1 |
| GemmaX2-28-9B | `ModelSpace/GemmaX2-28-9B-v0.1` | QLoRA, 4-bit NF4 | contrastive-2 |

Everything was trained on a single RTX 6000 Ada (49 GB) with seed 42. The fine-tuned
weights are on the Hub:
[pushkarsharma/wmt26-arabic-asian-mt](https://huggingface.co/pushkarsharma/wmt26-arabic-asian-mt).

## Official results

Ranks on the blind challenge test set. Primary and contrastive systems are ranked on
separate leaderboards, so the columns are not directly comparable.

| Direction | MADLAD-400 (primary) | NLLB-200 (contr-1) | GemmaX2 (contr-2) |
|---|:--:|:--:|:--:|
| en→ar | **1** | 4 | 3 |
| hi→ar | **1** | 3 | 2 |
| ur→ar | **1** | 3 | 2 |
| ar→en | 2 | 4 | 2 |
| ar→hi | **1** | 1 | 2 |
| ar→ur | 3 | 1 | 2 |

## Zero-shot vs fine-tuned

Macro-averages over the six directions on the 1,000-sentence dev split
(`--split dev`), zero-shot → fine-tuned:

| System | COMET-22 | ChrF2++ | BLEU | TER ↓ |
|---|---|---|---|---|
| NLLB-200 | 0.814 → 0.829 | 44.7 → 49.0 | 20.1 → 25.6 | 75.6 → 68.7 |
| MADLAD-400 | 0.812 → 0.835 | 43.2 → 50.4 | 18.0 → 27.3 | 77.0 → 66.4 |
| GemmaX2-28-9B | 0.576 → 0.827 | 10.3 → 48.9 | 4.3 → 25.7 | 108.3 → 70.2 |

Per-direction numbers are in `outputs/zeroshot_all_results.json` and
`outputs/eval_finetuned/all_results_dev.json`. Fine-tuning helps every system on
every metric in every direction. GemmaX2 zero-shot is only usable for ar→en; the
other five directions come out largely off-target, which is what its outsized gain
reflects rather than a stronger final system.

## Setup

Python 3.10+ and a CUDA GPU (the entry points abort early if none is visible).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python utlis/setup_env.py     # checks imports; --fix resolves the comet-ml / unbabel-comet clash
```

Paths, model ids, language codes and all hyper-parameters live in
[utlis/config.py](utlis/config.py). Set `WMT_ROOT` if the repo is not the working
directory.

## Data

```
dataset/<Pair>/{train,dev,devtest}_{lang}_{pair}.txt
```

Line *n* of the source file aligns with line *n* of the target file, and both
directions of a pair share one folder.

| Pair | train | dev | devtest |
|---|---:|---:|---:|
| Ar–En | 21,000 | 1,000 | 500 |
| Ar–Hi | 20,100 | 1,000 | 500 |
| Ar–Ur | 20,300 | 1,000 | 500 |

Because the task is Arabic-centric the same Arabic sentences are reused across
pairs: the aggregated Arabic side is 65.7% duplicated, while duplication inside any
single file stays under 0.05%. The blind challenge test (1,500 segments per
direction) is not redistributed here — only our outputs, under `submission/`.

## Running the pipeline

```bash
# zero-shot baselines
python run_pipeline.py --stage zero_shot --split dev

# fine-tuning: all models and directions, or one at a time
python run_pipeline.py --stage finetune
python pipelines/train_madlad.py --direction ar-hi

# evaluate saved checkpoints
python pipelines/run_eval_finetuned.py --split dev
python pipelines/run_eval_finetuned.py --model nllb --direction ar-en
```

`run_eval_finetuned.py` looks for checkpoints under `outputs/finetuned/{model}/{dir}`
or `checkpoints/{model}/{dir}`, which is the layout the Hub repo already uses:

```bash
# one direction (NLLB is 40 GB in full, so pull only what you need)
hf download pushkarsharma/wmt26-arabic-asian-mt --include "madlad/ar-en/*" --local-dir outputs/finetuned
```

`resume_finetune.py` skips directions that already have a saved checkpoint and
trains the rest.

Challenge-test run and submission files:

```bash
# point CHALLENGE_ROOT in the script at the organisers' Sub-Task tree first
python pipelines/run_eval_finetuned_A.py
python pipelines/run_noref_eval.py --comet-qe   # reference-free checks before submitting
bash utlis/rename_submission.sh --apply         # paths at the top of the script are absolute; edit them
```

The rename script writes `<team>_<model>_<Sub-Task>.txt`; the files we sent were then
relabelled primary / contrastive1 / contrastive2.

## Training configuration

| | NLLB-200 | MADLAD-400 | GemmaX2-28-9B |
|---|---|---|---|
| Adaptation | full FT | LoRA | QLoRA |
| Precision | bf16 | 8-bit base | 4-bit NF4, bf16 compute |
| Epochs | 5 | 5 | 3 |
| Batch (per-device × accum) | 16 × 2 | 8 × 4 | 4 × 8 |
| Learning rate | 5e-5 | 5e-5 | 2e-4 |
| Warmup / weight decay | 0.10 / 0.01 | 0.10 / 0.01 | 0.05 / 0.01 |
| LoRA r / α / dropout | — | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Optimizer | 8-bit AdamW | AdamW | AdamW |
| Checkpoint selection | COMET-22 | eval loss | eval loss |

Effective batch size is 32 everywhere, with gradient checkpointing, a linear
schedule with warmup and early stopping at patience 2. LoRA targets
`q, k, v, o, wi_0, wi_1, wo` for MADLAD and `q, k, v, o, gate, up, down_proj` for
GemmaX2.

## Preprocessing and decoding

No heuristic normalisation before tokenisation. The usual Arabic rules (alef and ya
unification, diacritic stripping) collapse Urdu characters that are contrastive, and
each model's subword vocabulary was learned over unnormalised text anyway. Empty
pairs are dropped, inputs are truncated at 512 tokens, and each model keeps its own
language signal: `src_lang` plus `forced_bos_token_id` for NLLB, a `<2xx>` prefix for
MADLAD, and an instruction prompt for GemmaX2 (loss over the full
prompt-source-target sequence; first non-empty output line kept at inference).

Decoding is deterministic and identical across systems: beam 4,
`no_repeat_ngram_size=3`, repetition penalty 1.3, length penalty 0.8,
`max_new_tokens = min(512, 2.5 × source length)`. Outputs are normalised to Unicode
NFC after decoding so codepoint variants are not scored as errors.

## Evaluation

[pipelines/evaluate.py](pipelines/evaluate.py) reports COMET-22, ChrF2++, BLEU and
TER, with the SacreBLEU tokeniser chosen per target language (`intl` for Arabic,
`flores101` for Hindi and Urdu). COMET-Kiwi is available for reference-free QE and is
what the challenge-test checks rely on, since references were never released.

Results land in `outputs/` as one metrics `.txt` and one hypothesis file per
model / direction / split, aggregated into the `all_results*.json` files.

## Corpus analysis

```bash
python EDA/eda_train.py       --dataset-root dataset --output-root EDA/eda_train
python EDA/cross_split_eda.py --dataset-root dataset --output-root EDA/eda_split
```

Written up in [EDA/CORPUS_ANALYSIS.md](EDA/CORPUS_ANALYSIS.md), with per-split CSVs,
plots and a leakage report under `EDA/eda_split/`. The short version: Arabic is the
outlier in this corpus — the shortest sentences (26.6 tokens against 38.3 for Hindi
and 41.4 for Urdu) but the largest vocabulary (98K types, TTR 0.176), a dev OOV rate
near 11% against roughly 2% for Hindi and Urdu, and train↔eval trigram overlap of
only 9–12%. There is no exact-match leakage between splits. That sparsity is what
keeps into-Arabic BLEU low while COMET-22 stays high.

## Layout

```
dataset/                official parallel data
pipelines/              per-model training, inference and evaluation
utlis/                  config, data loading, GPU checks, submission helpers
EDA/                    corpus analysis scripts and generated reports
outputs/                hypotheses and metrics (zero-shot, dev, challenge test)
submission/             the files sent to the organisers
run_pipeline.py         orchestrator for the zero-shot / finetune / eval stages
resume_finetune.py      restart training for directions without a checkpoint
```

## Acknowledgment

We thank the WMT 2026 Low-Resource Arabic–Asian MT organisers for the datasets, and
the COIL-D (Centre of Indian Language Data) project under Bhashini, funded by MeitY,
Government of India, for the compute.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) —
https://creativecommons.org/licenses/by-nc/4.0/
