#!/usr/bin/env bash
# rename_submission.sh
# Copies WMT challenge hypothesis files into submission/ with the official format:
#   <TEAM_NAME>_<MODEL>_<SUB-Task No.>.txt
#
# Note: outputs/eval_finetuned_challenge/ is read-only, so files are copied
# (not moved) into /mnt/storage/Pushkar/challenge/wmt/submission/
#
# Usage:
#   bash rename_submission.sh          # dry-run (shows what will happen)
#   bash rename_submission.sh --apply  # actually copy the files

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR="/mnt/storage/Pushkar/challenge/wmt/outputs/eval_finetuned_challenge"
OUT_DIR="/mnt/storage/Pushkar/challenge/wmt/submission"
TEAM_NAME="NLP-IIT-Patna"
DRY_RUN=true
[[ "${1:-}" == "--apply" ]] && DRY_RUN=false

# language pair → sub-task ID
declare -A SUBTASK=(
    [en-ar]="Sub-Task-1A"
    [hi-ar]="Sub-Task-1B"
    [ur-ar]="Sub-Task-1E"
    [ar-en]="Sub-Task-2A"
    [ar-hi]="Sub-Task-2B"
    [ar-ur]="Sub-Task-2E"
)

# ── Header ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  WMT Submission File Copier"
echo "  Team     : ${TEAM_NAME}"
echo "  Source   : ${BASE_DIR}"
echo "  Dest     : ${OUT_DIR}/{madlad,nllb,gemmax2}/"
echo "  Mode     : $( $DRY_RUN && echo 'DRY-RUN (pass --apply to copy)' || echo 'LIVE — files will be copied' )"
echo "══════════════════════════════════════════════════════════════════════"
printf "\n  %-46s  →  %s\n" "SOURCE FILE" "DEST / SUBMISSION FILENAME"
echo "  $(printf '%.0s─' {1..80})"

renamed=0
skipped=0
errors=0

# ── Main loop ────────────────────────────────────────────────────────────────
for folder in madlad nllb gemmax2; do
    folder_path="${BASE_DIR}/${folder}"

    if [[ ! -d "${folder_path}" ]]; then
        echo "  [WARN] Directory not found: ${folder_path}" >&2
        continue
    fi

    echo ""
    echo "  ── ${folder^^} ──────────────────────────────────"

    for filepath in "${folder_path}"/hyp_*_challenge.txt; do
        filename="$(basename "${filepath}")"

        # Extract language pair: match pattern like 'ar-en', 'hi-ar', etc.
        # Filename format: hyp_<model>_ft_<lang-pair>_challenge.txt
        if [[ "${filename}" =~ _([a-z]{2}-[a-z]{2})_challenge\.txt$ ]]; then
            lang_pair="${BASH_REMATCH[1]}"
        else
            echo "  [SKIP] Could not extract language pair from: ${filename}" >&2
            (( skipped++ )) || true
            continue
        fi

        # Look up sub-task ID
        if [[ -z "${SUBTASK[${lang_pair}]+_}" ]]; then
            echo "  [SKIP] No sub-task mapping for language pair '${lang_pair}' in: ${filename}" >&2
            (( skipped++ )) || true
            continue
        fi
        subtask_id="${SUBTASK[${lang_pair}]}"

        # Build new filename — named by model, organized per-model subdir
        new_filename="${TEAM_NAME}_${folder}_${subtask_id}.txt"
        model_out_dir="${OUT_DIR}/${folder}"
        new_filepath="${model_out_dir}/${new_filename}"

        printf "  %-46s  →  %s/%s\n" "${filename}" "${folder}" "${new_filename}"

        if $DRY_RUN; then
            (( renamed++ )) || true
        else
            mkdir -p "${model_out_dir}"
            if cp "${filepath}" "${new_filepath}"; then
                (( renamed++ )) || true
            else
                echo "  [ERROR] Failed to copy: ${filename}" >&2
                (( errors++ )) || true
            fi
        fi
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "  $(printf '%.0s─' {1..80})"
if $DRY_RUN; then
    echo "  DRY-RUN complete: ${renamed} file(s) would be copied, ${skipped} skipped."
    echo ""
    echo "  To apply: bash rename_submission.sh --apply"
else
    echo "  Done: ${renamed} file(s) copied to ${OUT_DIR}, ${skipped} skipped, ${errors} error(s)."
    if [[ ${renamed} -gt 0 ]]; then
        echo ""
        for model in madlad nllb gemmax2; do
            model_dir="${OUT_DIR}/${model}"
            if [[ -d "${model_dir}" ]]; then
                echo "  ${model_dir}/"
                ls -lh "${model_dir}"/*.txt 2>/dev/null | awk '{printf "    %s  %s\n", $5, $9}'
            fi
        done
    fi
fi
echo "══════════════════════════════════════════════════════════════════════"
echo ""
