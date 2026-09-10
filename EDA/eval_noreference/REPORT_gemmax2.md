# Reference-Free Evaluation Report — `challenge_gemmax2` split

*No target references available. Checks are based on source-hypothesis*
*consistency, language validity, and statistical patterns.*

---

## Overall Verdict

| Direction | Sanity | Language | Copying | Length | Fluency | Ready? |
|-----------|--------|----------|---------|--------|---------|--------|
| ar-en | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ SUBMIT |
| en-ar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ SUBMIT |
| ar-hi | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ SUBMIT |
| hi-ar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ SUBMIT |
| ar-ur | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ SUBMIT |
| ur-ar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ SUBMIT |

## ar-en

**Sanity** ✅

- src_lines: 1500
- hyp_lines: 1500
- lines_match: True
- empty_hyp: 0
- empty_pct: 0.0

**Language** ✅

- target_lang: en
- expected_script: LATIN
- correct_script_pct: 100.0

**Source Copying** ✅

- exact_copies: 0
- exact_copy_pct: 0.0
- high_overlap_lines: 0
- high_overlap_pct: 0.0
- mean_token_overlap: 0.0001

**Length Consistency** ✅

- hyp_mean_tokens: 42.67
- src_mean_tokens: 33.23
- mean_length_ratio: 1.3008
- std_length_ratio: 0.2557
- expected_ratio: 1.2
- drift_pct: 8.4

**Fluency** ✅

- mean_repetition: 0.0056
- high_repetition_pct: 0.0
- mean_ttr: 0.8862

## en-ar

**Sanity** ✅

- src_lines: 1500
- hyp_lines: 1500
- lines_match: True
- empty_hyp: 0
- empty_pct: 0.0

**Language** ✅

- target_lang: ar
- expected_script: ARABIC
- correct_script_pct: 100.0

**Source Copying** ✅

- exact_copies: 0
- exact_copy_pct: 0.0
- high_overlap_lines: 0
- high_overlap_pct: 0.0
- mean_token_overlap: 0.0003

**Length Consistency** ✅

- hyp_mean_tokens: 32.77
- src_mean_tokens: 41.34
- mean_length_ratio: 0.8146
- std_length_ratio: 0.1665
- expected_ratio: 0.85
- drift_pct: 4.2

**Fluency** ✅

- mean_repetition: 0.0011
- high_repetition_pct: 0.0
- mean_ttr: 0.9576

## ar-hi

**Sanity** ✅

- src_lines: 1500
- hyp_lines: 1500
- lines_match: True
- empty_hyp: 0
- empty_pct: 0.0

**Language** ✅

- target_lang: hi
- expected_script: DEVANAGARI
- correct_script_pct: 100.0

**Source Copying** ✅

- exact_copies: 0
- exact_copy_pct: 0.0
- high_overlap_lines: 0
- high_overlap_pct: 0.0
- mean_token_overlap: 0.0003

**Length Consistency** ✅

- hyp_mean_tokens: 48.98
- src_mean_tokens: 33.23
- mean_length_ratio: 1.5008
- std_length_ratio: 0.3118
- expected_ratio: 1.45
- drift_pct: 3.5

**Fluency** ✅

- mean_repetition: 0.0041
- high_repetition_pct: 0.0
- mean_ttr: 0.8815

## hi-ar

**Sanity** ✅

- src_lines: 1500
- hyp_lines: 1500
- lines_match: True
- empty_hyp: 0
- empty_pct: 0.0

**Language** ✅

- target_lang: ar
- expected_script: ARABIC
- correct_script_pct: 100.0

**Source Copying** ✅

- exact_copies: 0
- exact_copy_pct: 0.0
- high_overlap_lines: 0
- high_overlap_pct: 0.0
- mean_token_overlap: 0.0005

**Length Consistency** ✅

- hyp_mean_tokens: 32.51
- src_mean_tokens: 48.83
- mean_length_ratio: 0.6845
- std_length_ratio: 0.1345
- expected_ratio: 0.7
- drift_pct: 2.2

**Fluency** ✅

- mean_repetition: 0.0011
- high_repetition_pct: 0.0
- mean_ttr: 0.9572

## ar-ur

**Sanity** ✅

- src_lines: 1500
- hyp_lines: 1500
- lines_match: True
- empty_hyp: 0
- empty_pct: 0.0

**Language** ✅

- target_lang: ur
- expected_script: ARABIC
- correct_script_pct: 100.0

**Source Copying** ✅

- exact_copies: 0
- exact_copy_pct: 0.0
- high_overlap_lines: 0
- high_overlap_pct: 0.0
- mean_token_overlap: 0.0046

**Length Consistency** ✅

- hyp_mean_tokens: 53.2
- src_mean_tokens: 33.23
- mean_length_ratio: 1.6333
- std_length_ratio: 0.3367
- expected_ratio: 1.55
- drift_pct: 5.4

**Fluency** ✅

- mean_repetition: 0.0052
- high_repetition_pct: 0.0
- mean_ttr: 0.8629

## ur-ar

**Sanity** ✅

- src_lines: 1500
- hyp_lines: 1500
- lines_match: True
- empty_hyp: 0
- empty_pct: 0.0

**Language** ✅

- target_lang: ar
- expected_script: ARABIC
- correct_script_pct: 100.0

**Source Copying** ✅

- exact_copies: 0
- exact_copy_pct: 0.0
- high_overlap_lines: 0
- high_overlap_pct: 0.0
- mean_token_overlap: 0.0053

**Length Consistency** ✅

- hyp_mean_tokens: 32.67
- src_mean_tokens: 53.02
- mean_length_ratio: 0.6357
- std_length_ratio: 0.1312
- expected_ratio: 0.65
- drift_pct: 2.2

**Fluency** ✅

- mean_repetition: 0.001
- high_repetition_pct: 0.0
- mean_ttr: 0.9582

## Action Items

No issues found. All directions ready for submission. ✅
