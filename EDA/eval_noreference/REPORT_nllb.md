# Reference-Free Evaluation Report — `challenge_nllb` split

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
- mean_token_overlap: 0.0059

**Length Consistency** ✅

- hyp_mean_tokens: 40.45
- src_mean_tokens: 33.23
- mean_length_ratio: 1.2194
- std_length_ratio: 0.1855
- expected_ratio: 1.2
- drift_pct: 1.6

**Fluency** ✅

- mean_repetition: 0.007
- high_repetition_pct: 0.0
- mean_ttr: 0.8801

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
- mean_token_overlap: 0.0054

**Length Consistency** ✅

- hyp_mean_tokens: 33.29
- src_mean_tokens: 41.34
- mean_length_ratio: 0.8169
- std_length_ratio: 0.1218
- expected_ratio: 0.85
- drift_pct: 3.9

**Fluency** ✅

- mean_repetition: 0.0008
- high_repetition_pct: 0.0
- mean_ttr: 0.9587

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
- mean_token_overlap: 0.0067

**Length Consistency** ✅

- hyp_mean_tokens: 47.17
- src_mean_tokens: 33.23
- mean_length_ratio: 1.4253
- std_length_ratio: 0.2259
- expected_ratio: 1.45
- drift_pct: 1.7

**Fluency** ✅

- mean_repetition: 0.0074
- high_repetition_pct: 0.0
- mean_ttr: 0.8705

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
- mean_token_overlap: 0.0061

**Length Consistency** ✅

- hyp_mean_tokens: 32.69
- src_mean_tokens: 48.83
- mean_length_ratio: 0.6859
- std_length_ratio: 0.1062
- expected_ratio: 0.7
- drift_pct: 2.0

**Fluency** ✅

- mean_repetition: 0.0009
- high_repetition_pct: 0.0
- mean_ttr: 0.9581

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
- mean_token_overlap: 0.011

**Length Consistency** ✅

- hyp_mean_tokens: 51.25
- src_mean_tokens: 33.23
- mean_length_ratio: 1.5538
- std_length_ratio: 0.2558
- expected_ratio: 1.55
- drift_pct: 0.2

**Fluency** ✅

- mean_repetition: 0.0079
- high_repetition_pct: 0.0
- mean_ttr: 0.8559

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
- mean_token_overlap: 0.0113

**Length Consistency** ✅

- hyp_mean_tokens: 32.49
- src_mean_tokens: 53.02
- mean_length_ratio: 0.6278
- std_length_ratio: 0.0995
- expected_ratio: 0.65
- drift_pct: 3.4

**Fluency** ✅

- mean_repetition: 0.0009
- high_repetition_pct: 0.0
- mean_ttr: 0.9579

## Action Items

No issues found. All directions ready for submission. ✅
