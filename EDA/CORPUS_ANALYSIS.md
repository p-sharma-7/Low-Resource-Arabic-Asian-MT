# Detailed Corpus Analysis

## 1. Scope of the Analysis

The corpus analysis examined the Arabic-English (Ar-En), Arabic-Hindi (Ar-Hi), and
Arabic-Urdu (Ar-Ur) parallel datasets at two complementary levels. First, the
training corpora were analyzed independently by file and jointly by language
using the artifacts in `eda_train`. Second, the representativeness and integrity
of the train, development, and development-test partitions were assessed using
the artifacts in `eda_split`. The analysis covered corpus size, sentence
alignment, duplicate and empty sentences, sentence-length distributions,
vocabulary size and diversity, token-length distributions, frequency
distributions, Zipfian behavior, vocabulary coverage, out-of-vocabulary (OOV)
rates, n-gram overlap, semantic similarity, distribution divergence, frequency
shift, and cross-split leakage.

## 2. Training-Corpus Composition and Alignment

The three parallel corpora are similar in scale. Ar-En contains 21,000 aligned
sentence pairs, Ar-Hi contains 20,100 pairs, and Ar-Ur contains 20,300 pairs.
The `alignment_check.json` artifact confirms exact source-target line-count
agreement for every language pair, with a delta of zero in all cases. No empty
sentences were detected in any training file. These findings establish that the
corpora are structurally suitable for supervised machine translation without
requiring line-level realignment or empty-line removal.

| Pair | Sentence pairs | Arabic tokens | Target tokens | Mean Arabic length | Mean target length |
|---|---:|---:|---:|---:|---:|
| Ar-En | 21,000 | 558,133 | 670,845 | 26.58 | 31.95 |
| Ar-Hi | 20,100 | 539,522 | 770,493 | 26.84 | 38.33 |
| Ar-Ur | 20,300 | 543,873 | 840,793 | 26.79 | 41.42 |

The Arabic sides are highly consistent across the three pairs: their mean
sentence lengths range only from 26.58 to 26.84 whitespace-delimited tokens,
their medians are uniformly 24 tokens, and their 99th percentiles range from 80
to 81 tokens. This similarity indicates that the three datasets draw heavily
from the same Arabic source material. When all Arabic files are aggregated,
40,314 of 61,400 sentences are marked as duplicates (65.66%). This should not be
interpreted as poor within-file quality: each individual Arabic file contains
zero duplicates. Rather, it reflects extensive reuse of Arabic source sentences
across translation directions.

The target sides contain only negligible within-file duplication: English has
6 duplicate sentences (0.0286%), Hindi has 9 (0.0448%), and Urdu has 6
(0.0296%). Thus, duplicate-induced training bias within each parallel corpus is
unlikely to be substantial.

## 3. Sentence-Length Characteristics

All sentence-length histograms in `eda_train/per_file` and
`eda_train/per_language` are positively skewed: most sentences occupy a
moderate-length central region, followed by a progressively thinning
right-hand tail. Arabic is the shortest and most compact language side, with a
mean of approximately 26.7 tokens and a median of 24. English is longer
(mean 31.95; median 28), Hindi longer still (mean 38.33; median 33), and Urdu
the longest (mean 41.42; median 36).

The target-to-source mean-length ratios are approximately 1.20 for English,
1.43 for Hindi, and 1.55 for Urdu. These ratios indicate systematic expansion
under whitespace tokenization, particularly for Hindi and Urdu. They should not
be treated solely as evidence of semantic verbosity because orthographic and
morphosyntactic conventions affect whitespace-token counts.

The sentence-length histograms also identify long-tail outliers. Maximum lengths
are 120 tokens for each Arabic file, compared with 509 for English, 421 for
Hindi, and 547 for Urdu. The more robust percentiles remain substantially lower:
the 95th/99th percentiles are 66/101 for English, 79.05/122 for Hindi, and
86/130 for Urdu. Consequently, the extreme maxima represent rare cases rather
than typical corpus behavior. The histograms are clipped at the 99th percentile
for readability and explicitly report the excluded tail: 205, 198, and 198
Arabic sentences; 210 English sentences; 198 Hindi sentences; and 200 Urdu
sentences lie beyond the displayed 99th-percentile boundary.

## 4. Lexical Diversity and Token Characteristics

Arabic exhibits the largest raw vocabulary on every individual source side:
98,062 types in Ar-En, 95,932 in Ar-Hi, and 96,320 in Ar-Ur. Its type-token
ratios (TTRs) are correspondingly high, ranging from 0.1757 to 0.1778. English
contains 58,233 raw types (TTR 0.0868), Hindi contains 35,427 (0.0460), and Urdu
contains 32,960 (0.0392). Although TTR is sensitive to corpus size and
tokenization, the consistent corpus scales make the contrast informative:
Arabic presents the sparsest lexical surface distribution, whereas Hindi and
Urdu concentrate more observations into fewer whitespace-delimited types.

The number of singleton types further demonstrates the long lexical tail:
approximately 58,000-59,000 Arabic types occur once in each source file,
compared with 31,205 English, 18,345 Hindi, and 16,815 Urdu types. Conversely,
only about 544-563 Arabic types occur more than 100 times, compared with 749
English, 1,037 Hindi, and 1,143 Urdu types. This pattern has direct modeling
implications: Arabic is more exposed to rare-form sparsity, while Hindi and
Urdu contain a stronger high-frequency core.

Normalization has no effect on Arabic, removes only one Hindi type, and removes
26 Urdu types. Its strongest impact is on English, where the vocabulary falls
from 58,233 to 53,325 types and the TTR declines from 0.0868 to 0.0795. This
suggests that case variation or related normalization-sensitive distinctions
account for a meaningful portion of apparent English vocabulary diversity.

The token-character-length histograms provide a complementary view. Arabic
tokens average 5.10 characters, English tokens 5.53, Hindi tokens 4.36, and
Urdu tokens 3.77. Arabic is centered around five-character tokens; English has
a broader distribution extending toward longer forms; Hindi is concentrated
around short-to-medium forms; and Urdu is concentrated most strongly between
approximately two and five characters. The very long observed maxima
(125 characters in Arabic and English, 124 in Urdu, and 49 in Hindi) are rare
outliers and may include malformed tokens, URLs, concatenations, or punctuation
artifacts. The plotted distributions are clipped at the 99th percentile,
approximately 11 characters for Arabic and Hindi, 13 for English, and 8 for
Urdu, so the graphs emphasize typical rather than exceptional behavior.

## 5. Frequency Profiles and Zipfian Behavior

Every token-frequency histogram is strongly right-skewed on logarithmic axes.
Most vocabulary types occur only once or a few times, while a small set of
function words accounts for a large proportion of token occurrences. The
`top100_words.txt` files confirm this interpretation. The Arabic lists are
dominated by function words such as `في`, `من`, `على`, `إلى`, and `أن`; the
English list by `the`, `of`, `to`, `and`, and `in`; the Hindi list by `के`,
`में`, `की`, `और`, and `को`; and the Urdu list by `کے`, `کی`, `میں`, `اور`,
and `سے`.

The three per-file Arabic top-word lists and frequency plots are almost
identical, reinforcing the conclusion that the Arabic source corpora are
substantially shared. The aggregated Arabic frequency plot differs because
repeated source material raises many mid- and high-frequency counts; this also
explains why the aggregated Arabic singleton count is only 1,735 despite
individual Arabic files each containing approximately 58,000 singleton types.

All ten Zipf plots, comprising six per-file and four per-language plots, display
the expected near-linear rank-frequency relationship on log-log axes. The
observed curves depart from the idealized `C/rank` reference at the head and
tail, which is typical of finite natural-language corpora. Arabic remains above
the idealized line across much of its tail, reflecting its large inventory of
rare surface forms. Hindi and Urdu exhibit a stronger high-frequency core and a
steeper decline near the tail. English follows the idealized relation relatively
closely through much of the middle ranks. Collectively, the frequency and Zipf
plots support the linguistic plausibility of the corpora while demonstrating
substantial long-tail sparsity.

## 6. Train-Development-DevTest Size and Length Comparison

For every translation direction, the development set contains 1,000 sentences
and the development-test set contains 500. Arabic development sentences are
reused across the three directions, as reflected by identical or near-identical
statistics. Target-side development and development-test sets contain one
within-split duplicate each, corresponding to duplicate rates of 0.1% and 0.2%,
respectively; Arabic evaluation splits contain no duplicates.

The `sentence_length_distribution_comparison.png` figure shows that the
evaluation distributions broadly preserve the right-skewed shape of the
training data but are shifted toward longer sentences. Mean length rises from
26.58-26.84 in Arabic training data to approximately 31.8 in development and
31.7-31.8 in development-test. English rises from 31.95 to approximately 39.7
and 39.2; Hindi from 38.33 to approximately 47.0 and 46.7; and Urdu from 41.42
to approximately 51.2 and 50.5. The evaluation partitions are therefore
systematically more length-demanding than training, which may reduce measured
translation performance but also provides a more challenging generalization
test.

## 7. Vocabulary Coverage and OOV Analysis

The `vocabulary_coverage_barplot.png` figure contrasts token-level coverage with
unique-type coverage. In every condition, token coverage is substantially
higher than unique-type coverage, demonstrating that most unseen types are
rare and contribute comparatively few running tokens. Average token coverage is
89.89% for Arabic, 95.89% for English, 97.80% for Hindi, and 98.15% for Urdu.
The corresponding development/development-test ranges are 89.08%-90.70% for
Arabic, 95.43%-96.36% for English, 97.52%-98.07% for Hindi, and
97.86%-98.44% for Urdu.

The `oov_rate_barplot.png` presents the inverse pattern. Arabic has the greatest
OOV burden, with token OOV rates of 10.80%-10.93% on development and
9.30%-9.40% on development-test; approximately 25% of unique Arabic development
types and 19% of unique Arabic development-test types are unseen in training.
English has moderate token OOV rates of 4.57% and 3.64%. Hindi and Urdu are
considerably lower, with Hindi at 2.48%/1.93% and Urdu at 2.14%/1.56%.

The `top_oov_tokens.csv` artifact shows that many prominent OOVs are named
entities, transliterations, specialized terms, numbers, or punctuation-bound
forms. Examples include Arabic `جنينة`, `سايكس`, and `بيكو`; English `Genena`
and `Sykes-Picot`; Hindi `साइक्स-पिकोट`; and Urdu `جنینا`, `LNG`, and
`Lutnick`. Numeric variants such as `1.147`, `1.313`, and comma-separated
forms also recur. Thus, the OOV results reflect both genuine topical novelty
and tokenization/normalization sensitivity.

## 8. N-gram Overlap and Leakage

The `ngram_overlap_heatmap.png` shows the expected monotonic decline from
unigram to four-gram overlap. Arabic has unigram overlap near 89%-91%, but
bigram overlap falls to 36.6%-40.5%, trigram overlap to 9.1%-12.2%, and
four-gram overlap to 2.6%-4.6%. English retains more shared phrase structure,
with bigram overlap of 61.3%-64.4%, trigram overlap of 24.6%-27.4%, and
four-gram overlap of 8.5%-10.0%. Hindi and Urdu exhibit the greatest overlap:
Hindi reaches 76.0%-79.1% bigram and 40.6%-44.5% trigram overlap, while Urdu
reaches 79.9%-82.9% bigram and 46.7%-50.5% trigram overlap.

High target-side n-gram overlap should be interpreted together with the much
lower exact-duplicate rates. Only three train-development exact matches were
found: the same quoted attribution appears in English, Hindi, and Urdu. Each
match represents approximately 0.10% of the corresponding development set, and
no train-development-test exact matches were detected. The leakage report
therefore classifies every comparison as low risk. The phrase-level overlap in
Hindi and Urdu is more consistent with recurring syntactic and formulaic
patterns than with wholesale sentence leakage.

## 9. Semantic Similarity

The `semantic_similarity_histogram.png` figure contains twelve TF-IDF cosine
similarity distributions, one for each direction-side and evaluation split.
The distributions are generally unimodal and positively skewed, with isolated
high-similarity cases in the tail. Arabic has the lowest mean similarities,
approximately 0.220-0.237, followed by English at 0.257-0.271 and Urdu at
0.287-0.309. Hindi has the highest mean similarity, 0.410-0.426, and its
distribution is visibly centered farther to the right.

Development-test similarity is consistently higher than development similarity
for every language side. This agrees with its higher vocabulary coverage and
n-gram overlap, although distribution-divergence measures show that the same
split can still differ more strongly in global token proportions. TF-IDF
similarity is lexical rather than fully semantic, so these values should be
understood as nearest lexical-content resemblance rather than model-based
semantic equivalence.

## 10. Distribution Divergence and Frequency Shift

The two panels of `distribution_divergence_barplot.png` report
Jensen-Shannon divergence (JSD) and Kullback-Leibler divergence (KLD). Both
metrics yield the same language ordering. Urdu has the smallest divergence
(JSD 0.071-0.097; KLD 1.218-1.823), followed by Hindi (JSD 0.080-0.107; KLD
1.399-2.024), English (JSD 0.134-0.171; KLD 2.492-3.373), and Arabic
(JSD approximately 0.230-0.286; KLD approximately 4.175-5.385). Arabic
therefore presents the greatest train-evaluation distribution mismatch.

For every language, development-test has higher JSD and KLD than development,
despite usually having better token coverage and lexical similarity. This is
not contradictory: coverage measures whether evaluation items have appeared,
whereas divergence measures how their probability mass is redistributed. The
`frequency_shift.csv` artifact illustrates this distinction. High-frequency
function words remain dominant across splits, but their proportions shift; for
example, Arabic `في` decreases from roughly 3.62%-3.64% in training to 3.41%
in evaluation, while `أن` rises from roughly 1.12%-1.14% to about 1.44%-1.47%.
Comparable but generally smaller shifts occur among common English, Hindi, and
Urdu function words.

## 11. Interpretation of Every Plot Family

The 24 per-file training plots comprise four plots for each of the six corpus
files. The sentence-length plots establish the positive skew, target-side
length expansion, and rare long outliers. The token-character-length plots show
language-specific orthographic profiles and exceptional malformed or
concatenated tokens. The token-frequency plots quantify the dominance of rare
types and the small high-frequency core. The Zipf plots verify broadly
natural-language-like rank-frequency behavior while exposing language-specific
tail differences.

The 16 per-language plots repeat these four views after language-level
aggregation. For English, Hindi, and Urdu, these plots are equivalent to their
single-file counterparts because each language occurs in one target file. For
Arabic, aggregation combines three strongly overlapping source files; its
plots must therefore be interpreted as a weighted view of reused source
material rather than as 61,400 independent Arabic sentences.

The six cross-split plots address distinct aspects of representativeness:
`sentence_length_distribution_comparison.png` shows that evaluation sentences
are longer; `vocabulary_coverage_barplot.png` shows high running-token coverage
but lower type coverage; `oov_rate_barplot.png` identifies Arabic as the most
lexically challenging side; `ngram_overlap_heatmap.png` shows rapid overlap
decay with increasing phrase length and particularly high Hindi/Urdu phrase
reuse; `semantic_similarity_histogram.png` shows low-to-moderate lexical
similarity without broad evidence of near duplication; and
`distribution_divergence_barplot.png` identifies the largest global shift on
Arabic and consistently greater divergence for development-test.

## 12. Overall Assessment and Modeling Implications

Overall, the corpus is structurally clean, exactly aligned, free of empty
sentences, and affected by only negligible within-file and cross-split exact
duplication. The development and development-test sets are sufficiently related
to training to support meaningful evaluation, yet they remain challenging
because they contain longer sentences and measurable lexical/distributional
shift. The principal difficulty is concentrated on the Arabic side, which has
the highest lexical diversity, singleton prevalence, OOV rate, and
train-evaluation divergence. Hindi and Urdu show stronger vocabulary coverage
and phrase overlap, but their longer token sequences may impose greater
decoding and memory demands.

These findings motivate subword-aware modeling, careful normalization,
length-aware batching, and explicit handling of named entities, numbers, and
punctuation-bound forms. They also motivate reporting results separately by
translation direction because corpus difficulty is not uniform across
languages. Finally, the shared Arabic source material should be acknowledged
when aggregating statistics or comparing multilingual training regimes, since
naive aggregation would overstate the amount of independent Arabic evidence.

## 13. Methodological Caveats

All length and vocabulary results are based on whitespace-delimited tokens and
therefore reflect both linguistic structure and orthographic conventions.
Type-token ratio is corpus-size sensitive and should be treated as descriptive,
not as an absolute measure of linguistic complexity. The semantic-similarity
analysis uses TF-IDF cosine similarity and consequently captures lexical
relatedness rather than deep semantic equivalence. Finally, the cross-split
leakage conclusion is based on exact matches and n-gram overlap; it does not by
itself exclude paraphrastic or externally sourced overlap.

## 14. Artifact-by-Artifact Guide

The `eda_train` directory contains 62 artifacts. `alignment_check.json` verifies
the line-level alignment of the three parallel corpora, and `summary.csv`
consolidates all principal per-file statistics. Each of the six `per_file`
directories contains a `stats.json` file, a `top100_words.txt` list, and four
PNG figures: `sentence_length_hist.png`, `token_char_len_hist.png`,
`token_freq_hist.png`, and `zipf_curve.png`. Together, these account for
24 per-file PNGs. Each of the four `per_language` directories contains the same
six-artifact structure, accounting for a further 16 PNGs. The per-language
English, Hindi, and Urdu artifacts restate their corresponding single target
files, whereas the Arabic artifacts aggregate the three overlapping Arabic
source files.

The `eda_split` directory contains 20 artifacts. `TRAIN_DEV_ANALYSIS_REPORT.md`
is the consolidated narrative and tabular summary, while `LEAKAGE_REPORT.md`
provides the leakage-risk classification for every train-evaluation comparison.
The `csv` directory contains 12 machine-readable tables:
`split_statistics.csv` records corpus sizes and duplicates;
`sentence_length_distribution.csv` records length moments and percentiles;
`vocabulary_coverage.csv` and `oov_statistics.csv` quantify lexical coverage;
`top_oov_tokens.csv` lists the most frequent unseen forms;
`ngram_overlap.csv` records unigram through four-gram overlap;
`semantic_similarity.csv` records TF-IDF similarity summaries;
`distribution_divergence.csv` records JSD and KLD;
`frequency_shift.csv` compares token proportions across splits;
`exact_duplicate_summary.csv` and `exact_duplicate_sentences.csv` document
cross-split exact matches; and `language_summary.csv` provides the
cross-language aggregate.

The six PNGs in `eda_split/plots` visualize the principal cross-split findings.
`sentence_length_distribution_comparison.png` compares length distributions;
`vocabulary_coverage_barplot.png` contrasts token and type coverage;
`oov_rate_barplot.png` contrasts token and type OOV rates;
`ngram_overlap_heatmap.png` visualizes phrase-overlap decay;
`semantic_similarity_histogram.png` visualizes all twelve lexical-similarity
distributions; and `distribution_divergence_barplot.png` compares JSD and KLD.
Accordingly, all 82 artifacts across `eda_train` and `eda_split`, including all
46 PNG figures, are represented in the analysis above.
