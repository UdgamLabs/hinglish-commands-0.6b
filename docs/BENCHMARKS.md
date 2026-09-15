# Benchmark results

Udgam improved complete-command parsing on the frozen Hinglish-TOP test from **9.90% to 54.88%** relative to the prompted original Qwen3-0.6B. This is a result for a specific structured-output task. It does not mean that general Hindi intelligence improved by this amount. There are still **5,877 incorrect complete trees out of 13,026**.

## What was compared

| Model / matched task prompt | All 13,026 | Hinglish 6,513 | English 6,513 |
| --- | --- | --- | --- |
| Qwen3-0.6B + four retrieved examples | 9.90% | 7.54% | 12.27% |
| KingNish Hinglish checkpoint + four retrieved examples | 9.19% | 5.94% | 12.44% |
| Nearest training answer, unchanged | 0.00% | 0.00% | 0.00% |
| Udgam selected adapter + four retrieved examples | 54.88% | 46.84% | 62.92% |

The two language views contain the paired English and Romanized Hinglish forms of 6,513 original test entries. They are related observations. After normalizing repeated English sources, the bootstrap contains **6,302 source clusters**. These are not 13,026 independent people, tasks or real-world conversations.

We first compared zero-shot and four-retrieved-example prompts on validation data. Four examples won for both Qwen and KingNish; that setup was frozen. The before/after comparison uses the same pinned Qwen base, task/schema instructions, training-only retrieval, non-thinking template, greedy decoding, BF16, batch size 16, input limit 4,096 tokens, output limit 768 tokens and seed 20260912. Training prompts exclude their own English-source group from retrieved examples.

The separate nearest-neighbour baseline returns the closest training answer unchanged. It has no mechanism to replace names, times or other arguments for a new command; its 0% complete-tree score is a limitation of that specific baseline. KingNish is a small general Hinglish checkpoint, not a model trained for this exact schema. These comparisons establish an advantage on the measured interface, not broad superiority over all Hinglish models, commercial assistants, or prior research using different tasks.

## Metric and uncertainty

**Complete-tree exact match** requires the predicted and inherited reference TOP trees to parse and match after whitespace-only canonicalization. Case, punctuation, words, labels, nesting, child order and repeated slots remain significant. Truncated or failed generations receive no correctness credit. The whole original test denominator is retained.

| View | Udgam minus Qwen (percentage points) | Paired 95% interval |
| --- | --- | --- |
| Original: all | +44.98 | [+43.93, +46.03] |
| Original: Hinglish | +39.31 | [+38.00, +40.59] |
| Original: English | +50.65 | [+49.39, +51.94] |
| Deduplicated: all | +44.58 | [+43.56, +45.61] |

The intervals use **2,000 paired bootstrap resamples, seed 42**, sampled by normalized English-source cluster. Each draw preserves the related rows in that cluster and uses the same sample for candidate and baseline. The statistic is the difference in row-weighted exact-match accuracy, with percentile endpoints at 2.5% and 97.5%. These intervals describe uncertainty within this benchmark; they are not a guarantee for new users or applications. Subgroup intervals are exploratory and not adjusted for multiple comparisons.

The original-view improvement over KingNish is **+45.69 percentage points [44.63, 46.71]**. The corresponding improvement over the nearest-neighbour baseline is **+54.88 points [53.77, 55.92]**. These are stored paired results, not a newly computed evaluation.

## Repeated inputs

The original test remains unchanged. A separate sensitivity view keeps the lexicographically smallest ID for each `(language, normalized input)` group. Normalization here is NFKC, casefold and collapsed whitespace; selection never depends on model correctness. Different languages remain separate.

| Model | Correct / 12,692 | Complete-tree accuracy |
| --- | --- | --- |
| Qwen3-0.6B | 1,217 / 12,692 | 9.59% |
| KingNish | 1,123 / 12,692 | 8.85% |
| Nearest neighbour | 0 / 12,692 | 0.00% |
| Udgam adapter | 6,875 / 12,692 | 54.17% |

The deduplicated view has **6,390 Hinglish and 6,302 English** commands. Its composition differs from the original test. Potentially inconsistent inherited targets are documented; they were not corrected to increase a model's score.

## Results by domain

| Domain | Test rows | Qwen | Udgam | Difference (pp) |
| --- | --- | --- | --- | --- |
| Alarm | 2,390 | 16.15% | 61.63% | +45.48 |
| Event | 922 | 7.81% | 55.97% | +48.16 |
| Messaging | 1,012 | 9.98% | 63.83% | +53.85 |
| Music | 1,446 | 8.64% | 53.46% | +44.81 |
| Navigation | 1,996 | 9.22% | 42.74% | +33.52 |
| Reminder | 1,930 | 2.64% | 36.37% | +33.73 |
| Timer | 1,386 | 9.38% | 54.55% | +45.17 |
| Weather | 1,944 | 12.40% | 73.56% | +61.16 |

All eight domains improved over prompted Qwen in the original test. The weakest absolute domain remains reminder parsing at 36.37%, while weather reaches 73.56%; aggregate improvement should not hide this spread.

## Supporting diagnostics

| Original test diagnostic | Qwen | Udgam |
| --- | --- | --- |
| Root intent accuracy | 78.77% | 86.55% |
| Slot micro precision | 46.30% | 74.72% |
| Slot micro recall | 36.15% | 71.22% |
| Slot micro F1 | 40.60% | 72.93% |
| Syntactically valid output | 96.81% | 99.96% |
| Source-copy-valid examples | 49.29% | 99.76% |

Slot metrics count `(label, descendant text)` with multiplicity, including repeated and nested slots. They can remain high when hierarchy or order is wrong. The source-copy check is a permissive substring diagnostic after NFC/whitespace handling, not a span-offset or meaning verifier. In particular, **99.96% valid syntax is not 99.96% correct commands**. The adapter has three generation-failure records; these remain in the denominator.

## Recorded successes and regressions

The following examples are verbatim saved prediction fields from the frozen final run. They are explicitly chosen illustrations from a previously fixed, stratified report collection, not a random sample, new live inference, or evidence of how often a particular error occurs. The display omits the stored terminal `<|im_end|>` token; it does not repair model output. References are unchanged inherited human annotations and can themselves be imperfect. Source dataset attribution and CC BY-SA lineage apply to the excerpts.

### Hinglish improvement: requested time and event category

Recorded test ID: `htop-test-000288-hinglish`. Input:

```text
Kya is month koi beer festival he
```

Original Qwen output:

```text
[IN:GET_EVENT Kya is month koi beer festival he]
```

Udgam output:

```text
[IN:GET_EVENT Kya [SL:DATE_TIME is month ] koi [SL:CATEGORY_EVENT beer festival ] he ]
```

Unchanged benchmark reference:

```text
[IN:GET_EVENT Kya [SL:DATE_TIME is month ] koi [SL:CATEGORY_EVENT beer festival ] he ]
```

Udgam matches the reference by extracting `is month` and `beer festival`. The original model gets the general event intent but misses both slots. This parser does not search for events or know whether a festival exists.

### English improvement: nested recipient and message content

Recorded test ID: `htop-test-002257-english`. Input:

```text
text my mom i ' ll be there in um ten minutes .
```

Original Qwen output:

```text
[IN:SEND_MESSAGE text [SL:RECIPIENT me ] [SL:DATE_TIME tomorrow ] [SL:DATE_TIME ten minutes ] [SL:CONTENT_EXACT i ' ll be there in ] [SL:CONTENT_EMOJI ] [SL:CONTENT_MESSAGE ] ]
```

Udgam output:

```text
[IN:SEND_MESSAGE text [SL:RECIPIENT [IN:GET_CONTACT [SL:CONTACT_RELATED my ] [SL:TYPE_RELATION mom ] ] ] [SL:CONTENT_EXACT i ' ll be there in um ten minutes ] . ]
```

Unchanged benchmark reference:

```text
[IN:SEND_MESSAGE text [SL:RECIPIENT [IN:GET_CONTACT [SL:CONTACT_RELATED my ] [SL:TYPE_RELATION mom ] ] ] [SL:CONTENT_EXACT i ' ll be there in um ten minutes ] . ]
```

Udgam preserves the nested relationship identifying the recipient and the quoted message content. The original prediction invents values such as `tomorrow`; no message is sent by this system.

### Hinglish regression: an extra location

Recorded test ID: `htop-test-005035-hinglish`. Input:

```text
Oahu ke island me events he
```

Original Qwen output:

```text
[IN:GET_EVENT [SL:LOCATION Oahu ] ke island me events he]
```

Udgam output:

```text
[IN:GET_EVENT [SL:LOCATION Oahu ] ke [SL:LOCATION island ] me events he ]
```

Unchanged benchmark reference:

```text
[IN:GET_EVENT [SL:LOCATION Oahu ] ke island me events he ]
```

The base matches the reference. Udgam incorrectly labels `island` as another location. Both outputs are syntactically valid and copy their words from the input, illustrating why those checks cannot establish meaning.

### English regression: a recurring time

Recorded test ID: `htop-test-000329-english`. Input:

```text
Schedule an alarm for 8 am on Monday am
```

Original Qwen output:

```text
[IN:CREATE_ALARM Schedule an alarm [SL:DATE_TIME for 8 am on Monday am ] ]
```

Udgam output:

```text
[IN:CREATE_ALARM Schedule an alarm [SL:DATE_TIME_RECURRING for 8 am on Monday am ] ]
```

Unchanged benchmark reference:

```text
[IN:CREATE_ALARM Schedule an alarm [SL:DATE_TIME for 8 am on Monday am ] ]
```

Udgam changes the slot type from `DATE_TIME` to `DATE_TIME_RECURRING`, disagreeing with the reference where the base was correct. It illustrates the need to confirm a proposed schedule before an application acts.

## Reproducibility and limits

The checkpoint and all inference choices were frozen on development data before the final test. One logical final generation was completed per model; after an interruption, saved prefixes were verified and resumed. Model caches and pinned core packages matched, while the A40 host's kernel and NVIDIA driver changed. This is a disclosed host migration, not bit-identical execution on an unchanged machine. There was no test-driven model or prompt revision.

The benchmark applies to the **selected source adapter with its four-example retrieval interface**. Final package loading and repeat/reload checks found 16/16 agreement between source and packaged adapter on a fixed development sample. An internal merged diagnostic agreed on 14/16; its separate full-test accuracy is unknown and it is excluded from the proposed public payload. The aggregate verification status was `review_required` because of that divergence; it is not honestly described as universal export parity.

Public benchmark examples may have appeared in base pretraining. Exact overlap removal during our adaptation cannot rule this out or remove all semantic paraphrases. The dataset contains inherited annotation choices and international places/names; it is not a representative sample of contemporary India. These results do not establish native-script Hindi ability, speech recognition, general chat quality, production reliability, adoption, or performance on low-end phones.

## Source identity

Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), revision `c1899de289a04d12100db370d81485cdf75e47ca`. Dataset: [Google Research Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset/tree/fdd3998a6573130659bfa1ce4b1ebe698df2bf3a), revision `fdd3998a6573130659bfa1ce4b1ebe698df2bf3a`, building on [TOPv2](https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip). The original authors have not endorsed this adaptation.

Selected adapter fingerprint: `85be8189b09cefa38dc7e50723042d606af19d9f2ad8ceab1cdb123f5eec65af`. Candidate final prediction file SHA-256: `78ae8a658cb0575dfc03a76aa6c0a12c41528870a0bc9ccac0e37125da17314f`. Original Qwen prediction file SHA-256: `b410302f455e80b73722eebdd8468151feedaa984bc96612faeb724b77223d4c`. Final freeze SHA-256: `c0ee8b093fab8c263739f284a64ce4b7eb2dce1c7435ae61c35d5b49c71b4401`.

The companion release evidence preserves stored metrics, paired comparisons, source filenames and hashes, and the exact source IDs of examples. Reading those records does not require rerunning a GPU benchmark. See [Data and training](DATA_AND_TRAINING.md) and [Licences](LICENSES.md).
