# Data and training

Udgam Hinglish Commands is a LoRA adaptation of Qwen3-0.6B for a narrow parsing task. Udgam did not pretrain a new foundation model. The selected checkpoint learned from **existing human-labelled Hinglish-TOP examples and their paired English counterparts**. No new teacher-model labels or newly authored synthetic data were used for this selected adapter.

## Final data counts

| Human split | Original source pairs | Final source pairs | Final language rows |
| --- | ---: | ---: | ---: |
| Train | 2,993 | 2,310 | 4,620 |
| Validation | 1,390 | 1,060 | 2,120 |
| Test | 6,513 | 6,513 | 13,026 |

A pair contains one English and one Romanized Hinglish example. Do not describe 4,620 rows as 4,620 independent bilingual conversations. Earlier research-stage cleaned counts of 2,312 train pairs and 1,064 validation pairs preceded the final conflict/structure checks and are not the training counts.

The scope is eight assistant-task domains: alarm, event, messaging, music, navigation, reminder, timer and weather. Hindi written in Devanagari was not separately trained or evaluated as a target language in this experiment.

## Cleaning and separation

The audit found 596 original training rows matching a test English source or Hinglish query under the recorded normalization. The test was preserved; matching training and validation material was excluded. Normalization for overlap detection used NFKC, lowercase and collapsed whitespace. It was not used to rewrite utterances or answer values.

Training removed 679 overlapping pairs, two duplicate pairs and two pairs with conflicting targets, leaving 2,310. Validation removed 325 overlapping pairs, one duplicate and four structurally unusable pairs, leaving 1,060. Training was separated from validation as well as from the test. Row-level removal reasons and pinned manifests were preserved.

This establishes separation for the supplementary training under the specified rules. It does not establish absence from Qwen's pretraining or eliminate every semantic paraphrase. Existing human labels can still contain semantic errors. The full test denominator was preserved; its repeated examples were also reported in a separate deduplicated sensitivity view.

## Selected recipe

| Setting | Recorded value |
| --- | --- |
| Base | Qwen/Qwen3-0.6B |
| Base revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Training seed | 20260912 |
| Selected completed epoch | 2 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | 0.0001 |
| Microbatch / accumulation | 8 / 4; effective batch 32 |
| Warmup fraction | 0.05 |
| Maximum training sequence length | 4,096 tokens |
| Adapter target modules | q/k/v/o projections and gate/up/down MLP projections |
| Trainable adapter parameters | 10,092,544 |
| Training precision | BF16 |
| Loss | Assistant target and EOS only; base weights frozen |
| Prompt | Explicit TOP schema with four retrieved human-training examples |
| Hardware used | NVIDIA A40 |

The run allowed a maximum of six epochs and a cooperative 90-minute stopping setting checked during training and validation. This was not a hard deadline: recorded elapsed time including validation was 6,101.04 seconds (about 101.7 minutes). It did not complete six epochs. Only completed epochs were eligible; the selected checkpoint is epoch 2. Partial later checkpoints were retained as history and not selected. The validation rule chose the highest complete-tree exact match, with the earlier epoch on a tie.

The selected checkpoint represents **9,240 human example exposures, zero existing-synthetic exposures and 312,810 supervised tokens**. These count two passes through 4,620 rows, not 9,240 unique examples. Zero rows were dropped for excessive token length. The retrieval corpus is pinned separately even though it is the same clean human file: inference retrieval alone is not evidence of which rows updated weights.

Recorded GPU core packages were torch 2.8.0, Transformers 5.16.1, PEFT 0.20.0, Accelerate 1.14.0, tokenizers 0.23.2 and safetensors 0.8.0. Local export verification used a different torch 2.14.0 MPS environment. Matching versions alone does not guarantee identical numerical outputs on different hardware.

## Alternatives that did not replace it

| Development variant | Complete-tree accuracy / 2,120 | Outcome |
| --- | ---: | --- |
| Prompted base Qwen, four examples | 9.39% | Matched baseline |
| Human-only seed 20260912, complete epoch 2 | 51.56% | Selected |
| Existing-synthetic augmentation, complete epoch 1 | 48.44% | Rejected |
| Human-only seed 20260913, complete epoch 2 | 50.90% | Not eligible to replace seed 1 |

The first human-only run exceeded the predeclared five-percentage-point validation improvement gate. We then tested one augmentation variant. The available optional pool contained up to 20,000 prepared synthetic examples, but the actual mixed optimizer file used **4,620 human and 4,620 existing-synthetic rows**: half the example exposures were human. The selected augmentation checkpoint completed one epoch. Its human supervised-token fraction was approximately 49.76%, so a 50% example fraction is not a 50% token guarantee. The synthetic annotations were not newly independently human-reviewed.

Augmentation lowered validation accuracy by 3.11 points, increased malformed/failed outputs from two to 16, and reduced five domains by more than three points. It failed the acceptance rule and is **not part of the released adapter's training mixture**. The original synthetic dataset contained 170,083 rows; blindly training on that entire collection was not done.

The second human-only seed was a stability check. It scored 0.66 points lower overall and dropped timer accuracy by 3.68 points. The replacement guard required a strict overall improvement, no increase in malformed/failed count, and no domain drop over three points; ties retained seed 1. Therefore seed 2 did not replace it. Neither alternate checkpoint received a final-test run. Two development seeds do not prove final-test stability over many seeds.

## Frozen final evaluation

After development selection, the chosen adapter, data hashes, prompt, retrieval source, model revisions and decoding settings were frozen. Final predictions were compared with inherited human labels by deterministic code, not by a language-model judge. Interrupted generation resumed verified saved prefixes without changing the selected recipe. No final-test failures were used to tune the model afterward.

The [benchmark report](BENCHMARKS.md) explains exact match, grouped confidence intervals, repeated examples, actual successes, regressions and runtime caveats. The model outputs hierarchical TOP trees; the deterministic interface preserves ordered repeated slots in nested JSON. It does not resolve relative dates, send messages, set alarms, search for weather or execute other actions.

## Reproduce from the research directory

Use the `research/` directory in the source repository. Every command below runs from that directory with a suitable Python environment already active. Saved-prediction scoring uses only the Python standard library. Training needs the recorded CUDA/BF16 environment and dependencies, including datasets 5.0.1; wrapper-only dependencies are not a full GPU-training setup. No command provisions or pays for a GPU.

To check the stored paired result without loading a model:

```sh
cd research
python scripts/score.py \
  --predictions evidence/final-test/candidate.jsonl \
  --baseline evidence/final-test/base.jsonl \
  --out reproduction/recheck-candidate-vs-base.json \
  --bootstrap-samples 2000 --seed 42
```

Choose a fresh output filename and retain the original evidence. The same scorer can inspect the saved KingNish and nearest-neighbour files. This creates a derivative verification report, not a new model run.

For new training, use a separate writable copy of `research/` and a fresh output directory. Before running, separately cache the exact Qwen revision, activate a compatible single-GPU CUDA environment and arrange your own compute budget and shutdown controls. From the copied research directory:

```sh
python scripts/train.py \
  --train-data data/processed/train.jsonl \
  --retrieval-train-data data/processed/train.jsonl \
  --validation-data data/processed/validation.jsonl \
  --out reproduction/human-only-seed20260912 \
  --shots 4 --config configs/training.json \
  --seed 20260912 --device cuda --local-files-only
```

The prepared human splits are already included, so this does not require the optional synthetic download. A new timed run can complete different numbers of epochs on another machine; the six-epoch maximum and 90-minute setting do not guarantee the historical epoch-2 choice or a hard provider shutdown. Do not overwrite the selected historical model or tune a new run on the published test failures.

For a separately recorded generation check of the selected downloaded adapter package, substitute its actual local path below. Run this only after deciding the model/prompt settings, and write fresh outputs:

```sh
python scripts/predict.py \
  --data data/processed/test.jsonl \
  --freeze-manifest evidence/test-freeze.json \
  --train-data data/processed/train.jsonl \
  --adapter /path/to/hinglish-commands-0.6b/model \
  --model-id Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --shots 4 --device cuda --dtype bfloat16 --batch-size 16 \
  --max-input-tokens 4096 --max-new-tokens 768 --seed 20260912 \
  --out reproduction/selected-adapter-test.jsonl --local-files-only
```

This is a new run with explicit recorded settings, not a replacement for the original frozen prediction evidence. Remove the `--adapter` option to measure the unchanged base with the same interface. Other hardware or versions can change outputs.

The release omits the heavyweight, unused original synthetic source to reduce repository size, while retaining its pin in the original source manifest. **The unchanged preparation script checks every manifest-listed source before preparing even the human splits.** For a full data-preparation reproduction, restore missing sources first in a separate writable research copy:

```sh
python scripts/restore_sources.py --download
mkdir -p records reports
python scripts/prepare_data.py --offline --with-synthetic
```

`restore_sources.py --download` explicitly downloads missing recorded source files and checks their hashes. It is not a model download or new label generation. Without restoration, missing source files should fail the original manifest check rather than be silently skipped. The optional synthetic outputs remain separate from human training. Original source/data manifests and the final selected human-only recipe are preserved as historical evidence.

## Pinned provenance

Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), revision `c1899de289a04d12100db370d81485cdf75e47ca`. Dataset: [Google Research Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset/tree/fdd3998a6573130659bfa1ce4b1ebe698df2bf3a), revision `fdd3998a6573130659bfa1ce4b1ebe698df2bf3a`, building on [TOPv2](https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip). The original authors have not endorsed this adaptation.

| Prepared file | SHA-256 |
| --- | --- |
| Human train / retrieval | `a5c62edc4bb8e0b395b95c991fc364bf1ad25850d7a46c2bf6edf503f494dc11` |
| Validation | `166ab068a57b7c1af8916d1e4e9d9d5fd943bb0155a7bc6c5164c6ef8258b6e8` |
| Original full test | `27b810557f30badbf98729ba71a6388798c2e8744d182402a206be9488c2e48f` |

Preserve the Google Apache notice and original TOPv2 CC BY-SA 4.0 lineage. This is not an Apache-only dataset. See [Licences](LICENSES.md).
