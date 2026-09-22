# Python integration

## Preload once, parse locally

```python
from udgam_hinglish import CommandParser

parser = CommandParser.from_local(
    "/path/to/adapter-package",
    device="auto",  # also: mps, cuda, cpu
)

result = parser.parse("kal subah 7 baje alarm lagao")
print(result["raw_top"])

if result["ok"]:
    print(result["tree"])  # Show this as a preview for review.
else:
    print(result["status"], result["message"])
```

Local loading is offline by default. A complete adapter package includes its sealed manifest, configuration, schema, retrieval corpus and `model/` files. The loader checks every manifest file, the adapter fingerprint and the corpus-derived schema. The SDK validates the selected recipe before loading and executes its own preserved runtime, not Python files inside the model package. Do not modify package files to work around an integrity failure.

`parse()` uses the original input string, selects four training examples with the preserved deterministic retriever, and performs one greedy generation. It does not interpret the input as an instruction to change the parser. An instance serializes generation calls; do not modify its private model or tokenizer. The demo uses the same lock if attached to that instance.

## Explicit Hub download

```python
parser = CommandParser.from_hub(
    "UdgamLabs/hinglish-commands-0.6b",
    revision="584f51c3421e1efb3fdb6996f769013572dd709d",
    device="auto",
    allow_base_download=True,
)
```

The shown 40-character commit identifies the verified complete adapter package. The method downloads a public adapter snapshot at that commit; `allow_base_download=True` separately permits the pinned base download. It does not discover a newest release or accept `main`. The adapter snapshot request explicitly omits Hub credentials; the optional base download uses the preserved loader's normal Hugging Face configuration. Downloads occur only during loading.

Adapter downloads use Hugging Face's explicit `local_dir` mode to materialize regular files under `HF_HOME/udgam-packages/<org>/<repo>/<commit>/`. Normal Hub cache snapshots contain symlinks into a sibling blob directory, which the preserved export loader correctly rejects as paths outside its package. Materialization retains those containment and byte-integrity checks rather than weakening them. Set `cache_dir=` to override the parent directory of `udgam-packages/`; otherwise the usual `HF_HOME` (or XDG default) is used. The destination and its ancestors must not be symlinks. The base model independently uses the normal Hugging Face cache configured by the environment.

The public Udgam release does not require Hub authentication. For other, private repositories, explicitly set `token=True` to use your existing Hugging Face authentication, or add CLI `--use-auth`. The default is `token=False`. The SDK accepts only this boolean switch, never a token string, and does not create credentials or persist them. Access must already be configured through Hugging Face tooling.

## Result contract

| Field | Meaning |
| --- | --- |
| `ok` / `status` | Whether a structured preview passed the limited checks |
| `raw_top` | The generated TOP string, retained even when rejected |
| `tree` | Nested `label` / `children` objects, or `None` after rejection |
| `checks` | Syntax, copied-text diagnostics, unknown labels, truncation/error flags |
| `example_ids` | The four training examples retrieved for this input |
| `executed_action` | Always `False` |
| `base_model`, `base_revision`, `source_adapter_sha256` | Exact model identity |
| `shots`, `seed`, `loaded_dtype` | Actual prompt/seed and loaded precision |
| `precision_matches_recorded_recipe` | Whether the loaded precision is BF16 |

`children` is an ordered list, preserving repeated slot nodes and nesting. Leaf strings preserve the parser output's spelling and case. `raw_top` retains the generated TOP text. The source-copy diagnostic uses Unicode NFC and whitespace-insensitive containment; it does **not** certify exact character spans, whitespace fidelity, correct slot assignment or semantic correctness. The API does not translate values, fix spelling, resolve dates/timezones, or invent missing fields. Unsupported requests can be mapped to an in-domain intent.

Preview statuses are `preview`, `truncated`, `generation_error`, `malformed`, `unknown_labels` and `source_mismatch`. An accepted tree is not permission to send a message, create a reminder, pay money or perform another action. Build a separate domain-specific confirmation and execution layer if your application needs one.

The result also contains preserved generation metadata such as raw output, token counts and `batch_latency_seconds`. That timing covers synchronized model generation only, excluding retrieval, tokenization, decoding and HTTP. It is not end-to-end request latency; device memory fields retain their original sampling/peak scope.

## Fixed release contract

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Adapter fingerprint: `85be8189b09cefa38dc7e50723042d606af19d9f2ad8ceab1cdb123f5eec65af`.
- Four retrieved training examples; seed `20260912`; non-thinking greedy generation.
- BF16, 4,096 input tokens and 768 new tokens. Overlong input is rejected rather than truncated.
- Input must be a nonempty string of at most 2,000 characters; no output action execution.

`from_local(..., dtype="float32", device="cpu")` is an explicit compatibility option. It reports `precision_matches_recorded_recipe=False`; published results do not automatically transfer to that precision. Hardware, driver, kernel and library changes can also affect output. Matching precision alone does not establish identical reproduction. The code pins the release recipe, not every device's numerical behavior.

## CLI and failure handling

```bash
udgam-hinglish parse --model-dir /path/to/adapter-package --text "set an alarm for 7 am"
udgam-hinglish demo --model-dir /path/to/adapter-package --device mps --port 8765
```

Parsing prints one JSON result. Exit code `0` means an accepted preview, `2` means a rejected preview or invalid command-line arguments, and `1` means a local loading/processing error. Argument errors appear on stderr; rejected previews retain JSON on stdout. Help and package imports do not load PyTorch or use the network. API calls raise exceptions for invalid input, missing assets, unsupported settings, integrity failures and runtime failures; catch these at your application boundary and avoid logging user text or unsanitized exception details.

The browser command uses the preserved loopback HTTP server. It binds `127.0.0.1` only; there is no public `--host` option, no action endpoint and no request-time downloads. It exposes preview/status/aggregate measurement endpoints. The original local HTTP integration was exercised, while browser visual QA remained blocked; the new SDK wrapper's offline tests alone do not establish browser or model-output parity.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m udgam_hinglish --help
```

These artificial unit fixtures require no model files, downloads, credentials or real listening sockets. They check recipe enforcement, offline/default behavior, pinned explicit downloads with mocks, literal/ordered previews, failure paths, CLI contracts and demo cleanup. Fresh dependency installation and real-model wrapper verification are separate checks. `src/udgam_hinglish/VENDORED_SOURCES.json` records the original core-module hashes and the small demo adaptation; the historical scientific source remains unchanged.
