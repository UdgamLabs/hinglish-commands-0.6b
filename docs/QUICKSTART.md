# Try the parser locally

Udgam Hinglish Commands converts short Romanized Hinglish and English commands into a nested TOP tree and an ordered JSON preview. It supports alarm, event, messaging, music, navigation, reminder, timer and weather intents. It does not execute the command.

## 1. Use the source package

From this repository, with a compatible Python environment activated:

```bash
python -m pip install -e .
udgam-hinglish --help
```

The wrapper itself has no mandatory dependencies. Model loading needs PyTorch, Transformers, PEFT, Accelerate, Tokenizers, Safetensors and Hugging Face Hub. The optional `python -m pip install -e '.[inference]'` installs the exact versions observed in our local inference environment. A fresh installation across operating systems has not been verified. The observed local environment was Python 3.11.5 on Apple Silicon; the package requires Python 3.11 or newer. Choose a compatible PyTorch installation for your hardware. No PyPI publication is implied by these source-install instructions.

## 2. Load the adapter package

`--model-dir` must point to the **complete adapter package**, not just its weight file or its inner `model/` directory:

```text
adapter-package/
├── artifact-manifest.json
├── udgam_config.json
├── schema.json
├── retrieval_train.jsonl
└── model/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files…
```

Use an existing local package and the cached pinned Qwen base:

```bash
udgam-hinglish parse --model-dir /path/to/adapter-package \
  --text "kal subah 7 baje alarm lagao" --device auto
```

This invocation downloads nothing. If the base lives in a custom Hugging Face cache, set `HF_HOME` to the directory containing its `hub/` subdirectory before starting Python. For example, when snapshots live under `/cache/huggingface/hub/`, set `HF_HOME=/cache/huggingface`. A missing package, missing cache or failed integrity check produces an error rather than silently selecting another model.

For an explicitly requested download, use the published release's full commit:

```bash
udgam-hinglish parse \
  --repo-id UdgamLabs/hinglish-commands-0.6b \
  --revision 584f51c3421e1efb3fdb6996f769013572dd709d \
  --allow-base-download \
  --text "kal subah 7 baje alarm lagao"
```

The shown commit identifies the complete reviewed adapter package; `main` and tags are rejected. `--repo-id` opts into downloading that public adapter snapshot. `--allow-base-download` separately permits startup download of the exact pinned Qwen base. Neither is needed for a complete cached local setup. The adapter snapshot request omits Hub credentials; the optional base download uses the preserved loader's normal Hugging Face configuration. The SDK does not import code from the downloaded model directory.

For a private Hub repository you already have access to, add `--use-auth` to explicitly use your existing Hugging Face authentication. Configure that authentication through Hugging Face tooling; the CLI provides no token-string option and does not save credentials.

Hub loading materializes a self-contained adapter package under `HF_HOME/udgam-packages/<org>/<repo>/<commit>/` (the standard Hugging Face home is used if `HF_HOME` is unset). This avoids the external blob symlinks in normal Hub snapshots while retaining all package integrity checks. Keep that directory writable and free of symlinks. The cached base model still uses the normal Hugging Face cache.

## 3. Open the browser preview

```bash
udgam-hinglish demo --model-dir /path/to/adapter-package --device auto
```

Wait for `local_demo_ready`, then open **http://127.0.0.1:8765** in your browser. Type a command and select **Preview interpretation**. The model is loaded once before the server starts; requests are processed locally. Stop with **Ctrl-C**. Use `--port 0` if the default port is occupied, then open the printed address.

The server accepts only loopback host/origin headers, limits request size, and does not log your input. This is a local development UI, not an authenticated or public production service.

## Settings and interpretation

The SDK fixes the selected adapter, pinned base, four retrieved training examples, seed `20260912`, non-thinking greedy generation, 4,096 input-token limit and 768 output-token limit. Default precision is **BF16**, including on CPU. Explicit `--device mps` selects Apple Silicon; `--device cuda` selects CUDA. If a CPU cannot run BF16, use `--device cpu --dtype float32`; that changes the evaluated precision and its outputs have no automatic claim to the published benchmark scores.

Accepted results have `ok: true`, `status: "preview"`, a `tree`, and `executed_action: false`. The preview checks tree syntax, known labels and whether copied text occurs in the command. These checks can still accept a wrong interpretation. Rejected results retain `raw_top`, but set `tree` to `null`; they are not repaired. Relative words such as `kal` are left as text, not converted into a calendar date.

The SDK intentionally accepts the selected **adapter** release only. The separately inspected merged export produced different outputs on two local test inputs; adapter benchmark scores must not be assigned to it.

See [Python integration](INTEGRATION.md) for API results, errors and the exact base revision.
