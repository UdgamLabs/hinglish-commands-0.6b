"""Pinned, deterministic model runtime. Importing this module downloads nothing."""
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_MAX_NEW_TOKENS = 768
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("id", "pair_id", "group_id", "split", "language", "domain", "text", "target", "english_source")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_rows(path, split=None):
    rows, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if any(k not in row for k in REQUIRED):
                raise ValueError(f"Missing required data fields at line {line_number}")
            if any(not isinstance(row[k], str) or not row[k] for k in REQUIRED):
                raise ValueError(f"Expected nonempty string fields at line {line_number}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate input id: {row['id']}")
            if row["language"] not in ("hinglish", "english"):
                raise ValueError("Unexpected language")
            if row["split"] not in ("train", "validation", "test") or (split and row["split"] != split):
                raise ValueError(f"Unexpected split in {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def adapter_fingerprint(path):
    if path is None:
        return None
    path = Path(path)
    names = ["adapter_config.json", "adapter_model.safetensors"]
    if any(not (path / n).is_file() for n in names):
        raise ValueError("Adapter requires config and safetensors weights")
    return canonical_hash({n: sha256_file(path / n) for n in names})


def validate_test_freeze(rows, freeze_path, data_hash, model_id, revision, adapter_hash, shots):
    if not any(r["split"] == "test" for r in rows):
        return
    if not freeze_path:
        raise ValueError("Final test locked: provide the parent-created freeze manifest")
    freeze = json.loads(Path(freeze_path).read_text())
    expected = {"model_id": model_id, "revision": revision,
                "adapter_sha256": adapter_hash, "shots": shots}
    if freeze.get("status") != "frozen" or data_hash not in freeze.get("test_data_sha256", []):
        raise ValueError("Freeze manifest does not authorize this test file")
    if expected not in freeze.get("allowed_runs", []):
        raise ValueError("Freeze manifest does not authorize this exact model/prompt")


def code_hashes():
    paths = [*sorted((ROOT / "udgam_commands").glob("*.py")), ROOT / "scripts" / "predict.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths if p.is_file()}


def environment_info():
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def resolve_device(requested="auto"):
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cuda", "mps", "cpu"):
        raise ValueError("Device must be auto/cuda/mps/cpu")
    return requested


def resolve_dtype(dtype_name, device):
    import torch
    name = dtype_name or ("bfloat16" if device == "cuda" else "float32")
    if name not in ("bfloat16", "float32"):
        raise ValueError("Supported model dtypes are bfloat16 and float32")
    return getattr(torch, name)


def synchronize_device(device):
    """Complete queued GPU work before and after a measured interval."""
    import torch
    kind = getattr(device, "type", str(device).split(":")[0])
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def load_tokenizer(model_id=BASE_MODEL, revision=BASE_REVISION, local_files_only=False):
    from transformers import AutoTokenizer
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable 40-character revision is required")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
        trust_remote_code=False, local_files_only=local_files_only)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_id=BASE_MODEL, revision=BASE_REVISION, adapter=None, device="auto", local_files_only=False,
               *, dtype_name=None):
    import torch
    from transformers import AutoModelForCausalLM
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable 40-character revision is required")
    device = resolve_device(device)
    dtype = resolve_dtype(dtype_name, device)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
        dtype=dtype, attn_implementation="sdpa", trust_remote_code=False,
        local_files_only=local_files_only).to(device)
    if adapter:
        from peft import PeftModel
        adapter_config = json.loads((Path(adapter) / "adapter_config.json").read_text())
        if adapter_config.get("base_model_name_or_path") != model_id or adapter_config.get("revision") != revision:
            raise ValueError("Adapter's pinned base identity does not match request")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model


def load_export(path, device="auto", *, allow_base_download=False, dtype=None):
    """Verify a package before loading; downloading a pinned adapter base is opt-in."""
    root = Path(path).resolve()
    cfg = json.loads((root / "udgam_config.json").read_text())
    if (cfg.get("base_model"), cfg.get("revision")) != (BASE_MODEL, BASE_REVISION) or cfg.get("format") not in ("merged", "adapter"):
        raise ValueError("Expected an export from the pinned Qwen3 base")
    if cfg.get("shots") not in (0, 4) or cfg.get("max_new_tokens") != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError("Export prompt or generation protocol changed")
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    required = {"udgam_config.json", "schema.json", "retrieval_train.jsonl", "model/tokenizer_config.json"}
    if cfg["format"] == "merged":
        required.add("model/config.json")
    else:
        required.update({"model/adapter_config.json", "model/adapter_model.safetensors"})
    if not required.issubset(manifest) or not any(
            key.startswith("model/") and key.endswith(".safetensors") for key in manifest):
        raise ValueError("Export manifest lacks required model or protocol files")
    for name, expected in manifest.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError("Invalid export manifest path")
        if candidate.stat().st_size != expected["bytes"] or sha256_file(candidate) != expected["sha256"]:
            raise ValueError(f"Export file integrity check failed: {name}")
    if sha256_file(root / "retrieval_train.jsonl") != cfg.get("train_sha256"):
        raise ValueError("Export retrieval training data identity changed")
    from .prompts import derive_schema
    if derive_schema(load_rows(root / "retrieval_train.jsonl", "train")) != json.loads((root / "schema.json").read_text()):
        raise ValueError("Export schema is not the exact train-derived schema")
    if cfg["format"] == "adapter":
        adapter_config = json.loads((root / "model/adapter_config.json").read_text())
        if (adapter_config.get("base_model_name_or_path"), adapter_config.get("revision")) != (BASE_MODEL, BASE_REVISION):
            raise ValueError("Export adapter does not match the pinned base identity")
        if adapter_fingerprint(root / "model") != cfg.get("source_adapter_sha256"):
            raise ValueError("Export adapter fingerprint changed")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = resolve_device(device)
    selected_dtype = dtype or cfg.get("dtype") or ("bfloat16" if device == "cuda" else "float32")
    model_dtype = resolve_dtype(selected_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Export tokenizer lacks EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if cfg["format"] == "merged":
        model = AutoModelForCausalLM.from_pretrained(root / "model", local_files_only=True,
            trust_remote_code=False, dtype=model_dtype,
            attn_implementation="sdpa").to(device).eval()
    else:
        model = load_model(BASE_MODEL, BASE_REVISION, root / "model", device,
            local_files_only=not allow_base_download, dtype_name=selected_dtype)
    return model, tokenizer, cfg


def training_tokens(tokenizer, prompt, target, max_length):
    """Fail instead of dropping/truncating a target or changing the prompt boundary."""
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt + target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if full[:len(prefix)] != prefix:
        raise ValueError("Tokenizer merged across assistant boundary; cannot mask safely")
    if len(full) > max_length:
        raise ValueError(f"Training sequence exceeds limit: {len(full)} > {max_length}")
    if len(full) <= len(prefix) or full[-1] != tokenizer.eos_token_id:
        raise ValueError("Missing supervised target/EOS")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


def collate_training(batch, pad_token_id):
    import torch
    width = max(len(x["input_ids"]) for x in batch)
    return {key: torch.tensor([x[key] + [pad] * (width - len(x[key])) for x in batch], dtype=torch.long)
            for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def trim_generated(tokens, eos_ids):
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1], True
    return tokens, False


def generate_batch(model, tokenizer, prompts, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_input_tokens=4096):
    import torch
    encoded = tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
    lengths = encoded["attention_mask"].sum(1).tolist()
    if max(lengths) > max_input_tokens:
        raise ValueError(f"Input exceeds limit ({max(lengths)} > {max_input_tokens}); no truncation permitted")
    width = encoded["input_ids"].shape[1]
    if width + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Input plus output budget exceeds model context")
    encoded = encoded.to(model.device)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    synchronize_device(model.device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True, eos_token_id=sorted(eos_ids),
            pad_token_id=tokenizer.pad_token_id)
    synchronize_device(model.device)
    elapsed = time.perf_counter() - start
    memory_after = {"device": model.device.type, "measurement": "post_generation_sample_not_transient_peak"}
    if model.device.type == "mps":
        memory_after.update(current_allocated_bytes=torch.mps.current_allocated_memory(),
            driver_allocated_bytes=torch.mps.driver_allocated_memory())
    elif model.device.type == "cuda":
        memory_after.update(current_allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    results = []
    for ids, length in zip(generated[:, width:].tolist(), lengths):
        ids, ended = trim_generated(ids, eos_ids)
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
        results.append({"prediction": prediction, "raw_output": raw,
            "input_tokens": length, "generated_tokens": len(ids), "eos_reached": ended,
            "truncated": not ended, "error": None if ended else "generation_limit_without_eos",
            "batch_latency_seconds": elapsed, "batch_size": len(prompts),
            "device_memory_after_generation": memory_after})
    return results


def validate_resume(path, metadata, rows):
    """The existing output must be an exact ordered prefix for the same experiment."""
    path = Path(path)
    sidecar = Path(str(path) + ".meta.json")
    if not path.exists() and not sidecar.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(sidecar, metadata)
        return []
    if not sidecar.exists():
        raise ValueError("Existing output lacks experiment metadata")
    old = json.loads(sidecar.read_text())
    if old != metadata:
        raise ValueError("Resume rejected: configuration, data, code or runtime changed")
    if not path.exists():
        return []
    completed = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.endswith("\n"):
                raise ValueError("Incomplete final output line; recover explicitly before resuming")
            row = json.loads(line)
            if index >= len(rows) or row.get("id") != rows[index]["id"]:
                raise ValueError("Resume rejected: duplicate, foreign or reordered prediction id")
            if row.get("record_sha256") != canonical_hash(rows[index]):
                raise ValueError("Resume rejected: record content changed")
            completed.append(row)
    return completed
