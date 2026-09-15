#!/usr/bin/env python3
"""Resumable, identity-checked greedy prediction. Test requires a frozen manifest."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udgam_commands.prompts import derive_schema, make_prompt, prompt_hash
from udgam_commands.runtime import (BASE_MODEL, BASE_REVISION, adapter_fingerprint,
    atomic_json, canonical_hash, code_hashes, environment_info, generate_batch,
    load_model, load_rows, load_tokenizer, resolve_device, sha256_file,
    validate_resume, validate_test_freeze)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default=BASE_MODEL)
    p.add_argument("--revision", default=BASE_REVISION)
    p.add_argument("--adapter")
    p.add_argument("--shots", type=int, choices=[0, 4], default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], help="Explicit model precision, recorded in resume identity")
    p.add_argument("--freeze-manifest")
    p.add_argument("--local-files-only", action="store_true")
    return p


def main():
    a = parser().parse_args()
    if min(a.batch_size, a.max_new_tokens, a.max_input_tokens) < 1:
        raise ValueError("Positive batch and length limits required")
    rows, train = load_rows(a.data), load_rows(a.train_data, "train")
    data_hash, train_hash = sha256_file(a.data), sha256_file(a.train_data)
    adapter_hash = adapter_fingerprint(a.adapter)
    validate_test_freeze(rows, a.freeze_manifest, data_hash, a.model_id, a.revision, adapter_hash, a.shots)
    device = resolve_device(a.device)
    dtype = a.dtype or ("bfloat16" if device == "cuda" else "float32")
    schema = derive_schema(train)
    metadata = {"format_version": 1, "data_sha256": data_hash, "train_sha256": train_hash,
        "model_id": a.model_id, "revision": a.revision, "adapter_sha256": adapter_hash,
        "shots": a.shots, "batch_size": a.batch_size, "max_new_tokens": a.max_new_tokens,
        "max_input_tokens": a.max_input_tokens, "seed": a.seed, "device": device, "dtype": dtype,
        "do_sample": False, "enable_thinking": False, "schema": schema,
        "code_sha256": code_hashes(), "environment": environment_info(), "input_rows": len(rows)}
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = validate_resume(output, metadata, rows)
        if len(completed) == len(rows):
            print(json.dumps({"status": "already_complete", "completed": len(rows)}), flush=True)
            return
        # Resume boundaries remain original batches; interrupted full lines from a
        # partial batch are retained, but remaining rows are regenerated with their
        # original batch peers, preserving padding/numerical conditions.
        from transformers import set_seed
        from udgam_commands.retrieval import Retriever
        set_seed(a.seed)
        tokenizer = load_tokenizer(a.model_id, a.revision, a.local_files_only)
        retriever = Retriever(train) if a.shots else None
        model = load_model(a.model_id, a.revision, a.adapter, device, a.local_files_only, dtype_name=dtype)
        import torch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        count = len(completed)
        with output.open("a", encoding="utf-8") as handle:
            for start in range(count // a.batch_size * a.batch_size, len(rows), a.batch_size):
                batch = rows[start:start + a.batch_size]
                prompts_and_ids = [make_prompt(tokenizer, r, schema, retriever, a.shots) for r in batch]
                outputs = generate_batch(model, tokenizer, [x[0] for x in prompts_and_ids],
                    a.max_new_tokens, a.max_input_tokens)
                for index, (row, result, (prompt, example_ids)) in enumerate(zip(batch, outputs, prompts_and_ids), start):
                    if index < count:
                        continue
                    record = {**row, **result, "record_sha256": canonical_hash(row),
                        "prompt_sha256": prompt_hash(prompt), "example_ids": example_ids,
                        "batch_index": start // a.batch_size}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                total = min(start + a.batch_size, len(rows))
                progress = {"completed": total, "total": len(rows), "resumed_from": count,
                    "session_elapsed_seconds": time.perf_counter() - started,
                    "last_batch_seconds": outputs[0]["batch_latency_seconds"],
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                    "status": "complete" if total == len(rows) else "running"}
                atomic_json(str(output) + ".progress.json", progress)
                print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
