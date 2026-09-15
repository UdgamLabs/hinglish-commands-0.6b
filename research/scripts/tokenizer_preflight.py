#!/usr/bin/env python3
"""Measure real train/dev token lengths; test access is label inventory only."""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udgam_commands.prompts import LABEL_RE, derive_schema, make_prompt
from udgam_commands.retrieval import Retriever
from udgam_commands.runtime import (BASE_MODEL, BASE_REVISION, atomic_json,
    environment_info, load_rows, sha256_file, training_tokens, code_hashes)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {"min": ordered[0], "max": ordered[-1], "mean": sum(ordered) / len(ordered),
        **{f"p{q}": ordered[max(0, math.ceil(len(ordered) * q / 100) - 1)] for q in (50, 95, 99)}}


def label_coverage(path, schema):
    """No prompts, predictions, examples or inspection of individual test errors."""
    labels, affected, total = Counter(), 0, 0
    for row in load_rows(path):
        current = {f"{kind}:{name}" for kind, name in LABEL_RE.findall(row["target"])}
        missing = current - set(schema["labels"])
        labels.update(missing)
        affected += bool(missing)
        total += 1
    return {"rows": total, "rows_with_unseen_labels": affected,
        "unseen_label_row_counts": dict(sorted(labels.items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", required=True, help="Local files downloaded from the pinned official revision")
    p.add_argument("--train-data", required=True)
    p.add_argument("--validation-data", required=True)
    p.add_argument("--test-labels-only")
    p.add_argument("--out", required=True)
    p.add_argument("--input-limit", type=int, default=4096)
    p.add_argument("--training-limit", type=int, default=4096)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer_path = Path(a.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    train, validation = load_rows(a.train_data, "train"), load_rows(a.validation_data, "validation")
    schema, retriever = derive_schema(train), Retriever(train)
    report = {"model_id": BASE_MODEL, "revision": BASE_REVISION, "environment": environment_info(),
        "code_sha256": {**code_hashes(), "scripts/tokenizer_preflight.py": sha256_file(__file__)},
        "tokenizer_files": {str(x.name): {"sha256": sha256_file(x), "bytes": x.stat().st_size}
                            for x in sorted(tokenizer_path.iterdir()) if x.is_file()},
        "train_sha256": sha256_file(a.train_data), "validation_sha256": sha256_file(a.validation_data),
        "limits": {"input": a.input_limit, "training": a.training_limit, "output": 768},
        "schema_label_count": len(schema["labels"]), "schema_from": "training only", "measurements": [],
        "validation_label_coverage": label_coverage(a.validation_data, schema),
        "model_weights_loaded": False, "model_generation_performed": False}
    if a.test_labels_only:
        report["test_label_coverage_only"] = label_coverage(a.test_labels_only, schema)
        report["test_sha256"] = sha256_file(a.test_labels_only)
    started = time.perf_counter()
    for split, rows in (("train", train), ("validation", validation)):
        for shots in (0, 4):
            prompt_lengths, total_lengths, target_lengths = [], [], []
            failures = []
            for index, row in enumerate(rows, 1):
                prompt, examples = make_prompt(tokenizer, row, schema, retriever if shots else None, shots)
                prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                try:
                    encoded = training_tokens(tokenizer, prompt, row["target"], max_length=1000000)
                except ValueError as error:
                    failures.append({"id": row["id"], "error": str(error)})
                    continue
                prompt_lengths.append(len(prefix))
                total_lengths.append(len(encoded["input_ids"]))
                target_lengths.append(sum(x != -100 for x in encoded["labels"]))
                if index % 500 == 0:
                    print(json.dumps({"split": split, "shots": shots, "completed": index, "total": len(rows)}), flush=True)
            result = {"split": split, "shots": shots, "rows": len(rows), "boundary_failures": failures,
                "prompt_tokens": distribution(prompt_lengths), "prompt_plus_target_eos_tokens": distribution(total_lengths),
                "target_eos_tokens": distribution(target_lengths),
                "over_input_limit": sum(x > a.input_limit for x in prompt_lengths),
                "over_training_limit": sum(x > a.training_limit for x in total_lengths)}
            report["measurements"].append(result)
            report["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(a.out, report)
            print(json.dumps(result), flush=True)
    report["status"] = "pass" if all(not x["boundary_failures"] and not x["over_input_limit"] and not x["over_training_limit"]
        for x in report["measurements"]) else "preflight_failed_no_truncation_allowed"
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(a.out, report)


if __name__ == "__main__":
    main()
