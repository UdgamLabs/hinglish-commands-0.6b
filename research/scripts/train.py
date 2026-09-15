#!/usr/bin/env python3
"""Human-labelled supervised LoRA, full generation validation after each epoch."""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udgam_commands.prompts import LABEL_RE, derive_schema, make_prompt, prompt_hash
from udgam_commands.runtime import (ROOT, atomic_json, canonical_hash, code_hashes,
    collate_training, environment_info, generate_batch, load_model, load_rows,
    load_tokenizer, sha256_file, training_tokens)


def validate_training_indexes(train, validation, retrieval_train):
    if {r["group_id"] for r in train} & {r["group_id"] for r in validation}:
        raise ValueError("Training and validation English-source groups overlap")
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("Training and validation IDs overlap")
    originals = {r["id"]: canonical_hash(r) for r in train}
    if not retrieval_train or any(r.get("split") != "train" for r in retrieval_train):
        raise ValueError("Retrieval index must contain nonempty training records only")
    if len({r["id"] for r in retrieval_train}) != len(retrieval_train):
        raise ValueError("Retrieval index has duplicate IDs")
    if any(originals.get(r["id"]) != canonical_hash(r) for r in retrieval_train):
        raise ValueError("Retrieval index must be an unchanged subset of the training records")
    permitted = set(derive_schema(retrieval_train)["labels"])
    if any({f"{kind}:{name}" for kind, name in LABEL_RE.findall(r["target"])} - permitted for r in train):
        raise ValueError("Training targets use labels absent from the fixed retrieval/schema index")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", required=True)
    p.add_argument("--retrieval-train-data", help="Fixed clean index for schema/examples; defaults to training data")
    p.add_argument("--validation-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shots", type=int, choices=[0, 4], required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "training.json"))
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["cuda", "mps"], default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.seed is not None:
        cfg["seed"] = a.seed
    if not 1 <= cfg["epochs"] <= 6 or cfg["batch_size"] * cfg["gradient_accumulation"] != 32:
        raise ValueError("Protocol requires 1–6 epochs and effective batch size 32")
    if cfg["max_new_tokens"] != 768:
        raise ValueError("Protocol fixes greedy validation output budget at 768")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from udgam_commands.retrieval import Retriever
    from udgam_commands.scoring import score_prediction
    if a.device == "mps":
        raise RuntimeError("Full MPS training remains disabled pending the fixed-shape performance smoke; CUDA default is unchanged")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a BF16-capable CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This protocol is single-GPU only")
    train, validation = load_rows(a.train_data, "train"), load_rows(a.validation_data, "validation")
    retrieval_path = a.retrieval_train_data or a.train_data
    retrieval_train = load_rows(retrieval_path, "train")
    validate_training_indexes(train, validation, retrieval_train)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError("Every training attempt needs a new output directory")
    out.mkdir(parents=True)
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg["model_id"], cfg["revision"], a.local_files_only)
    schema, retriever = derive_schema(retrieval_train), Retriever(retrieval_train) if a.shots else None
    encoded, used = [], []
    for row in train:
        prompt, ids = make_prompt(tokenizer, row, schema, retriever, a.shots)
        try:
            tokens = training_tokens(tokenizer, prompt, row["target"], cfg["max_length"])
        except ValueError as error:
            atomic_json(out / "preprocessing-error.json", {"id": row["id"], "error": str(error)})
            raise
        encoded.append(tokens)
        used.append({"id": row["id"], "record_sha256": canonical_hash(row), "prompt_sha256": prompt_hash(prompt),
            "example_ids": ids, "input_tokens": len(tokens["input_ids"]),
            "supervised_tokens": sum(x != -100 for x in tokens["labels"])})
    val_prompts = [make_prompt(tokenizer, r, schema, retriever, a.shots) for r in validation]
    # Fail before spending training time if any validation prompt is too long.
    if any(len(tokenizer(x[0], add_special_tokens=False)["input_ids"]) > cfg["max_input_tokens"] for x in val_prompts):
        raise ValueError("Validation prompt exceeds input budget; no truncation allowed")
    total_steps = math.ceil(math.ceil(len(encoded) / cfg["batch_size"]) / cfg["gradient_accumulation"]) * cfg["epochs"]
    manifest = {"settings": cfg, "shots": a.shots, "schema": schema, "device": a.device,
        "train_sha256": sha256_file(a.train_data), "validation_sha256": sha256_file(a.validation_data),
        "retrieval_train_sha256": sha256_file(retrieval_path), "retrieval_training_rows": len(retrieval_train),
        "training_rows": len(train), "validation_rows": len(validation), "overlength_rows_dropped": 0,
        "input_tokens_per_epoch": sum(x["input_tokens"] for x in used),
        "supervised_tokens_per_epoch": sum(x["supervised_tokens"] for x in used),
        "environment": environment_info(), "gpu": torch.cuda.get_device_name(),
        "code_sha256": {**code_hashes(), "scripts/train.py": sha256_file(__file__)},
        "planned_optimizer_steps": total_steps, "selection": "highest full-validation exact match, earliest epoch on tie",
        "publication": "PRIVATE; separate owner approval required"}
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "used-records.json", used)
    model = load_model(cfg["model_id"], cfg["revision"], device="cuda", local_files_only=a.local_files_only)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.peft_config["default"].revision = cfg["revision"]
    model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
    model.enable_input_require_grads()
    model.config.use_cache = False
    manifest.update({"trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
                     "total_parameters": sum(x.numel() for x in model.parameters())})
    atomic_json(out / "manifest.json", manifest)
    started = time.perf_counter()
    history = []

    class EpochValidation(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.perf_counter() - started >= cfg["max_minutes"] * 60:
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = float(state.epoch)
            full_epoch = abs(epoch - round(epoch)) < 1e-6
            label = f"epoch-{epoch:.6f}"
            checkpoint = out / "epoch-checkpoints" / label
            checkpoint.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            model.eval()
            predictions, correct = [], 0
            validation_started = time.perf_counter()
            for start in range(0, len(validation), cfg["validation_batch_size"]):
                batch = validation[start:start + cfg["validation_batch_size"]]
                prompts = val_prompts[start:start + cfg["validation_batch_size"]]
                results = generate_batch(model, tokenizer, [x[0] for x in prompts], cfg["max_new_tokens"], cfg["max_input_tokens"])
                for row, result, (prompt, ids) in zip(batch, results, prompts):
                    score = score_prediction(result["prediction"], row["target"], row["text"],
                        truncated=result["truncated"], error=result["error"])
                    passed = bool(score["exact_match"])
                    correct += passed
                    predictions.append({**row, **result, "exact_match": passed,
                        "record_sha256": canonical_hash(row), "example_ids": ids, "prompt_sha256": prompt_hash(prompt)})
                print(json.dumps({"epoch": epoch, "validation_completed": len(predictions), "total": len(validation)}), flush=True)
            with (out / f"{label}-validation.jsonl").open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = {"epoch": epoch, "full_epoch": full_epoch, "correct": correct,
                "total": len(validation), "exact_match": correct / len(validation),
                "truncated": sum(r["truncated"] for r in predictions), "checkpoint": str(checkpoint.relative_to(out)),
                "validation_seconds": time.perf_counter() - validation_started}
            history.append(summary)
            atomic_json(out / "validation-history.json", history)
            print(json.dumps(summary), flush=True)
            model.train()
            if time.perf_counter() - started >= cfg["max_minutes"] * 60:
                control.should_training_stop = True
            return control

    args = TrainingArguments(output_dir=str(out / "trainer-state"), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation"],
        learning_rate=cfg["learning_rate"], lr_scheduler_type="cosine",
        warmup_steps=max(1, round(total_steps * cfg["warmup_fraction"])),
        optim="adamw_torch", bf16=True, gradient_checkpointing=True,
        logging_steps=10, logging_first_step=True, save_strategy="no", eval_strategy="no",
        report_to="none", push_to_hub=False, seed=cfg["seed"], data_seed=cfg["seed"],
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(encoded),
        data_collator=lambda batch: collate_training(batch, tokenizer.pad_token_id), callbacks=[EpochValidation()])
    result = trainer.train()
    trainer.save_state()
    eligible = [x for x in history if x["full_epoch"]]
    if not eligible:
        atomic_json(out / "metrics.json", {**result.metrics, "status": "incomplete_no_full_epoch"})
        raise RuntimeError("No full epoch completed; no candidate selected")
    best = max(eligible, key=lambda row: (row["exact_match"], -row["epoch"]))
    shutil.copytree(out / best["checkpoint"], out / "best-adapter")
    atomic_json(out / "selection.json", {**best, "selected_adapter": "best-adapter", "shots": a.shots,
        "training_complete": len(eligible) == cfg["epochs"], "test_evaluated": False})
    atomic_json(out / "metrics.json", {**result.metrics, "elapsed_including_validation_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "status": "completed"})
    print(json.dumps({"status": "completed", "selected_epoch": best["epoch"], "validation_exact_match": best["exact_match"]}), flush=True)


if __name__ == "__main__":
    main()
