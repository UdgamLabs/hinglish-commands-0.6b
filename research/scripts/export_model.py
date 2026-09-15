#!/usr/bin/env python3
"""Build a private adapter/merged export with provenance and checked inference."""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udgam_commands.prompts import derive_schema
from udgam_commands.runtime import (ROOT, BASE_MODEL, BASE_REVISION, adapter_fingerprint,
    atomic_json, canonical_hash, load_model, load_rows, load_tokenizer, resolve_device, sha256_file)

INFERENCE = '''#!/usr/bin/env python3
"""Verified private package inference. Offline by default; never executes actions."""
import argparse
import json
from pathlib import Path
from udgam_commands.prompts import make_prompt
from udgam_commands.runtime import load_export, load_rows, generate_batch
from udgam_commands.retrieval import Retriever
from udgam_commands.top import inspect_top, parse_top, to_json_tree, walk_nodes


def preview_result(output, text, schema):
    prediction = output.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError("The model returned an invalid prediction type")
    checks = inspect_top(prediction, text)
    labels = {node.label for node in walk_nodes(parse_top(prediction))} if checks["syntax_valid"] else set()
    unknown = sorted(labels - set(schema["labels"]))
    truncated = bool(output.get("truncated", False))
    if truncated:
        status, message = "truncated", "Generation stopped before its end marker. No structured preview is accepted."
    elif output.get("error"):
        status, message = "generation_error", "The engine reported a generation error. No structured preview is accepted."
    elif not checks["syntax_valid"]:
        status, message = "malformed", "The output is not one valid TOP tree. It has not been repaired."
    elif unknown:
        status, message = "unknown_labels", "The output uses labels outside the training schema."
    elif not checks["source_valid"]:
        status, message = "source_mismatch", "Some output words are absent from this command. No structured preview is accepted."
    else:
        status, message = "preview", "Structure and source-text checks passed. Check the interpretation; no action was executed."
    accepted = status == "preview"
    return {**output, "ok": accepted, "status": status, "message": message,
            "raw_top": prediction, "tree": to_json_tree(prediction) if accepted else None,
            "checks": {**checks, "unknown_labels": unknown, "truncated": truncated,
                       "generation_error": bool(output.get("error"))},
            "executed_action": False,
            "limitation": "Experimental TOP preview only. Structure and copied-text checks do not prove semantic correctness; unsupported requests may be misclassified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], help="Defaults to the package's recorded inference dtype")
    parser.add_argument("--allow-base-download", action="store_true", help="Adapter packages only: explicitly allow the exact pinned base to download at startup")
    args = parser.parse_args()
    if not args.text.strip() or len(args.text) > 2000:
        parser.error("Provide a nonempty command of at most 2000 characters")
    root = Path(__file__).resolve().parent
    # This shared loader verifies manifest, pinned base, adapter fingerprint,
    # retrieval-corpus hash and schema before loading. No remote code is trusted.
    model, tokenizer, cfg = load_export(root, args.device,
        allow_base_download=args.allow_base_download, dtype=args.dtype)
    from transformers import set_seed
    set_seed(cfg["seed"])
    train = load_rows(root / "retrieval_train.jsonl", "train")
    schema = json.loads((root / "schema.json").read_text())
    retriever = Retriever(train) if cfg["shots"] else None
    prompt, ids = make_prompt(tokenizer, {"text": args.text}, schema, retriever, cfg["shots"])
    output = generate_batch(model, tokenizer, [prompt], cfg["max_new_tokens"], cfg["max_input_tokens"])[0]
    result = preview_result(output, args.text, schema)
    result.update({"example_ids": ids, "format": cfg["format"], "base_model": cfg["base_model"],
                   "base_revision": cfg["revision"], "source_adapter_sha256": cfg["source_adapter_sha256"],
                   "shots": cfg["shots"], "loaded_dtype": str(model.dtype)})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--train-data", required=True,
                   help="Canonical human retrieval/schema corpus; this does not establish the optimizer training mixture")
    p.add_argument("--out", required=True)
    p.add_argument("--shots", required=True, type=int, choices=[0, 4])
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--dtype", choices=["bfloat16", "float32"], help="Merged computation and preferred package inference dtype")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--model-card-dir", help="Completed private assembler output; attach exact card/evidence before sealing, without authorizing publication")
    return p


def validate_evidence():
    """Use only verified, already-snapshotted source/license files."""
    source_path = ROOT / "data/source-manifest.json"
    data_path = ROOT / "data/processed/manifest.json"
    source = json.loads(source_path.read_text())
    data = json.loads(data_path.read_text())
    if (source.get("model_id"), source.get("model_revision")) != (BASE_MODEL, BASE_REVISION):
        raise ValueError("Source snapshot uses a different base model")
    if source.get("dataset_revision") != data.get("dataset_revision"):
        raise ValueError("Dataset source and processed manifests disagree")
    if data.get("source_manifest_sha256") != sha256_file(source_path):
        raise ValueError("Source manifest no longer matches prepared-data provenance")
    selected = [entry for entry in source["files"] if entry["path"].startswith("data/licenses/")
                or entry["path"].startswith("data/model_metadata/qwen3-0.6b/")]
    required = {"data/licenses/google-hinglish-top-APACHE-2.0.txt",
                "data/licenses/google-hinglish-top-README.md",
                "data/licenses/meta-topv2-LICENSE.txt", "data/licenses/meta-topv2-README.txt",
                "data/model_metadata/qwen3-0.6b/LICENSE"}
    if not required.issubset({entry["path"] for entry in selected}):
        raise ValueError("Required original license/README snapshots are missing")
    for entry in selected:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("Invalid source snapshot path")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("A source or license snapshot changed: " + entry["path"])
    return source, data, selected


def validate_model_card(directory, adapter, fingerprint, adapter_config, train_hash, train_rows, train_schema, source, data, shots, seed):
    """Bind a completed private card to this export, without reading test records.

    The assembler's fixed renderer links the card text to its machine evidence;
    its file map links both bytes and the proposed licence document. This is
    local provenance, not a signature, publication approval or fresh load proof.
    """
    def need(ok, message):
        if not ok:
            raise ValueError("Model-card attachment: " + message)

    root = ROOT.resolve()
    def project_file(name):
        need(isinstance(name, str) and name and "\\" not in name, "invalid evidence path")
        parts = name.split("/")
        need(not Path(name).is_absolute() and all(p not in ("", ".", "..") for p in parts), "unsafe evidence path")
        need(not any(p in (".private", ".git", ".ssh", ".cache") or p.startswith(".env") for p in parts), "private operational path")
        need(parts[-1] != "test.jsonl" and not ("final-test" in parts and parts[-1].endswith(".jsonl")), "held-out data/prediction paths are forbidden")
        path = root
        for part in parts:
            path /= part
            need(not path.is_symlink(), "symlink evidence is forbidden")
        need(path.is_file(), "required evidence file is missing")
        return path

    supplied = Path(directory).absolute()
    # Permit macOS /var aliases for the project itself; reject symlinks beneath it.
    lexical_root = ROOT.absolute()
    relative = supplied.relative_to(lexical_root) if supplied.is_relative_to(lexical_root) else supplied.relative_to(root)
    need(relative.is_relative_to("outputs/model-card-drafts") and relative != Path("outputs/model-card-drafts"), "use an assembler output below outputs/model-card-drafts")
    mapping_path = project_file(str(relative / "bundle-entries.json"))
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)
    need(mapping.get("publication_authorized") is False and mapping.get("candidate_adapter_sha256") == fingerprint, "file map does not identify this private adapter")
    entries = mapping.get("files", [])
    expected = {"MODEL_CARD.md": str(relative / "MODEL_CARD.md"),
                "evidence/MODEL_CARD_EVIDENCE.json": str(relative / "MODEL_CARD_EVIDENCE.json"),
                "RELEASE_LICENSE_REVIEW.md": "reports/RELEASE_LICENSE_REVIEW.md"}
    need(len(entries) == len(expected) and {x.get("destination") for x in entries} == set(expected), "file map must contain exactly the assembled card, evidence and licence review")
    payload = {}
    for entry in entries:
        name = entry["destination"]
        need(set(entry) == {"source", "destination", "bytes", "sha256"} and entry["source"] == expected[name], "file-map source differs from the assembler contract")
        content = project_file(entry["source"]).read_bytes()
        need(len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == entry["sha256"], "card/evidence/licence byte hash mismatch")
        payload[Path(name).name] = content
    evidence = json.loads(payload["MODEL_CARD_EVIDENCE.json"])
    need(evidence.get("status") == "private_model_card_draft" and evidence.get("publication_authorized") is False, "only private assembler evidence is accepted")
    need((evidence.get("model_id"), evidence.get("base_revision"), evidence.get("candidate_adapter_sha256")) == (BASE_MODEL, BASE_REVISION, fingerprint), "base revision or selected adapter differs")
    assembler_path = Path(__file__).resolve().with_name("assemble_model_card.py")
    need(evidence.get("assembler_sha256") == sha256_file(assembler_path), "assembler version differs; assemble a fresh private draft")
    spec = importlib.util.spec_from_file_location("udgam_card_attachment_renderer", assembler_path)
    assembler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembler)
    need(payload["MODEL_CARD.md"] == assembler.render_card(evidence).encode("utf-8"), "card text is not the exact rendering of its evidence")
    assembler.assert_candidate_role(adapter.resolve(), root)
    selected, manifest, freeze = evidence["selected_training"], evidence["training_manifest"], evidence["freeze"]
    need(selected.get("adapter_sha256") == fingerprint and evidence["stage2_decision"].get("selected_candidate") == selected, "selected-run evidence differs")
    need(freeze.get("status") == "frozen" and freeze.get("selection_data") == "validation_only" and
         freeze.get("publication_authorized") is False and freeze.get("tuning_after_freeze_authorized") is False and
         freeze.get("decision_sha256") == canonical_hash(evidence["stage2_decision"]), "selection freeze is inconsistent")
    candidate = [item for item in freeze.get("run_specs", []) if item.get("label") == "candidate"]
    need(len(candidate) == 1 and candidate[0] == {"label": "candidate", "model_id": BASE_MODEL, "revision": BASE_REVISION,
         "adapter_sha256": fingerprint, "shots": shots, "adapter": selected["adapter"]}, "frozen candidate/prompt differs")
    need(shots == selected.get("shots") == manifest.get("shots") and seed == freeze.get("seed"), "inference prompt or seed differs from frozen evidence")
    settings = manifest["settings"]
    need((settings.get("model_id"), settings.get("revision")) == (BASE_MODEL, BASE_REVISION), "training base differs")
    for config_key, setting_key in (("r", "lora_rank"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        need(adapter_config.get(config_key) == settings.get(setting_key), "LoRA configuration differs from selected training")
    need(set(adapter_config.get("target_modules", [])) == set(settings.get("target_modules", [])) and bool(settings.get("target_modules")), "LoRA target modules differ")
    inputs = evidence["input_files"]
    for path in ("data/source-manifest.json", "data/processed/manifest.json", "reports/RELEASE_LICENSE_REVIEW.md"):
        identity = inputs.get(path, {})
        actual = project_file(path)
        need(identity.get("sha256") == sha256_file(actual) and identity.get("bytes") == actual.stat().st_size, "source/processed/licence provenance changed")
    def selected_identity(suffix, expected_hash=None):
        matches = [(name, item) for name, item in inputs.items() if name.endswith("/" + suffix)]
        need(len(matches) == 1 and (expected_hash is None or matches[0][1].get("sha256") == expected_hash), "selected-run source hash is not uniquely linked")
        name, item = matches[0]
        path = project_file(name)
        need(path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"), "selected-run source file changed")
        return path
    manifest_path = selected_identity(selected["directory"] + "/manifest.json", selected["training_manifest_sha256"])
    need(json.loads(manifest_path.read_text()) == manifest, "embedded training manifest differs from its source")
    selection = json.loads(selected_identity(selected["directory"] + "/selection.json").read_text())
    need(selection == evidence["selection"] and selection.get("epoch") == selected.get("epoch") and
         selection.get("shots") == shots and selection.get("full_epoch") is True, "selected checkpoint differs from its source")
    need(json.loads(selected_identity("records/test-freeze.json").read_text()) == freeze, "canonical frozen selection differs from its source")
    selected_identity(selected["directory"] + "/used-records.json", selected["used_records_sha256"])
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        selected_identity(selected["adapter"] + "/" + name, sha256_file(adapter / name))
    optimizer, retrieval, exposure = evidence["optimizer_data"], evidence["retrieval_data"], evidence["verified_exposures"]
    need(retrieval.get("sha256") == train_hash == freeze.get("train_sha256") == manifest.get("retrieval_train_sha256", manifest.get("train_sha256")) and
         retrieval.get("rows") == train_rows and manifest.get("schema") == train_schema, "retrieval/schema corpus differs")
    need(optimizer.get("sha256") == selected.get("training_data_sha256") == manifest.get("train_sha256") and
         type(optimizer.get("rows")) is int and optimizer["rows"] > 0 and
         optimizer.get("rows") == manifest.get("training_rows") and manifest.get("overlength_rows_dropped") == 0, "optimizer corpus identity differs")
    need(sha256_file(project_file(optimizer["path"])) == optimizer["sha256"], "optimizer data bytes changed")
    epoch = selected.get("epoch")
    need(type(epoch) in (int, float) and math.isfinite(epoch) and epoch == int(epoch) and epoch >= 1, "selected epoch is incomplete")
    need(exposure == selected.get("actual_selected_checkpoint_exposures"), "selected optimizer exposure record differs")
    need(all(type(exposure.get(k)) is int and exposure[k] >= 0 for k in ("human", "synthetic", "human_supervised_tokens", "total_supervised_tokens")), "invalid optimizer exposure counts")
    count, tokens = exposure["human"] + exposure["synthetic"], exposure["total_supervised_tokens"]
    need(count == optimizer["rows"] * epoch and exposure["human"] == train_rows * epoch and tokens > 0 and
         tokens == manifest.get("supervised_tokens_per_epoch", 0) * epoch and 0 < exposure["human_supervised_tokens"] <= tokens and
         exposure.get("human_fraction") == exposure["human"] / count >= 0.5 and
         exposure.get("human_supervised_token_fraction") == exposure["human_supervised_tokens"] / tokens, "optimizer exposure totals/fractions disagree")
    need((evidence["stage2_decision"].get("selected_recipe") == "human_only") == (exposure["synthetic"] == 0), "recipe and optimizer origins disagree")
    final = evidence["final_metrics"]
    need(final.get("status") == "complete_stored_metrics" and set(final.get("scores", {})) == {"base", "kingnish", "nearest", "candidate"} and
         set(final.get("comparisons", {})) == {"base", "kingnish", "nearest"}, "completed stored final metrics are required")
    prepared = {entry["path"]: entry for entry in data["files"]}
    test = prepared["data/processed/test.jsonl"]
    need(evidence.get("dataset_revision") == source["dataset_revision"] and final.get("test_data_sha256") == test["sha256"] and
         freeze.get("test_data_sha256") == [test["sha256"]] and evidence.get("prepared_data", {}).get("test") == test, "final test/source identity differs")
    for label, value in final["scores"].items():
        stored = json.loads(selected_identity("final-test/" + label + "-score.json").read_text())
        need(assembler.compact_score(stored) == value, "embedded final score differs from its source")
        need(value["original"]["overall"]["n"] == test["rows"], "final test denominator differs")
        need(value["deduplicated"]["overall"]["n"] <= test["rows"], "deduplicated test denominator differs")
        for view in ("original", "deduplicated"):
            assembler.validate_score_view(value[view])
    for label, comparison in final["comparisons"].items():
        stored = json.loads(selected_identity("final-test/candidate-vs-" + label + ".json").read_text())
        need(stored.get("paired_comparison") == comparison, "embedded final comparison differs from its source")
        for view in ("original", "deduplicated"):
            baseline, candidate_scores = final["scores"][label][view], final["scores"]["candidate"][view]
            assembler.validate_paired(comparison[view]["overall"], baseline["overall"], candidate_scores["overall"])
            for group in ("by_language", "by_domain"):
                need(set(comparison[view][group]) == set(candidate_scores[group]) == set(baseline[group]), "paired breakdown groups differ")
                for name, paired in comparison[view][group].items():
                    assembler.validate_paired(paired, baseline[group][name], candidate_scores[group][name])
    if evidence.get("export_verification") is not None:
        need(evidence["export_verification"].get("source_adapter_sha256") == fingerprint, "earlier export check identifies another adapter")
    attachment = {"status": "attached_completed_model_card_evidence", "evidence_path": "MODEL_CARD_EVIDENCE.json",
        "evidence_sha256": hashlib.sha256(payload["MODEL_CARD_EVIDENCE.json"]).hexdigest(),
        "optimizer_corpus": optimizer, "retrieval_schema_corpus": retrieval,
        "selected_recipe": evidence["stage2_decision"]["selected_recipe"], "selected_seed": selected["seed"], "selected_epoch": epoch,
        "verified_selected_checkpoint_exposures": exposure,
        "training_manifest_sha256": selected["training_manifest_sha256"], "used_records_sha256": selected["used_records_sha256"]}
    provenance = {"publication_authorized": False, "source_adapter_sha256": fingerprint,
        "source_directory": str(relative), "source_map_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()} for name, content in payload.items()},
        "origin_note": "The verbatim card describes its origin outside exports. These exact bytes were attached before this package's seal; no existing export or proof was modified.",
        "final_metric_scope": "Stored final metrics identify the frozen source adapter; this is not a full merged-format benchmark.",
        "recorded_earlier_export_verification": evidence.get("export_verification"),
        "current_package_export_and_http_proof": "Not established by attachment. Obtain a fresh proof after this seal and retain it separately in the private review bundle.",
        "publication_note": "Private proposed component terms only; owner approval remains required."}
    return {"payload": payload, "source_map": mapping_bytes, "optimizer_training_evidence": attachment, "provenance": provenance}


def attach_evidence(out, source, data, selected, adapter, fingerprint, train_hash, card_attachment=None):
    for entry in selected:
        original = ROOT / entry["path"]
        if entry["path"].startswith("data/licenses/"):
            destination = out / "licenses" / original.name
        else:
            destination = out / "provenance/base-model" / original.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    shutil.copy2(ROOT / "data/model_metadata/qwen3-0.6b/LICENSE", out / "licenses/Qwen3-APACHE-2.0.txt")
    shutil.copy2(ROOT / "data/source-manifest.json", out / "provenance/source-manifest.json")
    shutil.copy2(ROOT / "data/processed/manifest.json", out / "provenance/data-manifest.json")
    atomic_json(out / "provenance/adapter-origin.json", {
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "adapter_fingerprint": fingerprint,
        "adapter_files": {name: {"sha256": sha256_file(adapter / name), "bytes": (adapter / name).stat().st_size}
                          for name in ("adapter_config.json", "adapter_model.safetensors")},
        "retrieval_corpus_sha256": train_hash,
        "retrieval_matches_prepared_artifact": any(entry["sha256"] == train_hash for entry in data["files"]),
        "optimizer_training_evidence": card_attachment["optimizer_training_evidence"] if card_attachment else {
            "status": "pending_selected_run_assembly",
            "note": "This export identifies the inference retrieval/schema corpus only. The optimizer training corpus, mixture, exposures and selected checkpoint must be established from the exact selected training-run evidence; they are not inferred from --train-data."},
        "legacy_config_key": "udgam_config.json train_sha256 identifies retrieval_train.jsonl for the verified inference loader; it does not establish optimizer training data.",
        "dataset_revision": source["dataset_revision"],
        "source_manifest_sha256": sha256_file(ROOT / "data/source-manifest.json"),
        "processed_manifest_sha256": sha256_file(ROOT / "data/processed/manifest.json"),
        "base_weight_provenance": "Pinned upstream model revision; original base weights are not copied into provenance. Merged output weights receive new file hashes in artifact-manifest.json."})
    (out / "NOTICE.md").write_text(
        "# Attribution and private license review\n\n"
        "This experiment adapts Qwen/Qwen3-0.6B, revision " + BASE_REVISION + ". "
        "Its actual Apache 2.0 license snapshot is included in licenses/Qwen3-APACHE-2.0.txt.\n\n"
        "The included retrieval/schema corpus comes from Google's Hinglish-TOP dataset, revision " + source["dataset_revision"] + ". "
        "Exact optimizer training data and any supplementary mixture require the selected training-run evidence. "
        "Google's Apache 2.0 notice and README are included separately from the original Meta TOPv2 "
        "CC-BY-SA 4.0 LICENSE and README. Preserve their attribution and terms. This package is not "
        "declared Apache-only, and this notice does not resolve how share-alike applies to trained weights. "
        "Final license wording requires private publication review.\n\n"
        "Official sources:\n\n"
        "- https://huggingface.co/Qwen/Qwen3-0.6B/tree/" + BASE_REVISION + "\n"
        "- https://github.com/google-research-datasets/Hinglish-TOP-Dataset/tree/" + source["dataset_revision"] + "\n"
        "- https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip\n\n"
        "This is a supplementary fine-tune; Udgam did not pretrain the original model. "
        "Hashes prove file identity relative to this local record, not a cryptographic publisher signature.\n",
        encoding="utf-8")


def model_card(cfg):
    return ("# Udgam Hinglish Commands - Private model card template\n\n"
        "**PRIVATE. EVALUATION SCORES PENDING ASSEMBLY. NOT APPROVED FOR PUBLICATION.**\n\n"
        "The export process does not establish that this adapter is the selected candidate. "
        "Compatibility/smoke-test adapters are not release candidates. Populate this card only from "
        "the recorded selection and matched evaluation evidence before requesting owner approval.\n\n"
        "## Identity\n\n"
        f"- Base: {cfg['base_model']} at {cfg['revision']}\n"
        f"- Format: {cfg['format']}\n"
        f"- Source adapter fingerprint: {cfg['source_adapter_sha256']}\n"
        f"- Exact inference retrieval/schema corpus SHA256: {cfg['train_sha256']}\n"
        "- The legacy configuration key `train_sha256` identifies `retrieval_train.jsonl`; it does not establish optimizer training data.\n"
        "- Optimizer training corpus, mixture, exposures and checkpoint selection: pending exact selected-run evidence.\n"
        f"- Prompt examples: {cfg['shots']}; non-thinking greedy generation\n"
        f"- Preferred inference dtype: {cfg['dtype']} (not a claim about every stored tensor)\n\n"
        "## Intended use and limits\n\n"
        "Experimental Romanized Hinglish/English TOP semantic parsing for eight assistant-task domains. "
        "The included inference command produces a preview only and executes no actions. It rejects "
        "truncated/error outputs, invalid grammar, unsupported labels and source-copy mismatches. "
        "A passed structure/copy check does not prove semantic correctness. Unseen requests may be "
        "misclassified. No Hindi-script, general-chat, low-end-phone, or production-reliability claim is made.\n\n"
        "## Required measured results\n\n"
        "| Comparison | Status |\n|---|---|\n"
        "| Original zero/four-shot full validation | Pending assembly |\n"
        "| Selected candidate full validation | Pending assembly |\n"
        "| Independent same-size comparator | Pending assembly |\n"
        "| Frozen test, language/domain scores, paired intervals | Pending assembly |\n"
        "| Failures, actual hardware latency/memory and export parity | Pending assembly |\n\n"
        "These are placeholders, not zero scores. Never fill them from a tiny compatibility sample, "
        "software tests, training loss, or nearest-neighbour performance alone. Public pretraining "
        "contamination cannot be excluded.\n\n"
        "## Attribution and release gate\n\n"
        "Read NOTICE.md and licenses/. Google Apache 2.0 and original Meta TOPv2 CC-BY-SA 4.0 "
        "lineage are retained separately; this is not an Apache-only package declaration. "
        "Publication requires explicit owner approval using the Udgam Labs/hackingfix identity.\n")


def build_export(args):
    adapter = Path(args.adapter)
    fingerprint = adapter_fingerprint(adapter)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != BASE_MODEL or config.get("revision") != BASE_REVISION:
        raise ValueError("Only the experiment's pinned Qwen3 base may be exported")
    train = load_rows(args.train_data, "train")
    train_hash = sha256_file(args.train_data)
    schema = derive_schema(train)
    source, data, selected = validate_evidence()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Export needs a fresh private output directory")
    device = resolve_device(args.device)
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    card_attachment = validate_model_card(args.model_card_dir, adapter, fingerprint, config, train_hash,
        len(train), schema, source, data, args.shots, args.seed) if getattr(args, "model_card_dir", None) else None
    out.mkdir(parents=True)
    marker = out / "EXPORT_INCOMPLETE.txt"
    marker.write_text("This export is incomplete until artifact-manifest.json is written. Do not use or publish.\n")
    attach_evidence(out, source, data, selected, adapter, fingerprint, train_hash, card_attachment)
    target = out / "model"
    if args.merge:
        model = load_model(BASE_MODEL, BASE_REVISION, adapter, device, args.local_files_only, dtype_name=dtype)
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(target, safe_serialization=True, max_shard_size="2GB")
    else:
        target.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(adapter / name, target / name)
    tokenizer = load_tokenizer(BASE_MODEL, BASE_REVISION, args.local_files_only)
    tokenizer.save_pretrained(target)
    shutil.copy2(args.train_data, out / "retrieval_train.jsonl")
    atomic_json(out / "schema.json", schema)
    cfg = {"base_model": BASE_MODEL, "revision": BASE_REVISION,
        "source_adapter_sha256": fingerprint, "shots": args.shots, "seed": args.seed,
        "max_new_tokens": 768, "max_input_tokens": 4096, "enable_thinking": False,
        "format": "merged" if args.merge else "adapter", "train_sha256": train_hash,
        "dtype": dtype, "export_device": device, "export_merge_dtype": dtype if args.merge else None,
        "dataset_revision": source["dataset_revision"],
        "source_manifest_sha256": sha256_file(ROOT / "data/source-manifest.json"),
        "publication": "PRIVATE; owner approval required", "executes_actions": False}
    atomic_json(out / "udgam_config.json", cfg)
    package = out / "udgam_commands"
    package.mkdir()
    for name in ("__init__.py", "runtime.py", "prompts.py", "retrieval.py", "top.py"):
        shutil.copy2(ROOT / "udgam_commands" / name, package / name)
    (out / "inference.py").write_text(INFERENCE, encoding="utf-8")
    (out / "requirements.txt").write_text(
        "# Install a hardware-appropriate PyTorch build first.\n"
        "transformers==5.16.1\npeft==0.20.0\naccelerate==1.14.0\n"
        "safetensors==0.8.0\nhuggingface-hub==1.30.0\ntokenizers==0.23.2\n")
    (out / "PUBLICATION_HOLD.md").write_text(
        "# Private review package\n\nDo not publish without explicit owner approval. "
        "Use the approved Udgam Labs/hackingfix identity. This package produces parse previews only "
        "and executes no actions. Compatibility/smoke adapters are excluded from release candidates.\n")
    if card_attachment:
        for name, content in card_attachment["payload"].items():
            (out / name).write_bytes(content)
        (out / "provenance/model-card-source-map.json").write_bytes(card_attachment["source_map"])
        atomic_json(out / "provenance/model-card-attachment.json", card_attachment["provenance"])
    else:
        (out / "MODEL_CARD.md").write_text(model_card(cfg), encoding="utf-8")
    marker.unlink()
    manifest = {str(path.relative_to(out)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in sorted(out.rglob("*")) if path.is_file()}
    atomic_json(out / "artifact-manifest.json", manifest)
    return {"status": "private_export_complete", "out": str(out), "format": cfg["format"],
            "dtype": dtype, "files": len(manifest), "bytes": sum(x["bytes"] for x in manifest.values()),
            "source_adapter_sha256": fingerprint, "publication_authorized": False,
            "evaluation_scores": "attached stored frozen-source-adapter metrics; current export proof remains separate and pending" if card_attachment
                                 else "pending assembly; no quality claim added by export"}


def main():
    print(json.dumps(build_export(parser().parse_args())))


if __name__ == "__main__":
    main()
