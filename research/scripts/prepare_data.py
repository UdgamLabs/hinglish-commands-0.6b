#!/usr/bin/env python3
"""Fetch pinned public evidence and prepare leakage-audited TOP parsing data.

No cloud credentials, model weights, paid APIs, or model generation are used.
The original held-out test is preserved, including any imperfect gold examples.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import ssl
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA_REV = "fdd3998a6573130659bfa1ce4b1ebe698df2bf3a"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
COMPARATOR_ID = "KingNish/Qwen3-0.6b-hinglish-2"
RAW_ROOT = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/"
TOPV2_URL = "https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip"
EXPECTED = {
    "train": (2993, "8bcf9a01e467a97195631f1ec9983af27ef9a752b6d8f8537764e81814102360"),
    "validation": (1390, "c713c7fb367713c3a5ed65b727d3d6329dd944f1032e786b06bcaedcf3cdc332"),
    "test": (6513, "9f7bd4f80b15956143b4188a113981735264944ac825d6231b60cc2444533791"),
    "synthetic": (170083, "b7309e51e831e43cd4d8c11524df8749e3ce8bddce8031a0d5f1dfee70b56f1f"),
}
DOMAINS = {"alarm", "event", "messaging", "music", "navigation", "reminder", "timer", "weather"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalization(text: str) -> str:
    """Only for overlap grouping; never applied to model input or gold target."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def canonical_whitespace(text: str) -> str:
    return " ".join(text.split())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_source_manifest(manifest: dict) -> None:
    """Refuse changed or missing cached evidence, including license snapshots."""
    if manifest.get("dataset_revision") != DATA_REV or manifest.get("model_revision") != MODEL_REV:
        raise ValueError("Source manifest does not use the approved dataset/base revisions")
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Source manifest path escapes project")
        content = path.read_bytes()
        if len(content) != item["bytes"] or sha256(content) != item["sha256"]:
            raise ValueError(f"Changed source snapshot: {item['path']}")


def fetch(url: str, path: Path, expected_sha: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        content = path.read_bytes()
        if expected_sha and sha256(content) != expected_sha:
            raise ValueError(f"Cached source hash mismatch: {path}")
        return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": len(content), "sha256": sha256(content)}
    context = ssl.create_default_context(cafile="/etc/ssl/cert.pem") if Path("/etc/ssl/cert.pem").exists() else ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, context=context, timeout=120) as response:
            content = response.read()
    except (OSError, urllib.error.URLError):
        # A certificate-validating fallback; no shell and no credentials.
        content = subprocess.check_output(["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "120", url])
    if expected_sha and sha256(content) != expected_sha:
        raise ValueError(f"Downloaded source hash mismatch: {url}")
    path.write_bytes(content)
    return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": len(content), "sha256": sha256(content)}


def snapshot_sources(with_synthetic: bool) -> dict:
    evidence: dict = {"dataset_revision": DATA_REV, "model_id": MODEL_ID, "model_revision": MODEL_REV, "files": []}
    for split in ["train", "validation", "test"] + (["synthetic"] if with_synthetic else []):
        source = f"Dataset/Human Annotated Data/{split}.tsv" if split != "synthetic" else "Dataset/Synthetically Generated Data/train.tsv"
        url = RAW_ROOT + DATA_REV + "/" + urllib.parse.quote(source)
        evidence["files"].append(fetch(url, ROOT / "data/raw" / f"{split}.tsv", EXPECTED[split][1]))
    for source, destination in [("LICENSE.md", "google-hinglish-top-APACHE-2.0.txt"), ("README.md", "google-hinglish-top-README.md")]:
        evidence["files"].append(fetch(RAW_ROOT + DATA_REV + "/" + source, ROOT / "data/licenses" / destination))
    archive = ROOT / "data/raw/TOPv2_Dataset.zip"
    evidence["files"].append(fetch(TOPV2_URL, archive))
    with zipfile.ZipFile(archive) as zf:
        for source in ["LICENSE", "README"]:
            value = zf.read(source)
            target = ROOT / "data/licenses" / f"meta-topv2-{source}.txt"
            target.write_bytes(value)
            evidence["files"].append({"url": TOPV2_URL, "archive_member": source, "path": str(target.relative_to(ROOT)), "bytes": len(value), "sha256": sha256(value)})
    meta_root = ROOT / "data/model_metadata/qwen3-0.6b"
    evidence["files"].append(fetch(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REV}", meta_root / "hub-metadata.json"))
    meta = json.loads((meta_root / "hub-metadata.json").read_text())
    if meta.get("sha") != MODEL_REV:
        raise ValueError("Model metadata revision does not match pin")
    for file in ["LICENSE", "README.md", "config.json", "generation_config.json"]:
        evidence["files"].append(fetch(f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REV}/{file}", meta_root / file))
    comparator_root = ROOT / "data/model_metadata/kingnish-qwen3-0.6b-hinglish-2"
    metadata = fetch(f"https://huggingface.co/api/models/{COMPARATOR_ID}", comparator_root / "hub-metadata.json")
    evidence["files"].append(metadata)
    comparator_meta = json.loads((comparator_root / "hub-metadata.json").read_text())
    comparator_revision = comparator_meta["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", comparator_revision):
        raise ValueError("Comparator metadata lacks an exact commit revision")
    evidence["comparator_model_id"] = COMPARATOR_ID
    evidence["comparator_model_revision"] = comparator_revision
    available = {item["rfilename"] for item in comparator_meta["siblings"]}
    for file in ["LICENSE", "LICENSE.md", "README.md", "config.json", "generation_config.json"]:
        if file in available:
            evidence["files"].append(fetch(f"https://huggingface.co/{COMPARATOR_ID}/resolve/{comparator_revision}/{file}", comparator_root / file))
    evidence["comparator_license_metadata"] = comparator_meta.get("cardData", {}).get("license")
    evidence["comparator_standalone_license_file_present"] = bool({"LICENSE", "LICENSE.md"} & available)
    return evidence


def read_source(split: str) -> list[dict]:
    path = ROOT / "data/raw" / f"{split}.tsv"
    if sha256(path.read_bytes()) != EXPECTED[split][1]:
        raise ValueError(f"Unexpected raw file hash: {split}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if len(header) != (4 if split == "synthetic" else 5):
            raise ValueError(f"Unexpected source schema: {split}: {header}")
        rows = []
        for index, fields in enumerate(reader, start=1):
            if len(fields) != len(header):
                raise ValueError(f"Wrong column count in {split} row {index}")
            rows.append({"source_split": split, "source_row": index, "english_source": fields[0], "hinglish": fields[1], "english_target": fields[2], "hinglish_target": fields[3], "domain": fields[4] if len(fields) == 5 else None})
    if len(rows) != EXPECTED[split][0]:
        raise ValueError(f"Unexpected raw row count: {split}: {len(rows)}")
    if split != "synthetic" and {r["domain"] for r in rows} != DOMAINS:
        raise ValueError(f"Unexpected domain inventory: {split}")
    return rows


def source_keys(rows: list[dict]) -> tuple[set[str], set[str]]:
    return ({normalization(r["english_source"]) for r in rows}, {normalization(r["hinglish"]) for r in rows})


def exclude_overlap(rows: list[dict], held: list[dict]) -> tuple[list[dict], list[dict]]:
    english, hinglish = source_keys(held)
    kept, removed = [], []
    for row in rows:
        reasons = []
        if normalization(row["english_source"]) in english:
            reasons.append("english_source_overlap")
        if normalization(row["hinglish"]) in hinglish:
            reasons.append("hinglish_query_overlap")
        (removed if reasons else kept).append(dict(row, removal_reasons=reasons) if reasons else row)
    return kept, removed


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = set()
    kept, removed = [], []
    for row in rows:
        key = normalization(row["hinglish"]), canonical_whitespace(row["hinglish_target"])
        if key in seen:
            removed.append(dict(row, removal_reasons=["repeated_hinglish_target_pair"]))
        else:
            seen.add(key)
            kept.append(row)
    return kept, removed


def audit_overlap(rows: list[dict], held: list[dict]) -> dict:
    a_en, a_hi = source_keys(rows)
    b_en, b_hi = source_keys(held)
    return {"english_unique_matches": len(a_en & b_en), "hinglish_unique_matches": len(a_hi & b_hi), "rows_matching_either": sum(normalization(r["english_source"]) in b_en or normalization(r["hinglish"]) in b_hi for r in rows)}


def parser_functions():
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module("udgam_commands.top")
    return module


def structural_check(target: str, text: str, module) -> list[str]:
    """Parser-specific adaptation is deliberately narrow and fail-closed."""
    result = module.inspect_top(target, text)
    if not isinstance(result, dict) or "syntax_valid" not in result or "source_valid" not in result:
        raise TypeError("inspect_top must return the documented validation dictionary")
    if not result["syntax_valid"]:
        return ["syntax: " + result["error"]]
    return ["source_leaf_mismatch: " + json.dumps(issue, ensure_ascii=False, sort_keys=True) for issue in result["source_issues"]]


def validate_rows(rows: list[dict], module, preserve: bool = False) -> tuple[list[dict], list[dict]]:
    kept, removed = [], []
    for row in rows:
        checks = {"english": structural_check(row["english_target"], row["english_source"], module), "hinglish": structural_check(row["hinglish_target"], row["hinglish"], module)}
        checked = dict(row, gold_checks=checks)
        if any(checks.values()) and not preserve:
            removed.append(dict(checked, removal_reasons=["gold_structure_or_source_mismatch"]))
        else:
            kept.append(checked)
    return kept, removed


def remove_conflicts(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Discard training inputs with multiple different original gold trees."""
    values = {"english": collections.defaultdict(set), "hinglish": collections.defaultdict(set)}
    for r in rows:
        values["english"][normalization(r["english_source"])].add(canonical_whitespace(r["english_target"]))
        values["hinglish"][normalization(r["hinglish"])].add(canonical_whitespace(r["hinglish_target"]))
    kept, removed = [], []
    for r in rows:
        conflict = len(values["english"][normalization(r["english_source"])]) > 1 or len(values["hinglish"][normalization(r["hinglish"])]) > 1
        (removed if conflict else kept).append(dict(r, removal_reasons=["conflicting_training_gold"]) if conflict else r)
    return kept, removed


def pair_id(row: dict) -> str:
    return "htop-" + row["source_split"] + "-" + str(row["source_row"]).zfill(6)


def processed_rows(rows: list[dict], split: str, languages=("hinglish", "english")):
    for row in rows:
        pid = pair_id(row)
        group_id = "en-" + sha256(normalization(row["english_source"]).encode())[:24]
        for language in languages:
            yield {"id": pid + "-" + language, "pair_id": pid, "group_id": group_id, "split": split, "language": language, "domain": row["domain"], "text": row["hinglish"] if language == "hinglish" else row["english_source"], "target": row[language + "_target"], "english_source": row["english_source"], "origin": "synthetic" if row["source_split"] == "synthetic" else "human", "source_row": row["source_row"], "dataset_revision": DATA_REV, "gold_validation_errors": row.get("gold_checks", {}).get(language, [])}


def write_jsonl(path: Path, rows) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return {"path": str(path.relative_to(ROOT)), "rows": count, "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes())}


def balanced_subset(rows: list[dict], count: int = 20000, seed: int = 20260912) -> list[dict]:
    buckets = collections.defaultdict(list)
    for row in rows:
        root_intent = re.match(r"\[IN:([^\s\]]+)", row["hinglish_target"].lstrip())
        if not root_intent:
            raise ValueError("Synthetic root intent missing after structure checks")
        buckets[root_intent.group(1)].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: sha256(f"{seed}|{pair_id(row)}".encode()))
    selected = []
    queues = {key: collections.deque(value) for key, value in buckets.items()}
    while len(selected) < min(count, len(rows)):
        for key in sorted(queues):
            if queues[key]:
                selected.append(queues[key].popleft())
                if len(selected) == min(count, len(rows)):
                    break
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-synthetic", action="store_true", help="Prepare a separate optional pool; never add it to human training")
    ap.add_argument("--offline", action="store_true", help="Use existing verified snapshots only")
    args = ap.parse_args()
    sources_path = ROOT / "data/source-manifest.json"
    if sources_path.exists():
        verify_source_manifest(json.loads(sources_path.read_text(encoding="utf-8")))
    if not args.offline:
        write_json(sources_path, snapshot_sources(args.with_synthetic))
    elif not sources_path.exists():
        raise ValueError("--offline requires source-manifest.json")
    parser = parser_functions()
    source = {split: read_source(split) for split in ["train", "validation", "test"] + (["synthetic"] if args.with_synthetic else [])}
    audit = {"dataset_revision": DATA_REV, "normalization_for_overlap_only": "NFKC, lowercase, collapsed whitespace", "original_counts": {key: len(rows) for key, rows in source.items()}, "overlap": {}, "cleanup": {}, "gold_validation": {}, "outputs": [], "limitations": ["Public benchmark may have appeared in the base model's pretraining; unknown and not excluded by these checks.", "Exact normalized source/query cleanup does not eliminate all semantic paraphrase overlap.", "Existing human gold can contain semantic errors that syntax/source checks cannot detect.", "Whole original test retained; repeated query groups must be respected when estimating confidence intervals.", "Google repository Apache2 and underlying Meta TOPv2 CC-BY-SA4 notices are retained separately; this is not an Apache-only dataset release."]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")] + ([("synthetic", "test"), ("synthetic", "validation")] if args.with_synthetic else []):
        audit["overlap"][a + "__" + b] = audit_overlap(source[a], source[b])
    removals = []
    stage = {}
    for split in ["train", "validation"] + (["synthetic"] if args.with_synthetic else []):
        held = source["test"] + (source["validation"] if split != "validation" else [])
        clean, overlapped = exclude_overlap(source[split], held)
        clean, repeated = deduplicate(clean)
        historical_count = len(clean)
        expected_count = {"train": 2312, "validation": 1064, "synthetic": 132895}[split]
        if historical_count != expected_count:
            raise ValueError(f"Preparation does not reproduce audited count: {split} {historical_count}")
        clean, invalid = validate_rows(clean, parser)
        conflicts = []
        if split in {"train", "synthetic"}:
            clean, conflicts = remove_conflicts(clean)
        stage[split] = clean
        audit["cleanup"][split] = {"overlap_removed": len(overlapped), "duplicate_pairs_removed": len(repeated), "research_reproduced_count": historical_count, "structure_removed_pairs": len(invalid), "conflicting_gold_removed_pairs": len(conflicts), "final_pairs": len(clean)}
        for row in overlapped + repeated + invalid + conflicts:
            removals.append({"pair_id": pair_id(row), "source_split": split, "source_row": row["source_row"], "reasons": row["removal_reasons"], "gold_checks": row.get("gold_checks", {})})
    stage["test"], _ = validate_rows(source["test"], parser, preserve=True)
    audit["gold_validation"]["test"] = {language: sum(bool(r["gold_checks"][language]) for r in stage["test"]) for language in ["english", "hinglish"]}
    audit["test_repeated_pair_count"] = len(source["test"]) - len(deduplicate(source["test"])[0])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if audit_overlap(stage[a], stage[b])["rows_matching_either"]:
            raise ValueError(f"Output split overlap remains: {a}, {b}")
    for split in ["train", "validation", "test"]:
        audit["outputs"].append(write_jsonl(ROOT / "data/processed" / f"{split}.jsonl", processed_rows(stage[split], split)))
    if args.with_synthetic:
        # Synthetic does not provide a domain. Map the root intent to the unique
        # human-training domain; retain unknown when genuinely unmappable.
        mapping = collections.defaultdict(set)
        for row in source["train"]:
            match = re.match(r"\[IN:([^\s\]]+)", row["hinglish_target"].lstrip())
            if match:
                mapping[match.group(1)].add(row["domain"])
        for row in stage["synthetic"]:
            root_intent = re.match(r"\[IN:([^\s\]]+)", row["hinglish_target"].lstrip()).group(1)
            domains = mapping.get(root_intent, set())
            row["domain"] = next(iter(domains)) if len(domains) == 1 else "unknown"
        candidate, overlap_human = exclude_overlap(stage["synthetic"], stage["train"])
        for row in overlap_human:
            removals.append({"pair_id": pair_id(row), "source_split": "synthetic", "source_row": row["source_row"], "reasons": ["optional_pool_human_train_" + reason for reason in row["removal_reasons"]], "gold_checks": row.get("gold_checks", {})})
        audit["cleanup"]["synthetic"]["human_train_overlap_removed_for_optional_pool"] = len(overlap_human)
        audit["cleanup"]["synthetic"]["optional_pool_pairs"] = len(candidate)
        audit["outputs"].append(write_jsonl(ROOT / "data/processed/synthetic.cleaned.jsonl", processed_rows(candidate, "train", languages=("hinglish",))))
        selected = balanced_subset(candidate)
        audit["outputs"].append(write_jsonl(ROOT / "data/processed/synthetic.stage2-20k.jsonl", processed_rows(selected, "train", languages=("hinglish",))))
    audit["outputs"].append(write_jsonl(ROOT / "reports/data-audit-removals.jsonl", removals))
    sources = json.loads(sources_path.read_text())
    audit["source_manifest_sha256"] = sha256(sources_path.read_bytes())
    audit["code_sha256"] = sha256(Path(__file__).read_bytes())
    write_json(ROOT / "reports/data-audit.json", audit)
    write_json(ROOT / "data/processed/manifest.json", {"dataset_revision": DATA_REV, "base_model_id": MODEL_ID, "base_model_revision": MODEL_REV, "comparator_model_id": sources.get("comparator_model_id"), "comparator_model_revision": sources.get("comparator_model_revision"), "source_manifest_sha256": audit["source_manifest_sha256"], "prepare_script_sha256": audit["code_sha256"], "parser_sha256": sha256(Path(parser.__file__).read_bytes()), "stage1_source": "human_only", "files": audit["outputs"], "license_notice_paths": [x["path"] for x in sources["files"] if "/licenses/" in x["path"]]})
    write_audit_markdown(audit)
    append_work_record(audit)
    print(json.dumps({"status": "prepared", "cleanup": audit["cleanup"], "test_gold_validation": audit["gold_validation"], "outputs": audit["outputs"]}, indent=2))


def write_audit_markdown(audit: dict) -> None:
    lines = ["# Hinglish-TOP data audit", "", "Private preparation record. No newly generated labels or human review were used.", "", "| Split | Original pairs | Overlap + duplicate cleanup | Additional invalid/conflicting pairs | Final pairs |", "|---|---:|---:|---:|---:|"]
    for split, stats in audit["cleanup"].items():
        lines.append(f"| {split} | {audit['original_counts'][split]} | {stats['research_reproduced_count']} | {stats['structure_removed_pairs'] + stats['conflicting_gold_removed_pairs']} | {stats.get('optional_pool_pairs', stats['final_pairs'])} |")
    if "synthetic" in audit["cleanup"]:
        synthetic = audit["cleanup"]["synthetic"]
        lines += ["", f"Synthetic structure/conflict cleanup retains {synthetic['final_pairs']:,} pairs. A final {synthetic['human_train_overlap_removed_for_optional_pool']:,} pairs are excluded because their normalized English source or Hinglish query matches retained human training data; the separate optional output therefore has {synthetic['optional_pool_pairs']:,} rows. This last exclusion is additional to the invalid/conflicting column above."]
    lines += ["", "The original 6,513 test pairs are preserved without changing text or targets. Each human pair produces one Hinglish and one English JSONL record. Pair IDs retain the original source split and row; group IDs hash normalized English sources.", "", "## Leakage policy", "", "Test has priority. Training excludes normalized English-source OR Hinglish-query matches to original test and validation. Validation excludes test matches. Normalization is NFKC + lowercase + collapsed whitespace; it is used only to compare duplicates. Duplicates are keyed by normalized Hinglish and whitespace-canonicalized target. Test targets are never corrected based on model behavior.", "", "## Automatic gold checks", "", "Training/development pairs with invalid structures or source-leaf mismatches are excluded using the shared parser; all test pairs remain and carry validation-error flags. Training pairs whose normalized input has conflicting gold are excluded. These checks do not establish semantic correctness of the inherited human labels.", "", "## Optional synthetic data", "", "Synthetic data is separate and never included in stage-one train.jsonl. Its original file has four columns, omitting domain. Any domain shown is inherited only where the root intent has one unambiguous human-training domain. A deterministic 20,000-pair selection balances root intents by round-robin, using a seeded SHA256 order. It is an optional later ablation, not newly human-verified data.", "", "## Licenses and scope", "", "The Google repository's Apache2 license and the original Meta TOPv2 archive's CC-BY-SA4 license are both preserved. Do not describe the entire dataset/release as Apache-only. Weight-license treatment remains part of the private publication review. Base-model pretraining contamination cannot be excluded.", "", "Full machine-readable counts, source hashes, output hashes, and removal reason IDs are in data-audit.json and data-audit-removals.jsonl."]
    (ROOT / "reports/data-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_work_record(audit: dict) -> None:
    path = ROOT / "records/data-work.md"
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {timestamp} — deterministic preparation\n\n")
        handle.write(f"Pinned Google dataset {DATA_REV}; Qwen base {MODEL_REV}. Downloaded only public dataset, license and small metadata files; no weights or paid APIs. Verified pre-researched source SHA256s and split sizes. Reproduced 2,312 human train / 1,064 development pairs before shared-parser checks. Original test targets remain untouched.\n\n")
        for split, stats in audit["cleanup"].items():
            handle.write(f"- {split}: {json.dumps(stats, sort_keys=True)}\n")
        handle.write("\nSee reports/data-audit.json and data/processed/manifest.json for hashes and exclusions. Synthetic output, when present, is optional and separate from stage-one human training.\n")


if __name__ == "__main__":
    main()
