#!/usr/bin/env python3
"""Score preserved JSONL predictions without contacting an API or running a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udgam_commands.scoring import evaluation_report


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", "--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = evaluation_report(read_jsonl(args.predictions),
                               read_jsonl(args.baseline) if args.baseline else None,
                               replicates=args.bootstrap_samples, seed=args.seed)
    report["inputs"] = {"predictions": str(args.predictions),
                        "baseline": str(args.baseline) if args.baseline else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out), "original": report["original"]["overall"],
                      "deduplicated": report["deduplicated"]["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
