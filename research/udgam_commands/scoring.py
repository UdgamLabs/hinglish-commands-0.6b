"""Deterministic TOP scoring, transparent denominators and paired uncertainty."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import unicodedata
from typing import Iterable

from .top import (TopNode, TopParseError, canonicalize_top, check_source_values,
                  inspect_top, leaf_tokens, parse_top, walk_nodes)


def _slots(tree: TopNode | None) -> Counter:
    return Counter((node.label, " ".join(leaf_tokens(node)))
                   for node in walk_nodes(tree) if node.label.startswith("SL:")) if tree else Counter()


def _parsed(text: str) -> TopNode | None:
    try:
        return parse_top(text)
    except TopParseError:
        return None


def score_prediction(prediction: str, target: str, text: str | None = None, *,
                     truncated: bool = False, error: str | None = None) -> dict:
    """Full EM preserves all hierarchy/order/case. Slot scores are multisets.

    Malformed gold never silently becomes a correct example: full-set EM marks
    it incorrect, and the report also supplies the valid-gold denominator.
    Source-copy diagnostics do not prove semantic correctness.
    """
    pred, gold = _parsed(prediction), _parsed(target)
    pred_info, gold_info = inspect_top(prediction, text), inspect_top(target, text)
    pred_slots, gold_slots = _slots(pred), _slots(gold)
    failed = bool(truncated or error)
    tp = sum((pred_slots & gold_slots).values()) if gold and not failed else 0
    return {
        "exact_match": bool(not failed and pred is not None and gold is not None
                            and canonicalize_top(pred) == canonicalize_top(gold)),
        "syntax_valid": pred is not None,
        "generation_failed": failed,
        "gold_syntax_valid": gold is not None,
        "intent_correct": bool(not failed and pred is not None and gold is not None and pred.label == gold.label),
        "slot_tp": tp,
        "slot_fp": sum(pred_slots.values()) - tp if gold else 0,
        "slot_fn": sum(gold_slots.values()) - tp if gold else 0,
        "slot_evaluable": gold is not None,
        "source_valid": False if failed and text is not None else pred_info["source_valid"],
        "source_issues": pred_info["source_issues"],
        "leaf_spans": pred_info["leaf_spans"],
        "copied_leaf_spans": 0 if failed else pred_info["copied_leaf_spans"],
        "gold_source_valid": gold_info["source_valid"],
        "gold_source_issues": gold_info["source_issues"],
        "prediction_error": pred_info["error"],
        "gold_error": gold_info["error"],
    }


def _score_row(row: dict) -> dict:
    return score_prediction(row["prediction"], row["target"], row.get("text"),
                            truncated=row.get("truncated", False), error=row.get("error"))


def _ratio(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def _summarize(scored: list[tuple[dict, dict]]) -> dict:
    n = len(scored)
    scores = [score for _, score in scored]
    correct = sum(score["exact_match"] for score in scores)
    valid_gold = sum(score["gold_syntax_valid"] for score in scores)
    tp, fp, fn = (sum(score[name] for score in scores) for name in ("slot_tp", "slot_fp", "slot_fn"))
    checked_source = [score for row, score in scored if isinstance(row.get("text"), str)]
    pred_spans = sum(score["leaf_spans"] for score in scores)
    return {
        "n": n, "correct": correct, "exact_match": _ratio(correct, n),
        "generation_failed": sum(score["generation_failed"] for score in scores),
        "generation_success_rate": _ratio(sum(not score["generation_failed"] for score in scores), n),
        "syntax_valid": sum(score["syntax_valid"] for score in scores),
        "syntax_valid_rate": _ratio(sum(score["syntax_valid"] for score in scores), n),
        "gold_syntax_valid": valid_gold, "gold_syntax_invalid": n - valid_gold,
        "valid_gold_exact_match": _ratio(correct, valid_gold),
        "root_intent_accuracy": _ratio(sum(score["intent_correct"] for score in scores), n),
        "slot_tp": tp, "slot_fp": fp, "slot_fn": fn,
        "slot_precision": _ratio(tp, tp + fp), "slot_recall": _ratio(tp, tp + fn),
        "slot_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "slot_evaluable_examples": valid_gold,
        "source_checked_examples": len(checked_source),
        "source_valid_rate": _ratio(sum(score["source_valid"] is True for score in checked_source), len(checked_source)),
        "predicted_leaf_spans": pred_spans,
        "copied_leaf_spans": sum(score["copied_leaf_spans"] for score in scores),
        "leaf_copy_rate": _ratio(sum(score["copied_leaf_spans"] for score in scores), pred_spans),
        "gold_source_invalid": sum(score["gold_source_valid"] is False for score in scores),
    }


def _unique_rows(rows: Iterable[dict]) -> list[dict]:
    rows = list(rows)
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Every prediction needs a nonempty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction ids are not allowed")
    for row in rows:
        if not isinstance(row.get("target"), str) or not isinstance(row.get("prediction"), str):
            raise ValueError(f"Row {row['id']} needs string target and prediction")
    return rows


def evaluate_predictions(rows: Iterable[dict]) -> dict:
    rows = _unique_rows(rows)
    scored = [(row, _score_row(row)) for row in rows]
    by_language, by_domain, by_domain_language = defaultdict(list), defaultdict(list), defaultdict(list)
    for row, score in scored:
        by_language[row.get("language", "unknown")].append((row, score))
        by_domain[row.get("domain", "unknown")].append((row, score))
        by_domain_language[f"{row.get('domain', 'unknown')}:{row.get('language', 'unknown')}"].append((row, score))
    issues = [{"id": row["id"], "target": row["target"], "text": row.get("text"),
               "syntax_error": score["gold_error"], "source_issues": score["gold_source_issues"]}
              for row, score in scored if not score["gold_syntax_valid"] or score["gold_source_valid"] is False]
    return {"overall": _summarize(scored),
            "by_language": {key: _summarize(value) for key, value in sorted(by_language.items())},
            "by_domain": {key: _summarize(value) for key, value in sorted(by_domain.items())},
            "by_domain_language": {key: _summarize(value) for key, value in sorted(by_domain_language.items())},
            "gold_issues": issues}


def normalize_dedup_text(text: str) -> str:
    """Deduplication only; this normalization is NEVER used for exact match."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def deduplicate_predictions(rows: Iterable[dict]) -> tuple[list[dict], dict]:
    """One representative per language + normalized query, independent of output.

    Conflicting gold remains in the original view and is reported explicitly.
    The deterministic lowest id is the representative, not the easiest target.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in _unique_rows(rows):
        if not isinstance(row.get("text"), str):
            raise ValueError("Deduplicated evaluation requires input text")
        groups[(row.get("language", "unknown"), normalize_dedup_text(row["text"]))].append(row)
    selected, removed, conflicts = [], [], []
    for _, group in sorted(groups.items()):
        group.sort(key=lambda row: row["id"])
        selected.append(group[0])
        removed.extend(row["id"] for row in group[1:])
        # Use canonical tree when valid, raw token whitespace otherwise. Never
        # repair malformed labels, remove them, or choose gold by prediction.
        targets = {inspect_top(row["target"])["canonical"] or " ".join(row["target"].split()) for row in group}
        if len(targets) > 1:
            conflicts.append({"representative_id": group[0]["id"], "ids": [row["id"] for row in group],
                              "targets": [row["target"] for row in group]})
    selected.sort(key=lambda row: row["id"])
    return selected, {"key": "language + NFKC/casefold/whitespace-normalized input text",
                      "representative": "lexicographically smallest id", "removed_ids": removed,
                      "removed_count": len(removed), "conflicting_gold_groups": conflicts}


def _align_pairs(baseline: Iterable[dict], candidate: Iterable[dict]) -> list[tuple[dict, dict]]:
    baseline, candidate = _unique_rows(baseline), _unique_rows(candidate)
    left, right = {row["id"]: row for row in baseline}, {row["id"]: row for row in candidate}
    if set(left) != set(right):
        raise ValueError("Paired evaluation requires exactly the same prediction ids")
    aligned = []
    for identifier in sorted(left):
        a, b = left[identifier], right[identifier]
        for key in ("target", "text", "language", "domain", "group_id", "english_source", "split"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Paired metadata mismatch for {identifier}: {key}")
        aligned.append((a, b))
    return aligned


def _percentile(values: list[float], quantile: float) -> float:
    at = (len(values) - 1) * quantile
    lower, upper = math.floor(at), math.ceil(at)
    return values[lower] + (values[upper] - values[lower]) * (at - lower)


def paired_bootstrap(baseline: Iterable[dict], candidate: Iterable[dict], *,
                     replicates: int = 2000, seed: int = 42) -> dict:
    """Paired micro-EM difference; resample whole English-source clusters.

    Each draw samples G groups with replacement, retaining all their English /
    Hinglish rows. Unequal cluster sizes therefore produce unequal denominators.
    Results are candidate minus baseline; CI is a percentile interval, not a
    guarantee about unseen populations or evidence of training decontamination.
    """
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    aligned = _align_pairs(baseline, candidate)
    clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    source_to_group, group_to_source = {}, {}
    for left, right in aligned:
        group = left.get("group_id")
        source = left.get("english_source")
        if not isinstance(group, str) or not group or not isinstance(source, str) or not source:
            raise ValueError("Clustered bootstrap requires group_id and english_source")
        source = normalize_dedup_text(source)
        if source in source_to_group and source_to_group[source] != group:
            raise ValueError("The same English source occurs in multiple clusters")
        if group in group_to_source and group_to_source[group] != source:
            raise ValueError("A cluster contains different normalized English sources")
        source_to_group[source], group_to_source[group] = group, source
        clusters[group][0] += 1
        clusters[group][1] += _score_row(left)["exact_match"]
        clusters[group][2] += _score_row(right)["exact_match"]
    stats = [clusters[key] for key in sorted(clusters)]
    n = sum(item[0] for item in stats)
    base_correct = sum(item[1] for item in stats)
    cand_correct = sum(item[2] for item in stats)
    result = {"metric": "complete_tree_exact_match", "difference_direction": "candidate_minus_baseline",
              "cluster_key": "group_id (normalized English source)", "n": n, "clusters": len(stats),
              "replicates": replicates, "seed": seed, "confidence": 0.95,
              "baseline_exact_match": _ratio(base_correct, n), "candidate_exact_match": _ratio(cand_correct, n),
              "difference": _ratio(cand_correct - base_correct, n), "ci_low": None, "ci_high": None,
              "warning": "Fewer than two clusters; interval is uninformative" if len(stats) < 2 else None}
    if not stats:
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        denominator, delta = 0, 0
        for _ in stats:
            count, a, b = stats[rng.randrange(len(stats))]
            denominator += count
            delta += b - a
        samples.append(delta / denominator)
    samples.sort()
    result.update(ci_low=_percentile(samples, 0.025), ci_high=_percentile(samples, 0.975))
    return result


def evaluation_report(rows: Iterable[dict], baseline_rows: Iterable[dict] | None = None, *,
                      replicates: int = 2000, seed: int = 42) -> dict:
    rows = _unique_rows(rows)
    deduped, dedup_audit = deduplicate_predictions(rows)
    report = {"protocol_version": "1.0", "rates_are_fractions": True,
              "original": evaluate_predictions(rows), "deduplicated": evaluate_predictions(deduped),
              "deduplication": dedup_audit}
    if baseline_rows is not None:
        baseline_rows = _unique_rows(baseline_rows)
        _align_pairs(baseline_rows, rows)
        baseline_map = {row["id"]: row for row in baseline_rows}
        selected_baseline = [baseline_map[row["id"]] for row in deduped]
        report["paired_comparison"] = {}
        for name, left, right in (("original", baseline_rows, rows),
                                  ("deduplicated", selected_baseline, deduped)):
            comparison = {"overall": paired_bootstrap(left, right, replicates=replicates, seed=seed)}
            for key in ("language", "domain"):
                comparison[f"by_{key}"] = {
                    value: paired_bootstrap([row for row in left if row.get(key, "unknown") == value],
                                            [row for row in right if row.get(key, "unknown") == value],
                                            replicates=replicates, seed=seed)
                    for value in sorted({row.get(key, "unknown") for row in right})}
            report["paired_comparison"][name] = comparison
    return report
