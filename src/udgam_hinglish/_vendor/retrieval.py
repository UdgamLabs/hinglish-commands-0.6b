"""Deterministic train-only character TF-IDF retrieval, with no dependencies."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import unicodedata
from typing import Iterable


def _features(text: str) -> Counter:
    normalized = " " + " ".join(unicodedata.normalize("NFC", text).casefold().split()) + " "
    return Counter(normalized[start:start + size] for size in (3, 4, 5)
                   for start in range(max(0, len(normalized) - size + 1)))


class Retriever:
    def __init__(self, train_rows: Iterable[dict]):
        self.rows = sorted((dict(row) for row in train_rows), key=lambda row: str(row["id"]))
        if not self.rows:
            raise ValueError("Retriever needs at least one training row")
        if any(row.get("split") != "train" for row in self.rows):
            raise ValueError("Retrieval index may contain training rows only")
        if len({row["id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Training retrieval ids must be unique")
        if any(not isinstance(row.get("text"), str) or not isinstance(row.get("target"), str)
               for row in self.rows):
            raise ValueError("Each training row needs string text and target")
        features = [_features(row["text"]) for row in self.rows]
        df = Counter(gram for counts in features for gram in counts)
        self.idf = {gram: math.log((1 + len(self.rows)) / (1 + count)) + 1
                    for gram, count in df.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, counts in enumerate(features):
            for gram, weight in self._vector(counts).items():
                self.postings[gram].append((index, weight))

    def _vector(self, counts: Counter) -> dict[str, float]:
        weights = {gram: (1 + math.log(count)) * self.idf[gram]
                   for gram, count in counts.items() if gram in self.idf}
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {gram: weight / norm for gram, weight in weights.items()} if norm else {}

    def retrieve(self, text: str, k: int = 4, *, exclude_group_id: str | None = None) -> list[dict]:
        if k < 0:
            raise ValueError("k cannot be negative")
        scores: dict[int, float] = defaultdict(float)
        for gram, query_weight in self._vector(_features(text)).items():
            for index, weight in self.postings.get(gram, ()):
                scores[index] += query_weight * weight
        eligible = [index for index, row in enumerate(self.rows)
                    if exclude_group_id is None or row.get("group_id") != exclude_group_id]
        eligible.sort(key=lambda index: (-scores[index], str(self.rows[index]["id"])))
        return [dict(self.rows[index], retrieval_score=scores[index]) for index in eligible[:k]]

    def predict(self, text: str, *, exclude_group_id: str | None = None) -> str:
        examples = self.retrieve(text, 1, exclude_group_id=exclude_group_id)
        return examples[0]["target"] if examples else ""


def retrieve_examples(query: str, train_rows: Iterable[dict], k: int = 4,
                      *, exclude_group_id: str | None = None) -> list[dict]:
    """Convenience function; reuse Retriever for inference on many queries."""
    return Retriever(train_rows).retrieve(query, k, exclude_group_id=exclude_group_id)
