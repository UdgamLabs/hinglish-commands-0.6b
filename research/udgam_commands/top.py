"""Lossless, case-sensitive TOP trees and deliberately limited source checks.

Whitespace between tokens is the only scoring normalization.  The JSON form uses
ordered children, never a label-keyed map, because TOP permits repeated slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterator


class TopParseError(ValueError):
    """A string is not one complete TOP tree."""


@dataclass(frozen=True)
class TopNode:
    label: str
    children: tuple["TopNode | str", ...]

    def to_dict(self) -> dict:
        return {"label": self.label, "children": [
            child.to_dict() if isinstance(child, TopNode) else child
            for child in self.children
        ]}


_TOKEN = re.compile(r"\[|\]|[^\s\[\]]+")
_LABEL = re.compile(r"(?:IN|SL):[^\s\[\]:]+\Z")


def parse_top(text: str, *, max_depth: int = 128) -> TopNode:
    """Parse exactly one IN-rooted tree; do not strip fences or repair output."""
    if not isinstance(text, str):
        raise TopParseError("TOP input must be a string")
    tokens = _TOKEN.findall(text)
    position = 0

    def node(depth: int) -> TopNode:
        nonlocal position
        if depth > max_depth:
            raise TopParseError(f"Tree exceeds maximum depth {max_depth}")
        if position >= len(tokens) or tokens[position] != "[":
            raise TopParseError(f"Expected '[' at token {position}")
        position += 1
        if position >= len(tokens) or not _LABEL.fullmatch(tokens[position]):
            raise TopParseError(f"Expected IN:/SL: label at token {position}")
        label = tokens[position]
        position += 1
        children: list[TopNode | str] = []
        while position < len(tokens) and tokens[position] != "]":
            if tokens[position] == "[":
                children.append(node(depth + 1))
            else:
                children.append(tokens[position])
                position += 1
        if position >= len(tokens):
            raise TopParseError(f"Unclosed node {label}")
        position += 1
        return TopNode(label, tuple(children))

    root = node(0)
    if position != len(tokens):
        raise TopParseError(f"Unexpected trailing content at token {position}")
    if not root.label.startswith("IN:"):
        raise TopParseError("The root must be an IN: node")
    return root


def serialize_top(tree: TopNode) -> str:
    return "[ " + tree.label + (" " if tree.children else "") + " ".join(
        serialize_top(child) if isinstance(child, TopNode) else child
        for child in tree.children
    ) + " ]"


def canonicalize_top(text: str | TopNode) -> str:
    """Ignore bracket/token whitespace only; preserve case and Unicode bytes."""
    return serialize_top(parse_top(text) if isinstance(text, str) else text)


def to_json_tree(tree: str | TopNode) -> dict:
    return (parse_top(tree) if isinstance(tree, str) else tree).to_dict()


def from_json_tree(value: dict) -> TopNode:
    """Read the exact ordered JSON representation and validate its grammar."""
    def convert(obj: dict, depth: int = 0) -> TopNode:
        if depth > 128:
            raise TopParseError("JSON tree exceeds maximum depth 128")
        if not isinstance(obj, dict) or set(obj) != {"label", "children"}:
            raise TopParseError("JSON node needs exactly label and children")
        if not isinstance(obj["label"], str) or not _LABEL.fullmatch(obj["label"]):
            raise TopParseError("Invalid JSON node label")
        if not isinstance(obj["children"], list):
            raise TopParseError("JSON children must be an ordered list")
        children = []
        for child in obj["children"]:
            if isinstance(child, dict):
                children.append(convert(child, depth + 1))
            elif isinstance(child, str) and _TOKEN.findall(child) == [child] and child not in "[]":
                children.append(child)
            else:
                raise TopParseError("JSON leaves must be individual nonempty TOP tokens")
        return TopNode(obj["label"], tuple(children))
    result = convert(value)
    if not result.label.startswith("IN:"):
        raise TopParseError("The root must be an IN: node")
    return result


def walk_nodes(tree: TopNode) -> Iterator[TopNode]:
    yield tree
    for child in tree.children:
        if isinstance(child, TopNode):
            yield from walk_nodes(child)


def leaf_tokens(tree: TopNode) -> list[str]:
    result = []
    for child in tree.children:
        result.extend(leaf_tokens(child) if isinstance(child, TopNode) else [child])
    return result


def leaf_spans(tree: TopNode, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], str]]:
    """Yield contiguous runs of leaf tokens, including unlabelled root words."""
    run: list[str] = []
    run_start = 0
    for index, child in enumerate(tree.children):
        if isinstance(child, TopNode):
            if run:
                yield path + (run_start,), " ".join(run)
                run = []
            yield from leaf_spans(child, path + (index,))
        else:
            if not run:
                run_start = index
            run.append(child)
    if run:
        yield path + (run_start,), " ".join(run)


def _copy_text(text: str) -> str:
    # NFC and whitespace-insensitive containment are diagnostics ONLY; neither
    # changes the target or exact-match normalization. Case/punctuation remain.
    return "".join(unicodedata.normalize("NFC", text).split())


def check_source_values(tree: str | TopNode, text: str) -> dict:
    tree = parse_top(tree) if isinstance(tree, str) else tree
    source = _copy_text(text)
    spans = list(leaf_spans(tree))
    issues = [{"path": list(path), "value": value} for path, value in spans
              if _copy_text(value) not in source]
    return {"source_valid": not issues, "source_issues": issues,
            "leaf_spans": len(spans), "copied_leaf_spans": len(spans) - len(issues)}


def inspect_top(target: str, text: str | None = None) -> dict:
    try:
        tree = parse_top(target)
    except TopParseError as error:
        return {"syntax_valid": False, "canonical": None, "error": str(error),
                "source_valid": None, "source_issues": [], "leaf_spans": 0,
                "copied_leaf_spans": 0}
    result = {"syntax_valid": True, "canonical": canonicalize_top(tree),
              "error": None, "source_valid": None, "source_issues": [],
              "leaf_spans": 0, "copied_leaf_spans": 0}
    if text is not None:
        result.update(check_source_values(tree, text))
    return result
