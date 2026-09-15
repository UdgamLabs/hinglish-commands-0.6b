"""Thin adapter-only API around the preserved scientific inference contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading

from ._vendor import runtime
from ._vendor.prompts import make_prompt
from ._vendor.retrieval import Retriever
from ._vendor.top import inspect_top, parse_top, to_json_tree, walk_nodes

MAX_TEXT_CHARS = 2000
EXPECTED_CONFIG = {
    "base_model": runtime.BASE_MODEL,
    "revision": runtime.BASE_REVISION,
    "source_adapter_sha256": "85be8189b09cefa38dc7e50723042d606af19d9f2ad8ceab1cdb123f5eec65af",
    "shots": 4, "seed": 20260912, "max_new_tokens": 768,
    "max_input_tokens": 4096, "enable_thinking": False, "format": "adapter",
    "train_sha256": "a5c62edc4bb8e0b395b95c991fc364bf1ad25850d7a46c2bf6edf503f494dc11",
    "dtype": "bfloat16", "executes_actions": False,
}
LIMITATION = (
    "Experimental preview for alarm, event, messaging, music, navigation, reminder, "
    "timer and weather commands in Romanized Hinglish or English. Other requests "
    "may be misclassified. Structure and copied-text checks do not establish semantic "
    "correctness. Dates, names and slot text are not translated or resolved. No action is executed."
)


def validate_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Provide a nonempty command of at most {MAX_TEXT_CHARS} characters")


def validate_config(config):
    """Reject a different model/recipe, including a diagnostic merged export."""
    if not isinstance(config, dict):
        raise ValueError("Expected an adapter package configuration")
    for key, expected in EXPECTED_CONFIG.items():
        if type(config.get(key)) is not type(expected) or config[key] != expected:
            raise ValueError(f"This SDK requires the selected adapter's exact {key} setting")


def validate_load_options(device, dtype, allow_base_download):
    if device not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps or cuda")
    if dtype not in (None, "bfloat16", "float32"):
        raise ValueError("dtype must be bfloat16 or float32")
    if type(allow_base_download) is not bool:
        raise ValueError("allow_base_download must be a boolean")


def _check_package_cache(path):
    """Reject links before downloading into or loading a materialized package."""
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("The materialized package cache must not contain symlink ancestors")
    if path.exists():
        if not path.is_dir():
            raise ValueError("The materialized package cache is not a directory")
        if any(item.is_symlink() for item in path.rglob("*")):
            raise ValueError("The materialized package cache must contain files, not symlinks")


def _package_directory(repo_id, revision, cache_dir):
    default_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"
    base = Path(cache_dir) if cache_dir is not None else Path(os.environ.get("HF_HOME", str(default_home)))
    package = (base.expanduser().absolute() / "udgam-packages" / repo_id / revision)
    _check_package_cache(package)
    package = package.resolve()
    package.mkdir(parents=True, exist_ok=True)
    _check_package_cache(package)
    return package


def preview_result(output, text, schema):
    """Return an ordered tree only after the original structural/copy/schema checks.

    This is a preview filter, not an action authorization or a semantic verifier.
    No malformed output, spelling, case, relative date or intent is repaired.
    """
    validate_text(text)
    prediction = output.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError("The model returned an invalid prediction type")
    checks = inspect_top(prediction, text)
    labels = {n.label for n in walk_nodes(parse_top(prediction))} if checks["syntax_valid"] else set()
    unknown = sorted(labels - set(schema["labels"]))
    truncated = bool(output.get("truncated", False))
    if truncated:
        status, message = "truncated", "Generation stopped before its end marker; no preview accepted."
    elif output.get("error"):
        status, message = "generation_error", "Generation failed; no preview accepted."
    elif not checks["syntax_valid"]:
        status, message = "malformed", "The output is not one valid TOP tree; it has not been repaired."
    elif unknown:
        status, message = "unknown_labels", "The output uses labels outside the training schema."
    elif not checks["source_valid"]:
        status, message = "source_mismatch", "Some output words are absent from the command."
    else:
        status, message = "preview", "Structure and source-text checks passed. Review the interpretation."
    accepted = status == "preview"
    return {**output, "ok": accepted, "status": status, "message": message,
            "raw_top": prediction, "tree": to_json_tree(prediction) if accepted else None,
            "checks": {**checks, "unknown_labels": unknown, "truncated": truncated,
                       "generation_error": bool(output.get("error"))},
            "executed_action": False, "limitation": LIMITATION}


class CommandParser:
    """Preload once, then parse locally. Each instance serializes generation calls."""

    @classmethod
    def from_local(cls, model_dir, *, device="auto", dtype=None, allow_base_download=False):
        """Load a complete adapter package; all assets must be local by default.

        An explicit float32 override supports a CPU that cannot execute BF16,
        but it changes the benchmarked inference precision.
        """
        validate_load_options(device, dtype, allow_base_download)
        root = Path(model_dir).expanduser().resolve()
        config = json.loads((root / "udgam_config.json").read_text(encoding="utf-8"))
        validate_config(config)
        # The preserved loader hashes every manifest entry, verifies model identity,
        # adapter fingerprint and train-derived schema; it never imports package code.
        model, tokenizer, checked = runtime.load_export(root, device,
            allow_base_download=allow_base_download, dtype=dtype)
        if checked != config:
            raise ValueError("The model package configuration changed during loading")
        from transformers import set_seed
        set_seed(config["seed"])
        instance = cls()
        instance._root, instance._config = root, dict(config)
        instance._model, instance._tokenizer = model, tokenizer
        instance._schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
        instance._retriever = Retriever(runtime.load_rows(root / "retrieval_train.jsonl", "train"))
        instance._lock = threading.Lock()
        return instance

    @classmethod
    def from_hub(cls, repo_id, *, revision, cache_dir=None, device="auto", dtype=None,
                 allow_base_download=False, token=False):
        """Explicitly download one Hub adapter release at an immutable commit.

        No default revision or request-time network access. The adapter snapshot
        request omits Hub credentials by default. Set token=True to opt into the
        user's existing Hugging Face authentication for a private repository.
        Its pinned base is a separate opt-in and uses the preserved loader's
        normal Hugging Face configuration. Token strings are not accepted here.
        Adapter files are materialized with local_dir, so normal Hub blob-cache
        symlinks never bypass or conflict with the export's containment checks.
        """
        if not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
            raise ValueError("repo_id must be a namespace/repository identifier")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be the exact 40-character release commit, not main or a tag")
        if type(token) is not bool:
            raise ValueError("token must be False or True; configure credentials through Hugging Face")
        validate_load_options(device, dtype, allow_base_download)
        package = _package_directory(repo_id, revision, cache_dir)
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=repo_id, revision=revision,
                                 local_dir=str(package), token=token)
        _check_package_cache(package)
        if Path(path).absolute() != package:
            raise ValueError("Hub download did not return the requested materialized package directory")
        return cls.from_local(package, device=device, dtype=dtype,
                              allow_base_download=allow_base_download)

    @property
    def settings(self):
        """Return a copy of recipe metadata, never mutable model configuration."""
        return {**self._config, "loaded_dtype": str(self._model.dtype),
                "device": str(self._model.device),
                "precision_matches_recorded_recipe": str(self._model.dtype) == "torch.bfloat16"}

    def parse(self, text):
        validate_text(text)
        with self._lock:
            prompt, ids = make_prompt(self._tokenizer, {"text": text}, self._schema,
                                     self._retriever, self._config["shots"])
            output = runtime.generate_batch(self._model, self._tokenizer, [prompt],
                self._config["max_new_tokens"], self._config["max_input_tokens"])[0]
        result = preview_result(output, text, self._schema)
        return {**result, "example_ids": ids, "base_model": self._config["base_model"],
                "base_revision": self._config["revision"],
                "source_adapter_sha256": self._config["source_adapter_sha256"],
                "shots": self._config["shots"], "seed": self._config["seed"],
                "loaded_dtype": str(self._model.dtype),
                "precision_matches_recorded_recipe": str(self._model.dtype) == "torch.bfloat16"}

    def make_demo_app(self):
        """Create the existing guarded local UI around this preloaded adapter."""
        from ._demo import DemoApp, ModelEngine
        engine = ModelEngine(model=self._model, tokenizer=self._tokenizer,
            schema=self._schema, retriever=self._retriever, shots=self._config["shots"],
            engine_id="fine_tuned", label="Udgam Hinglish Commands · selected adapter",
            source="selected_adapter", max_new_tokens=self._config["max_new_tokens"],
            max_input_tokens=self._config["max_input_tokens"])
        # Share this instance's generation lock even if the API and UI are used together.
        engine.predict = self.parse
        return DemoApp([engine], self._schema)
