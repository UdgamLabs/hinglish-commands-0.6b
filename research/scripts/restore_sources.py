#!/usr/bin/env python3
"""Check historical source snapshots; --download explicitly restores missing files.

Run from any directory. Paths are relative to this script's research/ parent.
No model weights, credentials or dataset labels are interpreted. Default mode
only checks byte identities and lists missing sources. Existing changed files
are never overwritten. Historical API responses may no longer be recoverable
byte-for-byte; a changed upstream response fails its fixed hash check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import tempfile
from urllib.parse import urlsplit
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "7a263ec530b0eff88a9488b9db544b36817263555420ea79c829057b16e6ccef"
ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com", "huggingface.co", "dl.fbaipublicfiles.com"})
PINS = {"dataset_revision": "fdd3998a6573130659bfa1ce4b1ebe698df2bf3a",
        "model_id": "Qwen/Qwen3-0.6B", "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "comparator_model_id": "KingNish/Qwen3-0.6b-hinglish-2",
        "comparator_model_revision": "fecc11f386ae5d30f84cfd8bb90b1512b5ba6188"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("Invalid source path")
    parts = PurePosixPath(name)
    if (parts.is_absolute() or str(parts) != name or ".." in parts.parts
            or len(parts.parts) < 3 or parts.parts[:2] not in
            (("data", "raw"), ("data", "licenses"), ("data", "model_metadata"))):
        raise ValueError("Source path is outside the fixed data directories")
    root = Path(root).resolve()
    target = root / name
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symlink source paths are forbidden: {name}")
        current = current.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError("Source path escapes research root")
    return target


def validate_url(url):
    if not isinstance(url, str):
        raise ValueError("Source URL must be a string")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("Only HTTPS on the fixed upstream host allowlist is permitted")
    return url


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_download(url):
    validate_url(url)
    context = (ssl.create_default_context(cafile="/etc/ssl/cert.pem")
               if Path("/etc/ssl/cert.pem").is_file() else ssl.create_default_context())
    # No auth headers, implicit proxy configuration, retries or shell fallback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context), CheckedRedirect())
    response = opener.open(urllib.request.Request(url,
        headers={"User-Agent": "Udgam-Source-Restorer/1", "Accept-Encoding": "identity"}), timeout=120)
    try:
        validate_url(response.geturl())
        if response.status != 200:
            raise ValueError("Upstream did not return HTTP 200")
    except Exception:
        response.close()
        raise
    return response


def load_manifest(root):
    # No CLI override: only the preserved manifest is an authorized download list.
    root = Path(root).resolve()
    path = root / "data/source-manifest.json"
    if (root / "data").is_symlink() or path.is_symlink():
        raise ValueError("Symlink manifest paths are forbidden")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Historical source manifest identity changed")
    manifest = json.loads(raw)
    if any(manifest.get(k) != v for k, v in PINS.items()):
        raise ValueError("Historical public model/dataset pins changed")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Source manifest has no files")
    seen = set()
    for item in entries:
        if not isinstance(item, dict) or not {"path", "url", "bytes", "sha256"}.issubset(item):
            raise ValueError("Invalid source manifest entry")
        safe_path(root, item["path"]); validate_url(item["url"])
        if (item["path"] in seen or type(item["bytes"]) is not int or item["bytes"] < 0
                or not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid or repeated source identity")
        seen.add(item["path"])
        if "archive_member" in item and item["archive_member"] not in ("LICENSE", "README"):
            raise ValueError("Only the two historical archive notices may be read")
    return manifest


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"]


def publish_stream(root, entry, stream):
    """Publish verified bytes atomically without replacing a concurrently created file."""
    target = safe_path(root, entry["path"])
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing source: {entry['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target = safe_path(root, entry["path"])
    fd, name = tempfile.mkstemp(prefix=".restore-", dir=target.parent)
    temp = Path(name)
    try:
        total, h = 0, hashlib.sha256()
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = stream.read(min(1024 * 1024, entry["bytes"] - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > entry["bytes"]:
                    raise ValueError(f"Downloaded size exceeds fixed identity: {entry['path']}")
                h.update(chunk); handle.write(chunk)
            if total != entry["bytes"] or h.hexdigest() != entry["sha256"]:
                raise ValueError(f"Downloaded size/hash differs from historical source: {entry['path']}")
            handle.flush(); os.fsync(handle.fileno())
        target = safe_path(root, entry["path"])
        # Same-directory hard-link publication is atomic and fails if target appeared.
        # Unlike os.replace, it cannot overwrite a changed source during a race.
        os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def restore(root=ROOT, *, download=False):
    manifest = load_manifest(root)
    missing, changed = [], []
    for item in manifest["files"]:
        path = safe_path(root, item["path"])
        if not path.exists():
            missing.append(item)
        elif not matches(path, item):
            changed.append(item["path"])
    result = {"manifest_sha256": MANIFEST_SHA256, "download_enabled": download,
              "checked_files": len(manifest["files"]), "verified_existing": len(manifest["files"]) - len(missing) - len(changed),
              "missing": [{k: row[k] for k in ("path", "bytes", "sha256")} for row in missing],
              "changed": changed, "restored": [], "status": "changed_sources" if changed else "missing_sources" if missing else "complete"}
    # A changed source blocks the whole operation before any request or write.
    if changed or not download:
        return result
    direct = {item["url"]: item for item in manifest["files"] if "archive_member" not in item}
    for item in sorted(missing, key=lambda x: "archive_member" in x):
        if "archive_member" in item:
            archive = direct.get(item["url"])
            if not archive:
                raise ValueError("Archive member lacks a separately pinned source archive")
            path = safe_path(root, archive["path"])
            if not matches(path, archive):
                raise ValueError("Source archive failed verification before notice extraction")
            with zipfile.ZipFile(path) as zf:
                if zf.namelist().count(item["archive_member"]) != 1:
                    raise ValueError("Archive notice is missing or duplicated")
                if zf.getinfo(item["archive_member"]).file_size != item["bytes"]:
                    raise ValueError("Archive notice size differs from recorded identity")
                with zf.open(item["archive_member"]) as stream:
                    publish_stream(root, item, stream)
        else:
            with open_download(item["url"]) as stream:
                publish_stream(root, item, stream)
        result["restored"].append(item["path"])
    result["status"] = "complete"
    result["missing"] = []
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Restore missing sources at fixed URLs, sizes and hashes")
    parser.add_argument("--check", action="store_true", help="Explicit read-only check (the default)")
    args = parser.parse_args(argv)
    if args.check and args.download:
        parser.error("Choose --check or --download")
    try:
        result = restore(download=args.download)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "message": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
