"""Artificial source snapshots and mocked downloads; never uses the network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.request
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "research/scripts/restore_sources.py"
spec = importlib.util.spec_from_file_location("restore_sources", SCRIPT)
restore = importlib.util.module_from_spec(spec); spec.loader.exec_module(restore)


def entry(path="data/raw/synthetic.tsv", data=b"artificial training source", **extra):
    return {"path": path, "url": "https://raw.githubusercontent.com/example/pinned/source.tsv",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **extra}


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); (self.root / "data").mkdir()

    def manifest(self, entries):
        raw = (json.dumps({**restore.PINS, "files": entries}, indent=2) + "\n").encode()
        (self.root / "data/source-manifest.json").write_bytes(raw)
        pin = patch.object(restore, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
        pin.start(); self.addCleanup(pin.stop)

    def test_default_only_lists_missing_identity_without_network_or_writes(self):
        item = entry(); self.manifest([item])
        with patch.object(restore, "open_download") as fetch:
            result = restore.restore(self.root)
        fetch.assert_not_called()
        self.assertEqual(result["status"], "missing_sources")
        self.assertEqual(result["missing"][0]["bytes"], item["bytes"])
        self.assertFalse((self.root / "data/raw").exists())

    def test_valid_existing_snapshot_is_checked_and_never_redownloaded(self):
        payload = b"kept"; item = entry(data=payload); self.manifest([item])
        target = self.root / item["path"]; target.parent.mkdir(); target.write_bytes(payload)
        with patch.object(restore, "open_download") as fetch:
            result = restore.restore(self.root, download=True)
        self.assertEqual(result["status"], "complete"); fetch.assert_not_called()
        self.assertEqual(target.read_bytes(), payload)

    def test_changed_existing_file_blocks_entire_download_operation(self):
        existing = entry("data/raw/train.tsv"); missing = entry()
        self.manifest([missing, existing])
        target = self.root / existing["path"]; target.parent.mkdir(); target.write_bytes(b"changed")
        with patch.object(restore, "open_download") as fetch:
            result = restore.restore(self.root, download=True)
        self.assertEqual(result["status"], "changed_sources")
        self.assertEqual(result["changed"], [existing["path"]])
        self.assertEqual(target.read_bytes(), b"changed"); fetch.assert_not_called()

    def test_download_saves_only_exact_bytes_and_leaves_no_temporary_file(self):
        payload = b"exact snapshot"; item = entry(data=payload); self.manifest([item])
        with patch.object(restore, "open_download", return_value=io.BytesIO(payload)) as fetch:
            result = restore.restore(self.root, download=True)
        fetch.assert_called_once_with(item["url"])
        self.assertEqual(result["status"], "complete"); self.assertEqual(result["missing"], [])
        self.assertEqual((self.root / item["path"]).read_bytes(), payload)
        self.assertEqual(list(self.root.rglob(".restore-*")), [])

    def test_hash_mismatch_and_oversize_never_publish_or_leave_partial_files(self):
        item = entry(data=b"right"); self.manifest([item])
        for payload in [b"wrong", b"far too many bytes", b"tiny"]:
            with self.subTest(payload=payload), patch.object(restore, "open_download", return_value=io.BytesIO(payload)), self.assertRaises(ValueError):
                restore.restore(self.root, download=True)
            self.assertFalse((self.root / item["path"]).exists())
            self.assertEqual(list(self.root.rglob(".restore-*")), [])

    def test_target_created_during_download_is_not_overwritten(self):
        payload = b"correct"; item = entry(data=payload); self.manifest([item])
        target = self.root / item["path"]
        class Concurrent(io.BytesIO):
            def read(self, count=-1):
                if not target.exists(): target.write_bytes(b"concurrent owner file")
                return super().read(count)
        with self.assertRaises(FileExistsError):
            restore.publish_stream(self.root, item, Concurrent(payload))
        self.assertEqual(target.read_bytes(), b"concurrent owner file")
        self.assertEqual(list(self.root.rglob(".restore-*")), [])

    def test_manifest_hash_change_is_rejected_before_download(self):
        self.manifest([entry()])
        with (self.root / "data/source-manifest.json").open("a") as handle: handle.write(" ")
        with patch.object(restore, "open_download") as fetch, self.assertRaisesRegex(ValueError, "identity changed"):
            restore.restore(self.root, download=True)
        fetch.assert_not_called()

    def test_path_escape_and_symlinks_are_rejected(self):
        for name in ["../outside", "/tmp/outside", "data/raw/../../outside", "data/raw/./x", "other/source", "data\\raw\\x"]:
            with self.subTest(name=name), self.assertRaises(ValueError): restore.safe_path(self.root, name)
        outside = self.root / "elsewhere"; outside.mkdir()
        (self.root / "data/raw").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): restore.safe_path(self.root, "data/raw/file.tsv")
        (self.root / "data/raw").unlink(); (self.root / "data/raw").mkdir()
        (self.root / "data/raw/file.tsv").symlink_to(outside / "missing")
        with self.assertRaises(ValueError): restore.safe_path(self.root, "data/raw/file.tsv")

    def test_https_host_allowlist_and_redirects_reject_unsafe_targets(self):
        for url in ["http://huggingface.co/file", "https://huggingface.co.evil.test/file",
                    "https://user:password@huggingface.co/file", "https://huggingface.co:444/file", "file:///tmp/source"]:
            with self.subTest(url=url), self.assertRaises(ValueError): restore.validate_url(url)
        req = urllib.request.Request("https://huggingface.co/model")
        with self.assertRaises(ValueError):
            restore.CheckedRedirect().redirect_request(req, None, 302, "Found", {}, "https://other.test/source")
        self.assertEqual(restore.validate_url("https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip"),
                         "https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip")

    def test_notice_is_read_only_from_verified_archive_without_network(self):
        buffer = io.BytesIO(); notice = b"ARTIFICIAL LICENSE"
        with zipfile.ZipFile(buffer, "w") as zf: zf.writestr("LICENSE", notice)
        archive_bytes = buffer.getvalue(); url = "https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip"
        archive = entry("data/raw/TOPv2_Dataset.zip", archive_bytes, url=url)
        member = entry("data/licenses/meta-topv2-LICENSE.txt", notice, url=url, archive_member="LICENSE")
        self.manifest([archive, member])
        target = self.root / archive["path"]; target.parent.mkdir(); target.write_bytes(archive_bytes)
        with patch.object(restore, "open_download") as fetch:
            result = restore.restore(self.root, download=True)
        fetch.assert_not_called(); self.assertEqual(result["status"], "complete")
        self.assertEqual((self.root / member["path"]).read_bytes(), notice)


if __name__ == "__main__":
    unittest.main()
