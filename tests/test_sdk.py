"""Artificial fixtures only: no models, credentials, downloads or listening sockets."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from udgam_hinglish import CommandParser, preview_result
from udgam_hinglish import api, cli
from udgam_hinglish._vendor import runtime

SCHEMA = {"labels": ["IN:CREATE_ALARM", "SL:DATE_TIME", "SL:NAME"]}
TEXT = "kal subah 7 baje alarm lagao"
TOP = "[IN:CREATE_ALARM [SL:DATE_TIME kal subah 7 baje]]"


def output(top=TOP, **kwargs):
    return {"prediction": top, "raw_output": top + "<eos>", "eos_reached": True,
            "truncated": False, "error": None, **kwargs}


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.options = messages, kwargs
        return "ARTIFICIAL PROMPT"


class LocalFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = copy.deepcopy(api.EXPECTED_CONFIG)
        (self.root / "udgam_config.json").write_text(json.dumps(self.cfg))
        (self.root / "schema.json").write_text(json.dumps(SCHEMA))
        rows = [{"id": str(i), "pair_id": str(i), "group_id": str(i), "split": "train",
                 "language": "hinglish", "domain": "alarm", "text": f"subah {i} alarm",
                 "target": f"[IN:CREATE_ALARM [SL:DATE_TIME subah {i}]]", "english_source": f"alarm {i}"}
                for i in range(4)]
        (self.root / "retrieval_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.tokenizer = Tokenizer()
        self.model = types.SimpleNamespace(dtype="torch.bfloat16", device="mps",
                                           parameters=lambda: [], buffers=lambda: [])
        self.set_seed = Mock()
        fake = types.SimpleNamespace(set_seed=self.set_seed)
        self.mods = patch.dict(sys.modules, {"transformers": fake})
        self.mods.start(); self.addCleanup(self.mods.stop)

    def load(self, **kwargs):
        with patch.object(runtime, "load_export", return_value=(self.model, self.tokenizer, self.cfg)) as load:
            obj = CommandParser.from_local(self.root, **kwargs)
        self.last_load = load
        return obj


class LoadingTests(LocalFixture):
    def test_local_default_keeps_exact_recipe_and_downloads_disabled(self):
        obj = self.load(device="mps")
        self.last_load.assert_called_once_with(self.root.resolve(), "mps", allow_base_download=False, dtype=None)
        self.set_seed.assert_called_once_with(20260912)
        self.assertEqual(obj.settings["shots"], 4)
        self.assertTrue(obj.settings["precision_matches_recorded_recipe"])
        view = obj.settings; view["shots"] = 0
        self.assertEqual(obj.settings["shots"], 4)

    def test_each_fixed_configuration_field_is_checked_before_loading(self):
        for key in api.EXPECTED_CONFIG:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg); cfg[key] = None
                (self.root / "udgam_config.json").write_text(json.dumps(cfg))
                with patch.object(runtime, "load_export") as load, self.assertRaises(ValueError):
                    CommandParser.from_local(self.root)
                load.assert_not_called()

    def test_boolean_cannot_impersonate_numeric_contract(self):
        for key in ("shots", "seed", "max_input_tokens"):
            cfg = copy.deepcopy(self.cfg); cfg[key] = True
            with self.assertRaises(ValueError): api.validate_config(cfg)

    def test_merged_package_rejected(self):
        cfg = copy.deepcopy(self.cfg); cfg["format"] = "merged"
        (self.root / "udgam_config.json").write_text(json.dumps(cfg))
        with patch.object(runtime, "load_export") as load, self.assertRaises(ValueError):
            CommandParser.from_local(self.root)
        load.assert_not_called()

    def test_float32_override_explicitly_marks_precision_change(self):
        self.model.dtype = "torch.float32"
        obj = self.load(device="cpu", dtype="float32")
        self.assertFalse(obj.settings["precision_matches_recorded_recipe"])
        self.assertEqual(self.last_load.call_args.kwargs["dtype"], "float32")

    def test_changed_configuration_after_load_fails(self):
        changed = dict(self.cfg, shots=0)
        with patch.object(runtime, "load_export", return_value=(self.model, self.tokenizer, changed)), self.assertRaises(ValueError):
            CommandParser.from_local(self.root)

    def test_corrupt_manifest_file_rejected_before_ml_import(self):
        (self.root / "model").mkdir()
        for n in ["tokenizer_config.json", "adapter_config.json", "adapter_model.safetensors"]:
            (self.root / "model" / n).write_bytes(b"artificial")
        names = ["udgam_config.json", "schema.json", "retrieval_train.jsonl", "model/tokenizer_config.json",
                 "model/adapter_config.json", "model/adapter_model.safetensors"]
        manifest = {n: {"bytes": (self.root/n).stat().st_size, "sha256": runtime.sha256_file(self.root/n)} for n in names}
        (self.root / "artifact-manifest.json").write_text(json.dumps(manifest))
        (self.root / "model/adapter_model.safetensors").write_bytes(b"CORRUPTED")
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            CommandParser.from_local(self.root)


class HubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name).resolve()
        self.env = patch.dict(os.environ, {"HF_HOME": str(self.cache)})
        self.env.start(); self.addCleanup(self.env.stop)

    def test_existing_authentication_is_explicit_boolean_opt_in(self):
        download = Mock(side_effect=lambda **kwargs: kwargs["local_dir"])
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
             patch.object(CommandParser, "from_local"):
            CommandParser.from_hub("UdgamLabs/private-model", revision="a"*40, token=True)
            self.assertIs(download.call_args.kwargs["token"], True)
            download.reset_mock()
            with self.assertRaises(ValueError):
                CommandParser.from_hub("UdgamLabs/private-model", revision="a"*40, token="ARTIFICIAL_SECRET")
            download.assert_not_called()

    def test_explicit_download_pins_commit_and_does_not_use_credentials(self):
        download = Mock(side_effect=lambda **kwargs: kwargs["local_dir"])
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
             patch.object(CommandParser, "from_local", return_value="parser") as load:
            result = CommandParser.from_hub("UdgamLabs/hinglish-commands-0.6b", revision="a"*40)
        self.assertEqual(result, "parser")
        download.assert_called_once_with(repo_id="UdgamLabs/hinglish-commands-0.6b", revision="a"*40,
            local_dir=str(self.cache / "udgam-packages/UdgamLabs/hinglish-commands-0.6b" / ("a"*40)), token=False)
        self.assertFalse(load.call_args.kwargs["allow_base_download"])

    def test_normal_hub_blob_symlinks_are_materialized_before_verified_loading(self):
        hub = self.cache / "hub/models--UdgamLabs--fixture"
        blobs = hub / "blobs"; blobs.mkdir(parents=True)
        snapshot = hub / "snapshots" / ("a"*40); (snapshot / "model").mkdir(parents=True)
        source = {"udgam_config.json": json.dumps(api.EXPECTED_CONFIG).encode(),
                  "model/adapter_model.safetensors": b"ARTIFICIAL ADAPTER"}
        manifest = {}
        for name, data in source.items():
            h = runtime.canonical_hash({"fixture": name})
            blob = blobs / h; blob.write_bytes(data)
            (snapshot / name).symlink_to(blob)
            manifest[name] = {"bytes": len(data), "sha256": runtime.sha256_file(blob)}
        raw_manifest = json.dumps(manifest).encode()
        (blobs / "manifest").write_bytes(raw_manifest)
        (snapshot / "artifact-manifest.json").symlink_to(blobs / "manifest")
        self.assertFalse((snapshot / "model/adapter_model.safetensors").resolve().is_relative_to(snapshot))

        def download(**kwargs):
            # Simulate Hub's normal snapshot result if local_dir is omitted, and
            # its supported regular-file local_dir materialization otherwise.
            if "local_dir" not in kwargs: return str(snapshot)
            shutil.copytree(snapshot, kwargs["local_dir"], symlinks=False, dirs_exist_ok=True)
            return kwargs["local_dir"]

        def checked_load(path, **kwargs):
            path = Path(path)
            self.assertNotEqual(path, snapshot)
            for name, expected in manifest.items():
                candidate = path / name
                self.assertFalse(candidate.is_symlink())
                self.assertTrue(candidate.resolve().is_relative_to(path.resolve()))
                self.assertEqual(runtime.sha256_file(candidate), expected["sha256"])
                self.assertEqual(candidate.stat().st_size, expected["bytes"])
            return "checked parser"

        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
             patch.object(CommandParser, "from_local", side_effect=checked_load) as load:
            result = CommandParser.from_hub("UdgamLabs/fixture", revision="a"*40)
        self.assertEqual(result, "checked parser"); load.assert_called_once()
        self.assertTrue((snapshot / "model/adapter_model.safetensors").is_symlink())

    def test_package_cache_symlink_is_rejected_before_download(self):
        outside = self.cache / "elsewhere"; outside.mkdir()
        (self.cache / "udgam-packages").symlink_to(outside, target_is_directory=True)
        download = Mock()
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), self.assertRaises(ValueError):
            CommandParser.from_hub("UdgamLabs/fixture", revision="a"*40)
        download.assert_not_called()

    def test_unexpected_snapshot_or_returned_symlink_is_not_loaded(self):
        for use_link in (False, True):
            def download(**kwargs):
                if use_link:
                    (Path(kwargs["local_dir"]) / "external").symlink_to(self.cache / "outside")
                    return kwargs["local_dir"]
                return str(self.cache / "unexpected-snapshot")
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
                 patch.object(CommandParser, "from_local") as load, self.assertRaises(ValueError):
                CommandParser.from_hub("UdgamLabs/fixture", revision="a"*40)
            load.assert_not_called()

    def test_invalid_identity_or_options_never_download(self):
        download = Mock()
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
            for opts in [{"revision": "main"}, {"revision": "v1"}, {"revision": "A"*40},
                         {"revision": "a"*40, "device": "remote"},
                         {"revision": "a"*40, "allow_base_download": "yes"}]:
                with self.subTest(opts=opts), self.assertRaises(ValueError):
                    CommandParser.from_hub("UdgamLabs/hinglish-commands-0.6b", **opts)
            with self.assertRaises(ValueError):
                CommandParser.from_hub("https://example.test/model", revision="a"*40)
        download.assert_not_called()


class ParseTests(LocalFixture):
    def test_four_retrieval_examples_and_original_text_reach_fixed_prompt(self):
        obj = self.load()
        with patch.object(runtime, "generate_batch", return_value=[output()]) as generate:
            result = obj.parse(TEXT)
        self.assertTrue(result["ok"])
        self.assertEqual(len(set(result["example_ids"])), 4)
        self.assertEqual(self.tokenizer.messages[-1], {"role": "user", "content": TEXT})
        self.assertEqual(len(self.tokenizer.messages), 10)
        self.assertFalse(self.tokenizer.options["enable_thinking"])
        generate.assert_called_once_with(self.model, self.tokenizer, ["ARTIFICIAL PROMPT"], 768, 4096)
        self.assertEqual(result["seed"], 20260912)
        self.assertFalse(result["executed_action"])

    def test_invalid_inputs_do_not_generate(self):
        obj = self.load()
        with patch.object(runtime, "generate_batch") as generate:
            for text in [None, 12, "", " \n", "x"*2001]:
                with self.subTest(text=type(text).__name__), self.assertRaises(ValueError): obj.parse(text)
        generate.assert_not_called()

    def test_demo_reuses_parser_lock_and_selected_engine(self):
        obj = self.load(); app = obj.make_demo_app()
        self.assertEqual(set(app.engines), {"fine_tuned"})
        self.assertEqual(app.engines["fine_tuned"].predict, obj.parse)
        with patch.object(runtime, "generate_batch", return_value=[output()]):
            result = app.parse({"text": TEXT, "engine": "fine_tuned"})
        self.assertTrue(result["ok"])
        self.assertFalse(app.status()["logs_user_input"])


class PreviewTests(unittest.TestCase):
    def test_nested_repeated_slots_and_case_are_preserved(self):
        top = "[IN:CREATE_ALARM [SL:NAME AMIT] [SL:NAME Amit] [SL:DATE_TIME kal]]"
        result = preview_result(output(top), "AMIT Amit kal", SCHEMA)
        self.assertTrue(result["ok"])
        self.assertEqual(result["raw_top"], top)
        self.assertEqual([c["label"] for c in result["tree"]["children"]],
                         ["SL:NAME", "SL:NAME", "SL:DATE_TIME"])
        self.assertEqual(result["tree"]["children"][0]["children"], ["AMIT"])

    def test_bad_output_never_produces_a_tree_or_action(self):
        cases = [(output("[IN:CREATE_ALARM"), "malformed"),
                 (output("[IN:UNKNOWN]"), "unknown_labels"),
                 (output("[IN:CREATE_ALARM [SL:NAME invented]]"), "source_mismatch"),
                 (output(truncated=True), "truncated"),
                 (output(error="failure"), "generation_error")]
        for raw, status in cases:
            with self.subTest(status=status):
                result = preview_result(raw, TEXT, SCHEMA)
                self.assertFalse(result["ok"]); self.assertIsNone(result["tree"])
                self.assertEqual(result["status"], status); self.assertFalse(result["executed_action"])

    def test_copy_checks_do_not_casefold_or_resolve_dates(self):
        result = preview_result(output("[IN:CREATE_ALARM [SL:NAME amit]]"), "AMIT", SCHEMA)
        self.assertEqual(result["status"], "source_mismatch")
        result = preview_result(output("[IN:CREATE_ALARM [SL:DATE_TIME kal]]"), "kal", SCHEMA)
        self.assertEqual(result["tree"]["children"][0]["children"], ["kal"])


class CLITests(unittest.TestCase):
    def test_use_auth_is_forwarded_only_for_explicit_hub_loading(self):
        fake = Mock(); fake.parse.return_value = {"ok": True}
        with patch.object(CommandParser, "from_hub", return_value=fake) as load, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["parse", "--repo-id", "UdgamLabs/private-model", "--revision", "a"*40,
                                       "--use-auth", "--text", TEXT]), 0)
        self.assertIs(load.call_args.kwargs["token"], True)
        with patch.object(CommandParser, "from_local") as local, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["parse", "--model-dir", "fixture", "--use-auth", "--text", TEXT])
        local.assert_not_called()

    def test_runtime_failure_has_nonzero_exit_without_echoing_input(self):
        fake = Mock(); fake.parse.side_effect = RuntimeError("ARTIFICIAL_PRIVATE_INPUT")
        with patch.object(CommandParser, "from_local", return_value=fake), contextlib.redirect_stderr(io.StringIO()) as buf:
            code = cli.main(["parse", "--model-dir", "fixture", "--text", TEXT])
        self.assertEqual(code, 1)
        self.assertNotIn("ARTIFICIAL_PRIVATE_INPUT", buf.getvalue())
        self.assertIn("RuntimeError", buf.getvalue())

    def test_parse_prints_unicode_json_and_distinguishes_rejection_exit(self):
        for ok, code in [(True, 0), (False, 2)]:
            fake = Mock(); fake.parse.return_value = {"ok": ok, "tree": None, "executed_action": False}
            with patch.object(CommandParser, "from_local", return_value=fake) as load, contextlib.redirect_stdout(io.StringIO()) as buf:
                result = cli.main(["parse", "--model-dir", "fixture", "--text", TEXT])
            self.assertEqual(result, code); self.assertEqual(json.loads(buf.getvalue())["ok"], ok)
            self.assertFalse(load.call_args.kwargs["allow_base_download"])

    def test_bad_cli_arguments_never_load_model(self):
        for args in [["parse", "--repo-id", "UdgamLabs/model", "--text", TEXT],
                     ["parse", "--model-dir", "fixture", "--text", ""],
                     ["demo", "--model-dir", "fixture", "--port", "-1"],
                     ["demo", "--model-dir", "fixture", "--host", "0.0.0.0"]]:
            with patch.object(CommandParser, "from_local") as load, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(args)
            load.assert_not_called()

    def test_demo_closes_listener_on_interrupt_without_real_socket(self):
        server = Mock(); server.server_address = ("127.0.0.1", 32100)
        server.serve_forever.side_effect = KeyboardInterrupt
        fake = Mock()
        with patch.object(CommandParser, "from_local", return_value=fake), \
             patch("udgam_hinglish._demo.LocalServer", return_value=server) as construct, \
             contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli.main(["demo", "--model-dir", "fixture", "--port", "0"]), 0)
        self.assertEqual(construct.call_args.args[1], 0)
        server.server_close.assert_called_once()
        self.assertEqual(json.loads(buf.getvalue())["url"], "http://127.0.0.1:32100")

    def test_import_and_help_require_no_ml_packages(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        code = '''import sys, importlib.abc
class NoML(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','peft','huggingface_hub'}:
   raise AssertionError('Unexpected heavyweight import: ' + fullname)
sys.meta_path.insert(0, NoML())
sys.path.insert(0, SOURCE)
import udgam_hinglish
from udgam_hinglish.cli import main
main(['--help'])
'''.replace("SOURCE", repr(source))
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("udgam-hinglish", result.stdout)


if __name__ == "__main__":
    unittest.main()
