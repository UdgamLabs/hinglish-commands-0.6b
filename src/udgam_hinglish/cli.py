"""One-command JSON parsing or a loopback-only browser preview."""
import argparse
import json
import sys

from .api import CommandParser, validate_text


def argument_parser():
    p = argparse.ArgumentParser(prog="udgam-hinglish", description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name, help_text in [("parse", "Parse one command and print a JSON preview"),
                            ("demo", "Preload the adapter, then open a local browser UI")]:
        sub = commands.add_parser(name, help=help_text)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--model-dir", help="Complete local adapter package root")
        source.add_argument("--repo-id", help="Explicitly download a Hub adapter repository")
        sub.add_argument("--revision", help="Required immutable 40-character commit with --repo-id")
        sub.add_argument("--use-auth", action="store_true",
                         help="With --repo-id, explicitly use existing Hugging Face authentication")
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument("--dtype", choices=["bfloat16", "float32"],
                         help="Default BF16; float32 changes the benchmarked precision")
        sub.add_argument("--allow-base-download", action="store_true",
                         help="Explicitly permit startup download of the exact pinned Qwen base")
        if name == "parse":
            sub.add_argument("--text", required=True)
        else:
            sub.add_argument("--port", type=int, default=8765, help="127.0.0.1 only; 0 selects a free port")
    return p


def main(argv=None):
    cli = argument_parser()
    args = cli.parse_args(argv)
    if args.repo_id and not args.revision:
        cli.error("--repo-id requires --revision with the release's immutable commit")
    if args.model_dir and args.revision:
        cli.error("--revision applies only to --repo-id")
    if args.model_dir and args.use_auth:
        cli.error("--use-auth applies only to --repo-id")
    if args.command == "parse":
        try:
            validate_text(args.text)
        except ValueError as error:
            cli.error(str(error))
    elif not 0 <= args.port <= 65535:
        cli.error("--port must be between 0 and 65535")
    options = dict(device=args.device, dtype=args.dtype, allow_base_download=args.allow_base_download)
    try:
        parser = (CommandParser.from_hub(args.repo_id, revision=args.revision, token=args.use_auth, **options)
                  if args.repo_id else CommandParser.from_local(args.model_dir, **options))
        if args.command == "parse":
            result = parser.parse(args.text)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["ok"] else 2
        from ._demo import LocalServer
        server = LocalServer(parser.make_demo_app(), args.port)
        print(json.dumps({"status": "local_demo_ready", "url": f"http://127.0.0.1:{server.server_address[1]}",
                          "executes_actions": False, "logs_user_input": False}), flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        # Keep arbitrary exception text (potential paths/input) out of the CLI output.
        print(f"Local parsing could not start or complete ({type(error).__name__}). "
              "Check the complete package, pinned base cache and inference dependencies.", file=sys.stderr)
        return 1
