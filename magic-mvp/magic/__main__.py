"""Explicit model installation, local resolution, and a reproducible benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import MagicError, get_resolver, resolve
from .embeddings import DEFAULT_MODEL, DEFAULT_REVISION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="magic", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-model", help="Install weights once; requires internet")
    download.add_argument("--output", default="models/all-MiniLM-L6-v2")
    explain = commands.add_parser("resolve", help="Explain a semantic intent; execution is opt-in")
    explain.add_argument("intent")
    explain.add_argument("--args", default="[]", help="JSON positional arguments")
    explain.add_argument("--kwargs", default="{}", help="JSON keyword arguments")
    explain.add_argument("--top-k", type=int, default=12)
    explain.add_argument("--execute", action="store_true")
    explain.add_argument("--json", action="store_true")
    bench = commands.add_parser("benchmark", help="Evaluate local retrieval and argument mapping")
    bench.add_argument("--output", default="benchmark-results.json")
    commands.add_parser("registry", help="List curated functions")
    options = parser.parse_args(argv)
    try:
        if options.command == "download-model":
            from huggingface_hub import snapshot_download

            output = Path(options.output).resolve()
            snapshot_download(
                repo_id=DEFAULT_MODEL,
                revision=DEFAULT_REVISION,
                local_dir=str(output),
                allow_patterns=[
                    "config.json",
                    "config_sentence_transformers.json",
                    "modules.json",
                    "sentence_bert_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                    "model.safetensors",
                    "1_Pooling/config.json",
                    "README.md",
                ],
            )
            print(f"Installed {DEFAULT_MODEL} ({DEFAULT_REVISION}) at {output}")
            print(
                f"Set MAGIC_MODEL_PATH={output} or call magic.configure(model_path={str(output)!r})"
            )
        elif options.command == "registry":
            for d in get_resolver().registry.functions:
                print(f"{d.id}{d.signature} [{d.effect}]")
        elif options.command == "benchmark":
            from .benchmark import run_benchmark

            report = run_benchmark(get_resolver())
            Path(options.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
            print(f"Detailed results: {options.output}")
        else:
            args = json.loads(options.args)
            kwargs = json.loads(options.kwargs)
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                parser.error("--args must be a JSON list and --kwargs a JSON object")
            result = resolve(
                options.intent, *args, execute=options.execute, top_k=options.top_k, **kwargs
            )
            if options.json:
                print(
                    json.dumps(
                        result.to_dict() if hasattr(result, "to_dict") else result,
                        indent=2,
                        default=repr,
                    )
                )
            else:
                print(result)
            if not options.execute and result.status != "resolved":
                return 2
    except (MagicError, ValueError, TypeError, ImportError) as e:
        parser.exit(2, f"magic: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
