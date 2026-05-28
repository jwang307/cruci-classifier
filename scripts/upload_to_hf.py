#!/usr/bin/env python3
"""Upload cruci-classifier artifacts to a Hugging Face Hub repo."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_IGNORE_PATTERNS = [
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    # Keep HF uploads small by default. The portable head checkpoints are kept.
    "**/best.pt",
    "**/epoch_*.pt",
    # Prepared CSVs contain sequence data and are reproducible from source FASTAs.
    "**/prepared/**/*.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. user/cruci-esm35m-grid")
    parser.add_argument("--repo_type", choices=["model", "dataset", "space"], default="dataset")
    parser.add_argument("--path", type=Path, required=True, help="Local file or directory to upload.")
    parser.add_argument("--path_in_repo", default=".", help="Destination path inside the HF repo.")
    parser.add_argument("--revision", default=None, help="Optional branch/revision to upload to.")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it does not exist.")
    parser.add_argument(
        "--include_full_checkpoints",
        action="store_true",
        help="Do not ignore legacy full ESM checkpoints such as best.pt.",
    )
    parser.add_argument(
        "--ignore_pattern",
        action="append",
        default=[],
        help="Additional ignore glob. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allow_pattern",
        action="append",
        default=None,
        help="Optional allow glob. Can be passed multiple times.",
    )
    parser.add_argument("--commit_message", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install with `pip install huggingface_hub` "
            "or `uv run --with huggingface_hub ...`."
        ) from exc

    path = args.path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    if args.include_full_checkpoints:
        ignore_patterns = [p for p in ignore_patterns if p not in {"**/best.pt", "**/epoch_*.pt"}]
    ignore_patterns.extend(args.ignore_pattern)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    commit_message = args.commit_message or f"Upload cruci-classifier artifacts from {path.name}"
    if path.is_dir():
        result = api.upload_folder(
            folder_path=str(path),
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=args.allow_pattern,
            ignore_patterns=ignore_patterns,
            commit_message=commit_message,
        )
    else:
        destination = args.path_in_repo
        if destination in {"", "."}:
            destination = path.name
        result = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=commit_message,
        )
    print(result)


if __name__ == "__main__":
    main()
