"""Publish exported model artifacts to the Hugging Face model repo.

Uploads ``artifacts/<model>/<category>/{model.onnx,preprocess.json,metrics.json}``
and ``artifacts/benchmark/*`` to ``MDDF_HF_MODEL_REPO`` (default
``bhadra244131/mddf-artifacts``). The running Space pulls these lazily per
category, so the deployed image ships no weights.

Usage:
    python deploy/publish_artifacts.py [--repo user/name] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mddf.config import get_settings  # noqa: E402

ALLOW = ("*/*/model.onnx", "*/*/preprocess.json", "*/*/metrics.json", "benchmark/*")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=get_settings().hf_model_repo)
    parser.add_argument("--artifacts", type=Path, default=get_settings().artifacts_dir)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.artifacts.is_dir():
        sys.stderr.write(f"No artifacts directory at {args.artifacts}\n")
        return 1

    per_category = {"model.onnx", "preprocess.json", "metrics.json"}

    def wanted(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name in per_category:
            return True
        return p.parent.name == "benchmark" and p.suffix in {".json", ".md"}

    uploads = [p for p in args.artifacts.rglob("*") if wanted(p)]
    print(f"{len(uploads)} files -> {args.repo}")
    for p in sorted(uploads):
        print("  ", p.relative_to(args.artifacts))
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(args.artifacts),
        allow_patterns=list(ALLOW),
        commit_message="Publish MDDF exported artifacts",
    )
    print(f"done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
