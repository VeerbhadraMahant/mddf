"""Publish exported model artifacts to the Hugging Face model repo.

Uploads ``artifacts/<model>/<category>/{model.int8.onnx,preprocess.json,metrics.json}``
plus ``artifacts/{benchmark,report}/*`` to ``MDDF_HF_MODEL_REPO`` (default
``bhadra244131/mddf-artifacts``). The running Space pulls these lazily per
category, so the deployed image ships no weights. fp32 ``model.onnx`` is kept
local (only needed for ``mddf verify`` / analysis) unless ``--include-fp32``.

Usage:
    python deploy/publish_artifacts.py [--repo user/name] [--include-fp32] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mddf.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=get_settings().hf_model_repo)
    parser.add_argument("--artifacts", type=Path, default=get_settings().artifacts_dir)
    parser.add_argument("--include-fp32", action="store_true", help="Also upload fp32 model.onnx.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.artifacts.is_dir():
        sys.stderr.write(f"No artifacts directory at {args.artifacts}\n")
        return 1

    per_category = {"model.int8.onnx", "preprocess.json", "metrics.json"}
    if args.include_fp32:
        per_category.add("model.onnx")
    allow = [f"*/*/{n}" for n in sorted(per_category)] + ["benchmark/*", "report/*"]

    def wanted(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name in per_category:
            return True
        return p.parent.name in {"benchmark", "report"} and p.suffix in {".json", ".md"}

    uploads = [p for p in args.artifacts.rglob("*") if wanted(p)]
    total_mb = sum(p.stat().st_size for p in uploads) / 1e6
    print(f"{len(uploads)} files ({total_mb:.0f} MB) -> {args.repo}")
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
        allow_patterns=allow,
        commit_message="Publish MDDF exported artifacts",
    )
    print(f"done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
