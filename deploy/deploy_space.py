"""Create/update the Hugging Face Docker Space and push the app.

Uploads the Dockerfile, package metadata, ``src/``, ``configs/`` and the ``web/``
source (HF builds the SPA in the image's node stage). The Space README carries
the required YAML frontmatter (``deploy/hf_space_README.md``).

Usage:
    python deploy/deploy_space.py [--space user/name] [--dry-run]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPACE = "bhadra244131/mddf"
INCLUDE_DIRS = ("src", "web")
INCLUDE_FILES = ("Dockerfile", "pyproject.toml", ".dockerignore")
IGNORE = (
    "web/node_modules/*",
    "web/dist/*",
    "web/.vite/*",
    "**/__pycache__/*",
    "**/*.pyc",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    staged: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        # README with the HF frontmatter.
        (stage / "README.md").write_text(
            (REPO_ROOT / "deploy" / "hf_space_README.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        staged.append("README.md")
        for name in INCLUDE_FILES:
            src = REPO_ROOT / name
            if src.is_file():
                (stage / name).write_bytes(src.read_bytes())
                staged.append(name)
        for d in INCLUDE_DIRS:
            src_dir = REPO_ROOT / d
            for p in src_dir.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(REPO_ROOT)
                if "node_modules" in rel.parts or "__pycache__" in rel.parts:
                    continue
                if rel.parts[:2] == ("web", "dist"):
                    continue
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
                staged.append(str(rel).replace("\\", "/"))

        print(f"{len(staged)} files -> space {args.space}")
        for s in sorted(staged)[:40]:
            print("  ", s)
        if len(staged) > 40:
            print(f"   ... (+{len(staged) - 40} more)")
        if args.dry_run:
            return 0

        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=str(stage),
            ignore_patterns=list(IGNORE),
            commit_message="Deploy MDDF",
        )
    print(f"done: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
