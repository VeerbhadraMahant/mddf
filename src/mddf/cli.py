"""``mddf`` command-line entrypoint.

``serve`` and ``data`` need only the base install plus (for ``data``) the ``train``
extra. ``train``/``export``/``benchmark`` are wired in later milestones. Heavy
imports are done inside the handlers so this module stays importable in the
Torch-free deployment image.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mddf.config import project_version


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "mddf.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )
    return 0


def _data(args: argparse.Namespace) -> int:
    from mddf.catalog import load_catalog
    from mddf.data.download import ensure_categories
    from mddf.logging import configure_logging

    configure_logging(level="INFO", json=False)
    categories = args.category or load_catalog().names
    try:
        status = ensure_categories(categories)
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    print(f"\nMVTec AD ready at {status.root}")
    print(f"{'category':<14} {'train/good':>10} {'test':>6} {'defects':>8}  status")
    for c in status.categories:
        print(
            f"{c.name:<14} {c.train_good:>10} {c.test_total:>6} "
            f"{len(c.defect_types):>8}  {'ok' if c.ok else '; '.join(c.issues)}"
        )
    return 0


def _not_yet(name: str) -> int:
    sys.stderr.write(
        f"`mddf {name}` is added in a later milestone. Install the training extra with "
        f"`pip install -e .[train]` first.\n"
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mddf", description="Manufacturing Defect Detection")
    parser.add_argument("--version", action="version", version=f"mddf {project_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI service.")
    serve.add_argument("--host", default="0.0.0.0")  # container entrypoint binds all ifaces
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    data = sub.add_parser("data", help="Download and verify the MVTec AD dataset.")
    data.add_argument(
        "--category",
        action="append",
        metavar="NAME",
        help="Restrict to one category (repeatable). Default: all 15.",
    )
    data.set_defaults(func=_data)

    for name, helptext in [
        ("train", "Train PatchCore / EfficientAD for one or more categories."),
        ("export", "Export ONNX artifacts and split the backbone / memory bank."),
        ("benchmark", "Compute the accuracy + latency comparison table."),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.set_defaults(func=lambda _a, _n=name: _not_yet(_n))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
