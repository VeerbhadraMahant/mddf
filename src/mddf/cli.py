"""``mddf`` command-line entrypoint.

Only the ``serve`` and ``version`` subcommands work with the base (Torch-free)
install. ``data``/``train``/``export``/``benchmark`` need the ``train`` extra and are
wired in the corresponding milestones; they lazy-import so this module stays
importable in the deployed image.
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

    for name, helptext in [
        ("data", "Download and verify the MVTec AD dataset."),
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
