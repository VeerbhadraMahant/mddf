"""``mddf`` command-line entrypoint.

``serve`` runs on the base (Torch-free) install. ``data`` / ``train`` / ``export`` /
``benchmark`` need the ``train`` extra. Heavy imports live inside each handler so
this module stays importable in the deployment image.
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


def _train(args: argparse.Namespace) -> int:
    from mddf.logging import configure_logging
    from mddf.training.run import ALL_MODELS, train_matrix

    configure_logging(level="INFO", json=False)
    models = list(ALL_MODELS) if args.model == "all" else [args.model]
    try:
        outcome = train_matrix(
            models,
            args.category,
            accelerator=args.accelerator,
            force=args.force,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    cols = (
        f"{'model':<14} {'category':<14} {'img AUROC':>10} "
        f"{'pix AUROC':>10} {'AUPRO':>8} {'sec':>7}"
    )
    print(f"\n{cols}")
    for r in outcome.results:
        print(
            f"{r.model:<14} {r.category:<14} "
            f"{r.metrics.get('image_auroc', float('nan')):>10.4f} "
            f"{r.metrics.get('pixel_auroc', float('nan')):>10.4f} "
            f"{r.metrics.get('aupro', float('nan')):>8.4f} "
            f"{r.seconds:>7.0f}"
        )
    for model, category, err in outcome.failures:
        sys.stderr.write(f"FAILED {model}/{category}: {err}\n")
    return 0 if outcome.ok else 1


def _export(args: argparse.Namespace) -> int:
    from mddf.logging import configure_logging
    from mddf.training.export import export_matrix
    from mddf.training.run import ALL_MODELS, resolve_categories

    configure_logging(level="INFO", json=False)
    models = list(ALL_MODELS) if args.model == "all" else [args.model]
    try:
        categories = resolve_categories(args.category)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    results, failures = export_matrix(models, categories, force=args.force, quantize=args.quantize)
    for r in results:
        mb = r.onnx.stat().st_size / 1e6
        int8 = f" | int8 {r.int8_bytes / 1e6:.1f} MB" if r.int8_bytes else ""
        print(f"{r.model:<14} {r.category:<14} {mb:>7.1f} MB{int8}  {r.output_names}")
    for model, category, err in failures:
        sys.stderr.write(f"FAILED {model}/{category}: {err}\n")
    return 0 if not failures and results else 1


def _report(args: argparse.Namespace) -> int:
    from mddf.benchmark.report import build_report
    from mddf.logging import configure_logging
    from mddf.training.run import resolve_categories

    configure_logging(level="INFO", json=False)
    try:
        categories = resolve_categories(args.category)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    doc = build_report(categories=categories)
    n = sum(len(v) for v in doc["results"].values())
    if not n:
        sys.stderr.write("No exported models found; run `mddf export` first.\n")
        return 1
    print(f"{n} (model, category) operating-point sets -> artifacts/report/")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from mddf.benchmark.accuracy import aggregate, comparison_markdown
    from mddf.benchmark.latency import benchmark_all
    from mddf.config import get_settings
    from mddf.logging import configure_logging
    from mddf.training import artifacts as art
    from mddf.training.run import ALL_MODELS

    configure_logging(level="INFO", json=False)
    models = list(ALL_MODELS)
    if args.latency:
        benchmark_all(models, runs=args.runs)
    doc = aggregate(models)
    md = comparison_markdown()

    out_dir = get_settings().artifacts_dir / "benchmark"
    art.write_json(out_dir / "metrics.json", doc)
    (out_dir / "COMPARISON.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    n = sum(len(v) for v in doc["results"].values())
    print(f"\n{n} (model, category) results -> {out_dir / 'metrics.json'}")
    return 0 if doc["results"] else 1


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

    train = sub.add_parser("train", help="Train PaDiM / PatchCore / EfficientAD per category.")
    train.add_argument(
        "--model",
        choices=["padim", "patchcore", "efficient_ad", "all"],
        default="patchcore",
    )
    train.add_argument(
        "--category",
        action="append",
        metavar="NAME",
        help="Restrict to one category (repeatable). Default: all 15.",
    )
    train.add_argument("--accelerator", default="auto", help="auto | gpu | cpu")
    train.add_argument("--force", action="store_true", help="Retrain even if artifacts exist.")
    train.set_defaults(func=_train)

    bench = sub.add_parser("benchmark", help="Aggregate the accuracy + latency comparison table.")
    bench.add_argument("--latency", action="store_true", help="Also time the ONNX models on CPU.")
    bench.add_argument("--runs", type=int, default=50, help="Latency iterations per model.")
    bench.set_defaults(func=_benchmark)

    report = sub.add_parser(
        "report", help="Operating-point analysis (recall / false-alarm trade-off) from ONNX."
    )
    report.add_argument("--category", action="append", metavar="NAME")
    report.set_defaults(func=_report)

    export = sub.add_parser("export", help="Export ONNX artifacts for the trained models.")
    export.add_argument(
        "--model", choices=["padim", "patchcore", "efficient_ad", "all"], default="all"
    )
    export.add_argument("--category", action="append", metavar="NAME")
    export.add_argument("--force", action="store_true")
    export.add_argument(
        "--quantize", action="store_true", help="Also emit a dynamic-INT8 model.int8.onnx."
    )
    export.set_defaults(func=_export)

    return parser


def _force_utf8_stdio() -> None:
    # Anomalib / Lightning emit non-ASCII (emoji, box-drawing); a cp1252 Windows
    # console would raise UnicodeEncodeError mid-run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
