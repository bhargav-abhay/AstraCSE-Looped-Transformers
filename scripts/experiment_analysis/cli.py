"""Shared CLI plumbing for the experiment analysis scripts.

Exit codes: 0 success, 1 analysis error, 2 argparse usage error,
3 no result files found, 4 strict mode found warnings or errors.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .comparisons import build_comparisons
from .core import ComparisonCategory, ComparisonResult, MetricValue
from .discovery import AnalysisPaths, build_analysis, resolve_paths
from .reports import write_comparison_csvs, write_report_bundle

EXIT_OK = 0
EXIT_ANALYSIS_ERROR = 1
EXIT_NO_RESULTS = 3
EXIT_STRICT = 4


def add_common_arguments(parser: argparse.ArgumentParser, *, output: bool = True) -> None:
    """Add the directory and strictness arguments shared by all three CLIs."""
    parser.add_argument("--experiments-dir", default=None,
                        help="directory containing <run>.mmlu_cs.json and "
                             "<run>.benchmark.json files (default: paper/data/results)")
    parser.add_argument("--repo-root", default=None,
                        help="repository root (default: inferred from this script)")
    parser.add_argument("--configs-dir", default=None,
                        help="model config directory (default: <repo-root>/configs)")
    parser.add_argument("--metadata-dir", default=None,
                        help="directory with rescore/MCQ/curve/bootstrap files "
                             "(default: <repo-root>/paper/data)")
    if output:
        parser.add_argument("--output-dir", default="reports",
                            help="output directory for generated reports (default: reports)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 4 when any warning or validation error is present")


def resolve_common(args: argparse.Namespace) -> AnalysisPaths:
    return resolve_paths(
        repo_root=Path(args.repo_root) if getattr(args, "repo_root", None) else None,
        experiments_dir=(Path(args.experiments_dir)
                         if getattr(args, "experiments_dir", None) else None),
        configs_dir=(Path(args.configs_dir)
                     if getattr(args, "configs_dir", None) else None),
        metadata_dir=(Path(args.metadata_dir)
                      if getattr(args, "metadata_dir", None) else None),
    )


def to_jsonable(value: Any) -> Any:
    """Convert analysis dataclasses to deterministic JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MetricValue):
        return {
            "value": value.value,
            "status": value.status.value,
            "source_file": value.source_file,
            "source_key": value.source_key,
            "detail": value.detail,
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def dump_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n"


def _print_analysis_text(result: Any) -> None:
    print(f"Experiments directory: {result.paths.experiments_dir}")
    print(f"Runs discovered: {len(result.records)}")
    print(f"{'Run':<24}{'Type':<8}{'Seed':<18}{'Val loss':>12}{'Clean loss':>12}  Status")
    for record in sorted(result.records, key=lambda r: r.run_id):
        seed = (str(record.seed.value) if record.seed.known
                else record.seed.status.value.upper())
        print(f"{record.run_id:<24}{record.model.model_type.value:<8}{seed:<18}"
              f"{record.metric('val_loss').display():>12}"
              f"{record.metric('clean_loss').display():>12}  {record.status.value}")
    counts: dict[str, int] = {}
    for issue in result.issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    summary = ", ".join(f"{counts.get(level, 0)} {level}"
                        for level in ("error", "warning", "info"))
    print(f"\nFindings: {summary}")


def _print_comparison_text(comparisons: ComparisonResult) -> None:
    if not comparisons.comparisons:
        print("No eligible dense/loop pairs. Missing metadata is not substituted.")
        return
    print(f"{'Category':<18}{'Metric':<12}{'Loop':<22}{'Dense':<18}"
          f"{'Delta':>11}{'Delta %':>11}  Status")
    for comp in comparisons.comparisons:
        print(f"{comp.category.value:<18}{comp.metric:<12}{comp.loop_run:<22}"
              f"{comp.dense_run:<18}{comp.loss_difference.display():>11}"
              f"{comp.percent_difference.display():>11}  {comp.status}")
    if comparisons.skipped:
        print(f"\n{len(comparisons.skipped)} candidate pairs were not comparable.")


def _strict_exit_code(result: Any, comparisons: Optional[ComparisonResult]) -> int:
    issues = list(result.issues)
    if comparisons is not None:
        issues.extend(i for comp in comparisons.comparisons for i in comp.issues)
    if any(i.severity in ("error", "warning") for i in issues):
        print("ERROR: strict mode: warnings or validation errors are present",
              file=sys.stderr)
        return EXIT_STRICT
    return EXIT_OK


def _load_or_exit(args: argparse.Namespace):
    paths = resolve_common(args)
    result = build_analysis(paths)
    if not result.records:
        print(f"ERROR: no result files found under {paths.experiments_dir}",
              file=sys.stderr)
        return None, EXIT_NO_RESULTS
    return result, EXIT_OK


def analysis_main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``analyze_experiments.py``."""
    parser = argparse.ArgumentParser(
        description="Discover, validate and summarise existing experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--format", choices=("text", "json", "both"), default="text",
                        help="stdout format; 'both' also writes analysis.json")
    parser.add_argument("--json-out", default=None,
                        help="optional path for a JSON analysis file")
    args = parser.parse_args(argv)

    result, code = _load_or_exit(args)
    if result is None:
        return code
    if args.format in ("text", "both"):
        _print_analysis_text(result)
    if args.format in ("json", "both"):
        payload = dump_json(result)
        if args.format == "json" and args.json_out is None:
            sys.stdout.write(payload)
        else:
            out = Path(args.json_out) if args.json_out else Path(args.output_dir) / "analysis.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
            print(f"Analysis JSON written to {out}", file=sys.stderr)
    if args.strict:
        return _strict_exit_code(result, None)
    return EXIT_OK


def compare_main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``compare_models.py``."""
    parser = argparse.ArgumentParser(
        description="Compare dense and loop models under matched conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--category",
                        choices=("both", "matched-parameter", "matched-compute"),
                        default="both", help="which matched category to report")
    parser.add_argument("--metric", action="append", default=None,
                        choices=("val_loss", "clean_loss"),
                        help="loss metric to compare (repeatable; default: all available)")
    parser.add_argument("--format", choices=("text", "csv", "json"), default="text",
                        help="output format for comparisons")
    args = parser.parse_args(argv)

    result, code = _load_or_exit(args)
    if result is None:
        return code
    category = (ComparisonCategory(args.category) if args.category != "both" else None)
    comparisons = build_comparisons(result, category=category, metrics=args.metric)
    if args.format == "text":
        _print_comparison_text(comparisons)
    elif args.format == "json":
        sys.stdout.write(dump_json(comparisons))
    else:
        written = write_comparison_csvs(comparisons, Path(args.output_dir))
        for name, path in written.items():
            print(f"{name}: {path}")
    if args.strict:
        return _strict_exit_code(result, comparisons)
    return EXIT_OK


def report_main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``generate_report.py``."""
    parser = argparse.ArgumentParser(
        description="Generate reproducible Markdown and CSV experiment reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--format", choices=("both", "markdown", "csv"), default="both",
                        help="report formats to generate")
    parser.add_argument("--model-types", choices=("both", "dense", "loop"), default="both",
                        help="filter runs in summary sections; comparisons always "
                             "require both model families")
    args = parser.parse_args(argv)

    result, code = _load_or_exit(args)
    if result is None:
        return code
    comparisons = build_comparisons(result)
    written = write_report_bundle(result, comparisons, Path(args.output_dir),
                                  model_types=args.model_types, formats=args.format)
    for name, path in written.items():
        print(f"{name}: {path}")
    if args.strict:
        return _strict_exit_code(result, comparisons)
    return EXIT_OK


def dispatch(kind: str, argv: Optional[list[str]] = None) -> int:
    """Run one CLI with consistent error handling and exit codes."""
    handlers = {"analyze": analysis_main, "compare": compare_main,
                "report": report_main}
    try:
        return handlers[kind](argv)
    except (OSError, ValueError) as exc:
        print(f"ERROR: analysis failed: {exc}", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
