"""Deterministic Markdown and CSV report writers.

Reports contain no timestamps or random ordering: the same inputs always
produce byte-identical output. Every row keeps the source file paths and every
report includes a SHA-256 source appendix.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .core import (
    TOOLKIT_VERSION,
    AnalysisResult,
    Comparison,
    ComparisonCategory,
    ComparisonResult,
    ExperimentRecord,
    MetricValue,
    RunStatus,
)
from .metrics import SeedCoverage, seed_coverage

SUMMARY_FIELDS = [
    "run_id", "model_type", "config_file", "declared_params", "analytic_params",
    "layer_applications", "unique_blocks", "loops", "d_model", "peak_lr",
    "warmup_ratio", "seed_value", "seed_status", "seed_label", "corpus",
    "val_loss", "clean_loss", "perplexity", "mcq_accuracy",
    "benchmark_mcq_accuracy", "cf_norm_accuracy", "forward_gflops",
    "seconds_per_batch", "subsampling_k", "checkpoint", "checkpoint_step",
    "status", "source_files",
]
COMPARISON_FIELDS = [
    "category", "metric", "loop_run", "dense_run", "loop_loss", "dense_loss",
    "loss_difference", "percent_difference", "seed_label", "corpus", "status",
    "match_basis", "bootstrap_mean_difference", "bootstrap_ci95",
    "warnings", "source_files",
]
SEED_FIELDS = ["run_id", "seed_value", "seed_status", "seed_label",
               "seed_source", "seed_detail", "source_file"]
ISSUE_FIELDS = ["scope", "run_id", "code", "severity", "message", "source_file"]
SOURCE_FIELDS = ["source_file", "schema", "size_bytes", "sha256"]


def _display(value: Any, decimals: int = 6) -> str:
    if isinstance(value, MetricValue):
        return value.display(decimals)
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.{decimals}g}"
    return str(value)


def _sources(value: Iterable[str]) -> str:
    return "; ".join(sorted(set(value)))


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(_md(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_md(cell) for cell in row) + " |")
    return lines


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _filter_records(records: list[ExperimentRecord], model_types: str) -> list[ExperimentRecord]:
    if model_types == "both":
        return records
    return [r for r in records if r.model.model_type.value == model_types]


def summary_rows(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    """Build one traceable CSV row per run."""
    rows = []
    for record in sorted(records, key=lambda r: r.run_id):
        model = record.model
        compute = record.compute
        rows.append({
            "run_id": record.run_id,
            "model_type": model.model_type.value,
            "config_file": model.config_file or "MISSING",
            "declared_params": _display(model.params, 0),
            "analytic_params": _display(model.analytic_params, 0),
            "layer_applications": _display(compute.layer_applications, 0),
            "unique_blocks": _display(compute.unique_blocks, 0),
            "loops": _display(compute.loops, 0),
            "d_model": _display(model.d_model, 0),
            "peak_lr": _display(model.peak_lr, 6),
            "warmup_ratio": _display(model.warmup_ratio, 6),
            "seed_value": _display(record.seed.value, 0),
            "seed_status": record.seed.status.value,
            "seed_label": record.seed.label,
            "corpus": record.corpus.value or record.corpus.status.value.upper(),
            "val_loss": _display(record.metric("val_loss")),
            "clean_loss": _display(record.metric("clean_loss")),
            "perplexity": _display(record.metric("perplexity")),
            "mcq_accuracy": _display(record.metric("mcq_accuracy")),
            "benchmark_mcq_accuracy": _display(record.metric("benchmark_mcq_accuracy")),
            "cf_norm_accuracy": _display(record.metric("cf_norm_accuracy")),
            "forward_gflops": _display(compute.forward_gflops),
            "seconds_per_batch": _display(compute.seconds_per_batch),
            "subsampling_k": _display(compute.subsampling_k, 0),
            "checkpoint": record.training.checkpoint or "MISSING",
            "checkpoint_step": _display(record.training.checkpoint_step, 0),
            "status": record.status.value,
            "source_files": _sources(record.sources),
        })
    return rows


def comparison_rows(comparisons: list[Comparison]) -> list[dict[str, Any]]:
    """Build CSV rows for model comparisons."""
    rows = []
    for comp in comparisons:
        rows.append({
            "category": comp.category.value,
            "metric": comp.metric,
            "loop_run": comp.loop_run,
            "dense_run": comp.dense_run,
            "loop_loss": _display(comp.loop_loss),
            "dense_loss": _display(comp.dense_loss),
            "loss_difference": _display(comp.loss_difference),
            "percent_difference": _display(comp.percent_difference),
            "seed_label": comp.seed_label,
            "corpus": comp.corpus or "MISSING",
            "status": comp.status,
            "match_basis": comp.match_basis,
            "bootstrap_mean_difference": _display(comp.bootstrap_mean_difference),
            "bootstrap_ci95": comp.bootstrap_ci95 or "UNAVAILABLE",
            "warnings": _sources(i.message for i in comp.warnings),
            "source_files": _sources(comp.source_files),
        })
    return rows


def seed_rows(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    return [{
        "run_id": record.run_id,
        "seed_value": _display(record.seed.value, 0),
        "seed_status": record.seed.status.value,
        "seed_label": record.seed.label,
        "seed_source": record.seed.source_file or record.seed.detail or "MISSING",
        "seed_detail": record.seed.detail or "MISSING",
        "source_file": record.result_file or "MISSING",
    } for record in sorted(records, key=lambda r: r.run_id)]


def issue_rows(result: AnalysisResult, comparisons: ComparisonResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue in result.global_issues:
        rows.append({"scope": "analysis", "run_id": "", "code": issue.code,
                     "severity": issue.severity, "message": issue.message,
                     "source_file": issue.source_file or "MISSING"})
    for record in sorted(result.records, key=lambda r: r.run_id):
        for issue in record.issues:
            rows.append({"scope": "run", "run_id": record.run_id, "code": issue.code,
                         "severity": issue.severity, "message": issue.message,
                         "source_file": issue.source_file or "MISSING"})
    for comp in comparisons.comparisons:
        for issue in comp.issues:
            rows.append({"scope": "comparison",
                         "run_id": f"{comp.loop_run} vs {comp.dense_run}",
                         "code": issue.code, "severity": issue.severity,
                         "message": issue.message,
                         "source_file": issue.source_file or "MISSING"})
    for skipped in comparisons.skipped:
        rows.append({"scope": "skipped_pair",
                     "run_id": f"{skipped.loop_run} vs {skipped.dense_run}",
                     "code": "PAIR_NOT_COMPARABLE", "severity": "info",
                     "message": skipped.reason, "source_file": "MISSING"})
    return rows


def write_comparison_csvs(comparisons: ComparisonResult,
                          output_dir: Path) -> dict[str, Path]:
    """Write only the comparison CSVs (used by ``compare_models.py``)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matched = [c for c in comparisons.comparisons
               if c.category is ComparisonCategory.MATCHED_PARAMETER]
    compute = [c for c in comparisons.comparisons
               if c.category is ComparisonCategory.MATCHED_COMPUTE]
    written: dict[str, Path] = {}
    for name, rows in (("model_comparisons", comparison_rows(comparisons.comparisons)),
                       ("matched_parameter", comparison_rows(matched)),
                       ("matched_compute", comparison_rows(compute))):
        path = output_dir / f"{name}.csv"
        _write_csv(path, COMPARISON_FIELDS, rows)
        written[name] = path
    return written


def write_csv_reports(result: AnalysisResult, comparisons: ComparisonResult,
                      output_dir: Path, *, model_types: str = "both") -> dict[str, Path]:
    """Write the CSV report bundle. Returns a mapping of name to path."""
    written = write_comparison_csvs(comparisons, output_dir)
    records = _filter_records(result.records, model_types)
    sources = [info.to_row() for _, info in sorted(result.sources.items())]
    issues = issue_rows(result, comparisons)
    specs = [
        ("experiment_summary", SUMMARY_FIELDS, summary_rows(records)),
        ("seed_coverage", SEED_FIELDS, seed_rows(records)),
        ("incomplete_experiments", ISSUE_FIELDS,
         [r for r in issues if r["severity"] in ("error", "warning")]),
        ("warnings", ISSUE_FIELDS, issues),
        ("source_files", SOURCE_FIELDS, sources),
    ]
    for name, fields, rows in specs:
        path = output_dir / f"{name}.csv"
        _write_csv(path, fields, rows)
        written[name] = path
    return written


def _status_counts(result: AnalysisResult) -> list[tuple[str, int]]:
    counts = {status.value: 0 for status in RunStatus}
    for record in result.records:
        counts[record.status.value] += 1
    return sorted(counts.items())


def write_markdown_report(result: AnalysisResult, comparisons: ComparisonResult,
                          output_dir: Path, *, model_types: str = "both") -> Path:
    """Write the Markdown report and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_report.md"
    records = _filter_records(result.records, model_types)
    coverage: SeedCoverage = seed_coverage(records)
    summary = summary_rows(records)
    comps = comparison_rows(comparisons.comparisons)
    issues = issue_rows(result, comparisons)
    lines: list[str] = [
        "# AstraCSE experiment analysis report",
        "",
        f"Toolkit version: `{TOOLKIT_VERSION}`. All values are read from existing "
        "files; no model was trained, fine-tuned or re-evaluated.",
        "",
        "## Provenance",
        "",
    ]
    for key, value in result.paths.display().items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        f"- Result files discovered: {len(result.records)}",
        f"- Input files hashed: {len(result.sources)}",
        "- Hash algorithm: SHA-256 (`source_files.csv`); no generation timestamp "
        "is included, so re-running on unchanged inputs is byte-identical.",
        "",
        "## Status summary",
        "",
    ]
    lines += _table(["Run status", "Count"],
                    [[name, count] for name, count in _status_counts(result)])
    lines += [
        "",
        f"Seeds: {coverage.known} of {coverage.total} runs have a seed value "
        f"({coverage.explicit} explicit, {coverage.inferred} inferred from the run id, "
        f"{coverage.missing} missing). "
        f"Known ratio: {_display(coverage.known_ratio, 3)}. "
        "Missing seeds are never replaced by zero.",
        "",
        "## Experiment summaries",
        "",
    ]
    summary_headers = ["Run", "Type", "Config", "Params", "Apps", "K x T", "Seed",
                       "Corpus", "Val loss", "Clean loss", "PPL", "MCQ-650",
                       "MCQ-10", "CF-norm", "GFLOPs", "s/batch", "Status", "Sources"]
    summary_table = []
    for row in summary:
        summary_table.append([
            row["run_id"], row["model_type"], row["config_file"],
            row["declared_params"], row["layer_applications"],
            f"{row['unique_blocks']} x {row['loops']}",
            f"{row['seed_value']} ({row['seed_status']})", row["corpus"],
            row["val_loss"], row["clean_loss"], row["perplexity"],
            row["mcq_accuracy"], row["benchmark_mcq_accuracy"],
            row["cf_norm_accuracy"], row["forward_gflops"],
            row["seconds_per_batch"], row["status"], row["source_files"],
        ])
    lines += _table(summary_headers, summary_table)

    comparison_headers = ["Category", "Metric", "Loop", "Dense", "Loop loss",
                          "Dense loss", "Delta (nats)", "Delta (%)", "Seed",
                          "Status", "Match basis", "Sources"]
    for title, category in (
        ("Model comparisons (overview)", None),
        ("Matched-parameter results", ComparisonCategory.MATCHED_PARAMETER),
        ("Matched-compute results", ComparisonCategory.MATCHED_COMPUTE),
    ):
        rows = [c for c in comps if category is None or c["category"] == category.value]
        lines += ["", f"## {title}", ""]
        if not rows:
            lines.append("No eligible pairs: missing metadata is not substituted.")
            continue
        lines += _table(comparison_headers, [[
            row["category"], row["metric"], row["loop_run"], row["dense_run"],
            row["loop_loss"], row["dense_loss"], row["loss_difference"],
            row["percent_difference"], row["seed_label"], row["status"],
            row["match_basis"], row["source_files"],
        ] for row in rows])
        if category is ComparisonCategory.MATCHED_PARAMETER:
            lines += [
                "",
                "Matched-parameter pairs share the unique-block count and differ only "
                "by the loop's depth-conditioning embedding. Dense controls in this "
                "category run fewer layer applications, so this category is kept "
                "separate from matched-compute pairs.",
            ]
        if category is ComparisonCategory.MATCHED_COMPUTE:
            lines += [
                "",
                "Matched-compute pairs run the same number of layer applications per "
                "token at the same width. Their parameter counts are allowed to differ.",
            ]

    lines += ["", "## Seed coverage", ""]
    lines += _table(["Run", "Seed", "Status", "Label", "Provenance", "Result file"], [
        [r["run_id"], r["seed_value"], r["seed_status"], r["seed_label"],
         r["seed_source"], r["source_file"]] for r in seed_rows(records)
    ])
    lines += ["", "## Incomplete experiments and validation findings", ""]
    problem_rows = [r for r in issues if r["severity"] in ("error", "warning")]
    if problem_rows:
        lines += _table(["Scope", "Run", "Code", "Severity", "Message", "Source"], [
            [r["scope"], r["run_id"], r["code"], r["severity"], r["message"],
             r["source_file"]] for r in problem_rows
        ])
    else:
        lines.append("No validation findings.")
    lines += ["", "## Warnings and informational findings", ""]
    if issues:
        lines += _table(["Scope", "Run", "Code", "Severity", "Message", "Source"], [
            [r["scope"], r["run_id"], r["code"], r["severity"], r["message"],
             r["source_file"]] for r in issues
        ])
    else:
        lines.append("No findings.")
    skipped_rows = [[s.loop_run, s.dense_run, s.category or "unclassified", s.reason]
                    for s in comparisons.skipped]
    lines += ["", "## Pairs not compared", ""]
    if skipped_rows:
        lines += _table(["Loop", "Dense", "Category", "Reason"], skipped_rows)
    else:
        lines.append("No candidate pairs were rejected.")

    lines += [
        "",
        "## Metric definitions",
        "",
        "- Losses are mean token cross-entropy in nats (lower is better). "
        "`val_loss` is the official strided evaluation; `clean_loss` is the "
        "token-weighted re-score on non-copied validation tokens.",
        "- `loss_difference = loop_loss - dense_loss` (nats). Positive means the "
        "loop model has the higher loss.",
        "- `percent_difference = 100 * (loop_loss - dense_loss) / abs(dense_loss)`. "
        "If `dense_loss == 0`, the percentage is undefined and reported as "
        "`UNAVAILABLE`; it is never replaced by zero. A measured zero loss is "
        "reported as `0.000000` with status `available`.",
        "- `UNAVAILABLE` means the source format does not provide the metric; "
        "`MISSING` means an expected value is absent; `INVALID` means a present "
        "value failed validation.",
        "- Analytic GFLOPs use `2 * (n_layer * params_per_unique_block + "
        "vocab_size * d_model) / 1e9` forward operations per token, excluding "
        "attention score products. `s/batch` values are CPU measurements from "
        "`cpu_bootstrap.json` and include a subsampling factor where recorded.",
        "",
        "## Limitations",
        "",
        "- Random seeds are absent from the result JSONs. Only `_sN` run-id "
        "suffixes provide a seed value; other runs are reported as missing.",
        "- Corpus regime is inferred from the `_v2` run-id marker; continuation "
        "runs without that marker are not paired.",
        "- Checkpoint binaries and GPU timing traces are not in the repository. "
        "Checkpoint filenames come from re-score metadata; compute is analytic "
        "plus the recorded CPU proxy.",
        "- MCQ-650 values from different scoring rules are kept in separate "
        "columns; the old mean-loss rule and CF-norm are not interchangeable.",
        "- Absent metadata is reported as missing/unavailable and never "
        "imputed, so some experiments cannot be compared.",
        "",
        "## Source files",
        "",
    ]
    source_rows = [[info.path, info.schema, info.size_bytes, info.sha256]
                   for _, info in sorted(result.sources.items())]
    lines += _table(["File", "Schema", "Bytes", "SHA-256"], source_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report_bundle(result: AnalysisResult, comparisons: ComparisonResult,
                        output_dir: Path, *, model_types: str = "both",
                        formats: str = "both") -> dict[str, Path]:
    """Write the selected report formats; returns name-to-path for all files."""
    written: dict[str, Path] = {}
    if formats in ("both", "csv"):
        written.update(write_csv_reports(result, comparisons, output_dir,
                                         model_types=model_types))
    if formats in ("both", "markdown"):
        written["experiment_report"] = write_markdown_report(
            result, comparisons, output_dir, model_types=model_types
        )
    return written
