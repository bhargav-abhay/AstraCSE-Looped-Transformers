"""Analyse AstraCSE experiment outputs without retraining or modifying them.

This package implements the shared pipeline used by
``scripts/analyze_experiments.py``, ``scripts/compare_models.py`` and
``scripts/generate_report.py``. The design and the exact formulas are
documented in ``docs/EXPERIMENT_ANALYSIS.md``.
"""

from .core import (
    TOOLKIT_VERSION,
    AnalysisResult,
    Comparison,
    ComparisonCategory,
    ComparisonResult,
    ComputeSpec,
    ExperimentRecord,
    Issue,
    LabelInfo,
    MetricValue,
    ModelSpec,
    ModelType,
    RunStatus,
    SeedInfo,
    SeedStatus,
    SkippedPair,
    SourceInfo,
    TrainingInfo,
    ValueStatus,
)
from .comparisons import build_comparisons, classify_pair
from .discovery import AnalysisPaths, build_analysis, resolve_paths
from .metrics import (
    SeedCoverage,
    absolute_loss_difference,
    derive_compute_cost,
    percentage_loss_difference,
    perplexity_consistent,
    seed_coverage,
)
from .reports import write_csv_reports, write_markdown_report

__all__ = [
    "TOOLKIT_VERSION",
    "AnalysisPaths",
    "AnalysisResult",
    "Comparison",
    "ComparisonCategory",
    "ComparisonResult",
    "ComputeSpec",
    "ExperimentRecord",
    "Issue",
    "LabelInfo",
    "MetricValue",
    "ModelSpec",
    "ModelType",
    "RunStatus",
    "SeedCoverage",
    "SeedInfo",
    "SeedStatus",
    "SkippedPair",
    "SourceInfo",
    "TrainingInfo",
    "ValueStatus",
    "absolute_loss_difference",
    "build_analysis",
    "build_comparisons",
    "classify_pair",
    "derive_compute_cost",
    "percentage_loss_difference",
    "perplexity_consistent",
    "resolve_paths",
    "seed_coverage",
    "write_csv_reports",
    "write_markdown_report",
]
