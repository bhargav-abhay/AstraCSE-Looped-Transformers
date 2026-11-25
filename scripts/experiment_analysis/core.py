"""Core value types for the experiment analysis toolkit.

Every extracted quantity is wrapped in :class:`MetricValue` so that a missing
value, an invalid value, an unavailable metric and a genuine zero are always
distinguishable. No absent metadata is ever converted to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

TOOLKIT_VERSION = "1.0.0"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
SEVERITIES = (SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO)


class ValueStatus(str, Enum):
    """Availability state of a single extracted value."""

    AVAILABLE = "available"
    MISSING = "missing"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MetricValue:
    """A number with explicit provenance and availability status.

    ``value`` is not ``None`` if and only if ``status`` is ``AVAILABLE``.
    A value of ``0.0`` with status ``AVAILABLE`` is a measured zero and is
    never equivalent to ``MISSING``, ``INVALID`` or ``UNAVAILABLE``.
    """

    value: Optional[float] = None
    status: ValueStatus = ValueStatus.MISSING
    source_file: Optional[str] = None
    source_key: Optional[str] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is ValueStatus.AVAILABLE:
            if self.value is None:
                raise ValueError("an available MetricValue must have a value")
            if isinstance(self.value, float) and not math.isfinite(self.value):
                raise ValueError("an available MetricValue must be finite")
        elif self.value is not None:
            raise ValueError("a non-available MetricValue cannot have a value")

    @classmethod
    def available(
        cls,
        value: Any,
        source_file: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> "MetricValue":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return cls.invalid(source_file, source_key, "not a number")
        number = float(value)
        if not math.isfinite(number):
            return cls.invalid(source_file, source_key, "not finite (NaN or Inf)")
        return cls(number, ValueStatus.AVAILABLE, source_file, source_key)

    @classmethod
    def missing(
        cls,
        source_file: Optional[str] = None,
        source_key: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> "MetricValue":
        return cls(None, ValueStatus.MISSING, source_file, source_key, detail)

    @classmethod
    def invalid(
        cls,
        source_file: Optional[str] = None,
        source_key: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> "MetricValue":
        return cls(None, ValueStatus.INVALID, source_file, source_key, detail)

    @classmethod
    def unavailable(
        cls,
        source_file: Optional[str] = None,
        source_key: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> "MetricValue":
        return cls(None, ValueStatus.UNAVAILABLE, source_file, source_key, detail)

    @property
    def usable(self) -> bool:
        return self.status is ValueStatus.AVAILABLE

    @property
    def is_zero(self) -> bool:
        return self.usable and self.value == 0.0

    def display(self, decimals: int = 6) -> str:
        if self.usable:
            return f"{self.value:.{decimals}f}"
        return self.status.value.upper()


class SeedStatus(str, Enum):
    """How a random seed was obtained (or why it is unavailable)."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    MISSING = "missing"


@dataclass(frozen=True)
class SeedInfo:
    """Seed value, its provenance and a label used for pairwise matching."""

    value: Optional[int] = None
    status: SeedStatus = SeedStatus.MISSING
    label: str = "unlabelled"
    source_file: Optional[str] = None
    detail: Optional[str] = None

    @property
    def known(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class LabelInfo:
    """A categorical label such as the corpus regime of a run."""

    value: Optional[str] = None
    status: ValueStatus = ValueStatus.MISSING
    source: Optional[str] = None
    detail: Optional[str] = None


class ModelType(str, Enum):
    DENSE = "dense"
    LOOP = "loop"
    UNKNOWN = "unknown"


@dataclass
class ModelSpec:
    """Architecture metadata, normally resolved from a YAML config."""

    model_type: ModelType = ModelType.UNKNOWN
    config_file: Optional[str] = None
    config_source: Optional[str] = None
    params: MetricValue = field(default_factory=MetricValue)
    analytic_params: MetricValue = field(default_factory=MetricValue)
    d_model: Optional[int] = None
    n_layer: Optional[int] = None
    n_unique: Optional[int] = None
    n_loops: Optional[int] = None
    vocab_size: Optional[int] = None
    d_ff: Optional[int] = None
    max_seq_len: Optional[int] = None
    peak_lr: Optional[float] = None
    warmup_ratio: Optional[float] = None


@dataclass
class ComputeSpec:
    """Compute metadata: analytic cost and, when recorded, measured cost."""

    layer_applications: Optional[int] = None
    unique_blocks: Optional[int] = None
    loops: Optional[int] = None
    forward_gflops: MetricValue = field(default_factory=MetricValue)
    seconds_per_batch: MetricValue = field(default_factory=MetricValue)
    subsampling_k: MetricValue = field(default_factory=MetricValue)
    batches: MetricValue = field(default_factory=MetricValue)
    sources: tuple[str, ...] = ()


@dataclass
class TrainingInfo:
    """Training/checkpoint metadata; all fields optional and traceable."""

    checkpoint: Optional[str] = None
    checkpoint_step: Optional[int] = None
    precision: Optional[str] = None
    batches: MetricValue = field(default_factory=MetricValue)
    clean_tokens: MetricValue = field(default_factory=MetricValue)
    copied_tokens: MetricValue = field(default_factory=MetricValue)
    truncated_items: MetricValue = field(default_factory=MetricValue)
    final_step: Optional[str] = None
    started: Optional[str] = None
    finished: Optional[str] = None


@dataclass(frozen=True)
class Issue:
    """A completeness, validation or provenance finding."""

    code: str
    severity: str
    message: str
    source_file: Optional[str] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown issue severity: {self.severity}")


class RunStatus(str, Enum):
    COMPLETE = "complete"
    COMPLETE_WITH_WARNINGS = "complete_with_warnings"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass
class ExperimentRecord:
    """One run, assembled from its result files and all available metadata."""

    run_id: str
    result_file: Optional[str] = None
    benchmark_file: Optional[str] = None
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    seed: SeedInfo = field(default_factory=SeedInfo)
    corpus: LabelInfo = field(default_factory=LabelInfo)
    model: ModelSpec = field(default_factory=ModelSpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    training: TrainingInfo = field(default_factory=TrainingInfo)
    issues: list[Issue] = field(default_factory=list)
    sources: tuple[str, ...] = ()
    status: RunStatus = RunStatus.INCOMPLETE

    def metric(self, name: str) -> MetricValue:
        return self.metrics.get(name, MetricValue.unavailable(detail=f"unknown metric {name}"))

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == SEVERITY_WARNING)


class ComparisonCategory(str, Enum):
    MATCHED_PARAMETER = "matched-parameter"
    MATCHED_COMPUTE = "matched-compute"


@dataclass
class Comparison:
    """A single dense/loop comparison for one loss metric."""

    category: ComparisonCategory
    metric: str
    loop_run: str
    dense_run: str
    loop_loss: MetricValue
    dense_loss: MetricValue
    loss_difference: MetricValue
    percent_difference: MetricValue
    seed_label: str
    corpus: Optional[str]
    match_basis: str
    status: str = "ok"
    issues: list[Issue] = field(default_factory=list)
    bootstrap_mean_difference: MetricValue = field(default_factory=MetricValue)
    bootstrap_ci95: Optional[str] = None
    source_files: tuple[str, ...] = ()

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity != SEVERITY_INFO)


@dataclass(frozen=True)
class SkippedPair:
    """A candidate dense/loop pair that failed an eligibility check."""

    loop_run: str
    dense_run: str
    reason: str
    category: Optional[str] = None


@dataclass
class ComparisonResult:
    comparisons: list[Comparison] = field(default_factory=list)
    skipped: list[SkippedPair] = field(default_factory=list)


@dataclass(frozen=True)
class SourceInfo:
    """Traceability record for one input file."""

    path: str
    sha256: str
    size_bytes: int
    schema: str

    def to_row(self) -> dict[str, Any]:
        return {
            "source_file": self.path,
            "schema": self.schema,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass
class AnalysisResult:
    """The complete validated analysis of one experiments directory."""

    paths: "AnalysisPaths"
    records: list[ExperimentRecord] = field(default_factory=list)
    global_issues: list[Issue] = field(default_factory=list)
    sources: dict[str, SourceInfo] = field(default_factory=dict)
    bootstrap_pairs: list[Any] = field(default_factory=list)

    @property
    def issues(self) -> tuple[Issue, ...]:
        return tuple(self.global_issues) + tuple(
            issue for record in self.records for issue in record.issues
        )

    def record(self, run_id: str) -> ExperimentRecord:
        for record in self.records:
            if record.run_id == run_id:
                return record
        raise KeyError(run_id)


def relative_path(path: Optional[Path], root: Path) -> Optional[str]:
    """Return a stable, repo-relative POSIX path for reporting."""
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
