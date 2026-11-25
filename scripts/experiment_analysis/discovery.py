"""Discovery, validation and metadata enrichment for existing result files.

Supported inputs are the repository's own formats: ``<run>.mmlu_cs.json`` and
``<run>.benchmark.json`` in the results directory, ``rescore_*.json``, the
conditional-MCQ files, ``curve_v2.json``, ``cpu_bootstrap.json``, the YAML
configs and the SHA-256 manifest. Unsupported files are never silently
reinterpreted; they are reported as warnings.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .core import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    AnalysisResult,
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
    SourceInfo,
    ValueStatus,
    relative_path,
)
from .metrics import derive_compute_cost, perplexity_consistent

OFFICIAL_SUFFIX = ".mmlu_cs.json"
BENCHMARK_SUFFIX = ".benchmark.json"
MALFORMED_CODES = frozenset({
    "MALFORMED_JSON", "MALFORMED_RECORD", "INVALID_METRIC", "INVALID_CONFIG"
})
CORE_METRICS = ("val_loss", "perplexity", "mcq_accuracy", "mcq_correct", "mcq_total")


@dataclass(frozen=True)
class AnalysisPaths:
    """Directories used by one analysis run."""

    repo_root: Path
    experiments_dir: Path
    configs_dir: Path
    metadata_dir: Path

    def display(self) -> dict[str, str]:
        return {
            "repo_root": relative_path(self.repo_root, self.repo_root) or ".",
            "experiments_dir": relative_path(self.experiments_dir, self.repo_root) or ".",
            "configs_dir": relative_path(self.configs_dir, self.repo_root) or ".",
            "metadata_dir": relative_path(self.metadata_dir, self.repo_root) or ".",
        }


def resolve_paths(repo_root: Optional[Path] = None,
                  experiments_dir: Optional[Path] = None,
                  configs_dir: Optional[Path] = None,
                  metadata_dir: Optional[Path] = None) -> AnalysisPaths:
    """Resolve analysis directories, defaulting to the repository layout."""
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    return AnalysisPaths(
        repo_root=root,
        experiments_dir=(experiments_dir or root / "paper" / "data" / "results").resolve(),
        configs_dir=(configs_dir or root / "configs").resolve(),
        metadata_dir=(metadata_dir or root / "paper" / "data").resolve(),
    )


@dataclass
class ConfigInfo:
    path: Path
    raw: dict[str, Any]
    model: dict[str, Any]
    declared_params: MetricValue
    analytic_params: MetricValue


@dataclass
class RescoreInfo:
    name: str
    source_file: str
    config: Optional[str] = None
    checkpoint: Optional[str] = None
    precision: Optional[str] = None
    full_val_loss: MetricValue = field(default_factory=MetricValue)
    clean_loss: MetricValue = field(default_factory=MetricValue)
    copied_loss: MetricValue = field(default_factory=MetricValue)
    reported_val_loss: MetricValue = field(default_factory=MetricValue)
    reproduces_reported: Optional[bool] = None
    batches: MetricValue = field(default_factory=MetricValue)
    clean_tokens: MetricValue = field(default_factory=MetricValue)
    copied_tokens: MetricValue = field(default_factory=MetricValue)


@dataclass
class McqInfo:
    name: str
    source_file: str
    accuracies: dict[str, MetricValue] = field(default_factory=dict)
    checkpoint: Optional[str] = None
    truncated_items: MetricValue = field(default_factory=MetricValue)


@dataclass
class CurveInfo:
    name: str
    source_file: str
    final_step: Optional[str] = None
    full_val_loss: MetricValue = field(default_factory=MetricValue)
    clean_loss: MetricValue = field(default_factory=MetricValue)
    reported_val_loss: MetricValue = field(default_factory=MetricValue)
    reproduces_reported: Optional[bool] = None


@dataclass(frozen=True)
class BootstrapPair:
    name: str
    loop: str
    dense: str
    category: str
    seed_label: str
    mean_difference: MetricValue
    ci95: Optional[str]
    source_file: str


@dataclass
class CpuInfo:
    source_file: str
    seconds_per_batch: dict[str, MetricValue] = field(default_factory=dict)
    k_subsampling: dict[str, MetricValue] = field(default_factory=dict)
    model_batches: dict[str, MetricValue] = field(default_factory=dict)
    pairs: list[BootstrapPair] = field(default_factory=list)
    started: Optional[str] = None
    finished: Optional[str] = None


@dataclass
class Metadata:
    configs_by_file: dict[str, ConfigInfo] = field(default_factory=dict)
    configs_by_key: dict[str, ConfigInfo] = field(default_factory=dict)
    rescore: dict[str, RescoreInfo] = field(default_factory=dict)
    mcq: dict[str, McqInfo] = field(default_factory=dict)
    curve: dict[str, CurveInfo] = field(default_factory=dict)
    cpu: Optional[CpuInfo] = None
    declared_runs: dict[str, tuple[str, str]] = field(default_factory=dict)


def _issue(code: str, severity: str, message: str,
           source_file: Optional[str] = None,
           run_id: Optional[str] = None) -> Issue:
    return Issue(code=code, severity=severity, message=message,
                 source_file=source_file, run_id=run_id)


def _register_source(path: Path, paths: AnalysisPaths,
                     sources: dict[str, SourceInfo], schema: str) -> str:
    rel = relative_path(path, paths.repo_root) or path.name
    if rel not in sources:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
        except OSError:
            digest, size = "", -1
        sources[rel] = SourceInfo(path=rel, sha256=digest, size_bytes=size, schema=schema)
    return rel


def _read_json(path: Path, paths: AnalysisPaths, issues: list[Issue],
               sources: dict[str, SourceInfo], schema: str,
               run_id: Optional[str] = None) -> Optional[Any]:
    rel = _register_source(path, paths, sources, schema)
    if not path.is_file():
        issues.append(_issue("MISSING_SOURCE_FILE", SEVERITY_ERROR,
                             f"expected input file is missing: {rel}", rel, run_id))
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(_issue("MALFORMED_JSON", SEVERITY_ERROR,
                             f"could not parse {rel}: {exc}", rel, run_id))
        return None


def _number(record: Any, key: str, rel: str, issues: list[Issue],
            run_id: Optional[str], *, integer: bool = False,
            minimum: Optional[float] = None, maximum: Optional[float] = None,
            required: bool = True, source_key: Optional[str] = None) -> MetricValue:
    source_key = source_key or key
    if not isinstance(record, dict) or key not in record:
        if required:
            issues.append(_issue("MISSING_METRIC", SEVERITY_ERROR,
                                 f"missing field '{source_key}'", rel, run_id))
        return MetricValue.missing(rel, source_key, "field absent")
    value = record[key]
    if integer and (isinstance(value, bool) or not isinstance(value, int)):
        issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                             f"field '{source_key}' must be an integer", rel, run_id))
        return MetricValue.invalid(rel, source_key, "not an integer")
    if not integer and (isinstance(value, bool) or not isinstance(value, (int, float))):
        issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                             f"field '{source_key}' must be numeric", rel, run_id))
        return MetricValue.invalid(rel, source_key, "not numeric")
    metric = MetricValue.available(value, rel, source_key)
    if not metric.usable:
        issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                             f"field '{source_key}' is {metric.detail}", rel, run_id))
        return metric
    assert metric.value is not None
    if minimum is not None and metric.value < minimum:
        issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                             f"field '{source_key}' is below {minimum}", rel, run_id))
        return MetricValue.invalid(rel, source_key, f"below {minimum}")
    if maximum is not None and metric.value > maximum:
        issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                             f"field '{source_key}' is above {maximum}", rel, run_id))
        return MetricValue.invalid(rel, source_key, f"above {maximum}")
    return metric


def _optional_number(record: Any, key: str, rel: str,
                     source_key: Optional[str] = None) -> MetricValue:
    source_key = source_key or key
    if not isinstance(record, dict) or key not in record:
        return MetricValue.missing(rel, source_key, "field absent")
    return MetricValue.available(record[key], rel, source_key)


def discover_result_files(paths: AnalysisPaths,
                          issues: list[Issue]) -> dict[str, dict[str, Optional[Path]]]:
    """Find official result files and group them by run id."""
    found: dict[str, dict[str, Optional[Path]]] = {}
    if not paths.experiments_dir.is_dir():
        issues.append(_issue(
            "MISSING_EXPERIMENTS_DIR", SEVERITY_ERROR,
            f"experiments directory does not exist: {relative_path(paths.experiments_dir, paths.repo_root)}",
            relative_path(paths.experiments_dir, paths.repo_root),
        ))
        return found
    for suffix, slot in ((OFFICIAL_SUFFIX, "official"),
                         (BENCHMARK_SUFFIX, "benchmark")):
        for path in sorted(paths.experiments_dir.rglob(f"*{suffix}")):
            run_id = path.name[:-len(suffix)]
            if not run_id:
                continue
            rel = relative_path(path, paths.repo_root) or path.name
            entry = found.setdefault(run_id, {"official": None, "benchmark": None})
            if entry[slot] is not None:
                issues.append(_issue(
                    "DUPLICATE_RESULT_FILE", SEVERITY_WARNING,
                    f"two {suffix} files for run '{run_id}'; using {rel}", rel, run_id,
                ))
                continue
            entry[slot] = path
    return found


def parse_official_metrics(path: Path, paths: AnalysisPaths, issues: list[Issue],
                           sources: dict[str, SourceInfo], run_id: str,
                           *, benchmark: bool,
                           raw_out: Optional[list] = None) -> dict[str, MetricValue]:
    """Validate one official result file and extract its metrics."""
    schema = "official.benchmark" if benchmark else "official.mmlu_cs"
    data = _read_json(path, paths, issues, sources, schema, run_id)
    if raw_out is not None:
        raw_out.append(data)
    rel = relative_path(path, paths.repo_root) or path.name
    metrics: dict[str, MetricValue] = {}
    if data is not None and not isinstance(data, dict):
        issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                             f"{rel} must contain a JSON object", rel, run_id))
        data = None
    if data is None:
        data = {}

    if benchmark:
        for name in ("val_loss", "perplexity"):
            metrics[name] = MetricValue.unavailable(
                rel, name, "the benchmark schema does not provide validation loss"
            )
    else:
        metrics["val_loss"] = _number(data, "val_loss", rel, issues, run_id)
        metrics["perplexity"] = _number(
            data, "perplexity", rel, issues, run_id, minimum=0.0
        )
    metrics["mcq_accuracy"] = _number(
        data, "mcq_accuracy", rel, issues, run_id, minimum=0.0, maximum=1.0
    )
    metrics["mcq_correct"] = _number(
        data, "mcq_correct", rel, issues, run_id, integer=True, minimum=0
    )
    metrics["mcq_total"] = _number(
        data, "mcq_total", rel, issues, run_id, integer=True, minimum=1
    )

    accuracy = metrics["mcq_accuracy"]
    correct = metrics["mcq_correct"]
    total = metrics["mcq_total"]
    if all(m.usable for m in (accuracy, correct, total)):
        assert accuracy.value is not None and correct.value is not None
        assert total.value is not None
        if correct.value > total.value:
            issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                                 f"mcq_correct ({correct.value:.0f}) exceeds mcq_total "
                                 f"({total.value:.0f})", rel, run_id))
        expected = correct.value / total.value
        if abs(accuracy.value - expected) > 1e-6:
            issues.append(_issue("METRIC_INCONSISTENT", SEVERITY_WARNING,
                                 f"mcq_accuracy {accuracy.value:.6f} disagrees with "
                                 f"correct/total {expected:.6f}", rel, run_id))
    if not benchmark and "val_loss" in metrics:
        if not perplexity_consistent(metrics["val_loss"], metrics["perplexity"]):
            issues.append(_issue("METRIC_INCONSISTENT", SEVERITY_WARNING,
                                 "perplexity is not exp(val_loss) within 1e-3", rel, run_id))

    per_subject = data.get("per_subject")
    if not isinstance(per_subject, dict) or not per_subject:
        issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                             "per_subject must be a non-empty object", rel, run_id))
    else:
        for subject in sorted(per_subject):
            entry = per_subject[subject]
            if not isinstance(entry, dict):
                issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                                     f"per_subject.{subject} is not an object", rel, run_id))
                continue
            for field_name in ("correct", "total", "accuracy"):
                _number(entry, field_name, rel, issues, run_id,
                        source_key=f"per_subject.{subject}.{field_name}",
                        required=(field_name != "accuracy"), integer=(field_name != "accuracy"))
    for name, metric in metrics.items():
        if metric.is_zero:
            issues.append(_issue("ZERO_VALUED_METRIC", SEVERITY_INFO,
                                 f"{name} is exactly zero (a valid measured value)", rel, run_id))
    return metrics


def load_configs(paths: AnalysisPaths, issues: list[Issue],
                 sources: dict[str, SourceInfo]) -> Metadata:
    """Load and validate all YAML model configs in the repository."""
    meta = Metadata()
    if not paths.configs_dir.is_dir():
        issues.append(_issue("MISSING_CONFIGS_DIR", SEVERITY_WARNING,
                             f"config directory does not exist: "
                             f"{relative_path(paths.configs_dir, paths.repo_root)}"))
        return meta
    scripts_dir = str(paths.repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from verify_param_count import count_params as analytic_count
    except ImportError as exc:
        issues.append(_issue("PARAM_COUNTER_UNAVAILABLE", SEVERITY_WARNING,
                             f"cannot import scripts/verify_param_count.py: {exc}"))
        analytic_count = None

    for path in sorted(paths.configs_dir.glob("*.yaml")):
        rel = _register_source(path, paths, sources, "model_config")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            issues.append(_issue("INVALID_CONFIG", SEVERITY_ERROR,
                                 f"could not parse config {rel}: {exc}", rel))
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
            issues.append(_issue("INVALID_CONFIG", SEVERITY_ERROR,
                                 f"{rel} lacks a model: section", rel))
            continue
        model = raw["model"]
        required = ("n_layer", "d_model", "vocab_size", "expected_param_count")
        missing = [key for key in required if key not in model]
        if missing:
            issues.append(_issue("INVALID_CONFIG", SEVERITY_ERROR,
                                 f"{rel} lacks model keys: {', '.join(missing)}", rel))
            continue
        declared = MetricValue.available(model["expected_param_count"], rel,
                                         "model.expected_param_count")
        if analytic_count is None:
            analytic = MetricValue.unavailable(rel, "model", "parameter counter unavailable")
        else:
            try:
                breakdown = analytic_count(model)
                analytic = MetricValue.available(breakdown["total"], rel, "model")
            except (KeyError, TypeError, ValueError) as exc:
                analytic = MetricValue.invalid(rel, "model", str(exc))
        info = ConfigInfo(path=path, raw=raw, model=model,
                          declared_params=declared, analytic_params=analytic)
        key = _normalize(path.stem)
        if key.startswith("model") and key != "model":
            key = key[len("model"):]
        meta.configs_by_file[path.name] = info
        meta.configs_by_key[key] = info
        if declared.usable and analytic.usable and declared.value != analytic.value:
            issues.append(_issue(
                "PARAM_COUNT_MISMATCH", SEVERITY_ERROR,
                f"{rel}: declared {declared.value:.0f}, analytic {analytic.value:.0f}", rel,
            ))
    return meta


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def infer_seed(run_id: str, explicit: list[tuple[Any, str, str]]) -> SeedInfo:
    """Infer seed metadata from explicit fields or the ``_sN`` run-id suffix.

    A run without either is recorded as MISSING. It is never assumed to be
    seed zero: the repository documents that early runs did not log a seed.
    """
    for value, source_file, source_key in explicit:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return SeedInfo(value=value, status=SeedStatus.EXPLICIT,
                        label=f"s{value}", source_file=source_file,
                        detail=f"explicit field {source_key}")
    match = re.search(r"(?:^|_)s(\d+)$", run_id)
    if match:
        value = int(match.group(1))
        return SeedInfo(
            value=value, status=SeedStatus.INFERRED, label=f"s{value}",
            source_file=None,
            detail="run id suffix '_sN'; docs/METHODS.md records --seed N for these runs",
        )
    return SeedInfo(
        value=None, status=SeedStatus.MISSING, label="unlabelled",
        detail="no seed field in result/metadata; absence is not seed 0",
    )


def infer_corpus(run_id: str) -> LabelInfo:
    """Infer the corpus regime from the run-id naming convention only."""
    if re.search(r"(?:^|_)v2(?:_|$)", run_id):
        return LabelInfo("v2", ValueStatus.AVAILABLE,
                         source="inferred from run id '_v2' marker")
    if "ctx" in run_id.lower():
        return LabelInfo(None, ValueStatus.MISSING,
                         detail="continuation run; corpus is not encoded in the run id")
    return LabelInfo("v1", ValueStatus.AVAILABLE,
                     source="inferred from run id convention: no '_v2' marker "
                            "(docs/RESULTS.md §6)")


def _config_candidates(run_id: str) -> list[str]:
    parts = run_id.split("_")
    variants: list[str] = []
    while parts and (parts[-1] == "v2" or re.fullmatch(r"s\d+", parts[-1])):
        parts.pop()
        variants.append("_".join(parts))
    base = "_".join(parts)
    if base == "baseline":
        variants.append("baseline")
    if not variants:
        variants = [run_id]
    variants.append(run_id)
    return list(dict.fromkeys(variants))


def _resolve_config(run_id: str, rescore: Optional[RescoreInfo],
                    meta: Metadata, issues: list[Issue],
                    rel: Optional[str]) -> Optional[ConfigInfo]:
    if rescore is not None and rescore.config:
        by_name = meta.configs_by_file.get(Path(rescore.config).name)
        if by_name is not None:
            return by_name
        issues.append(_issue("INVALID_CONFIG_REFERENCE", SEVERITY_WARNING,
                             f"rescore record references unknown config {rescore.config}",
                             rescore.source_file, run_id))
    for candidate in _config_candidates(run_id):
        key = "model" if candidate == "baseline" else _normalize(candidate)
        if key in meta.configs_by_key:
            return meta.configs_by_key[key]
    issues.append(_issue("MISSING_CONFIG", SEVERITY_WARNING,
                         f"no config could be resolved for run '{run_id}'", rel, run_id))
    return None


def _model_type(run_id: str, config: Optional[ConfigInfo],
                issues: list[Issue], rel: Optional[str]) -> ModelType:
    from_config: Optional[ModelType] = None
    if config is not None:
        n_unique = config.model.get("n_unique_layers")
        try:
            from_config = (ModelType.LOOP if int(n_unique) < int(config.model["n_layer"])
                           else ModelType.DENSE)
        except (TypeError, ValueError, KeyError):
            from_config = None
    from_name: Optional[ModelType] = None
    if re.search(r"(?:^|_)loop|loop\d|loop_", run_id):
        from_name = ModelType.LOOP
    elif "dense" in run_id or "baseline" in run_id or "d1024" in run_id:
        from_name = ModelType.DENSE
    if from_config is not None and from_name is not None and from_config != from_name:
        issues.append(_issue("MODEL_TYPE_CONFLICT", SEVERITY_WARNING,
                             f"run id suggests {from_name.value} but config says "
                             f"{from_config.value}", rel, run_id))
    return from_config or from_name or ModelType.UNKNOWN


def load_rescore_metadata(paths: AnalysisPaths, issues: list[Issue],
                          sources: dict[str, SourceInfo]) -> dict[str, RescoreInfo]:
    """Load clean-token re-score records from every rescore*.json file."""
    out: dict[str, RescoreInfo] = {}
    if not paths.metadata_dir.is_dir():
        return out
    for path in sorted(paths.metadata_dir.glob("rescore*.json")):
        rel = _register_source(path, paths, sources, "rescore")
        data = _read_json(path, paths, issues, sources, "rescore")
        if not isinstance(data, list):
            if data is not None:
                issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                     f"{rel} must contain a JSON array", rel))
            continue
        for index, raw in enumerate(data):
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                     f"{rel}[{index}] lacks a string name", rel))
                continue
            name = raw["name"]
            key = f"{rel}[{index}]"
            info = RescoreInfo(
                name=name,
                source_file=rel,
                config=raw.get("config") if isinstance(raw.get("config"), str) else None,
                checkpoint=raw.get("checkpoint") if isinstance(raw.get("checkpoint"), str) else None,
                precision=raw.get("precision") if isinstance(raw.get("precision"), str) else None,
                full_val_loss=_optional_number(raw, "full_val_loss", rel, f"{key}.full_val_loss"),
                clean_loss=_optional_number(raw, "clean_loss", rel, f"{key}.clean_loss"),
                copied_loss=_optional_number(raw, "copied_loss", rel, f"{key}.copied_loss"),
                reported_val_loss=_optional_number(raw, "reported_val_loss", rel,
                                                   f"{key}.reported_val_loss"),
                reproduces_reported=(raw.get("reproduces_reported")
                                     if isinstance(raw.get("reproduces_reported"), bool) else None),
                batches=_optional_number(raw, "batches", rel, f"{key}.batches"),
                clean_tokens=_optional_number(raw, "clean_tokens", rel, f"{key}.clean_tokens"),
                copied_tokens=_optional_number(raw, "copied_tokens", rel, f"{key}.copied_tokens"),
            )
            for metric in (info.full_val_loss, info.clean_loss, info.copied_loss,
                           info.reported_val_loss, info.batches, info.clean_tokens,
                           info.copied_tokens):
                if metric.status is ValueStatus.INVALID:
                    issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                                         f"{key}: {metric.source_key} is invalid", rel, name))
            if info.reproduces_reported is False:
                issues.append(_issue("REPRODUCTION_MISMATCH", SEVERITY_WARNING,
                                     f"{key}: reproduces_reported is false", rel, name))
            if (info.full_val_loss.usable and info.reported_val_loss.usable
                    and abs(info.full_val_loss.value - info.reported_val_loss.value) > 1e-3):
                issues.append(_issue("REPRODUCTION_MISMATCH", SEVERITY_WARNING,
                                     f"{key}: full_val_loss differs from reported_val_loss",
                                     rel, name))
            if name in out:
                issues.append(_issue("DUPLICATE_METADATA", SEVERITY_WARNING,
                                     f"duplicate rescore record for '{name}'; using {rel}",
                                     rel, name))
                continue
            out[name] = info
    return out


def load_mcq_metadata(paths: AnalysisPaths, issues: list[Issue],
                      sources: dict[str, SourceInfo]) -> dict[str, McqInfo]:
    """Load conditional-likelihood MCQ accuracies from mcq*.json files."""
    out: dict[str, McqInfo] = {}
    if not paths.metadata_dir.is_dir():
        return out
    for path in sorted(paths.metadata_dir.glob("mcq*.json")):
        rel = _register_source(path, paths, sources, "conditional_mcq")
        data = _read_json(path, paths, issues, sources, "conditional_mcq")
        if not isinstance(data, dict):
            if data is not None:
                issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                     f"{rel} must contain a JSON object", rel))
            continue
        if "rules_fixed_before_results" in data and data["rules_fixed_before_results"] is not True:
            issues.append(_issue("SCORING_RULE_CHANGED", SEVERITY_WARNING,
                                 f"{rel}: rules_fixed_before_results is not true", rel))
        runs = data.get("runs")
        if not isinstance(runs, dict):
            issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                 f"{rel} lacks a runs object", rel))
            continue
        for name in sorted(runs):
            entry = runs[name]
            if not isinstance(entry, dict):
                issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                     f"{rel}.runs.{name} is not an object", rel, name))
                continue
            accuracies: dict[str, MetricValue] = {}
            for rule in ("cf_norm", "cf_sum", "mcf"):
                rule_data = entry.get(rule)
                if isinstance(rule_data, dict):
                    accuracies[f"{rule}_accuracy"] = _optional_number(
                        rule_data, "accuracy", rel, f"runs.{name}.{rule}.accuracy"
                    )
                else:
                    accuracies[f"{rule}_accuracy"] = MetricValue.missing(
                        rel, f"runs.{name}.{rule}.accuracy"
                    )
            info = McqInfo(
                name=name,
                source_file=rel,
                accuracies=accuracies,
                checkpoint=(entry.get("checkpoint") if isinstance(entry.get("checkpoint"), str)
                            else None),
                truncated_items=_optional_number(entry, "truncated_items", rel,
                                                  f"runs.{name}.truncated_items"),
            )
            if name in out:
                issues.append(_issue("DUPLICATE_METADATA", SEVERITY_WARNING,
                                     f"duplicate MCQ record for '{name}'; using {rel}", rel, name))
                continue
            out[name] = info
            if info.truncated_items.usable and info.truncated_items.value > 0:
                issues.append(_issue("TRUNCATED_MCQ_ITEMS", SEVERITY_WARNING,
                                     f"{rel}: {info.truncated_items.value:.0f} truncated items",
                                     rel, name))
    return out


def load_curve_metadata(paths: AnalysisPaths, issues: list[Issue],
                        sources: dict[str, SourceInfo]) -> dict[str, CurveInfo]:
    """Load training-curve records, keeping only the final checkpoint per run."""
    out: dict[str, CurveInfo] = {}
    path = paths.metadata_dir / "curve_v2.json"
    if not path.is_file():
        return out
    rel = _register_source(path, paths, sources, "training_curve")
    data = _read_json(path, paths, issues, sources, "training_curve")
    if not isinstance(data, list):
        if data is not None:
            issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                 f"{rel} must contain a JSON array", rel))
        return out
    for index, raw in enumerate(data):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                 f"{rel}[{index}] lacks a string name", rel))
            continue
        name = raw["name"]
        step = raw.get("step")
        final = step == "final" or (isinstance(step, int) and name not in out)
        out.setdefault(name, CurveInfo(name=name, source_file=rel))
        info = out[name]
        if final and (isinstance(step, int) or isinstance(step, str)):
            info.final_step = str(step)
        for attr, key in (("full_val_loss", "full_val_loss"),
                          ("clean_loss", "clean_loss"),
                          ("reported_val_loss", "reported_val_loss")):
            if key in raw:
                setattr(info, attr, _optional_number(raw, key, rel, f"[{index}].{key}"))
        if isinstance(raw.get("reproduces_reported"), bool):
            info.reproduces_reported = raw["reproduces_reported"]
    return out


def load_cpu_metadata(paths: AnalysisPaths, issues: list[Issue],
                      sources: dict[str, SourceInfo]) -> Optional[CpuInfo]:
    """Load measured CPU timing, subsampling factors and bootstrap pairs."""
    path = paths.metadata_dir / "cpu_bootstrap.json"
    if not path.is_file():
        return None
    rel = _register_source(path, paths, sources, "cpu_bootstrap")
    data = _read_json(path, paths, issues, sources, "cpu_bootstrap")
    if not isinstance(data, dict):
        if data is not None:
            issues.append(_issue("MALFORMED_RECORD", SEVERITY_ERROR,
                                 f"{rel} must contain a JSON object", rel))
        return None
    info = CpuInfo(
        source_file=rel,
        started=data.get("started") if isinstance(data.get("started"), str) else None,
        finished=data.get("finished") if isinstance(data.get("finished"), str) else None,
    )
    seconds = data.get("seconds_per_batch")
    if isinstance(seconds, dict):
        for name in sorted(seconds):
            info.seconds_per_batch[name] = _optional_number(
                seconds, name, rel, f"seconds_per_batch.{name}"
            )
    factors = data.get("k_subsampling")
    if isinstance(factors, dict):
        for name in sorted(factors):
            info.k_subsampling[name] = _optional_number(
                factors, name, rel, f"k_subsampling.{name}"
            )
    models = data.get("models")
    if isinstance(models, dict):
        for name in sorted(models):
            entry = models[name]
            if isinstance(entry, dict):
                info.model_batches[name] = _optional_number(
                    entry, "batches", rel, f"models.{name}.batches"
                )
    pairs = data.get("pairs")
    if isinstance(pairs, dict):
        for name in sorted(pairs):
            entry = pairs[name]
            if not isinstance(entry, dict):
                continue
            if "compute" in name:
                category = "matched-compute"
            elif "matched" in name:
                category = "matched-parameter"
            else:
                category = "unspecified"
            seed_label = "s1" if "seed1" in name else "unlabelled"
            ci = entry.get("ci95")
            ci_text = ", ".join(f"{v:.6f}" for v in ci) if isinstance(ci, list) else None
            info.pairs.append(BootstrapPair(
                name=name,
                loop=str(entry.get("loop", "")),
                dense=str(entry.get("dense", "")),
                category=category,
                seed_label=seed_label,
                mean_difference=_optional_number(entry, "mean_difference", rel,
                                                 f"pairs.{name}.mean_difference"),
                ci95=ci_text,
                source_file=rel,
            ))
    return info


def _build_model_spec(run_id: str, config: Optional[ConfigInfo],
                      config_source: Optional[str], model_type: ModelType,
                      issues: list[Issue], rel: Optional[str],
                      repo_root: Path) -> ModelSpec:
    spec = ModelSpec(model_type=model_type, config_file=None, config_source=config_source)
    if config is None:
        return spec
    model = config.model
    spec.config_file = relative_path(config.path, repo_root)
    spec.params = config.declared_params
    spec.analytic_params = config.analytic_params
    for attr, key in (("d_model", "d_model"), ("n_layer", "n_layer"),
                      ("vocab_size", "vocab_size"), ("d_ff", "d_ff"),
                      ("max_seq_len", "max_seq_len")):
        value = model.get(key)
        setattr(spec, attr, int(value) if isinstance(value, int) and not isinstance(value, bool) else None)
    try:
        n_unique = int(model.get("n_unique_layers") or spec.n_layer)
        spec.n_unique = n_unique
        spec.n_loops = spec.n_layer // n_unique if spec.n_layer and n_unique else None
    except (TypeError, ValueError):
        issues.append(_issue("INVALID_CONFIG", SEVERITY_ERROR,
                             "n_unique_layers/n_layer are not usable integers", rel, run_id))
    lr = model.get("peak_lr", config.raw.get("lr_schedule", {}).get("peak_lr"))
    warmup = config.raw.get("lr_schedule", {}).get("warmup_ratio")
    spec.peak_lr = float(lr) if isinstance(lr, (int, float)) and not isinstance(lr, bool) else None
    spec.warmup_ratio = (float(warmup) if isinstance(warmup, (int, float))
                         and not isinstance(warmup, bool) else None)
    return spec


def _checkpoint_step(checkpoint: Optional[str]) -> Optional[int]:
    if not checkpoint:
        return None
    match = re.search(r"step-(\d+)-", checkpoint)
    return int(match.group(1)) if match else None


def _check_completeness(record: ExperimentRecord) -> None:
    issues = record.issues
    rel = record.result_file
    if record.result_file is None:
        issues.insert(0, _issue("MISSING_RESULT_FILE", SEVERITY_ERROR,
                                f"no {OFFICIAL_SUFFIX} file for run '{record.run_id}'"))
    if record.benchmark_file is None:
        issues.insert(0, _issue("MISSING_RESULT_FILE", SEVERITY_ERROR,
                                f"no {BENCHMARK_SUFFIX} file for run '{record.run_id}'"))
    for name in CORE_METRICS:
        metric = record.metric(name)
        if metric.status is ValueStatus.MISSING:
            issues.append(_issue("MISSING_METRIC", SEVERITY_ERROR,
                                 f"required metric '{name}' is missing",
                                 metric.source_file, record.run_id))
        elif metric.status is ValueStatus.INVALID:
            issues.append(_issue("INVALID_METRIC", SEVERITY_ERROR,
                                 f"required metric '{name}' is invalid",
                                 metric.source_file, record.run_id))
    if record.seed.status is SeedStatus.MISSING:
        issues.append(_issue("MISSING_SEED", SEVERITY_WARNING,
                             "random seed is not recorded for this run; "
                             "it is not assumed to be zero", rel, record.run_id))
    if record.model.config_file is None:
        issues.append(_issue("MISSING_CONFIG", SEVERITY_WARNING,
                             "model configuration unavailable: parameters and compute "
                             "cannot be reported", rel, record.run_id))
    elif record.model.params.usable and record.model.analytic_params.usable:
        if record.model.params.value != record.model.analytic_params.value:
            issues.append(_issue("PARAM_COUNT_MISMATCH", SEVERITY_ERROR,
                                 "declared and analytically computed parameter counts differ",
                                 record.model.config_file, record.run_id))
    if record.compute.seconds_per_batch.status is not ValueStatus.AVAILABLE:
        issues.append(_issue("MISSING_COMPUTE_TIMING", SEVERITY_INFO,
                             "no measured seconds/batch for this run", rel, record.run_id))
    if record.training.checkpoint is None:
        issues.append(_issue("MISSING_CHECKPOINT", SEVERITY_INFO,
                             "no checkpoint filename available for this run", rel, record.run_id))
    if any(issue.code in MALFORMED_CODES for issue in issues):
        record.status = RunStatus.INVALID
    elif any(issue.severity == SEVERITY_ERROR for issue in issues):
        record.status = RunStatus.INCOMPLETE
    elif any(issue.severity == SEVERITY_WARNING for issue in issues):
        record.status = RunStatus.COMPLETE_WITH_WARNINGS
    else:
        record.status = RunStatus.COMPLETE


def build_record(run_id: str, files: dict[str, Optional[Path]], paths: AnalysisPaths,
                 issues: list[Issue], sources: dict[str, SourceInfo],
                 meta: Metadata, official_metrics: dict[str, MetricValue],
                 benchmark_metrics: dict[str, MetricValue],
                 official_raw: Optional[dict] = None) -> ExperimentRecord:
    """Assemble one run from its parsed files and all joined metadata."""
    record = ExperimentRecord(run_id=run_id, issues=list(issues))
    if files["official"] is not None:
        record.result_file = relative_path(files["official"], paths.repo_root)
    if files["benchmark"] is not None:
        record.benchmark_file = relative_path(files["benchmark"], paths.repo_root)
    record.metrics.update(official_metrics)
    record.metrics.update({
        "benchmark_mcq_accuracy": benchmark_metrics["mcq_accuracy"],
        "benchmark_mcq_correct": benchmark_metrics["mcq_correct"],
    })

    rescore = meta.rescore.get(run_id)
    mcq = meta.mcq.get(run_id)
    curve = meta.curve.get(run_id)
    cpu = meta.cpu

    if rescore is not None:
        record.metrics.update({
            "full_val_loss": rescore.full_val_loss,
            "clean_loss": rescore.clean_loss,
            "copied_loss": rescore.copied_loss,
        })
        record.training.checkpoint = rescore.checkpoint
        record.training.precision = rescore.precision
        record.training.batches = rescore.batches
        record.training.clean_tokens = rescore.clean_tokens
        record.training.copied_tokens = rescore.copied_tokens
        if rescore.full_val_loss.usable and record.metric("val_loss").usable:
            assert rescore.full_val_loss.value is not None
            if abs(rescore.full_val_loss.value - record.metric("val_loss").value) > 1e-3:
                record.issues.append(_issue(
                    "REPORTED_LOSS_MISMATCH", SEVERITY_WARNING,
                    "re-scored full_val_loss differs from the official val_loss",
                    rescore.source_file, run_id,
                ))
    if mcq is not None:
        record.metrics.update(mcq.accuracies)
        if record.training.checkpoint is None:
            record.training.checkpoint = mcq.checkpoint
        record.training.truncated_items = mcq.truncated_items
    if curve is not None and curve.final_step is not None:
        record.training.final_step = curve.final_step
        if record.training.checkpoint_step is None:
            try:
                record.training.checkpoint_step = int(curve.final_step)
            except ValueError:
                pass

    explicit: list[tuple[Any, str, str]] = []
    if isinstance(official_raw, dict):
        for key in ("seed", "random_seed"):
            if key in official_raw:
                explicit.append((official_raw[key], record.result_file or "", key))
    record.seed = infer_seed(run_id, explicit)
    record.corpus = infer_corpus(run_id)

    config = _resolve_config(run_id, rescore, meta, record.issues,
                             record.result_file)
    config_source = None
    if config is not None:
        config_source = ("rescore record" if rescore is not None and rescore.config
                         else "run-id/config-stem match")
    record.model = _build_model_spec(run_id, config, config_source,
                                     _model_type(run_id, config, record.issues,
                                                 record.result_file),
                                     record.issues, record.result_file,
                                     paths.repo_root)

    cost = derive_compute_cost(config.model if config else None,
                               int(config.declared_params.value)
                               if config and config.declared_params.usable else None)
    record.compute = ComputeSpec(
        layer_applications=cost["layer_applications"],
        unique_blocks=cost["unique_blocks"],
        loops=cost["loops"],
        forward_gflops=cost["forward_gflops"],
        sources=(config_source or ""),
    )
    compute_sources: list[str] = []
    if config is not None:
        compute_sources.append(relative_path(config.path, paths.repo_root) or config.path.name)
    if cpu is not None:
        if run_id in cpu.seconds_per_batch:
            record.compute.seconds_per_batch = cpu.seconds_per_batch[run_id]
            compute_sources.append(cpu.source_file)
        if run_id in cpu.k_subsampling:
            record.compute.subsampling_k = cpu.k_subsampling[run_id]
            compute_sources.append(cpu.source_file)
        if run_id in cpu.model_batches:
            record.training.batches = cpu.model_batches[run_id]
            compute_sources.append(cpu.source_file)
        record.training.started = cpu.started
        record.training.finished = cpu.finished
    record.compute.sources = tuple(dict.fromkeys(compute_sources))

    if record.training.checkpoint_step is None:
        record.training.checkpoint_step = _checkpoint_step(record.training.checkpoint)

    source_candidates = [record.result_file, record.benchmark_file,
                         record.model.config_file]
    if rescore is not None:
        source_candidates.append(rescore.source_file)
    if mcq is not None:
        source_candidates.append(mcq.source_file)
    if curve is not None:
        source_candidates.append(curve.source_file)
    source_candidates.extend(record.compute.sources)
    record.sources = tuple(sorted({s for s in source_candidates if s}))
    _check_completeness(record)
    return record


def _verify_manifest(paths: AnalysisPaths, issues: list[Issue],
                     sources: dict[str, SourceInfo]) -> None:
    path = paths.metadata_dir / "MANIFEST.sha256"
    if not path.is_file():
        return
    rel = _register_source(path, paths, sources, "sha256_manifest")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        issues.append(_issue("MALFORMED_MANIFEST", SEVERITY_WARNING,
                             f"could not read {rel}: {exc}", rel))
        return
    listed: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^([0-9a-f]{64})\s+(.+)$", line.strip())
        if match:
            listed[match.group(2)] = match.group(1)
    for source_rel, info in sorted(sources.items()):
        if source_rel in listed and info.sha256 and listed[source_rel] != info.sha256:
            issues.append(_issue("MANIFEST_MISMATCH", SEVERITY_ERROR,
                                 f"{source_rel} does not match MANIFEST.sha256",
                                 rel))


def build_analysis(paths: Optional[AnalysisPaths] = None) -> AnalysisResult:
    """Discover and validate every available experiment result."""
    paths = paths or resolve_paths()
    issues: list[Issue] = []
    sources: dict[str, SourceInfo] = {}
    meta = load_configs(paths, issues, sources)
    meta.rescore = load_rescore_metadata(paths, issues, sources)
    meta.mcq = load_mcq_metadata(paths, issues, sources)
    meta.curve = load_curve_metadata(paths, issues, sources)
    meta.cpu = load_cpu_metadata(paths, issues, sources)

    files = discover_result_files(paths, issues)
    records: list[ExperimentRecord] = []
    for run_id in sorted(files):
        run_issues: list[Issue] = []
        official_raw: list = []
        official = parse_official_metrics(
            files[run_id]["official"] or paths.experiments_dir / f"{run_id}{OFFICIAL_SUFFIX}",
            paths, run_issues, sources, run_id, benchmark=False, raw_out=official_raw,
        ) if files[run_id]["official"] is not None else {
            name: MetricValue.missing(detail=f"no {OFFICIAL_SUFFIX} file")
            for name in CORE_METRICS
        }
        benchmark = parse_official_metrics(
            files[run_id]["benchmark"] or paths.experiments_dir / f"{run_id}{BENCHMARK_SUFFIX}",
            paths, run_issues, sources, run_id, benchmark=True,
        ) if files[run_id]["benchmark"] is not None else {
            "mcq_accuracy": MetricValue.missing(detail=f"no {BENCHMARK_SUFFIX} file"),
            "mcq_correct": MetricValue.missing(detail=f"no {BENCHMARK_SUFFIX} file"),
        }
        records.append(build_record(
            run_id, files[run_id], paths, run_issues, sources, meta, official, benchmark,
            official_raw[0] if official_raw else None,
        ))
    if not files:
        issues.append(_issue(
            "NO_RESULT_FILES", SEVERITY_WARNING,
            f"no '{OFFICIAL_SUFFIX}' or '{BENCHMARK_SUFFIX}' files found under "
            f"{relative_path(paths.experiments_dir, paths.repo_root)}",
        ))
    _verify_manifest(paths, issues, sources)
    return AnalysisResult(
        paths=paths, records=records, global_issues=issues, sources=sources,
        bootstrap_pairs=list(meta.cpu.pairs) if meta.cpu is not None else [],
    )
