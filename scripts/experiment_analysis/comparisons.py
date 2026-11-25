"""Dense/loop pairing under matched-parameter and matched-compute conditions.

The two categories are classified independently and are never combined. A pair
is only formed when the architecture, corpus regime and seed label agree. Any
remaining uncertainty is attached to the comparison as a warning or an
informational issue rather than being resolved by an assumption.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .core import (
    SEVERITY_INFO,
    SEVERITY_WARNING,
    AnalysisResult,
    Comparison,
    ComparisonCategory,
    ComparisonResult,
    ExperimentRecord,
    Issue,
    MetricValue,
    ModelType,
    RunStatus,
    SkippedPair,
)
from .metrics import absolute_loss_difference, percentage_loss_difference

DEFAULT_METRICS = ("val_loss", "clean_loss")


def _issue(code: str, severity: str, message: str,
           source_file: Optional[str] = None, run_id: Optional[str] = None) -> Issue:
    return Issue(code=code, severity=severity, message=message,
                 source_file=source_file, run_id=run_id)


def classify_pair(loop: ExperimentRecord,
                  dense: ExperimentRecord) -> tuple[Optional[ComparisonCategory], str, Optional[str]]:
    """Classify one candidate pair without ever mixing the two categories.

    Returns ``(category, match_basis, skip_reason)``. Exactly one of category
    and skip_reason is ``None``.
    """
    lm, dm = loop.model, dense.model
    if lm.model_type is not ModelType.LOOP or dm.model_type is not ModelType.DENSE:
        return None, "", "requires one loop model and one dense model"
    values = (lm.n_layer, lm.n_unique, lm.n_loops, lm.d_model,
              dm.n_layer, dm.n_unique, dm.d_model)
    if any(value is None for value in values):
        return None, "", "architecture metadata unavailable"
    assert lm.n_layer and lm.n_unique and lm.n_loops and lm.d_model
    assert dm.n_layer and dm.n_unique and dm.d_model
    if lm.n_loops < 2:
        return None, "", "loop model has only one application per unique block"
    if (lm.d_model != dm.d_model or lm.vocab_size != dm.vocab_size
            or lm.d_ff != dm.d_ff or lm.max_seq_len != dm.max_seq_len):
        return None, "", "models differ in width, vocabulary, FFN size or context"
    if loop.corpus.value is None or dense.corpus.value is None:
        return None, "", "corpus regime unavailable"
    if loop.corpus.value != dense.corpus.value:
        return None, "", f"different corpora ({loop.corpus.value} vs {dense.corpus.value})"
    if loop.seed.label != dense.seed.label:
        return None, "", f"different seed labels ({loop.seed.label} vs {dense.seed.label})"
    if loop.seed.known and dense.seed.known and loop.seed.value != dense.seed.value:
        return None, "", "different recorded seed values"
    if loop.status is RunStatus.INVALID or dense.status is RunStatus.INVALID:
        return None, "", "a record is malformed or invalid"

    if lm.n_unique == dm.n_layer:
        if not lm.params.usable or not dm.params.usable:
            return None, "", "parameter counts unavailable for matched-parameter check"
        assert lm.params.value is not None and dm.params.value is not None
        actual_gap = lm.params.value - dm.params.value
        expected_gap = lm.n_layer * lm.d_model
        if actual_gap != expected_gap:
            return None, "", (
                f"parameter gap {actual_gap:.0f} is not the depth embedding "
                f"{expected_gap:.0f}"
            )
        basis = (
            f"loop K={lm.n_unique} x T={lm.n_loops}; dense L={dm.n_layer}; "
            f"parameter gap {expected_gap:,} = n_layer x d_model"
        )
        return ComparisonCategory.MATCHED_PARAMETER, basis, None

    if lm.n_layer == dm.n_layer:
        basis = f"both apply {lm.n_layer} layers per token"
        return ComparisonCategory.MATCHED_COMPUTE, basis, None

    return None, "", "models are neither parameter-matched nor compute-matched"


def _pair_issues(loop: ExperimentRecord, dense: ExperimentRecord) -> list[Issue]:
    issues: list[Issue] = []
    for record in (loop, dense):
        if record.seed.status.value == "missing":
            issues.append(_issue(
                "SEED_METADATA_MISSING", SEVERITY_WARNING,
                f"no recorded seed for '{record.run_id}'; seed labels match by run id only",
                record.result_file, record.run_id,
            ))
        if record.corpus.value is not None and record.corpus.source:
            issues.append(_issue(
                "CORPUS_INFERRED", SEVERITY_INFO,
                f"corpus '{record.corpus.value}' {record.corpus.source}",
                record.result_file, record.run_id,
            ))
        if record.status is RunStatus.INCOMPLETE:
            issues.append(_issue(
                "INCOMPLETE_RECORD", SEVERITY_WARNING,
                f"run '{record.run_id}' has incomplete result metadata",
                record.result_file, record.run_id,
            ))
    if (loop.model.peak_lr is not None and dense.model.peak_lr is not None
            and loop.model.peak_lr != dense.model.peak_lr):
        issues.append(_issue(
            "HYPERPARAMETER_MISMATCH", SEVERITY_WARNING,
            f"peak LR {loop.model.peak_lr:g} vs {dense.model.peak_lr:g}",
        ))
    if (loop.model.warmup_ratio is not None and dense.model.warmup_ratio is not None
            and loop.model.warmup_ratio != dense.model.warmup_ratio):
        issues.append(_issue(
            "HYPERPARAMETER_MISMATCH", SEVERITY_WARNING,
            f"warmup ratio {loop.model.warmup_ratio:g} vs {dense.model.warmup_ratio:g}",
        ))
    return issues


def build_comparisons(result: AnalysisResult, *,
                      category: Optional[ComparisonCategory] = None,
                      metrics: Optional[Iterable[str]] = None) -> ComparisonResult:
    """Build all eligible dense/loop comparisons from a validated analysis."""
    metric_names = tuple(metrics) if metrics is not None else DEFAULT_METRICS
    loops = [r for r in result.records if r.model.model_type is ModelType.LOOP]
    denses = [r for r in result.records if r.model.model_type is ModelType.DENSE]
    output = ComparisonResult()
    for loop in loops:
        for dense in denses:
            pair_category, basis, skip_reason = classify_pair(loop, dense)
            if skip_reason is not None or pair_category is None:
                if skip_reason and any(token in skip_reason for token in (
                        "seed", "corpora", "corpus", "unavailable", "malformed")):
                    output.skipped.append(SkippedPair(
                        loop_run=loop.run_id, dense_run=dense.run_id,
                        reason=skip_reason, category=None,
                    ))
                continue
            if category is not None and pair_category is not category:
                continue
            pair_issues = _pair_issues(loop, dense)
            bootstrap = next(
                (p for p in result.bootstrap_pairs
                 if p.loop == loop.run_id and p.dense == dense.run_id
                 and p.category == pair_category.value
                 and p.seed_label == loop.seed.label),
                None,
            )
            for metric in metric_names:
                loop_loss = loop.metric(metric)
                dense_loss = dense.metric(metric)
                if not loop_loss.usable or not dense_loss.usable:
                    reasons = []
                    if not loop_loss.usable:
                        reasons.append(f"loop {metric} {loop_loss.status.value}")
                    if not dense_loss.usable:
                        reasons.append(f"dense {metric} {dense_loss.status.value}")
                    output.skipped.append(SkippedPair(
                        loop_run=loop.run_id, dense_run=dense.run_id,
                        reason=f"{metric}: {'; '.join(reasons)}",
                        category=pair_category.value,
                    ))
                    continue
                difference = absolute_loss_difference(loop_loss, dense_loss)
                percentage = percentage_loss_difference(loop_loss, dense_loss)
                issues = list(pair_issues)
                if not percentage.usable:
                    issues.append(_issue(
                        "PERCENT_DIFFERENCE_UNDEFINED", SEVERITY_WARNING,
                        percentage.detail or "percentage difference is undefined",
                    ))
                sources = sorted({s for s in loop.sources + dense.sources if s})
                output.comparisons.append(Comparison(
                    category=pair_category,
                    metric=metric,
                    loop_run=loop.run_id,
                    dense_run=dense.run_id,
                    loop_loss=loop_loss,
                    dense_loss=dense_loss,
                    loss_difference=difference,
                    percent_difference=percentage,
                    seed_label=loop.seed.label,
                    corpus=loop.corpus.value,
                    match_basis=basis,
                    status=("warning" if any(i.severity != SEVERITY_INFO for i in issues)
                            else "ok"),
                    issues=issues,
                    bootstrap_mean_difference=(bootstrap.mean_difference
                                               if bootstrap is not None
                                               else MetricValue()),
                    bootstrap_ci95=bootstrap.ci95 if bootstrap is not None else None,
                    source_files=tuple(sources),
                ))
    output.comparisons.sort(key=lambda c: (c.category.value, c.corpus or "",
                                           c.seed_label, c.metric, c.loop_run, c.dense_run))
    output.skipped.sort(key=lambda s: (s.category or "", s.loop_run, s.dense_run, s.reason))
    return output
