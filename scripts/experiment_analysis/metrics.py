"""Metric formulas used by the experiment analysis toolkit.

All losses are in nats per token. The sign convention is deliberately fixed:
``loop - dense`` is positive when the loop model has the higher (worse) loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .core import MetricValue

PERCENT_ZERO_DENOMINATOR = (
    "undefined: the dense (reference) loss is exactly zero; "
    "a relative difference cannot be computed"
)


def absolute_loss_difference(loop: MetricValue, dense: MetricValue) -> MetricValue:
    """Return ``loop_loss - dense_loss`` in nats (positive means loop is worse).

    Missing, invalid or unavailable inputs yield an ``UNAVAILABLE`` result with
    an explanatory detail. No input is ever coerced to zero.
    """
    if not loop.usable or not dense.usable:
        reasons = []
        if not loop.usable:
            reasons.append(f"loop loss {loop.status.value}")
        if not dense.usable:
            reasons.append(f"dense loss {dense.status.value}")
        return MetricValue.unavailable(detail="; ".join(reasons))
    return MetricValue.available(
        loop.value - dense.value,
        source_file=loop.source_file,
        source_key=f"({loop.source_key} - {dense.source_key})",
    )


def percentage_loss_difference(loop: MetricValue, dense: MetricValue) -> MetricValue:
    """Return ``100 * (loop - dense) / abs(dense)`` in percent.

    The reference is the dense loss. Its magnitude is used in the denominator
    so the sign of the difference is preserved for any finite non-zero dense
    loss. A dense loss of exactly zero makes the ratio mathematically
    undefined: the result is ``UNAVAILABLE``, never ``0``, ``inf`` or ``NaN``.
    A loop loss of exactly zero is a valid input and normally yields ``-100``.
    """
    difference = absolute_loss_difference(loop, dense)
    if not difference.usable:
        return difference
    assert dense.value is not None
    denominator = abs(dense.value)
    if denominator == 0.0:
        return MetricValue.unavailable(detail=PERCENT_ZERO_DENOMINATOR)
    return MetricValue.available(
        100.0 * difference.value / denominator,
        source_file=loop.source_file,
        source_key=f"100 * ({loop.source_key} - {dense.source_key}) / abs({dense.source_key})",
    )


def perplexity_consistent(loss: MetricValue, perplexity: MetricValue,
                          rtol: float = 1e-3) -> bool:
    """Check ``perplexity == exp(loss)`` within a relative tolerance."""
    if not loss.usable or not perplexity.usable:
        return False
    assert loss.value is not None and perplexity.value is not None
    if perplexity.value < 0:
        return False
    if loss.value > 709.0:
        return False
    expected = math.exp(loss.value)
    if expected == 0.0:
        return perplexity.value == 0.0
    return abs(perplexity.value - expected) / abs(expected) <= rtol


@dataclass(frozen=True)
class SeedCoverage:
    """Counts of recorded and unrecorded seeds across runs."""

    total: int
    explicit: int
    inferred: int
    missing: int
    known_ratio: Optional[float]

    @property
    def known(self) -> int:
        return self.explicit + self.inferred


def seed_coverage(records: list) -> SeedCoverage:
    """Summarise seed metadata without treating unknown seeds as any value."""
    explicit = sum(1 for r in records if r.seed.status.value == "explicit")
    inferred = sum(1 for r in records if r.seed.status.value == "inferred")
    missing = sum(1 for r in records if r.seed.status.value == "missing")
    total = len(records)
    return SeedCoverage(
        total=total,
        explicit=explicit,
        inferred=inferred,
        missing=missing,
        known_ratio=(explicit + inferred) / total if total else None,
    )


def derive_compute_cost(model_cfg: Optional[dict],
                        declared_params: Optional[int]) -> dict[str, Any]:
    """Derive analytic per-token compute from a raw ``model:`` config dict.

    This adapts the formula used by ``paper/make_tables.py``: a forward pass
    applies two operations per stored weight, once per layer application, plus
    the tied output projection. Attention score products are excluded because
    they are identical across models of a given width and sequence length.
    The result is an analytic estimate, not a hardware measurement.
    """
    result: dict[str, Any] = {
        "layer_applications": None,
        "unique_blocks": None,
        "loops": None,
        "forward_gflops": MetricValue.unavailable(detail="model config unavailable"),
    }
    if not isinstance(model_cfg, dict):
        return result
    required = ("n_layer", "d_model", "vocab_size")
    if any(key not in model_cfg for key in required):
        result["forward_gflops"] = MetricValue.unavailable(
            detail="model config lacks n_layer, d_model or vocab_size"
        )
        return result
    try:
        n_layer = int(model_cfg["n_layer"])
        d_model = int(model_cfg["d_model"])
        vocab = int(model_cfg["vocab_size"])
        n_unique = int(model_cfg.get("n_unique_layers") or n_layer)
        declared = int(declared_params) if declared_params is not None else None
    except (TypeError, ValueError):
        result["forward_gflops"] = MetricValue.invalid(detail="non-integer config values")
        return result
    if n_layer <= 0 or d_model <= 0 or vocab <= 0 or n_unique <= 0:
        result["forward_gflops"] = MetricValue.invalid(detail="non-positive config values")
        return result
    result["layer_applications"] = n_layer
    result["unique_blocks"] = n_unique
    result["loops"] = n_layer // n_unique
    if n_layer % n_unique != 0:
        result["forward_gflops"] = MetricValue.invalid(
            detail="n_layer is not divisible by n_unique_layers"
        )
        return result
    if declared is None or declared <= 0:
        result["forward_gflops"] = MetricValue.unavailable(
            detail="declared parameter count unavailable"
        )
        return result
    depth_embedding = n_layer * d_model if n_unique < n_layer else 0
    fixed = vocab * d_model + d_model + depth_embedding
    block_params = (declared - fixed) / n_unique
    if block_params <= 0:
        result["forward_gflops"] = MetricValue.invalid(
            detail="declared parameter count is inconsistent with the config"
        )
        return result
    gflops = 2.0 * (n_layer * block_params + vocab * d_model) / 1e9
    result["forward_gflops"] = MetricValue.available(gflops)
    return result
