"""Tests for the experiment analysis metric formulas (no model training)."""

from __future__ import annotations

import math

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experiment_analysis.core import MetricValue, SeedInfo, SeedStatus  # noqa: E402
from experiment_analysis.metrics import (  # noqa: E402
    PERCENT_ZERO_DENOMINATOR,
    SeedCoverage,
    absolute_loss_difference,
    derive_compute_cost,
    percentage_loss_difference,
    perplexity_consistent,
    seed_coverage,
)


def mv(value, key="val_loss"):
    return MetricValue.available(value, "result.json", key)


def test_absolute_difference_sign_convention():
    loop, dense = mv(4.6), mv(2.8)
    result = absolute_loss_difference(loop, dense)
    assert result.usable
    assert result.value == pytest.approx(1.8)
    assert absolute_loss_difference(dense, loop).value == pytest.approx(-1.8)


def test_absolute_difference_zero_is_available():
    result = absolute_loss_difference(mv(2.0), mv(2.0))
    assert result.status.value == "available"
    assert result.value == 0.0
    assert result.is_zero


def test_percentage_formula_uses_dense_loss_as_reference():
    result = percentage_loss_difference(mv(4.0), mv(2.0))
    assert result.usable
    assert result.value == pytest.approx(100.0)
    assert result.source_key == "100 * (val_loss - val_loss) / abs(val_loss)"


def test_percentage_negative_dense_uses_absolute_denominator():
    result = percentage_loss_difference(mv(1.0), mv(-2.0))
    assert result.value == pytest.approx(150.0)


def test_percentage_zero_loop_loss_is_valid_minus_one_hundred():
    result = percentage_loss_difference(mv(0.0), mv(2.0))
    assert result.status.value == "available"
    assert result.value == pytest.approx(-100.0)


def test_percentage_zero_dense_loss_is_undefined_not_zero():
    result = percentage_loss_difference(mv(0.0), mv(0.0))
    assert not result.usable
    assert result.value is None
    assert result.status.value == "unavailable"
    assert result.detail == PERCENT_ZERO_DENOMINATOR
    assert absolute_loss_difference(mv(0.0), mv(0.0)).value == 0.0


@pytest.mark.parametrize("loop,dense", [
    (MetricValue.missing(), mv(2.0)),
    (mv(2.0), MetricValue.missing()),
    (MetricValue.invalid(detail="NaN"), mv(2.0)),
    (mv(2.0), MetricValue.unavailable(detail="schema")),
])
def test_missing_invalid_unavailable_inputs_are_not_zero(loop, dense):
    assert absolute_loss_difference(loop, dense).value is None
    percent = percentage_loss_difference(loop, dense)
    assert percent.value is None
    assert percent.status.value == "unavailable"
    assert not percent.is_zero


def test_zero_metric_is_distinct_from_missing_and_invalid():
    zero = mv(0.0)
    assert zero.status.value == "available"
    assert zero.is_zero
    assert not MetricValue.missing().is_zero
    assert not MetricValue.invalid().is_zero
    assert not MetricValue.unavailable().is_zero
    assert zero.display() == "0.000000"
    assert MetricValue.missing().display() == "MISSING"
    assert MetricValue.invalid().display() == "INVALID"
    assert MetricValue.unavailable().display() == "UNAVAILABLE"


def test_metric_value_rejects_nan_and_inconsistent_status():
    assert MetricValue.available(float("nan")).status.value == "invalid"
    assert MetricValue.available("2.0").status.value == "invalid"
    with pytest.raises(ValueError):
        MetricValue(value=None, status=MetricValue.available(1.0).status)
    with pytest.raises(ValueError):
        MetricValue(value=1.0, status=MetricValue.missing().status)


def test_perplexity_consistency():
    loss, ppl = mv(math.log(10.0)), mv(10.0)
    assert perplexity_consistent(loss, ppl)
    assert not perplexity_consistent(loss, mv(11.0))
    assert not perplexity_consistent(MetricValue.missing(), ppl)
    assert not perplexity_consistent(mv(800.0), mv(float("inf")))


def test_seed_coverage_empty_and_counts():
    empty = seed_coverage([])
    assert empty == SeedCoverage(0, 0, 0, 0, None)
    records = [
        type("R", (), {"seed": SeedInfo(1, SeedStatus.INFERRED)})(),
        type("R", (), {"seed": SeedInfo(None, SeedStatus.MISSING)})(),
        type("R", (), {"seed": SeedInfo(7, SeedStatus.EXPLICIT)})(),
    ]
    coverage = seed_coverage(records)
    assert coverage.total == 3
    assert coverage.known == 2
    assert coverage.missing == 1
    assert coverage.known_ratio == pytest.approx(2 / 3)


def test_derive_compute_cost_matches_repo_formula():
    cfg = {"vocab_size": 15000, "d_model": 768, "n_layer": 12,
           "n_unique_layers": 6, "d_ff": 2048}
    cost = derive_compute_cost(cfg, 54_006_528)
    assert cost["layer_applications"] == 12
    assert cost["unique_blocks"] == 6
    assert cost["loops"] == 2
    assert cost["forward_gflops"].usable
    assert cost["forward_gflops"].value == pytest.approx(0.192946176, rel=1e-9)


def test_derive_compute_cost_missing_metadata_is_unavailable():
    cost = derive_compute_cost(None, None)
    assert cost["layer_applications"] is None
    assert cost["forward_gflops"].status.value == "unavailable"
    bad = derive_compute_cost({"vocab_size": 10, "d_model": 4, "n_layer": 3,
                               "n_unique_layers": 2}, 100)
    assert bad["forward_gflops"].status.value == "invalid"
