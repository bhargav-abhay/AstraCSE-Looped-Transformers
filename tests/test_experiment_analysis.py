"""Tests for experiment discovery, validation, matching and report generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_analysis.cli import analysis_main, compare_main, report_main  # noqa: E402
from experiment_analysis.comparisons import build_comparisons, classify_pair  # noqa: E402
from experiment_analysis.core import ComparisonCategory, RunStatus, ValueStatus  # noqa: E402
from experiment_analysis.discovery import (  # noqa: E402
    build_analysis,
    infer_corpus,
    infer_seed,
    resolve_paths,
)
from experiment_analysis.reports import write_report_bundle  # noqa: E402
import verify_param_count as vpc  # noqa: E402


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def official_payload(loss: float = 2.5, correct: int = 5, total: int = 10,
                     **extra) -> dict:
    payload = {
        "val_loss": loss,
        "perplexity": math.exp(loss) if isinstance(loss, (int, float)) else 1.0,
        "mcq_accuracy": correct / total,
        "mcq_correct": correct,
        "mcq_total": total,
        "per_subject": {
            "subject_a": {"accuracy": correct / total, "correct": correct, "total": total}
        },
    }
    payload.update(extra)
    return payload


def benchmark_payload(correct: int = 3) -> dict:
    return {
        "mcq_accuracy": correct / 10,
        "mcq_correct": correct,
        "mcq_total": 10,
        "per_subject": {
            "subject_a": {"accuracy": correct / 10, "correct": correct, "total": 10}
        },
    }


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.results = root / "results"
        self.configs = root / "configs"
        self.metadata = root / "metadata"
        for directory in (self.results, self.configs, self.metadata):
            directory.mkdir(parents=True, exist_ok=True)
        self.rescore: list[dict] = []
        self.paths = resolve_paths(
            repo_root=root, experiments_dir=self.results,
            configs_dir=self.configs, metadata_dir=self.metadata,
        )

    def add_config(self, name: str, n_layer: int, n_unique: int | None = None,
                   *, d_ff: int = 8, peak_lr: float = 0.001,
                   warmup: float = 0.02) -> None:
        model = {
            "vocab_size": 10, "d_model": 4, "n_layer": n_layer,
            "n_head": 2, "d_head": 2, "d_ff": d_ff, "max_seq_len": 16,
            "expected_param_count": 0,
        }
        if n_unique is not None:
            model["n_unique_layers"] = n_unique
        model["expected_param_count"] = vpc.count_params(model)["total"]
        payload = {"model": model, "lr_schedule": {"peak_lr": peak_lr,
                                                   "warmup_ratio": warmup}}
        (self.configs / name).write_text(yaml.safe_dump(payload), encoding="utf-8")

    def add_default_configs(self) -> None:
        self.add_config("model_loop_x2.yaml", 4, 2)
        self.add_config("model_dense_2l.yaml", 2)
        self.add_config("model_dense_4l.yaml", 4)

    def add_run(self, run: str, *, mmlu: dict | None = None, benchmark: bool = True,
                loss: float = 2.5, clean_loss: float | None = None,
                seed: int | None = None, config: str | None = None) -> None:
        payload = official_payload(loss=loss) if mmlu is None else mmlu
        if seed is not None:
            payload["seed"] = seed
        dump_json(self.results / f"{run}.mmlu_cs.json", payload)
        if benchmark:
            dump_json(self.results / f"{run}.benchmark.json", benchmark_payload())
        if clean_loss is not None:
            self.rescore.append({
                "name": run,
                "config": config or self._guess_config(run),
                "checkpoint": "step-0001000-final.pt",
                "precision": "fp32",
                "full_val_loss": payload.get("val_loss"),
                "clean_loss": clean_loss,
                "copied_loss": 0.5,
                "clean_tokens": 100,
                "copied_tokens": 10,
                "batches": 738,
                "reported_val_loss": payload.get("val_loss"),
                "reproduces_reported": True,
            })

    def _guess_config(self, run: str) -> str:
        base = run.split("_s1")[0].replace("_v2", "")
        if base.startswith("loop"):
            return "configs/model_loop_x2.yaml"
        if base.endswith("2l"):
            return "configs/model_dense_2l.yaml"
        return "configs/model_dense_4l.yaml"

    def add_bootstrap(self) -> None:
        dump_json(self.metadata / "cpu_bootstrap.json", {
            "started": "2026-11-13 16:46",
            "finished": "2026-11-13 17:50",
            "seconds_per_batch": {"loop_x2": 4.9, "dense_2l": 2.7},
            "k_subsampling": {"loop_x2": 8, "dense_2l": 1},
            "models": {"loop_x2": {"batches": 738}},
            "pairs": {
                "v1_matched": {"loop": "loop_x2", "dense": "dense_2l",
                               "mean_difference": 1.5, "ci95": [1.4, 1.6],
                               "subsampled": True, "batches": 93,
                               "unit": "batch"},
            },
        })

    def analysis(self):
        if self.rescore:
            dump_json(self.metadata / "rescore_fixture.json", self.rescore)
        return build_analysis(self.paths)


@pytest.fixture()
def project(tmp_path: Path) -> Project:
    return Project(tmp_path)


def test_discovery_extracts_metrics_config_and_compute(project):
    project.add_default_configs()
    project.add_run("loop_x2", loss=4.0, clean_loss=4.1, seed=1)
    result = project.analysis()
    assert len(result.records) == 1
    record = result.record("loop_x2")
    assert record.metric("val_loss").value == pytest.approx(4.0)
    assert record.metric("clean_loss").value == pytest.approx(4.1)
    assert record.metric("perplexity").value == pytest.approx(math.exp(4.0))
    assert record.metric("perplexity").source_file.endswith("loop_x2.mmlu_cs.json")
    assert record.metric("benchmark_mcq_accuracy").value == pytest.approx(0.3)
    assert record.metric("val_loss").status is ValueStatus.AVAILABLE
    assert record.model.config_file == "configs/model_loop_x2.yaml"
    assert record.model.model_type.value == "loop"
    assert record.model.params.value == vpc.count_params({
        "vocab_size": 10, "d_model": 4, "n_layer": 4, "n_unique_layers": 2,
        "d_ff": 8,
    })["total"]
    assert record.compute.layer_applications == 4
    assert record.compute.unique_blocks == 2
    assert record.compute.loops == 2
    assert record.compute.forward_gflops.usable
    assert record.status is RunStatus.COMPLETE
    assert "paper/data" not in record.result_file


def test_missing_result_file_seed_and_incomplete_status(project):
    project.add_default_configs()
    project.add_run("loop_x2", benchmark=False)
    record = project.analysis().record("loop_x2")
    codes = {issue.code for issue in record.issues}
    assert "MISSING_RESULT_FILE" in codes
    assert "MISSING_SEED" in codes
    assert record.seed.value is None
    assert record.seed.status.value == "missing"
    assert record.status is RunStatus.INCOMPLETE
    assert record.metric("benchmark_mcq_accuracy").status is ValueStatus.MISSING


def test_missing_metric_in_official_file_is_not_zero(project):
    project.add_default_configs()
    payload = official_payload()
    del payload["val_loss"]
    del payload["perplexity"]
    project.add_run("loop_x2", mmlu=payload)
    record = project.analysis().record("loop_x2")
    assert record.metric("val_loss").status is ValueStatus.MISSING
    assert record.metric("val_loss").value is None
    assert not record.metric("val_loss").is_zero
    assert any(issue.code == "MISSING_METRIC" for issue in record.issues)


def test_malformed_json_and_invalid_metric(project):
    project.add_default_configs()
    (project.results / "loop_x2.mmlu_cs.json").write_text("{not json", encoding="utf-8")
    dump_json(project.results / "loop_x2.benchmark.json", benchmark_payload())
    project.add_run("dense_2l", mmlu=official_payload(loss="oops"))
    result = project.analysis()
    assert result.record("loop_x2").status is RunStatus.INVALID
    assert any(i.code == "MALFORMED_JSON" for i in result.issues)
    assert result.record("dense_2l").metric("val_loss").status is ValueStatus.INVALID
    assert result.record("dense_2l").status is RunStatus.INVALID


def test_zero_val_loss_is_valid_and_percentage_undefined(project):
    project.add_default_configs()
    project.add_run("loop_x2", mmlu=official_payload(loss=0.0), seed=1)
    project.add_run("dense_2l", mmlu=official_payload(loss=0.0), seed=1)
    result = project.analysis()
    loop = result.record("loop_x2")
    assert loop.metric("val_loss").is_zero
    assert loop.metric("val_loss").status is ValueStatus.AVAILABLE
    assert not any(i.code == "MISSING_METRIC" for i in loop.issues)
    assert any(i.code == "ZERO_VALUED_METRIC" and i.severity == "info"
               for i in loop.issues)
    comp = build_comparisons(result).comparisons[0]
    assert comp.loss_difference.value == 0.0
    assert comp.percent_difference.value is None
    assert comp.percent_difference.status is ValueStatus.UNAVAILABLE


def test_seed_inference_explicit_field_takes_precedence(project):
    assert infer_seed("loop_x2_s1", []).value == 1
    assert infer_seed("loop_x2_s1", []).status.value == "inferred"
    assert infer_seed("loop_x2", []).value is None
    explicit = infer_seed("loop_x2_s1", [(7, "result.json", "seed")])
    assert explicit.value == 7
    assert explicit.status.value == "explicit"
    assert infer_seed("loop_x2", [(True, "result.json", "seed")]).value is None


def test_corpus_inference_and_continuation_is_unknown():
    assert infer_corpus("loop_x2_v2").value == "v2"
    assert infer_corpus("baseline_v2_s1").value == "v2"
    assert infer_corpus("loop_x2").value == "v1"
    assert infer_corpus("d1024_ctx2048").value is None


def test_matched_categories_are_strictly_separate(project):
    project.add_default_configs()
    for run, loss in (("loop_x2", 4.0), ("dense_2l", 2.2), ("dense_4l", 2.3)):
        project.add_run(run, loss=loss, clean_loss=loss + 0.1)
    result = project.analysis()
    comparisons = build_comparisons(result)
    categories = {(c.loop_run, c.dense_run): c.category for c in comparisons.comparisons}
    assert ("loop_x2", "dense_2l") in categories
    assert ("loop_x2", "dense_4l") in categories
    assert ("loop_x2", "dense_2l") not in {
        pair for pair, cat in categories.items() if cat is ComparisonCategory.MATCHED_COMPUTE
    }
    assert ("loop_x2", "dense_4l") not in {
        pair for pair, cat in categories.items() if cat is ComparisonCategory.MATCHED_PARAMETER
    }
    parameter = [c for c in comparisons.comparisons
                 if c.category is ComparisonCategory.MATCHED_PARAMETER]
    assert all(c.loss_difference.value == pytest.approx(1.8) for c in parameter)
    for comp in parameter:
        expected = (100 * (comp.loop_loss.value - comp.dense_loss.value)
                    / abs(comp.dense_loss.value))
        assert comp.percent_difference.value == pytest.approx(expected)
    assert all(c.metric in ("val_loss", "clean_loss") for c in comparisons.comparisons)


def test_pairs_with_different_seed_or_corpus_are_skipped(project):
    project.add_default_configs()
    for run in ("loop_x2_s1", "dense_2l", "dense_4l_v2"):
        project.add_run(run)
    result = project.analysis()
    comparisons = build_comparisons(result)
    assert comparisons.comparisons == []
    reasons = " ".join(s.reason for s in comparisons.skipped)
    assert "different seed labels" in reasons
    assert "different corpora" in reasons


def test_missing_parameter_counts_block_matched_parameter_pair(project):
    project.add_config("model_loop_x2.yaml", 4, 2)
    project.add_run("loop_x2", loss=4.0)
    project.add_run("dense_2l", loss=2.0)
    result = project.analysis()
    category, basis, reason = classify_pair(result.record("loop_x2"),
                                            result.record("dense_2l"))
    assert category is None
    assert "unavailable" in reason
    assert basis == ""
    assert build_comparisons(result).comparisons == []


def test_bootstrap_metadata_is_attached_to_comparison(project):
    project.add_default_configs()
    project.add_run("loop_x2", loss=4.0)
    project.add_run("dense_2l", loss=2.0)
    project.add_bootstrap()
    result = project.analysis()
    assert len(result.bootstrap_pairs) == 1
    comp = next(c for c in build_comparisons(result).comparisons
                if c.metric == "val_loss")
    assert comp.bootstrap_mean_difference.value == pytest.approx(1.5)
    assert comp.bootstrap_ci95 == "1.400000, 1.600000"
    assert "cpu_bootstrap.json" in result.record("loop_x2").sources[0] or any(
        "cpu_bootstrap.json" in s for s in result.record("loop_x2").sources
    )


def test_manifest_mismatch_is_reported(project):
    project.add_default_configs()
    project.add_run("loop_x2")
    result_file = project.results / "loop_x2.mmlu_cs.json"
    (project.metadata / "MANIFEST.sha256").write_text(
        f"{'0' * 64}  results/loop_x2.mmlu_cs.json\n", encoding="utf-8"
    )
    result = project.analysis()
    assert any(i.code == "MANIFEST_MISMATCH" for i in result.global_issues)
    assert hashlib.sha256(result_file.read_bytes()).hexdigest() in {
        info.sha256 for info in result.sources.values()
    }


def test_report_bundle_is_deterministic_and_traceable(project, tmp_path):
    project.add_default_configs()
    for run, loss in (("loop_x2", 4.0), ("dense_2l", 2.2), ("dense_4l", 2.3)):
        project.add_run(run, loss=loss, clean_loss=loss + 0.1)
    result = project.analysis()
    comparisons = build_comparisons(result)
    first = tmp_path / "report_a"
    second = tmp_path / "report_b"
    write_report_bundle(result, comparisons, first)
    write_report_bundle(result, comparisons, second)
    assert sorted(p.name for p in first.iterdir()) == sorted(
        p.name for p in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    markdown = (first / "experiment_report.md").read_text(encoding="utf-8")
    assert "Matched-parameter results" in markdown
    assert "Matched-compute results" in markdown
    assert "loop_x2.mmlu_cs.json" in markdown
    with (first / "experiment_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["run_id"] for row in rows} == {"loop_x2", "dense_2l", "dense_4l"}
    assert all("configs/" in row["source_files"] for row in rows)
    with (first / "source_files.csv").open(encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))
    assert all(len(row["sha256"]) == 64 for row in sources)
    param_rows = list(csv.DictReader((first / "matched_parameter.csv").open(encoding="utf-8")))
    compute_rows = list(csv.DictReader((first / "matched_compute.csv").open(encoding="utf-8")))
    assert {r["dense_run"] for r in param_rows} == {"dense_2l"}
    assert {r["dense_run"] for r in compute_rows} == {"dense_4l"}
    assert all(float(r["loss_difference"]) > 0 for r in param_rows)


def test_report_shows_missing_and_unavailable_explicitly(project, tmp_path):
    project.add_default_configs()
    project.add_run("loop_x2", mmlu=official_payload(loss=0.0), seed=1)
    project.add_run("dense_2l", mmlu=official_payload(loss=0.0), seed=1)
    result = project.analysis()
    comparisons = build_comparisons(result)
    out = tmp_path / "zero_report"
    write_report_bundle(result, comparisons, out)
    text = (out / "experiment_report.md").read_text(encoding="utf-8")
    assert "UNAVAILABLE" in text
    assert "0.000000" in text
    rows = list(csv.DictReader((out / "model_comparisons.csv").open(encoding="utf-8")))
    assert all(row["percent_difference"] == "UNAVAILABLE" for row in rows)


def test_cli_exit_codes_and_argument_validation(project, tmp_path):
    project.add_default_configs()
    project.add_run("loop_x2", loss=3.0)
    common = [
        "--experiments-dir", str(project.results),
        "--repo-root", str(project.root),
        "--configs-dir", str(project.configs),
        "--metadata-dir", str(project.metadata),
        "--output-dir", str(tmp_path / "out"),
    ]
    assert analysis_main(common + ["--format", "text"]) == 0
    assert compare_main(common + ["--format", "csv"]) == 0
    assert report_main(common + ["--format", "both"]) == 0
    assert analysis_main(common + ["--strict"]) == 4
    assert compare_main(common + ["--category", "matched-parameter"]) == 0
    with pytest.raises(SystemExit) as exc:
        compare_main(common + ["--category", "nonsense"])
    assert exc.value.code == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert analysis_main([
        "--experiments-dir", str(empty), "--repo-root", str(project.root),
        "--configs-dir", str(project.configs), "--metadata-dir", str(project.metadata),
    ]) == 3


def test_existing_param_counter_import_has_no_argv_side_effect():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "model_loop_6x2.yaml").read_text(
        encoding="utf-8"))["model"]
    assert vpc.count_params(cfg)["total"] == 54_006_528
    assert vpc.CONFIG_PATH == vpc.DEFAULT_CONFIG
