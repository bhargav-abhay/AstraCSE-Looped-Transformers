# Experiment Analysis & Reproducibility Toolkit

This toolkit reads the experiment outputs already committed under `paper/data/`
and produces traceable summaries, dense/loop comparisons and completeness
checks. It never trains, fine-tunes, re-evaluates or modifies a model, and it
never writes to the source data.

## Layout

```
scripts/experiment_analysis/
├── core.py          # validated value types, enums and result containers
├── metrics.py       # pure metric formulas (differences, percentages, seeds, compute)
├── discovery.py     # file discovery, parsing, metadata joins, validation
├── comparisons.py   # dense/loop pairing under strict categories
├── reports.py       # deterministic Markdown + CSV writers
└── cli.py           # shared argument parsing and exit codes

scripts/analyze_experiments.py   # run discovery/validation only
scripts/compare_models.py        # run comparisons only
scripts/generate_report.py       # full report bundle
```

The existing `scripts/verify_param_count.py` is reused as the analytic parameter
counter. It was refactored so importing it no longer parses `sys.argv` at import
time.

## What is read

- `paper/data/results/*.mmlu_cs.json` — official strided evaluation:
  `val_loss`, `perplexity`, `mcq_accuracy`, `mcq_correct`, `mcq_total`,
  `per_subject`.
- `paper/data/results/*.benchmark.json` — 10-item benchmark MCQ only. This
  format has no `val_loss` or `perplexity`; those fields are reported as
  `UNAVAILABLE`, never as zero.
- `paper/data/rescore*.json` — re-scored `full_val_loss`, `clean_loss`,
  `copied_loss`, token counts and reproduction flags.
- `paper/data/mcq*.json` — conditional-likelihood MCQ accuracies (`cf_norm`,
  `cf_sum`, `mcf`) kept in separate columns.
- `paper/data/curve_v2.json` — final training step and re-scored losses.
- `paper/data/cpu_bootstrap.json` — measured seconds/batch, subsampling factors
  and bootstrap comparison intervals.
- `configs/*.yaml` — architecture, declared `expected_param_count`, peak LR and
  warmup ratio.
- `paper/data/MANIFEST.sha256` — checked against every registered input file.

Every registered input is recorded in `source_files.csv` with its SHA-256.

## Missing, invalid, unavailable and zero

The toolkit distinguishes four states through `MetricValue`:

| Status | Meaning | Example |
| --- | --- | --- |
| `available` | A validated number was read; `0.0` is valid | `val_loss: 0.0` |
| `missing` | An expected field is absent | no `val_loss` in an official result |
| `invalid` | A present value failed validation | `NaN`, a string, a negative count |
| `unavailable` | The source format does not define the metric | `perplexity` in a benchmark-only file |

Zero is never used as a substitute for a missing value. A measured zero is
rendered as `0.000000`; missing/invalid/unavailable values are rendered by name.

Seeds follow the same rule: no numeric seed appears in the official result
JSONs. Only an explicit `seed`/`random_seed` field or a `_sN` run-id suffix
yields a value. All other runs are `MISSING` and are not assumed to be seed 0.
The corpus regime is inferred from the `_v2` marker; continuation runs without
it are not paired.

## Comparison rules

Losses are mean token cross-entropy in nats (lower is better).

- Absolute change: `loop_loss - dense_loss`. Positive means the loop model has
  the higher (worse) loss.
- Relative change: `100 * (loop_loss - dense_loss) / abs(dense_loss)`.
- If `dense_loss == 0`, the relative change is `UNAVAILABLE`, not zero or infinity.
- The denominator is the dense loss because dense models are the baseline.
- Only `val_loss` and `clean_loss` are compared. MCQ scores from different
  scoring rules are never mixed into a loss comparison.

Two comparison categories are kept strictly separate:

| Category | Rule | Interpretation |
| --- | --- | --- |
| matched-parameter | `loop.n_unique_layers == dense.n_layer` and `params(loop) - params(dense) == n_layer * d_model` | Same unique blocks; loop adds only depth conditioning, but runs fewer layer applications |
| matched-compute | `loop.n_layer == dense.n_layer` with equal width, vocabulary, FFN size and context | Same layer applications per token; parameter counts may differ |

A candidate pair is rejected if model types, corpora or seed labels disagree,
if a recorded seed value differs, if architecture metadata is unavailable, or
if the parameter gap is not exactly the depth embedding. Rejections are listed
in the "Pairs not compared" section and in `warnings.csv`; they are not silently
dropped.

## Commands

Run from the repository root:

```bash
# Discovery and validation only
python3 scripts/analyze_experiments.py --format text

# Comparisons only
python3 scripts/compare_models.py --format text
python3 scripts/compare_models.py --category matched-parameter --format csv

# Full report bundle into reports/
python3 scripts/generate_report.py --output-dir reports

# Strict mode: warnings/errors become exit code 4
python3 scripts/analyze_experiments.py --strict
```

Useful overrides:

- `--experiments-dir PATH` (default `paper/data/results`)
- `--repo-root PATH` (default current directory)
- `--configs-dir PATH` (default `configs`)
- `--metadata-dir PATH` (default `paper/data`)
- `--output-dir PATH` (default `reports`)
- `--model-types {both,loop,dense}` (`generate_report.py` only)
- `--format`: `text|json|both` for `analyze_experiments.py`,
  `text|csv|json` for `compare_models.py`, `both|markdown|csv` for
  `generate_report.py`
- `--category both|matched-parameter|matched-compute` and repeatable `--metric`
  for `compare_models.py`

Exit codes: `0` success, `1` analysis error, `2` argument error, `3` no result
files found, `4` strict mode found warnings/errors.

## Report bundle

`generate_report.py` writes nine deterministic files (no timestamps):

- `experiment_report.md` — human-readable report with provenance, status,
  summaries, both comparison categories, seed coverage, limitations and a
  SHA-256 source appendix.
- `experiment_summary.csv` — one row per run with metrics, config, params,
  compute, seed, status and source paths.
- `model_comparisons.csv` — all eligible comparisons.
- `matched_parameter.csv` — matched-parameter rows only.
- `matched_compute.csv` — matched-compute rows only.
- `seed_coverage.csv` — seed value, status, label and provenance per run.
- `incomplete_experiments.csv` — errors and warnings that need attention.
- `warnings.csv` — all findings, including rejected pairs.
- `source_files.csv` — every input file with schema, size and SHA-256.

Running the same command twice on unchanged inputs produces byte-identical
files.

## Known findings in the current data

- The five manifest entries below currently hash differently from their files
  and are reported as `MANIFEST_MISMATCH` errors. The toolkit does not edit the
  manifest or the source files; these are pre-existing data findings:
  - `configs/model.yaml`
  - `paper/data/cpu_bootstrap.json`
  - `paper/data/curve_v2.json`
  - `paper/data/rescore_loop6x2_v2.json`
  - `paper/data/rescore_v1_and_baseline_v2.json`
- Most runs have no recorded seed; only `_s1` runs have an inferred value.
- Checkpoint binaries and GPU timing traces are not in the repository. Compute
  values are analytic GFLOPs plus the recorded CPU seconds/batch proxy.
- Old mean-loss MCQ and CF-norm MCQ are not interchangeable, so they are never
  merged into a single score.

## Tests

The new tests use only the standard library, PyYAML, pytest and temporary
directories. They require no GPU, network, dataset, checkpoint or model
training:

```bash
python3 -m pytest tests/test_metrics.py tests/test_experiment_analysis.py -q
```

`tests/test_metrics.py` covers value states, zero handling, difference and
percentage formulas, seed coverage and compute derivation.
`tests/test_experiment_analysis.py` covers discovery, malformed/missing data,
config joins, strict category separation, bootstrap attachment, manifest
verification, deterministic report output and CLI exit codes.

The pre-existing test modules import `torch` and/or `numpy`; they are not
required by the toolkit tests and need those dependencies available separately.
