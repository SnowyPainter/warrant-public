# Warrant Benchmark

This repository runs the WarrantBlock benchmark from `docs/research_plan.pdf`.

## Main Benchmark

Default benchmark config:

```bash
experiments/main_benchmark/config.yaml
```

Run all configured domains, datasets, models, variants, and configured seeds:

```bash
/opt/conda/bin/python experiments/main_benchmark/run.py
/opt/conda/bin/python experiments/main_benchmark/compare_agg.py --metric primary_metric
```

## Filtering Runs

Run one domain:

```bash
/opt/conda/bin/python experiments/main_benchmark/run.py --domain mtpp
```

Run one dataset/model pair:

```bash
/opt/conda/bin/python experiments/main_benchmark/run.py \
  --domain mtpp \
  --dataset stackoverflow \
  --model SAHP
```

Run base vs Warrant for one seed:

```bash
/opt/conda/bin/python experiments/main_benchmark/run.py \
  --domain mtpp \
  --dataset stackoverflow \
  --model SAHP \
  --variant base,warrant \
  --seed 2024
```

Preview selected runs without training:

```bash
/opt/conda/bin/python experiments/main_benchmark/run.py --dry-run
```

Disable progress bars by setting this in `experiments/main_benchmark/config.yaml`:

```yaml
training:
  progress: false
```

Force rerun completed rows:

```bash
/opt/conda/bin/python experiments/main_benchmark/run.py --force
```

## Outputs

Per-run artifacts are saved under:

```bash
experiments/main_benchmark/outputs/{model}/{seed}/{domain}/{dataset}/{variant}/
```

Each run writes:

- `metrics.json`
- `run_config.json`
- `error.txt` if the run failed

The global result table is:

```bash
experiments/main_benchmark/outputs/results.csv
```

Completed rows in `results.csv` are skipped automatically unless `--force` is passed.
Rows now include `effective_implementation`, `warrant_active_blocks`, and `warrant_replacements` so Warrant runs can be audited directly from the CSV. Older rows without these audit columns are not treated as completed and will be replaced on rerun.


# Neural Dissection

```bash
/opt/conda/bin/python experiments/neural_dissection/run.py
```

# Mass Diagnostic

`experiments/mass_diagnostic` tests whether the localized Warrant path carries
labeled prediction evidence in CTDG, RAG, and TKG. It records evidence mass and
counterfactual metric changes after zeroing true-evidence or non-evidence mass.

```bash
/opt/conda/bin/python experiments/mass_diagnostic/run.py
/opt/conda/bin/python experiments/mass_diagnostic/agg.py
```

# Edge Query Adapter Ablation

```bash
/opt/conda/bin/python experiments/edge_query_adapter_ablation/run.py
```

# Warrant Operator Ablation

`experiments/warrant_ablation` decomposes the Warrant operator inside TKG
GDELT/xERTE, the TKG main-benchmark pair with the largest mean Warrant gain.
This avoids the CTDG edge-query adapter confound and tests whether the useful
part is the full query-conditioned value gate on historical-fact weighted terms,
rather than attention-only copy mass, attention-logit reweighting, or a
query-free/item-free gate.

Preview runs:

```bash
/opt/conda/bin/python experiments/warrant_ablation/run.py --dry-run
```

Run the default 3-seed operator ablation:

```bash
/opt/conda/bin/python experiments/warrant_ablation/run.py
```

Run a single mode:

```bash
/opt/conda/bin/python experiments/warrant_ablation/run.py \
  --variant full_value_gate \
  --seed 7
```

Aggregate:

```bash
/opt/conda/bin/python experiments/warrant_ablation/agg.py
```

# Path Localization

`experiments/path_localization` tests the central adaptation hypothesis:

> Warrant works best when it is applied to the weighted value path that directly determines the task metric.

This experiment is not a new benchmark leaderboard. It is a path-localization protocol for validating that Warrant was inserted at the right bottleneck instead of being attached to an arbitrary attention block.

## Path Localization Setup

Default config:

```bash
experiments/path_localization/config.yaml
```

The default setup runs one representative dataset/model per domain:

| Domain | Dataset | Model | Metric |
| --- | --- | --- | --- |
| CTDG | LastFM | DyGFormer | AUC |
| MTPP | Retweets | AttNHP | Mark MRR |
| RAG | HotpotQA | LED | Support MRR |
| STPP | Earthquake | Transformer-STPP | Location RMSE |
| TKG | ICEWS18 | xERTE | MRR |

Each domain is tested with four variants:

| Variant | Meaning |
| --- | --- |
| `base` | No Warrant on the metric-defining path |
| `generic_qk_warrant` | Warrant only in generic query-key attention; the localized metric path is disabled |
| `correct_path_warrant` | Warrant on the localized metric-defining weighted value path |
| `shuffled_pairing` | Correct-path Warrant evaluated after deliberately mismatching query-item pairing |

The expected proof pattern is:

```text
correct_path_warrant > generic_qk_warrant
correct_path_warrant > shuffled_pairing
correct_path_warrant > base
```

If this pattern holds, the result supports the path-localization hypothesis: Warrant improves performance because it gates the weighted value term that actually enters the metric-defining prediction path.

## Running Path Localization

Preview all selected runs:

```bash
/opt/conda/bin/python experiments/path_localization/run.py --dry-run
```

Run all configured domains, variants, and seeds:

```bash
/opt/conda/bin/python experiments/path_localization/run.py
```

Run one domain:

```bash
/opt/conda/bin/python experiments/path_localization/run.py --domain tkg
```

Run a quick smoke test:

```bash
/opt/conda/bin/python experiments/path_localization/run.py \
  --domain tkg \
  --seed 7 \
  --variant base,correct_path_warrant
```

Force rerun completed folders:

```bash
/opt/conda/bin/python experiments/path_localization/run.py --force
```

## Aggregating Path Localization

After runs finish, aggregate results:

```bash
/opt/conda/bin/python experiments/path_localization/agg.py
```

This writes:

```bash
experiments/path_localization/outputs/analysis/summary.csv
experiments/path_localization/outputs/analysis/paired_deltas.csv
experiments/path_localization/outputs/analysis/path_localization_verdicts.csv
experiments/path_localization/outputs/analysis/path_localization_report.md
```

The main run table is:

```bash
experiments/path_localization/outputs/results.csv
```

Per-run artifacts are saved under:

```bash
experiments/path_localization/outputs/{domain}/{dataset}/{model}/{variant}/seed_{seed}/
```
