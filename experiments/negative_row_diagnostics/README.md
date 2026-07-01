# Negative Row Diagnostics

This experiment audits the two main-benchmark negative rows that domain-level
WNS does not fully explain:

- `STPP / Gowalla / DeepSTPP`
- `TKG / ICEWS18 / CyGNet`

The experiment has two layers.

1. `micro-WNS`: recompute `PathReach`, `Correct > Generic`, and
   `Correct > Shuffled` on the exact negative row.
2. Failure diagnostics:
   - STPP false suppression: whether Warrant suppresses useful local
     history/place-prior mass together with noisy history.
   - TKG copy saturation: whether Warrant disrupts CyGNet's copy/generation
     balance by suppressing true-tail copy mass less or more than false-tail
     copy mass.

Run all diagnostics:

```bash
/opt/conda/bin/python experiments/negative_row_diagnostics/run.py
```

Run one row:

```bash
/opt/conda/bin/python experiments/negative_row_diagnostics/run.py --domain tkg
/opt/conda/bin/python experiments/negative_row_diagnostics/run.py --domain stpp
```

Aggregate existing outputs only:

```bash
/opt/conda/bin/python experiments/negative_row_diagnostics/run.py --aggregate-only
```

Main outputs:

- `outputs/analysis/warrant_need_score_report.md`
- `outputs/analysis/negative_row_diagnostic_report.md`
- `outputs/analysis/stpp_false_suppression.csv`
- `outputs/analysis/tkg_copy_saturation.csv`

Paper usage:

Use this as row-level failure analysis, not as a new main benchmark. WNS explains
cross-domain effect heterogeneity; this diagnostic explains why particular
negative rows can occur when Warrant collides with an existing useful prior path.
