# Mass Diagnostic Summary

This diagnostic asks whether the localized path carries labeled prediction evidence. It is not a new model leaderboard. The counterfactual columns are evaluation-time interventions that zero selected mass after the model has been trained.

## How To Read This

- `Evidence` means a labeled or task-derived support item: counterpart history for CTDG, supporting passage for RAG, and true-tail history for TKG.
- `Evidence Warrant Mass` is the effective mass on that evidence after the Warrant path: attention mass times gate when a gate exists.
- `Drop: Zero Evidence` is the metric decrease after zeroing evidence mass. A positive value means the path was carrying useful labeled evidence.
- `Drop: Zero Non-Evidence` is the metric decrease after zeroing the remaining mass. A negative value means non-evidence mass was hurting this oracle counterfactual.
- Base rows can still have attention mass, but they do not have the localized Warrant contribution path in the same sense; their zeroing columns are controls.

## Column Meanings

| Column | Meaning | What Supports The Claim |
| --- | --- | --- |
| `Primary` | The domain metric for the trained model. Higher is better for all rows here. | Warrant should improve over Base, but this table is mainly diagnostic. |
| `Evidence Present` | Fraction of eval samples where labeled evidence is available in the input. | Shows how often the diagnostic can observe the evidence path. |
| `Evidence Attention Mass` | Attention mass assigned to labeled evidence before permission scaling. | Indicates whether attention can find evidence at all. |
| `Evidence Warrant Mass` | Effective mass assigned to labeled evidence after Warrant scaling. | Larger values mean evidence is being carried by the localized path. |
| `Non-Evidence Warrant Mass` | Effective mass assigned to non-evidence items. | High values can still be useful context, but can also expose harmful mass. |
| `Drop: Zero Evidence` | `Primary - metric_after_zeroing_evidence_mass`. | Positive and large means evidence mass is causally important. |
| `Drop: Zero Non-Evidence` | `Primary - metric_after_zeroing_non_evidence_mass`. | Negative means removing non-evidence improves the oracle counterfactual. |

## Main Table

| Domain | Variant | Metric | Primary | Evidence Present | Evidence Attention Mass | Evidence Warrant Mass | Non-Evidence Warrant Mass | Drop: Zero Evidence | Drop: Zero Non-Evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ctdg | base | auc | 0.7892 +/- 0.0010 | 0.4053 +/- 0.0055 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| ctdg | warrant | auc | 0.8916 +/- 0.0073 | 0.4053 +/- 0.0055 | 0.6086 +/- 0.0234 | 0.6086 +/- 0.0234 | 1.3791 +/- 0.0236 | 0.0575 +/- 0.0068 | 0.0203 +/- 0.0012 |
| rag | base | support_mrr | 0.5511 +/- 0.0102 | 1.0000 +/- 0.0000 | 0.2571 +/- 0.0017 | 0.2571 +/- 0.0017 | 0.7429 +/- 0.0017 | -0.0036 +/- 0.0019 | 0.0029 +/- 0.0006 |
| rag | warrant | support_mrr | 0.5527 +/- 0.0113 | 1.0000 +/- 0.0000 | 0.2574 +/- 0.0013 | 0.2419 +/- 0.0008 | 0.6960 +/- 0.0082 | 0.0030 +/- 0.0015 | -0.0034 +/- 0.0008 |
| tkg | base | mrr | 0.0809 +/- 0.0047 | 0.3447 +/- 0.0000 | 0.0113 +/- 0.0019 | 0.0113 +/- 0.0019 | 0.9887 +/- 0.0019 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| tkg | warrant | mrr | 0.1173 +/- 0.0022 | 0.3447 +/- 0.0000 | 0.0849 +/- 0.0014 | 0.0815 +/- 0.0029 | 0.8787 +/- 0.0308 | 0.1162 +/- 0.0022 | -0.2245 +/- 0.0027 |

## Domain Notes

### CTDG / warrant

- Evidence definition: current counterpart appears in source/destination temporal history.
- Primary AUC: 0.8916.
- Evidence warrant mass: 0.6086; non-evidence warrant mass: 1.3791.
- Zeroing evidence changes the metric by -0.0575 relative to normal evaluation (`Drop: Zero Evidence` = +0.0575).
- Zeroing non-evidence changes the metric by -0.0203 relative to normal evaluation (`Drop: Zero Non-Evidence` = +0.0203).
- Reading: The evidence path is the stronger positive dependency. Removing non-evidence hurts, so remaining context still carries useful signal.

### RAG / warrant

- Evidence definition: retrieved passage contains supporting evidence.
- Primary Support MRR: 0.5527.
- Evidence warrant mass: 0.2419; non-evidence warrant mass: 0.6960.
- Zeroing evidence changes the metric by -0.0030 relative to normal evaluation (`Drop: Zero Evidence` = +0.0030).
- Zeroing non-evidence changes the metric by +0.0034 relative to normal evaluation (`Drop: Zero Non-Evidence` = -0.0034).
- Reading: The evidence path is useful, while non-evidence mass also changes the metric strongly. Removing non-evidence improves the oracle counterfactual, so some remaining mass is harmful or distracting.

### TKG / warrant

- Evidence definition: target tail appears in historical facts.
- Primary MRR: 0.1173.
- Evidence warrant mass: 0.0815; non-evidence warrant mass: 0.8787.
- Zeroing evidence changes the metric by -0.1162 relative to normal evaluation (`Drop: Zero Evidence` = +0.1162).
- Zeroing non-evidence changes the metric by +0.2245 relative to normal evaluation (`Drop: Zero Non-Evidence` = -0.2245).
- Reading: The evidence path is useful, while non-evidence mass also changes the metric strongly. Removing non-evidence improves the oracle counterfactual, so some remaining mass is harmful or distracting.

## Safe Wording

Use this table as mechanism evidence, not as a replacement for the main benchmark. A good sentence is:

> Mass diagnostics show that the localized Warrant path carries labeled prediction evidence: removing evidence mass reduces the task metric in CTDG, RAG, and especially TKG, while removing non-evidence mass reveals whether the remaining context is useful or distracting.

