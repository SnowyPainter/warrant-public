# Neural Dissection Report

This report is generated from per-epoch dissection runs. It is meant to show whether the Warrant operation changes internal evidence contribution, not only final benchmark scores.

## Generated Artifacts

- `analysis/tables/final_base_vs_warrant.csv`: final primary metric delta for each selected domain/model.
- `analysis/tables/warrant_gate_regimes.csv`: final gate level, gate slope, Warrant logit mean, and a coarse regime label.
- `analysis/tables/mass_diagnostics.csv`: available contribution diagnostics such as RAG support/distractor mass and TKG copy-tail mass.
- `analysis/plots/gate_mean_by_domain.png`: Warrant gate trajectory by domain.
- `analysis/plots/warrant_logit_mean_by_domain.png`: Warrant logit trajectory by domain.
- `analysis/plots/metric_vs_gate/*.png`: dual-axis primary metric and gate trajectory for each Warrant run.

## Final Base vs Warrant

| domain | dataset | model | primary_metric | direction | base | warrant | delta_warrant_minus_base | improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctdg | lastfm | DyGFormer | auc | higher | 0.7361 | 0.8232 | 0.0871 | 0.0871 |
| mtpp | retweets | AttNHP | mark_mrr | higher | 0.7727 | 0.7668 | -0.0058 | -0.0058 |
| rag | hotpotqa | LED | support_mrr | higher | 0.5514 | 0.5535 | 0.0021 | 0.0021 |
| stpp | earthquake | Transformer-STPP | rmse_location | lower | 1.4961 | 1.5543 | 0.0582 | -0.0582 |
| tkg | icews18 | xERTE | mrr | higher | 0.0127 | 0.0733 | 0.0606 | 0.0606 |

## Warrant Gate Regimes

| domain | dataset | model | primary_metric | direction | final_primary | final_gate_mean | gate_slope | final_warrant_logit_mean | regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctdg | lastfm | DyGFormer | auc | higher | 0.8232 | 0.8782 | 0.0427 | 7.7982 | selective, opening |
| mtpp | retweets | AttNHP | mark_mrr | higher | 0.7668 | 0.1267 | 0.0001 | 0.3856 | suppressive |
| rag | hotpotqa | LED | support_mrr | higher | 0.5535 | 0.8494 | -0.1042 | -0.0647 | selective, closing |
| stpp | earthquake | Transformer-STPP | rmse_location | lower | 1.5543 | 0.8488 | -0.0874 | 2.0897 | selective, closing |
| tkg | icews18 | xERTE | mrr | higher | 0.0733 | 0.9653 | 0.0152 | 4.2338 | open / near-identity |

## Mass Diagnostics

| domain | dataset | model | variant | warrant_tail_mass_mean | support_attention_mass | distractor_attention_mass | support_attention_ratio | support_warrant_mass | distractor_warrant_mass | support_mass_ratio | ratio_gain_warrant_minus_attention | edge_warrant_gate_mean | edge_warrant_attention_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tkg | icews18 | xERTE | warrant | 1.0000 |  |  |  |  |  |  |  |  |  |
| rag | hotpotqa | LED | warrant |  | 0.2589 | 0.7411 | 0.2589 | 0.2239 | 0.6345 | 0.2609 | 0.0020 |  |  |
| rag | hotpotqa | LED | base |  | 0.2593 | 0.7407 | 0.2593 | 0.2593 | 0.7407 | 0.2593 | 0.0000 |  |  |
| tkg | icews18 | xERTE | base | 0.0000 |  |  |  |  |  |  |  |  |  |
| ctdg | lastfm | DyGFormer | warrant |  |  |  |  |  |  |  |  | 1.0000 | 2.5219 |
