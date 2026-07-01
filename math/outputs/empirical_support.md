# Empirical Support for the Math Claims

## BERT HotpotQA

| Quantity | Value |
| --- | ---: |
| MRR delta | 0.0058 |
| MRR-deficit reduction | 6.32% |
| R@1 error reduction | 7.18% |
| Unsupported attribution reduction | 2.89% |
| Gold/Random attention ratio | 0.8663 |
| Gold/Random effective ratio | 3.8016 |
| Effective ratio gain over attention ratio | 4.39x |

## Mass Diagnostics

| Domain | Primary | Evidence attention | Evidence warrant | Non-evidence warrant | Drop zero evidence | Drop zero non-evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mass_ctdg | 0.8916 | 0.6086 | 0.6086 | 1.3791 | 0.0575 | 0.0203 |
| mass_rag | 0.5527 | 0.2574 | 0.2419 | 0.6960 | 0.0030 | -0.0034 |
| mass_tkg | 0.1173 | 0.0849 | 0.0815 | 0.8787 | 0.1162 | -0.2245 |

## Path Localization

| Domain | Dataset | Model | Metric | Base | Correct | Gain | Correct-Generic | Correct-Shuffled |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| path_ctdg | lastfm | DyGFormer | auc | 0.7860 | 0.8980 | 0.1119 | 0.1076 | 0.2603 |
| path_mtpp | retweets | AttNHP | mark_mrr | 0.7555 | 0.7710 | 0.0155 | -0.0003 | 0.0174 |
| path_rag | hotpotqa | FiD | support_mrr | 0.5576 | 0.5593 | 0.0018 | 0.0013 | 0.0003 |
| path_stpp | earthquake | DeepSTPP | rmse_location | 2.2519 | 2.2322 | 0.0197 | -0.0070 | 0.1316 |
| path_tkg | gdelt | xERTE | mrr | 0.0488 | 0.1175 | 0.0687 | 0.0683 | 0.0032 |
