# Dedicated Warrant Need Score

This report is produced from dedicated WNS diagnostic runs.  The model is trained and evaluated for each control variant under `experiments/warrant_need_score/outputs`; the score is not computed by reusing previous path-localization or mass-diagnostic outputs.

## Design

| Variant | Diagnostic role |
| --- | --- |
| `base` | No Warrant and no localized permission path. |
| `generic_qk_warrant` | Warrant on generic query-key attention while the localized metric path is disabled. |
| `open_path_no_gate` | Metric-facing path opened with `g=1` and no learned permission gate. |
| `correct_path_warrant` | Warrant on the metric-facing weighted value path. |
| `shuffled_pairing` | Correct path retained, but query-item pairing mismatched. |

A domain receives a high WNS when the correct path improves over base, beats generic placement, degrades when query-item pairing is broken, and improves over the open-path control.  The open-path control separates learned permission from merely exposing the metric-facing contribution path.

## Need Scores

| Domain | Dataset | Model | Metric | Seeds | WNS | Relative WNS | Tier | Main rel. gain % | Path reach | Correct-Generic | Correct-Shuffled | Correct-OpenPath |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CTDG | lastfm | DyGFormer | auc | 3 | 0.1127 | 0.1952 | strong | 2.896 | 0.1143 | 0.1159 | 0.1734 | 0.0028 |
| MTPP | retweets | AttNHP | mark_mrr | 3 | 0.0124 | 0.0208 | weak | 0.028 | 0.0000 | 0.0000 | 0.0496 | 0.0000 |
| RAG | hotpotqa | FiD | support_mrr | 3 | 0.0004 | 0.0000 | negligible | 0.157 | 0.0007 | 0.0002 | 0.0000 | 0.0008 |
| STPP | earthquake | DeepSTPP | rmse_location | 3 | 0.0983 | 0.1702 | moderate | 3.557 | 0.0074 | 0.0008 | 0.3822 | 0.0000 |
| TKG | gdelt | xERTE | mrr | 3 | 0.5758 | 1.0000 | very_high | 4.248 | 0.9385 | 0.9351 | 0.0437 | 0.0177 |

## Correlations

| Predictor | Target | n | Pearson | Spearman | Exact p |
| --- | --- | ---: | ---: | ---: | ---: |
| need_score_raw_mean | main_mean_relative_pct | 5 | 0.740 | 0.800 | 0.133 |
| need_score_raw_mean | main_win_rate | 5 | 0.193 | 0.051 | 1.000 |
| need_score_raw_mean | substantial_rate | 5 | 0.513 | 0.783 | 0.200 |
| need_score_raw_mean | drop_rate | 5 | -0.193 | -0.051 | 1.000 |
| warrant_need_score | main_mean_relative_pct | 5 | 0.740 | 0.800 | 0.133 |
| warrant_need_score | main_win_rate | 5 | 0.193 | 0.051 | 1.000 |
| warrant_need_score | substantial_rate | 5 | 0.513 | 0.783 | 0.200 |
| warrant_need_score | drop_rate | 5 | -0.193 | -0.051 | 1.000 |
| path_reach_rel_mean | main_mean_relative_pct | 5 | 0.634 | 0.900 | 0.083 |
| path_reach_rel_mean | main_win_rate | 5 | 0.177 | 0.410 | 0.500 |
| path_reach_rel_mean | substantial_rate | 5 | 0.501 | 0.783 | 0.200 |
| path_reach_rel_mean | drop_rate | 5 | -0.177 | -0.410 | 0.500 |
| correct_over_generic_rel_mean | main_mean_relative_pct | 5 | 0.630 | 0.900 | 0.083 |
| correct_over_generic_rel_mean | main_win_rate | 5 | 0.178 | 0.410 | 0.500 |
| correct_over_generic_rel_mean | substantial_rate | 5 | 0.505 | 0.783 | 0.200 |
| correct_over_generic_rel_mean | drop_rate | 5 | -0.178 | -0.410 | 0.500 |
| correct_over_shuffled_rel_mean | main_mean_relative_pct | 5 | 0.522 | 0.200 | 0.783 |
| correct_over_shuffled_rel_mean | main_win_rate | 5 | 0.060 | -0.359 | 0.567 |
| correct_over_shuffled_rel_mean | substantial_rate | 5 | -0.047 | 0.112 | 1.000 |
| correct_over_shuffled_rel_mean | drop_rate | 5 | -0.060 | 0.359 | 0.567 |
| correct_over_open_path_rel_mean | main_mean_relative_pct | 5 | 0.621 | 0.564 | 0.400 |
| correct_over_open_path_rel_mean | main_win_rate | 5 | 0.213 | 0.632 | 0.317 |
| correct_over_open_path_rel_mean | substantial_rate | 5 | 0.522 | 0.803 | 0.200 |
| correct_over_open_path_rel_mean | drop_rate | 5 | -0.213 | -0.632 | 0.317 |

## Figure

![Dedicated Warrant Need Score vs Gain](/workspace/warrant/experiments/warrant_need_score/outputs/analysis/warrant_need_score_vs_gain.png)

PDF: [warrant_need_score_vs_gain.pdf](/workspace/warrant/experiments/warrant_need_score/outputs/analysis/warrant_need_score_vs_gain.pdf)

## Reading

WNS is the raw direction-aware diagnostic effect size normalized by each domain's base metric.  `Relative WNS` is a min-max rescaling used only for ranking and plotting; it always assigns 0 to the smallest observed domain and 1 to the largest observed domain.  The tier column is a practical reading of the raw score: negligible (<0.005), weak (0.005-0.02), moderate (0.02-0.10), strong (0.10-0.25), and very_high (>=0.25).

WNS is high when the trained controls show that the metric-facing path is reachable, more useful than generic attention placement, and sensitive to the correct query-item pairing.  The score should be compared against the main benchmark only after being computed from these dedicated diagnostic runs.

This report does not treat WNS as a ground-truth label.  It is an operational measurement of the bottleneck that Warrant claims to solve.
