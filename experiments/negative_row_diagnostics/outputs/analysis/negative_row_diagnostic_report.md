# Negative Row Diagnostics

This diagnostic audits the two main-benchmark negative rows that are not fully explained by domain-level WNS alone: `Gowalla / DeepSTPP` and `ICEWS18 / CyGNet`.

The experiment has two layers.

1. **micro-WNS** recomputes `PathReach`, `Correct > Generic`, and `Correct > Shuffled` on the exact negative row.
2. **failure diagnostics** inspect whether the learned permission suppresses useful STPP history or disrupts CyGNet's copy/generation balance.

## micro-WNS

| domain | dataset | model | warrant_need_score | warrant_need_score_minmax | warrant_need_tier | main_row_relative_pct | main_row_tier | path_reach_rel_mean | correct_over_generic_rel_mean | correct_over_shuffled_rel_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stpp | gowalla | DeepSTPP | 12.3600 | 1.0000 | very_high | -0.4180 | drop | 0.0000 | 0.0217 | 37.4328 |
| tkg | icews18 | CyGNet | 0.1298 | 0.0000 | strong | -2.2398 | drop | 0.0000 | 0.0000 | 0.3933 |


## Gowalla / DeepSTPP: false suppression diagnostic

The STPP diagnostic treats the closest half of the valid history events to the target location as useful local history for that prediction.  It reports whether the permission gate preserves that useful local mass or suppresses it more than the remaining history.

| domain | dataset | model | variant | seeds | stpp_rmse_revisit_mean | stpp_rmse_new_place_mean | stpp_rmse_short_move_mean | stpp_rmse_long_move_mean | stpp_useful_mass_retention_mean | stpp_nonuseful_mass_retention_mean | stpp_false_suppression_index_mean | stpp_prior_rmse_delta_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stpp | gowalla | DeepSTPP | base | 3 | 0.0344 | 0.0434 | 0.0325 | 0.0535 |  |  |  |  |
| stpp | gowalla | DeepSTPP | correct_path_warrant | 3 | 0.0373 | 0.0444 | 0.0412 | 0.0468 | 0.9060 | 0.6276 | -0.2785 | -0.0000 |
| stpp | gowalla | DeepSTPP | generic_qk_warrant | 3 | 0.0345 | 0.0446 | 0.0340 | 0.0542 |  |  |  |  |
| stpp | gowalla | DeepSTPP | open_path_no_gate | 3 | 0.0530 | 0.0585 | 0.0487 | 0.0677 |  |  |  |  |
| stpp | gowalla | DeepSTPP | shuffled_pairing | 3 |  | 1.6357 | 0.7462 | 2.5253 |  |  |  |  |


Reading rule: positive `stpp_false_suppression_index_mean` means non-useful history is retained more than useful history.  Positive `stpp_prior_rmse_delta_mean` means the Warrant-weighted location prior is farther from the target than the raw-attention prior.

## ICEWS18 / CyGNet: copy-saturation diagnostic

The TKG diagnostic separates true-tail copy mass from the largest false-tail copy mass.  It checks whether Warrant preserves the useful copy path or disturbs CyGNet's existing copy/generation mixture.

| domain | dataset | model | variant | seeds | tkg_true_tail_seen_rate_mean | tkg_mrr_seen_mean | tkg_mrr_unseen_mean | tkg_true_copy_retention_mean | tkg_false_copy_retention_mean | tkg_copy_saturation_attention_mean | tkg_copy_saturation_warrant_mean | tkg_copy_gate_mean_seen_mean | tkg_copy_gate_mean_unseen_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tkg | icews18 | CyGNet | base | 3 | 0.2833 | 0.2841 | 0.0115 | 1.0000 | 1.0000 | 8.0931 | 8.0931 | 0.9999 | 0.9999 |
| tkg | icews18 | CyGNet | correct_path_warrant | 3 | 0.2833 | 0.2823 | 0.0115 | 0.9999 | 1.0004 | 8.5290 | 8.5375 | 0.9999 | 0.9999 |
| tkg | icews18 | CyGNet | generic_qk_warrant | 3 | 0.2833 | 0.2823 | 0.0115 | 0.9999 | 1.0004 | 8.5290 | 8.5375 | 0.9999 | 0.9999 |
| tkg | icews18 | CyGNet | open_path_no_gate | 3 | 0.2833 | 0.2841 | 0.0115 | 1.0000 | 1.0000 | 8.0931 | 8.0931 | 0.9999 | 0.9999 |
| tkg | icews18 | CyGNet | shuffled_pairing | 3 | 0.2833 | 0.1865 | 0.0007 | 1.0000 | 1.0004 | 10.2302 | 10.2403 | 0.9999 | 0.9999 |


Reading rule: `tkg_true_copy_retention_mean < tkg_false_copy_retention_mean` indicates that true-tail copy mass is suppressed more than false-tail copy mass.  A larger `tkg_copy_saturation_warrant_mean` than `tkg_copy_saturation_attention_mean` indicates a worse false-over-true copy imbalance after permission scaling.

## Suggested paper use

Use this report as a row-level failure audit rather than a new main benchmark.  WNS explains cross-domain effect heterogeneity, while this diagnostic explains why particular negative rows can occur when Warrant collides with an existing useful prior path: Gowalla revisit/place-prior dynamics or CyGNet copy saturation.

## Automatic verdict hints

- STPP false suppression index on correct path: `-0.2785`.
- STPP Warrant-prior minus attention-prior RMSE: `0.0000`.
- TKG true-copy retention vs false-copy retention: `0.9999` vs `1.0004`.
- TKG copy saturation attention vs Warrant: `8.5290` vs `8.5375`.

The STPP prior-RMSE delta is numerically near zero.  This indicates that Warrant changes mass retention between useful and non-useful history, but does not materially move the simple attention-weighted location prior in this diagnostic.
