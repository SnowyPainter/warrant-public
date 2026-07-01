# Path Localization Results

This report tests whether Warrant works best when placed on the metric-defining weighted value path.

## Verdict

| domain | dataset | model | metric | seeds | path_supported | mean_correct_gain_vs_base | mean_open_path_gain_vs_base | mean_correct_minus_open_path | mean_correct_minus_generic | mean_correct_minus_shuffled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctdg | lastfm | DyGFormer | auc | 3 | 3 | 0.1119 |  |  | 0.1076 | 0.2603 |
| mtpp | retweets | AttNHP | mark_mrr | 1 | 0 | 0.0155 |  |  | -0.0003 | 0.0174 |
| rag | hotpotqa | FiD | support_mrr | 1 | 1 | 0.0018 |  |  | 0.0013 | 0.0003 |
| stpp | earthquake | DeepSTPP | rmse_location | 1 | 0 | 0.0197 |  |  | -0.0070 | 0.1316 |
| tkg | gdelt | xERTE | mrr | 1 | 1 | 0.0687 | 0.0677 | 0.0010 | 0.0683 | 0.0032 |


## Variant Summary

| domain | dataset | model | variant | path_role | seeds | metric | direction | primary_mean | primary_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctdg | lastfm | DyGFormer | base | no_warrant | 3 | auc | higher | 0.7860 | 0.0014 |
| ctdg | lastfm | DyGFormer | generic_qk_warrant | weak_path_generic_attention | 3 | auc | higher | 0.7903 | 0.0003 |
| ctdg | lastfm | DyGFormer | correct_path_warrant | metric_defining_path | 3 | auc | higher | 0.8980 | 0.0029 |
| ctdg | lastfm | DyGFormer | shuffled_pairing | query_item_pairing_control | 3 | auc | higher | 0.6376 | 0.0011 |
| mtpp | retweets | AttNHP | base | no_warrant | 1 | mark_mrr | higher | 0.7555 | 0.0000 |
| mtpp | retweets | AttNHP | generic_qk_warrant | weak_path_generic_attention | 1 | mark_mrr | higher | 0.7713 | 0.0000 |
| mtpp | retweets | AttNHP | correct_path_warrant | metric_defining_path | 1 | mark_mrr | higher | 0.7710 | 0.0000 |
| mtpp | retweets | AttNHP | shuffled_pairing | query_item_pairing_control | 1 | mark_mrr | higher | 0.7536 | 0.0000 |
| rag | hotpotqa | FiD | base | no_warrant | 1 | support_mrr | higher | 0.5576 | 0.0000 |
| rag | hotpotqa | FiD | generic_qk_warrant | weak_path_generic_attention | 1 | support_mrr | higher | 0.5580 | 0.0000 |
| rag | hotpotqa | FiD | correct_path_warrant | metric_defining_path | 1 | support_mrr | higher | 0.5593 | 0.0000 |
| rag | hotpotqa | FiD | shuffled_pairing | query_item_pairing_control | 1 | support_mrr | higher | 0.5591 | 0.0000 |
| stpp | earthquake | DeepSTPP | base | no_warrant | 1 | rmse_location | lower | 2.2519 | 0.0000 |
| stpp | earthquake | DeepSTPP | generic_qk_warrant | weak_path_generic_attention | 1 | rmse_location | lower | 2.2253 | 0.0000 |
| stpp | earthquake | DeepSTPP | correct_path_warrant | metric_defining_path | 1 | rmse_location | lower | 2.2322 | 0.0000 |
| stpp | earthquake | DeepSTPP | shuffled_pairing | query_item_pairing_control | 1 | rmse_location | lower | 2.3639 | 0.0000 |
| tkg | gdelt | xERTE | base | no_warrant | 1 | mrr | higher | 0.0488 | 0.0000 |
| tkg | gdelt | xERTE | generic_qk_warrant | weak_path_generic_attention | 1 | mrr | higher | 0.0492 | 0.0000 |
| tkg | gdelt | xERTE | open_path_no_gate | metric_path_open_g_equals_one | 1 | mrr | higher | 0.1165 | 0.0000 |
| tkg | gdelt | xERTE | correct_path_warrant | metric_defining_path | 1 | mrr | higher | 0.1175 | 0.0000 |
| tkg | gdelt | xERTE | shuffled_pairing | query_item_pairing_control | 1 | mrr | higher | 0.1144 | 0.0000 |


## Paired Deltas

| domain | dataset | model | seed | variant | metric | direction_aware_delta_vs_base |
| --- | --- | --- | --- | --- | --- | --- |
| ctdg | lastfm | DyGFormer | 7 | base | auc | 0.0000 |
| ctdg | lastfm | DyGFormer | 7 | generic_qk_warrant | auc | 0.0031 |
| ctdg | lastfm | DyGFormer | 7 | correct_path_warrant | auc | 0.1099 |
| ctdg | lastfm | DyGFormer | 7 | shuffled_pairing | auc | -0.1489 |
| ctdg | lastfm | DyGFormer | 17 | base | auc | 0.0000 |
| ctdg | lastfm | DyGFormer | 17 | generic_qk_warrant | auc | 0.0057 |
| ctdg | lastfm | DyGFormer | 17 | correct_path_warrant | auc | 0.1107 |
| ctdg | lastfm | DyGFormer | 17 | shuffled_pairing | auc | -0.1467 |
| ctdg | lastfm | DyGFormer | 37 | base | auc | 0.0000 |
| ctdg | lastfm | DyGFormer | 37 | generic_qk_warrant | auc | 0.0040 |
| ctdg | lastfm | DyGFormer | 37 | correct_path_warrant | auc | 0.1151 |
| ctdg | lastfm | DyGFormer | 37 | shuffled_pairing | auc | -0.1496 |
| mtpp | retweets | AttNHP | 7 | base | mark_mrr | 0.0000 |
| mtpp | retweets | AttNHP | 7 | generic_qk_warrant | mark_mrr | 0.0158 |
| mtpp | retweets | AttNHP | 7 | correct_path_warrant | mark_mrr | 0.0155 |
| mtpp | retweets | AttNHP | 7 | shuffled_pairing | mark_mrr | -0.0019 |
| rag | hotpotqa | FiD | 7 | base | support_mrr | 0.0000 |
| rag | hotpotqa | FiD | 7 | generic_qk_warrant | support_mrr | 0.0005 |
| rag | hotpotqa | FiD | 7 | correct_path_warrant | support_mrr | 0.0018 |
| rag | hotpotqa | FiD | 7 | shuffled_pairing | support_mrr | 0.0015 |
| stpp | earthquake | DeepSTPP | 7 | base | rmse_location | -0.0000 |
| stpp | earthquake | DeepSTPP | 7 | generic_qk_warrant | rmse_location | 0.0266 |
| stpp | earthquake | DeepSTPP | 7 | correct_path_warrant | rmse_location | 0.0197 |
| stpp | earthquake | DeepSTPP | 7 | shuffled_pairing | rmse_location | -0.1119 |
| tkg | gdelt | xERTE | 7 | base | mrr | 0.0000 |
| tkg | gdelt | xERTE | 7 | generic_qk_warrant | mrr | 0.0004 |
| tkg | gdelt | xERTE | 7 | open_path_no_gate | mrr | 0.0677 |
| tkg | gdelt | xERTE | 7 | correct_path_warrant | mrr | 0.0687 |
| tkg | gdelt | xERTE | 7 | shuffled_pairing | mrr | 0.0656 |
