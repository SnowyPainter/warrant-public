# Negative-Row micro-WNS

This report recomputes WNS on the exact negative rows rather than using a domain-level representative setting.

The score uses three components: `PathReach`, `Correct > Generic`, and `Correct > Shuffled`.  The `Main row relative` column is copied from the exact row in the main benchmark table.

| domain | dataset | model | warrant_need_score | warrant_need_score_minmax | warrant_need_tier | main_row_relative_pct | main_row_tier | path_reach_rel_mean | correct_over_generic_rel_mean | correct_over_shuffled_rel_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stpp | gowalla | DeepSTPP | 12.3600 | 1.0000 | very_high | -0.4180 | drop | 0.0000 | 0.0217 | 37.4328 |
| tkg | icews18 | CyGNet | 0.1298 | 0.0000 | strong | -2.2398 | drop | 0.0000 | 0.0000 | 0.3933 |


Reading rule: a high micro-WNS on a negative row means the row is highly sensitive to path/pairing controls even if the final Warrant variant drops against Base.  That pattern points to path collision or false suppression, not absence of a Warrant-relevant bottleneck.
