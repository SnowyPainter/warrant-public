# Edge-Query Adapter Ablation

Results are aggregated over completed seeds. AUC and accuracy are reported as mean ± sample standard deviation.

| Variant | Seeds | AUC | Delta AUC vs Base | Accuracy | Delta Accuracy vs Base | Eval Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 0.7860 ± 0.0014 | 0.0000 ± 0.0000 | 0.7157 ± 0.0047 | 0.0000 ± 0.0000 | 0.6098 ± 0.0028 |
| Generic q-k Warrant | 3 | 0.7903 ± 0.0003 | 0.0043 ± 0.0013 | 0.7185 ± 0.0010 | 0.0028 ± 0.0053 | 0.5969 ± 0.0039 |
| Param control | 3 | 0.8072 ± 0.0058 | 0.0211 ± 0.0047 | 0.7314 ± 0.0056 | 0.0157 ± 0.0038 | 0.5826 ± 0.0108 |
| Hand-crafted-only control | 3 | 0.8197 ± 0.0052 | 0.0337 ± 0.0040 | 0.7429 ± 0.0081 | 0.0272 ± 0.0061 | 0.5379 ± 0.0081 |
| Edge-conditioned Warrant | 3 | 0.8980 ± 0.0029 | 0.1119 ± 0.0028 | 0.8329 ± 0.0060 | 0.1172 ± 0.0050 | 0.3915 ± 0.0073 |
| Shuffled edge query | 3 | 0.6376 ± 0.0011 | -0.1484 ± 0.0015 | 0.5846 ± 0.0116 | -0.1312 ± 0.0148 | 0.8660 ± 0.0059 |

## Paired Seed Deltas

| Variant | Seed | AUC | Base AUC | Delta AUC | Accuracy | Base Accuracy | Delta Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 7 | 0.7875 | 0.7875 | 0.0000 | 0.7212 | 0.7212 | 0.0000 |
| Base | 17 | 0.7847 | 0.7847 | 0.0000 | 0.7123 | 0.7123 | 0.0000 |
| Base | 37 | 0.7860 | 0.7860 | 0.0000 | 0.7138 | 0.7138 | 0.0000 |
| Generic q-k Warrant | 7 | 0.7906 | 0.7875 | 0.0031 | 0.7178 | 0.7212 | -0.0034 |
| Generic q-k Warrant | 17 | 0.7904 | 0.7847 | 0.0057 | 0.7180 | 0.7123 | 0.0057 |
| Generic q-k Warrant | 37 | 0.7900 | 0.7860 | 0.0040 | 0.7197 | 0.7138 | 0.0060 |
| Param control | 7 | 0.8107 | 0.7875 | 0.0232 | 0.7355 | 0.7212 | 0.0143 |
| Param control | 17 | 0.8004 | 0.7847 | 0.0157 | 0.7251 | 0.7123 | 0.0127 |
| Param control | 37 | 0.8104 | 0.7860 | 0.0244 | 0.7338 | 0.7138 | 0.0200 |
| Hand-crafted-only control | 7 | 0.8236 | 0.7875 | 0.0362 | 0.7479 | 0.7212 | 0.0267 |
| Hand-crafted-only control | 17 | 0.8138 | 0.7847 | 0.0291 | 0.7336 | 0.7123 | 0.0213 |
| Hand-crafted-only control | 37 | 0.8218 | 0.7860 | 0.0358 | 0.7473 | 0.7138 | 0.0335 |
| Edge-conditioned Warrant | 7 | 0.8974 | 0.7875 | 0.1099 | 0.8361 | 0.7212 | 0.1149 |
| Edge-conditioned Warrant | 17 | 0.8954 | 0.7847 | 0.1107 | 0.8260 | 0.7123 | 0.1137 |
| Edge-conditioned Warrant | 37 | 0.9011 | 0.7860 | 0.1151 | 0.8367 | 0.7138 | 0.1229 |
| Shuffled edge query | 7 | 0.6386 | 0.7875 | -0.1489 | 0.5790 | 0.7212 | -0.1422 |
| Shuffled edge query | 17 | 0.6380 | 0.7847 | -0.1467 | 0.5980 | 0.7123 | -0.1144 |
| Shuffled edge query | 37 | 0.6364 | 0.7860 | -0.1496 | 0.5768 | 0.7138 | -0.1369 |
