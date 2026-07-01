# Main Benchmark Aggregation

Metric: `primary_metric` (domain-aware primary metric)

- paired model/dataset groups: 32
- direction-aware positive mean changes: 27/32 (84.4%)
- exact two-sided sign test: p=0.000113 under H0: P(positive change)=0.5
- practical tiers: drop=5, marginal gain=1, positive, uncertain=8, substantial gain=10, tie/negligible=8
- mean improvement delta: +0.0146
- median improvement delta: +0.0058

## Top Improvements

| domain | dataset    | model            | metric_name   | metric_direction | base            | warrant         | improvement_delta | paired_ci95        | paired_t_p | practical_tier      | relative_delta_pct | n_seeds |
| ------ | ---------- | ---------------- | ------------- | ---------------- | --------------- | --------------- | ----------------- | ------------------ | ---------- | ------------------- | ------------------ | ------- |
| stpp   | earthquake | DeepSTPP         | rmse_location | lower            | 1.1383 ± 0.0288 | 1.0502 ± 0.0926 | +0.0881           | [-0.1454, +0.3216] | 0.2461     | positive, uncertain | +7.72%             | 3       |
| ctdg   | lastfm     | DyGFormer        | auc           | higher           | 0.8482 ± 0.0045 | 0.9023 ± 0.0022 | +0.0541           | [+0.0376, +0.0706] | 0.0050     | substantial gain    | +6.38%             | 3       |
| stpp   | earthquake | NSTPP            | rmse_location | lower            | 1.1589 ± 0.0612 | 1.1159 ± 0.1006 | +0.0430           | [-0.0575, +0.1435] | 0.2070     | positive, uncertain | +3.84%             | 3       |
| stpp   | earthquake | Transformer-STPP | rmse_location | lower            | 1.1147 ± 0.0893 | 1.0748 ± 0.0948 | +0.0399           | [-0.0690, +0.1488] | 0.2557     | positive, uncertain | +3.56%             | 3       |
| ctdg   | wikipedia  | DyGFormer        | auc           | higher           | 0.9440 ± 0.0015 | 0.9836 ± 0.0005 | +0.0396           | [+0.0349, +0.0443] | 0.0008     | substantial gain    | +4.20%             | 3       |
| ctdg   | wikipedia  | TGAT             | auc           | higher           | 0.9437 ± 0.0027 | 0.9824 ± 0.0010 | +0.0387           | [+0.0311, +0.0463] | 0.0021     | substantial gain    | +4.10%             | 3       |
| ctdg   | lastfm     | GraphMixer       | auc           | higher           | 0.8806 ± 0.0031 | 0.9139 ± 0.0009 | +0.0334           | [+0.0245, +0.0422] | 0.0038     | substantial gain    | +3.79%             | 3       |
| ctdg   | wikipedia  | GraphMixer       | auc           | higher           | 0.9529 ± 0.0040 | 0.9832 ± 0.0003 | +0.0303           | [+0.0212, +0.0394] | 0.0049     | substantial gain    | +3.18%             | 3       |
| ctdg   | lastfm     | TGAT             | auc           | higher           | 0.8598 ± 0.0044 | 0.8897 ± 0.0024 | +0.0298           | [+0.0155, +0.0442] | 0.0123     | substantial gain    | +3.47%             | 3       |
| tkg    | gdelt      | xERTE            | mrr           | higher           | 0.1277 ± 0.0022 | 0.1429 ± 0.0008 | +0.0151           | [+0.0095, +0.0207] | 0.0074     | substantial gain    | +11.85%            | 3       |
| tkg    | icews18    | RE-NET           | mrr           | higher           | 0.1423 ± 0.0002 | 0.1550 ± 0.0025 | +0.0127           | [+0.0067, +0.0188] | 0.0119     | substantial gain    | +8.95%             | 3       |
| tkg    | gdelt      | RE-NET           | mrr           | higher           | 0.1280 ± 0.0011 | 0.1395 ± 0.0010 | +0.0115           | [+0.0080, +0.0151] | 0.0051     | substantial gain    | +9.01%             | 3       |

## Largest Drops

| domain | dataset       | model      | metric_name   | metric_direction | base            | warrant         | improvement_delta | paired_ci95        | paired_t_p | practical_tier      | relative_delta_pct | n_seeds |
| ------ | ------------- | ---------- | ------------- | ---------------- | --------------- | --------------- | ----------------- | ------------------ | ---------- | ------------------- | ------------------ | ------- |
| mtpp   | retweets      | THP        | mark_mrr      | higher           | 0.7584 ± 0.0029 | 0.7497 ± 0.0115 | -0.0086           | [-0.0311, +0.0138] | 0.2388     | drop                | -1.14%             | 3       |
| tkg    | icews18       | CyGNet     | mrr           | higher           | 0.1536 ± 0.0007 | 0.1501 ± 0.0054 | -0.0035           | [-0.0185, +0.0116] | 0.4274     | drop                | -2.24%             | 3       |
| mtpp   | stackoverflow | SAHP       | mark_mrr      | higher           | 0.6076 ± 0.0017 | 0.6062 ± 0.0017 | -0.0013           | [-0.0030, +0.0003] | 0.0730     | drop                | -0.22%             | 3       |
| mtpp   | stackoverflow | AttNHP     | mark_mrr      | higher           | 0.6051 ± 0.0020 | 0.6041 ± 0.0045 | -0.0010           | [-0.0083, +0.0063] | 0.6081     | drop                | -0.17%             | 3       |
| stpp   | gowalla       | DeepSTPP   | rmse_location | lower            | 0.0481 ± 0.0012 | 0.0484 ± 0.0046 | -0.0002           | [-0.0101, +0.0096] | 0.9306     | drop                | -0.42%             | 3       |
| tkg    | gdelt         | CyGNet     | mrr           | higher           | 0.1478 ± 0.0006 | 0.1478 ± 0.0012 | +0.0000           | [-0.0027, +0.0028] | 0.9708     | tie/negligible      | +0.02%             | 3       |
| tkg    | icews14       | CyGNet     | mrr           | higher           | 0.2195 ± 0.0039 | 0.2196 ± 0.0031 | +0.0000           | [-0.0037, +0.0038] | 0.9689     | tie/negligible      | +0.02%             | 3       |
| rag    | hotpotqa      | LED        | support_mrr   | higher           | 0.6669 ± 0.0018 | 0.6670 ± 0.0002 | +0.0002           | [-0.0038, +0.0042] | 0.8816     | tie/negligible      | +0.02%             | 3       |
| ctdg   | mooc          | GraphMixer | auc           | higher           | 0.9770 ± 0.0000 | 0.9774 ± 0.0015 | +0.0004           | [-0.0032, +0.0040] | 0.6859     | tie/negligible      | +0.04%             | 3       |
| mtpp   | stackoverflow | THP        | mark_mrr      | higher           | 0.6066 ± 0.0017 | 0.6076 ± 0.0013 | +0.0009           | [-0.0019, +0.0038] | 0.2906     | tie/negligible      | +0.15%             | 3       |
| ctdg   | mooc          | DyGFormer  | auc           | higher           | 0.9763 ± 0.0003 | 0.9773 ± 0.0016 | +0.0010           | [-0.0032, +0.0053] | 0.3962     | tie/negligible      | +0.11%             | 3       |
| stpp   | gowalla       | NSTPP      | rmse_location | lower            | 0.0486 ± 0.0033 | 0.0474 ± 0.0017 | +0.0012           | [-0.0034, +0.0058] | 0.3862     | positive, uncertain | +2.26%             | 3       |
