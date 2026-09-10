# Additional Control Experiments

All controls use the 65,536-example, 30-epoch main budget and seeds 7/17/37.

| Domain | Dataset / model | Variant | n | Primary metric |
|---|---|---|---:|---:|
| ctdg | lastfm / DyGFormer | attention_adapter | 3 | 0.9032 ± 0.0006 |
| ctdg | lastfm / DyGFormer | combined_full | 3 | 0.9040 ± 0.0023 |
| ctdg | lastfm / DyGFormer | generic_open_path | 3 | 0.9022 ± 0.0012 |
| ctdg | lastfm / DyGFormer | generic_qk_warrant | 3 | 0.8484 ± 0.0074 |
| ctdg | lastfm / DyGFormer | item_only_gate | 3 | 0.9022 ± 0.0022 |
| ctdg | lastfm / DyGFormer | normalized_gate | 3 | 0.9061 ± 0.0008 |
| ctdg | lastfm / DyGFormer | scalar_gate | 3 | 0.9026 ± 0.0018 |
| mtpp | retweets / THP | attention_adapter | 3 | 0.7543 ± 0.0081 |
| mtpp | retweets / THP | item_only_gate | 3 | 0.7503 ± 0.0127 |
| mtpp | retweets / THP | normalized_gate | 3 | 0.7520 ± 0.0126 |
| mtpp | retweets / THP | scalar_gate | 3 | 0.7574 ± 0.0153 |
| rag | hotpotqa / FiD | attention_adapter | 3 | 0.6589 ± 0.0015 |
| rag | hotpotqa / FiD | frozen_attention_adapter | 3 | 0.6575 ± 0.0006 |
| rag | hotpotqa / FiD | frozen_gate | 3 | 0.6570 ± 0.0004 |
| rag | hotpotqa / FiD | item_only_gate | 3 | 0.6583 ± 0.0015 |
| rag | hotpotqa / FiD | normalized_gate | 3 | 0.6593 ± 0.0007 |
| rag | hotpotqa / FiD | scalar_gate | 3 | 0.6588 ± 0.0008 |
| stpp | earthquake / DeepSTPP | attention_adapter | 3 | 1.0564 ± 0.0717 |
| stpp | earthquake / DeepSTPP | item_only_gate | 3 | 1.0588 ± 0.0780 |
| stpp | earthquake / DeepSTPP | normalized_gate | 3 | 1.0623 ± 0.0806 |
| stpp | earthquake / DeepSTPP | scalar_gate | 3 | 1.0533 ± 0.0942 |
| tkg | gdelt / xERTE | attention_adapter | 3 | 0.1426 ± 0.0003 |
| tkg | gdelt / xERTE | frozen_attention_adapter | 3 | 0.1420 ± 0.0011 |
| tkg | gdelt / xERTE | frozen_gate | 3 | 0.1424 ± 0.0010 |
| tkg | gdelt / xERTE | item_only_gate | 3 | 0.1424 ± 0.0007 |
| tkg | gdelt / xERTE | normalized_gate | 3 | 0.1434 ± 0.0006 |
| tkg | gdelt / xERTE | scalar_gate | 3 | 0.1425 ± 0.0008 |
