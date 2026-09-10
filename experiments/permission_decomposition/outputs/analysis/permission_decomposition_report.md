# Permission Decomposition Results

## Controlled CTDG factorial (32,768 examples, 10 epochs)

All rows below were trained afresh with the same split, optimizer, example budget, and seeds.
The interrupted run left a subset of seeds for some variants; seed counts are reported explicitly.

| Variant | Seeds | AUC | Δ vs OpenPath | Gate mean |
| --- | ---: | ---: | ---: | ---: |
| base | 1 | 0.7875 ± 0.0000 | -0.1122 |  |
| open_path_no_gate | 1 | 0.8997 ± 0.0000 | +0.0000 | 1.0000 ± 0.0000 |
| scalar_gate | 1 | 0.8993 ± 0.0000 | -0.0003 | 0.9869 ± 0.0000 |
| item_only_gate | 1 | 0.8987 ± 0.0000 | -0.0010 | 0.9838 ± 0.0000 |
| normalized_gate | 1 | 0.8978 ± 0.0000 | -0.0019 | 0.1146 ± 0.0000 |
| correct_path_warrant | 1 | 0.8950 ± 0.0000 | -0.0046 | 1.0000 ± 0.0000 |
| generic_open_path | 1 | 0.8993 ± 0.0000 | -0.0004 | 1.0000 ± 0.0000 |
| combined_full | 1 | 0.8974 ± 0.0000 | -0.0023 | 1.0000 ± 0.0000 |

## Main-benchmark budget CTDG decomposition (65,536 examples, 30 epochs)

| Seed | Base | OpenPath g=1 | Existing Full | Full−OpenPath |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.8512 | 0.9047 | 0.9006 | -0.0041 |
| 17 | 0.8503 | 0.9053 | 0.9014 | -0.0039 |
| 37 | 0.8430 | 0.8992 | 0.9047 | +0.0055 |
| **Mean** | **0.8482** | **0.9031** | **0.9023** | **-0.0008** |

## Direct answers to the reviewer questions

- Path exposure accounts for the LastFM-DyGFormer main-budget gain: Base→OpenPath is `+0.0549` AUC, while OpenPath→Full is `-0.0008` AUC.
- The 10-epoch controlled seed-7 run also does not show an advantage for the unnormalized edge gate over OpenPath. The normalized query-item control also remains below OpenPath in that run.
- The earlier path-localization report mixed a 10,000-example OpenPath row with a 32,768-example Full row and labeled a model with generic attention Warrant enabled as correct-path-only. Those rows cannot support a gate-versus-path conclusion.
- The trained edge gate is nearly identity in the unnormalized Full variants. The matched inference-time g=1 intervention consequently changes AUC by approximately zero; this is consistent with the gate distribution and is not evidence of final-prediction suppression.

These findings answer the requested CTDG ablation but do not test all meanings of permission. They specifically show that the current learned unnormalized g does not explain the reported LastFM-DyGFormer gain under the present implementation.


## Auxiliary permission supervision diagnostic (seed 7)

The target is whether each endpoint-history item is the current counterpart, used as a CTDG evidence proxy. This is a diagnostic target rather than a general permission label.

| Variant | AUC | Gate mean | Gate std | High saturation | Gate grad norm | Inference native−g=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task loss only | 0.8950 | 1.0000 | 0.0000 | 1.0000 | 0.000001 | -0.0000 |
| + auxiliary evidence loss | 0.8897 | 0.1110 | 0.1817 | 0.0435 | 0.006259 | +0.0114 |
| + auxiliary evidence loss (0.01) | 0.8893 | 0.1096 | 0.1741 | 0.0060 | 0.008604 | +0.0098 |

The auxiliary objective prevents the identity-gate collapse and creates measurable inference-time dependence on g, but this first target/weight does not improve AUC over OpenPath.