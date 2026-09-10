# Frozen Marginal-Value Probe

The encoder and predictor are frozen. Splits are made by source example, so coalition rows from one example cannot cross train/test boundaries. Local and set-aware probes have exactly the same parameter count; the local probe's context branch is present but zeroed.

| Domain | Probe | RMSE | R2 | Pearson | Spearman | Sign acc. | Harmful AUC | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTDG | Counterfactual utility | 0.0274 +/- 0.0057 | -0.213 | 0.255 | 0.247 | 0.646 | 0.665 | 43393 |
| CTDG | Local | 0.0273 +/- 0.0059 | -0.191 | 0.175 | 0.207 | 0.626 | 0.645 | 86785 |
| CTDG | Set-aware | 0.0286 +/- 0.0065 | -0.325 | 0.159 | 0.119 | 0.601 | 0.602 | 86785 |
| CTDG | Shuffled-set | 0.0280 +/- 0.0064 | -0.256 | 0.164 | 0.130 | 0.582 | 0.596 | 86785 |
| RAG | Counterfactual utility | 0.1496 +/- 0.0199 | 0.067 | 0.310 | 0.278 | 0.592 | 0.608 | 37249 |
| RAG | Local | 0.1509 +/- 0.0198 | 0.049 | 0.284 | 0.227 | 0.557 | 0.567 | 74497 |
| RAG | Set-aware | 0.1495 +/- 0.0183 | 0.065 | 0.303 | 0.253 | 0.568 | 0.591 | 74497 |
| RAG | Shuffled-set | 0.1497 +/- 0.0196 | 0.064 | 0.292 | 0.237 | 0.566 | 0.579 | 74497 |

## Phase-B verdict

- **CTDG**: set-aware minus local R2 = `-0.134`; set-aware minus shuffled R2 = `-0.070`.
- **RAG**: set-aware minus local R2 = `+0.016`; set-aware minus shuffled R2 = `+0.001`.

The current pooled set-aware probe does not satisfy the Phase-B Go criterion. CTDG shows no set-aware advantage. RAG shows a small gain over the local probe, but not over the shuffled-set control. Phase C is therefore not justified by this representation. The result separates the existence of context-dependent marginal values (Phase A) from their learnable out-of-example predictability (Phase B).
