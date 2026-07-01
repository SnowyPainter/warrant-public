# SNR Simulation

Monte Carlo check for the theorem `SNR_W > SNR_B iff R_S > R_N` under evidence-aligned gates.

| Scenario | Support items | Noise items | E[g_S] | RMS(g_N) | R_S | R_N | Mean SNR Ratio | P(SNR improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weak_alignment_8_32 | 8 | 32 | 0.720 | 0.606 | 0.720 | 0.605 | 1.194 | 0.985 |
| moderate_alignment_8_32 | 8 | 32 | 0.850 | 0.508 | 0.850 | 0.507 | 1.686 | 1.000 |
| strong_alignment_8_32 | 8 | 32 | 0.920 | 0.360 | 0.920 | 0.359 | 2.584 | 1.000 |
| moderate_alignment_4_8 | 4 | 8 | 0.850 | 0.507 | 0.850 | 0.505 | 1.706 | 1.000 |
| moderate_alignment_16_64 | 16 | 64 | 0.850 | 0.508 | 0.850 | 0.507 | 1.680 | 1.000 |
| false_suppression | 8 | 32 | 0.450 | 0.803 | 0.450 | 0.803 | 0.561 | 0.000 |

Reading: SNR improves when retained support signal `R_S` exceeds retained noise standard deviation `R_N`.
The `false_suppression` row is an explicit counterexample where support gates are lower than noise gates.
