# Marginal Value Audit

This is a frozen-model diagnostic. It enumerates every subset of the top-attended items while keeping all other valid items as fixed background. The oracle column is label-informed diagnostic headroom, not a deployable result.

Top-k: `5`; requested examples/domain: `128`.

| Domain | N | Base primary | Oracle primary | Directional gain | Sign reversal | |Interaction| | Attn-Shapley rho | Harmful top-attn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTDG | 384 | 0.7929 | 0.8295 | +0.0366 | 0.142 | 0.0027 | 0.094 | 0.435 |
| MTPP | 384 | 0.7027 | 0.7157 | +0.0130 | 0.060 | 0.0036 | 0.019 | 0.451 |
| RAG | 384 | 0.5503 | 0.5928 | +0.0425 | 0.185 | 0.0137 | -0.083 | 0.542 |
| STPP | 384 | 0.8363 | 0.8271 | +0.0092 | 0.035 | 0.0001 | 0.083 | 0.500 |
| TKG | 384 | 0.0629 | 0.0630 | +0.0001 | 0.023 | 0.0004 | -0.010 | 0.544 |

## Go / No-Go

The hypothesis receives initial support when sign reversals or pair interactions are non-zero, attention and Shapley utility are imperfectly aligned, and subset selection exposes positive utility headroom. A zero-interaction, rank-aligned result would reject the need for a set-aware router in this setting.

RAG caveat: the audited final attention path is only the attention-mediated context/mass path; the passage scorer also receives passage representations directly.
