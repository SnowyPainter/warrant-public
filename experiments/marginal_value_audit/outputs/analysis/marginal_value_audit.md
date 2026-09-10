# Marginal Value Audit

This is a frozen-model diagnostic. It enumerates every subset of the top-attended items while keeping all other valid items as fixed background. The oracle column is label-informed diagnostic headroom, not a deployable result.

Top-k: `5`; requested examples/domain: `24`.

| Domain | N | Base primary | Oracle primary | Directional gain | Sign reversal | |Interaction| | Attn-Shapley rho | Harmful top-attn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTDG | 24 | 0.8194 | 0.8403 | +0.0208 | 0.292 | 0.0035 | 0.083 | 0.583 |
| MTPP | 24 | 0.7917 | 0.7917 | +0.0000 | 0.146 | 0.0054 | 0.133 | 0.458 |
| RAG | 24 | 0.5889 | 0.6118 | +0.0229 | 0.208 | 0.0097 | -0.200 | 0.708 |
| STPP | 24 | 1.1138 | 1.1018 | +0.0120 | 0.000 | 0.0000 | 0.271 | 0.333 |
| TKG | 24 | 0.0064 | 0.0082 | +0.0018 | 0.008 | 0.0003 | -0.008 | 0.583 |

## Go / No-Go

The hypothesis receives initial support when sign reversals or pair interactions are non-zero, attention and Shapley utility are imperfectly aligned, and subset selection exposes positive utility headroom. A zero-interaction, rank-aligned result would reject the need for a set-aware router in this setting.

RAG caveat: the audited final attention path is only the attention-mediated context/mass path; the passage scorer also receives passage representations directly.
