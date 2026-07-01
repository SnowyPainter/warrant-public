# Novelty Package Summary

이 실험은 HotpotQA multi-candidate RoBERTa 설정에서 gate가 곱해지는 계산 단위를 비교한다.
모든 variant는 같은 입력, 같은 candidate marker readout, 같은 train/eval split을 사용한다.

## Main Control Table

| Variant | Seeds | MRR | R@1 | F1 | AUPRC | Unsupported@GoldCount ↓ | Precision@GoldCount ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 0.9111 ± 0.0045 | 0.8371 ± 0.0073 | 0.7751 ± 0.0049 | 0.8110 ± 0.0065 | 0.2249 ± 0.0049 | 0.7751 ± 0.0049 |
| Param-MLP | 3 | 0.9036 ± 0.0057 | 0.8245 ± 0.0106 | 0.7688 ± 0.0033 | 0.8103 ± 0.0082 | 0.2312 ± 0.0033 | 0.7688 ± 0.0033 |
| Post-attention GLU | 3 | 0.9051 ± 0.0039 | 0.8295 ± 0.0098 | 0.7629 ± 0.0084 | 0.8034 ± 0.0116 | 0.2371 ± 0.0084 | 0.7629 ± 0.0084 |
| Attention-readout | 3 | 0.9125 ± 0.0042 | 0.8405 ± 0.0072 | 0.7730 ± 0.0076 | 0.8129 ± 0.0040 | 0.2270 ± 0.0076 | 0.7730 ± 0.0076 |
| Query-only gate | 3 | 0.9078 ± 0.0021 | 0.8312 ± 0.0048 | 0.7726 ± 0.0057 | 0.8104 ± 0.0053 | 0.2274 ± 0.0057 | 0.7726 ± 0.0057 |
| Shuffled Warrant | 3 | 0.9046 ± 0.0102 | 0.8270 ± 0.0166 | 0.7671 ± 0.0160 | 0.7979 ± 0.0086 | 0.2329 ± 0.0160 | 0.7671 ± 0.0160 |
| OpenPath-NoGate | 3 | 0.9129 ± 0.0092 | 0.8405 ± 0.0165 | 0.7755 ± 0.0110 | 0.8102 ± 0.0127 | 0.2245 ± 0.0110 | 0.7755 ± 0.0110 |
| Full Warrant | 3 | 0.9134 ± 0.0010 | 0.8414 ± 0.0052 | 0.7772 ± 0.0058 | 0.8042 ± 0.0027 | 0.2228 ± 0.0058 | 0.7772 ± 0.0058 |

## Full Warrant vs Base

| Metric | Base | Full Warrant | Absolute Δ | Relative change |
| --- | ---: | ---: | ---: | ---: |
| MRR | 0.9111 | 0.9134 | +0.0023 ↑ | +0.25% improved |
| R@1 | 0.8371 | 0.8414 | +0.0042 ↑ | +0.50% improved |
| Evidence F1 | 0.7751 | 0.7772 | +0.0021 ↑ | +0.27% improved |
| AUPRC | 0.8110 | 0.8042 | -0.0067 ↓ | -0.83% worse |
| Unsupported@GoldCount | 0.2249 | 0.2228 | -0.0021 ↓ | +0.94% improved |
| Precision@GoldCount | 0.7751 | 0.7772 | +0.0021 ↑ | +0.27% improved |

## Permission Diagnostics

| Variant | Gate μ | Gate σ | Gold g | Random g | Gold/Random α | Gold/Random αg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base |  |  |  |  |  |  |
| Param-MLP |  |  |  |  |  |  |
| Post-attention GLU |  |  |  |  |  |  |
| Attention-readout | 0.3671 ± 0.0646 | 0.4311 ± 0.0218 | 0.8688 ± 0.0654 | 0.1745 ± 0.0648 | 0.3811 ± 0.0898 | 2.2536 ± 1.1305 |
| Query-only gate | 0.5416 ± 0.3294 | 0.2716 ± 0.1945 | 0.8610 ± 0.1300 | 0.4161 ± 0.4149 | 0.9987 ± 0.7294 | 5.0091 ± 5.0603 |
| Shuffled Warrant | 0.9994 ± 0.0005 | 0.0020 ± 0.0011 | 0.9994 ± 0.0005 | 0.9994 ± 0.0005 | 0.9013 ± 0.3446 | 0.9015 ± 0.3448 |
| OpenPath-NoGate | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6265 ± 0.2263 | 0.6265 ± 0.2263 |
| Full Warrant | 0.5319 ± 0.3311 | 0.2709 ± 0.1916 | 0.8647 ± 0.0961 | 0.4049 ± 0.4209 | 0.8179 ± 0.1370 | 3.8522 ± 2.2126 |

## Same-Aggregate Constructive Check

| Variant | Item identity available | Attention renormalized | Expected accuracy | Interpretation |
| --- | --- | --- | ---: | --- |
| post_attention_gate | False | False | 0.5 | aggregate gate sees the same zero vector for both labels |
| attention_logit_gate | True | True | 1.0 | can separate items, but does so by redefining attention mass |
| warrant_value_term_gate | True | False | 1.0 | keeps alpha fixed and changes item-wise weighted value contribution |
