# Qiu G1/G2 vs Warrant

Qiu G1/G2 use the paper-style ordinary optimizer LR (gate multiplier 1). Warrant uses its established gate multiplier 10. All methods use the same HotpotQA/RoBERTa split, two epochs, and seeds 7/17/37/47/57.

| Variant | Params | Support MRR | R@1 | Evidence F1 | Unsupported ↓ | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 124.65M | 0.9083±0.0087 | 0.8314±0.0161 | 0.7719±0.0076 | 0.2281±0.0076 | 0.8043±0.0202 |
| OpenPath | 129.97M | 0.9111±0.0084 | 0.8390±0.0145 | 0.7729±0.0117 | 0.2271±0.0117 | 0.8125±0.0119 |
| Qiu G2 | 131.74M | 0.9094±0.0058 | 0.8344±0.0120 | 0.7747±0.0024 | 0.2253±0.0024 | 0.8126±0.0069 |
| Qiu G1 | 131.74M | 0.9090±0.0041 | 0.8349±0.0083 | 0.7704±0.0145 | 0.2296±0.0145 | 0.8048±0.0149 |
| Full Warrant | 129.97M | 0.9122±0.0070 | 0.8395±0.0141 | 0.7770±0.0090 | 0.2230±0.0090 | 0.8138±0.0098 |

## Paired Support-MRR deltas

- full_minus_base: +0.0040±0.0119; positive seeds 3/5
- full_minus_open_path: +0.0012±0.0055; positive seeds 3/5
- full_minus_qiu_g2: +0.0028±0.0119; positive seeds 3/5
- full_minus_qiu_g1: +0.0033±0.0046; positive seeds 4/5
