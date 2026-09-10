# Qiu G1/G2 Comparison

This experiment compares Full Warrant with faithful head-specific,
element-wise implementations of the G1 and G2 gates from Qiu et al., using the
same HotpotQA/RoBERTa data split, optimization budget, and five seeds.

- `qiu_g2`: gate each projected value from its key/value-token hidden state,
  before the attention-weighted sum.
- `qiu_g1`: gate each SDPA output from its query-token hidden state, after the
  attention-weighted sum.
- `full_warrant`: gate each metric-facing weighted-value term from the current
  query--item pair, before aggregation.

Run:

```bash
/opt/conda/bin/python experiments/bert_hotpotqa_warrant/run.py \
  --config experiments/bert_hotpotqa_warrant/qiu_comparison.yaml
```

The primary Qiu replication uses the paper's ordinary shared optimizer learning
rate (gate multiplier 1):

```bash
/opt/conda/bin/python experiments/bert_hotpotqa_warrant/run.py \
  --config experiments/bert_hotpotqa_warrant/qiu_comparison_lr1.yaml
```

The first configuration retains a multiplier-10 sensitivity run matching the
existing Warrant fine-tuning recipe.
