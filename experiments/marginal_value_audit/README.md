# Marginal Value Audit

This frozen-checkpoint experiment measures the alignment between attention and
marginal prediction utility. For each example, it masks post-softmax weighted
value terms without renormalizing attention and exactly enumerates all 32
subsets of the five top-attended items. The default configuration evaluates
128 examples for each of three seeds in one representative setting from CTDG,
MTPP, RAG, STPP, and TKG.

```bash
/opt/conda/bin/python experiments/marginal_value_audit/run.py
```

Quick smoke test:

```bash
/opt/conda/bin/python experiments/marginal_value_audit/run.py \
  --domain rag --examples 8 --top-k 4
```

Outputs are written to `experiments/marginal_value_audit/outputs_3seed`.
