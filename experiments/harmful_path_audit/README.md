# Harmful-path recovery audit

This is a pre-specified five-seed HotpotQA hard-distractor test of the complete
causal pattern: OpenPath exposes non-evidence mass, learned permission lowers
hard/random gates relative to gold evidence, and Full improves over OpenPath.

```bash
python experiments/novelty_package/run.py --config experiments/harmful_path_audit/config.yaml
python experiments/harmful_path_audit/aggregate.py
```

The report is written to `outputs/analysis/report.md`. All conditions are
reported together to avoid selecting only favorable internal diagnostics.
