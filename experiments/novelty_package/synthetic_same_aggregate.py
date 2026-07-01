#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "novelty_package" / "outputs" / "synthetic"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "variant": "post_attention_gate",
            "input_visible_to_gate": "h = 0.5u + 0.5(-u) = 0",
            "item_identity_available": False,
            "attention_renormalized": False,
            "expected_accuracy": 0.5,
            "interpretation": "aggregate gate sees the same zero vector for both labels",
        },
        {
            "variant": "attention_logit_gate",
            "input_visible_to_gate": "softmax relevance logits",
            "item_identity_available": True,
            "attention_renormalized": True,
            "expected_accuracy": 1.0,
            "interpretation": "can separate items, but does so by redefining attention mass",
        },
        {
            "variant": "warrant_value_term_gate",
            "input_visible_to_gate": "query-item pair before value aggregation",
            "item_identity_available": True,
            "attention_renormalized": False,
            "expected_accuracy": 1.0,
            "interpretation": "keeps alpha fixed and changes item-wise weighted value contribution",
        },
    ]
    with (OUT / "same_aggregate_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "same_aggregate_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(OUT / "same_aggregate_results.csv")


if __name__ == "__main__":
    main()
