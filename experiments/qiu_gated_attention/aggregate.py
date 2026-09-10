#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WARRANT_RESULTS = ROOT / "experiments/qiu_gated_attention/outputs/results.csv"
QIU_RESULTS = ROOT / "experiments/qiu_gated_attention/outputs_lr1/results.csv"
OUT = ROOT / "experiments/qiu_gated_attention/analysis"
ORDER = ["base", "open_path_no_gate", "qiu_g2", "qiu_g1", "full_warrant"]
METRICS = [
    "support_mrr",
    "recall_at_1",
    "evidence_f1",
    "unsupported_at_gold_count",
    "auprc",
]


def main() -> None:
    warrant = pd.read_csv(WARRANT_RESULTS)
    qiu = pd.read_csv(QIU_RESULTS)
    frame = pd.concat(
        [
            warrant[warrant.variant.isin(["base", "open_path_no_gate", "full_warrant"])],
            qiu[qiu.variant.isin(["qiu_g2", "qiu_g1"])],
        ],
        ignore_index=True,
    )
    frame["variant"] = pd.Categorical(frame.variant, ORDER, ordered=True)
    frame = frame.sort_values(["variant", "seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "selected_runs.csv", index=False)

    rows = []
    for variant, group in frame.groupby("variant", observed=True, sort=False):
        row = {
            "variant": str(variant),
            "seeds": int(group.seed.nunique()),
            "total_parameters": int(group.total_parameters.iloc[0]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    pivot = frame.pivot(index="seed", columns="variant", values="support_mrr")
    deltas = pd.DataFrame({
        "seed": pivot.index,
        "full_minus_base": pivot["full_warrant"] - pivot["base"],
        "full_minus_open_path": pivot["full_warrant"] - pivot["open_path_no_gate"],
        "full_minus_qiu_g2": pivot["full_warrant"] - pivot["qiu_g2"],
        "full_minus_qiu_g1": pivot["full_warrant"] - pivot["qiu_g1"],
    })
    deltas.to_csv(OUT / "paired_mrr_deltas.csv", index=False)

    labels = {
        "base": "Base",
        "open_path_no_gate": "OpenPath",
        "qiu_g2": "Qiu G2",
        "qiu_g1": "Qiu G1",
        "full_warrant": "Full Warrant",
    }
    lines = [
        "# Qiu G1/G2 vs Warrant",
        "",
        "Qiu G1/G2 use the paper-style ordinary optimizer LR (gate multiplier 1). "
        "Warrant uses its established gate multiplier 10. All methods use the same "
        "HotpotQA/RoBERTa split, two epochs, and seeds 7/17/37/47/57.",
        "",
        "| Variant | Params | Support MRR | R@1 | Evidence F1 | Unsupported ↓ | AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def f(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}"
        lines.append(
            f"| {labels[row['variant']]} | {row['total_parameters']/1e6:.2f}M | "
            f"{f('support_mrr')} | {f('recall_at_1')} | {f('evidence_f1')} | "
            f"{f('unsupported_at_gold_count')} | {f('auprc')} |"
        )
    lines += ["", "## Paired Support-MRR deltas", ""]
    for column in deltas.columns[1:]:
        values = deltas[column]
        lines.append(
            f"- {column}: {values.mean():+.4f}±{values.std(ddof=1):.4f}; "
            f"positive seeds {int((values > 0).sum())}/5"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
