#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "bert_hotpotqa_warrant" / "outputs" / "results.csv"
OUT = ROOT / "experiments" / "bert_hotpotqa_warrant" / "outputs" / "analysis"

VARIANT_ORDER = [
    "base",
    "qiu_g2",
    "qiu_g1",
    "post_mlp",
    "query_only_gate",
    "shuffled_warrant",
    "full_warrant",
]


def fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return ""
    return f"{mean:.4f} ± {std:.4f}"


def fmt_delta(value: float, *, lower_is_better: bool = False) -> str:
    if not np.isfinite(value):
        return ""
    arrow_good = "↓" if lower_is_better else "↑"
    arrow_bad = "↑" if lower_is_better else "↓"
    if abs(value) < 0.00005:
        return f"{value:+.4f}"
    arrow = arrow_good if value < 0 and lower_is_better else arrow_good if value > 0 and not lower_is_better else arrow_bad
    return f"{value:+.4f} {arrow}"


def summarize_metric(group: pd.DataFrame, metric: str) -> str:
    if metric not in group.columns:
        return ""
    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    if values.empty:
        return ""
    return fmt(float(values.mean()), float(values.std(ddof=0)))


def mean_metric(group: pd.DataFrame, metric: str) -> float:
    if metric not in group.columns:
        return float("nan")
    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def add_table(lines: list[str], title: str, headers: list[str], rows: list[list[object]]) -> None:
    lines.extend(["", f"## {title}", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")


def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    order = {variant: idx for idx, variant in enumerate(VARIANT_ORDER)}
    summary = summary.copy()
    summary["_order"] = summary["variant"].map(order).fillna(999).astype(int)
    return summary.sort_values(["_order", "variant"]).drop(columns=["_order"])


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}")

    frame = pd.read_csv(RESULTS)
    completed = frame[frame["status"].eq("completed")].copy()

    if completed.empty:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.md").write_text(
            "# BERT HotpotQA Warrant Summary\n\n_No completed rows._\n",
            encoding="utf-8",
        )
        print(OUT / "summary.md")
        return

    metrics = [
        # Ranking
        "support_mrr",
        "clean_support_mrr",
        "recall_at_1",
        "clean_recall_at_1",
        "recall_at_2",
        "evidence_f1",
        "auprc",

        # Evidence misattribution
        "unsupported_at_gold_count",
        "precision_at_gold_count",

        # Warrant diagnostics
        "candidate_gate_mean",
        "candidate_gate_std",
        "gold_gate_mean",
        "random_gate_mean",
        "gold_attention_mass",
        "random_attention_mass",
        "gold_effective_mass",
        "random_effective_mass",
        "gold_random_attention_ratio",
        "gold_random_effective_ratio",
    ]

    rows = []
    mean_rows = []
    for variant, group in completed.groupby("variant"):
        row: dict[str, object] = {
            "variant": variant,
            "seeds": int(group["seed"].nunique()),
        }
        mean_row: dict[str, object] = {
            "variant": variant,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[metric] = summarize_metric(group, metric)
            mean_row[metric] = mean_metric(group, metric)
        rows.append(row)
        mean_rows.append(mean_row)

    summary = sort_summary(pd.DataFrame(rows))
    means = sort_summary(pd.DataFrame(mean_rows))

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    means.to_csv(OUT / "summary_means.csv", index=False)

    lines: list[str] = ["# BERT HotpotQA Warrant Summary"]

    ranking_rows = []
    for _, row in summary.iterrows():
        ranking_rows.append(
            [
                row["variant"],
                row["seeds"],
                row["support_mrr"],
                row["clean_support_mrr"],
                row["recall_at_1"],
                row["clean_recall_at_1"],
                row["recall_at_2"],
                row["evidence_f1"],
                row["auprc"],
            ]
        )

    add_table(
        lines,
        "Ranking",
        ["Variant", "Seeds", "MRR", "Clean MRR", "R@1", "Clean R@1", "R@2", "F1", "AUPRC"],
        ranking_rows,
    )

    misattrib_rows = []
    for _, row in summary.iterrows():
        misattrib_rows.append(
            [
                row["variant"],
                row["seeds"],
                row["unsupported_at_gold_count"],
                row["precision_at_gold_count"],
            ]
        )

    add_table(
        lines,
        "Evidence Misattribution",
        [
            "Variant",
            "Seeds",
            "Unsupported@GoldCount ↓",
            "Precision@GoldCount ↑",
        ],
        misattrib_rows,
    )

    # Optional but useful: direct full_warrant - base comparison.
    if "base" in set(means["variant"]) and "full_warrant" in set(means["variant"]):
        base = means[means["variant"].eq("base")].iloc[0]
        full = means[means["variant"].eq("full_warrant")].iloc[0]

        delta_rows = [
            ["MRR", fmt_delta(float(full["support_mrr"]) - float(base["support_mrr"]))],
            ["R@1", fmt_delta(float(full["recall_at_1"]) - float(base["recall_at_1"]))],
            [
                "Unsupported@GoldCount",
                fmt_delta(
                    float(full["unsupported_at_gold_count"]) - float(base["unsupported_at_gold_count"]),
                    lower_is_better=True,
                ),
            ],
            [
                "Precision@GoldCount",
                fmt_delta(float(full["precision_at_gold_count"]) - float(base["precision_at_gold_count"])),
            ],
        ]

        add_table(
            lines,
            "Full Warrant - Base Delta",
            ["Metric", "Delta"],
            delta_rows,
        )

    warrant_rows = []
    for _, row in summary.iterrows():
        warrant_rows.append(
            [
                row["variant"],
                row["seeds"],
                row["candidate_gate_mean"],
                row["candidate_gate_std"],
                row["gold_gate_mean"],
                row["random_gate_mean"],
                row["gold_attention_mass"],
                row["random_attention_mass"],
                row["gold_effective_mass"],
                row["random_effective_mass"],
                row["gold_random_attention_ratio"],
                row["gold_random_effective_ratio"],
            ]
        )

    add_table(
        lines,
        "Warrant Diagnostics",
        [
            "Variant",
            "Seeds",
            "Gate μ",
            "Gate σ",
            "Gold g",
            "Random g",
            "Gold α",
            "Random α",
            "Gold αg",
            "Random αg",
            "Gold/Random α",
            "Gold/Random αg",
        ],
        warrant_rows,
    )

    mass_rows = []
    for _, row in summary.iterrows():
        mass_rows.append(
            [
                row["variant"],
                row["seeds"],
                row["gold_attention_mass"],
                row["random_attention_mass"],
                row["gold_effective_mass"],
                row["random_effective_mass"],
            ]
        )

    add_table(
        lines,
        "Mass Means",
        [
            "Variant",
            "Seeds",
            "Gold α",
            "Random α",
            "Gold αg",
            "Random αg",
        ],
        mass_rows,
    )

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
