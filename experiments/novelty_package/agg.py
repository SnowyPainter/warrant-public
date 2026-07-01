#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "novelty_package" / "outputs" / "results.csv"
OUT = ROOT / "experiments" / "novelty_package" / "outputs" / "analysis"

VARIANT_ORDER = [
    "base",
    "post_mlp",
    "post_attn_glu",
    "attention_readout",
    "query_only_gate",
    "shuffled_warrant",
    "open_path_no_gate",
    "full_warrant",
]

DISPLAY = {
    "base": "Base",
    "post_mlp": "Param-MLP",
    "post_attn_glu": "Post-attention GLU",
    "attention_readout": "Attention-readout",
    "query_only_gate": "Query-only gate",
    "shuffled_warrant": "Shuffled Warrant",
    "open_path_no_gate": "OpenPath-NoGate",
    "full_warrant": "Full Warrant",
}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return ""
    return f"{mean:.4f} ± {std:.4f}"


def fmt_delta(value: float, *, lower_is_better: bool = False) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < 0.00005:
        return f"{value:+.4f}"
    good = value < 0 if lower_is_better else value > 0
    arrow = "↑" if good and not lower_is_better else "↓" if good else "↓" if not lower_is_better else "↑"
    return f"{value:+.4f} {arrow}"


def pct_delta(new: float, old: float, *, lower_is_better: bool = False) -> str:
    if not np.isfinite(new) or not np.isfinite(old) or abs(old) < 1.0e-12:
        return ""
    raw = (new - old) / abs(old) * 100.0
    improved = raw < 0 if lower_is_better else raw > 0
    signed = -raw if lower_is_better else raw
    marker = "improved" if improved else "worse"
    return f"{signed:+.2f}% {marker}"


def sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    order = {variant: idx for idx, variant in enumerate(VARIANT_ORDER)}
    frame = frame.copy()
    frame["_order"] = frame["variant"].map(order).fillna(999).astype(int)
    return frame.sort_values(["_order", "variant"]).drop(columns=["_order"])


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}")
    frame = pd.read_csv(RESULTS)
    completed = frame[frame["status"].eq("completed")].copy()
    if completed.empty:
        raise SystemExit("no completed novelty rows")

    metrics = [
        "support_mrr",
        "recall_at_1",
        "recall_at_2",
        "evidence_f1",
        "auprc",
        "unsupported_at_gold_count",
        "precision_at_gold_count",
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

    summary_rows: list[dict[str, object]] = []
    mean_rows: list[dict[str, object]] = []
    for variant, group in completed.groupby("variant"):
        row: dict[str, object] = {"variant": variant, "display": DISPLAY.get(variant, variant), "seeds": group["seed"].nunique()}
        mean_row: dict[str, object] = {"variant": variant, "display": DISPLAY.get(variant, variant), "seeds": group["seed"].nunique()}
        for metric in metrics:
            values = numeric(group, metric).dropna()
            row[metric] = fmt(float(values.mean()), float(values.std(ddof=0))) if not values.empty else ""
            mean_row[metric] = float(values.mean()) if not values.empty else float("nan")
        summary_rows.append(row)
        mean_rows.append(mean_row)

    summary = sort_rows(pd.DataFrame(summary_rows))
    means = sort_rows(pd.DataFrame(mean_rows))
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    means.to_csv(OUT / "summary_means.csv", index=False)

    base = means[means["variant"].eq("base")].iloc[0] if "base" in set(means["variant"]) else None
    full = means[means["variant"].eq("full_warrant")].iloc[0] if "full_warrant" in set(means["variant"]) else None

    lines: list[str] = [
        "# Novelty Package Summary",
        "",
        "이 실험은 HotpotQA multi-candidate RoBERTa 설정에서 gate가 곱해지는 계산 단위를 비교한다.",
        "모든 variant는 같은 입력, 같은 candidate marker readout, 같은 train/eval split을 사용한다.",
    ]

    lines.extend(
        [
            "",
            "## Main Control Table",
            "",
            "| Variant | Seeds | MRR | R@1 | F1 | AUPRC | Unsupported@GoldCount ↓ | Precision@GoldCount ↑ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["display"]),
                    str(row["seeds"]),
                    str(row["support_mrr"]),
                    str(row["recall_at_1"]),
                    str(row["evidence_f1"]),
                    str(row["auprc"]),
                    str(row["unsupported_at_gold_count"]),
                    str(row["precision_at_gold_count"]),
                ]
            )
            + " |"
        )

    if base is not None and full is not None:
        lines.extend(
            [
                "",
                "## Full Warrant vs Base",
                "",
                "| Metric | Base | Full Warrant | Absolute Δ | Relative change |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        comparisons = [
            ("MRR", "support_mrr", False),
            ("R@1", "recall_at_1", False),
            ("Evidence F1", "evidence_f1", False),
            ("AUPRC", "auprc", False),
            ("Unsupported@GoldCount", "unsupported_at_gold_count", True),
            ("Precision@GoldCount", "precision_at_gold_count", False),
        ]
        for label, metric, lower in comparisons:
            b = float(base[metric])
            f = float(full[metric])
            lines.append(
                f"| {label} | {b:.4f} | {f:.4f} | {fmt_delta(f - b, lower_is_better=lower)} | {pct_delta(f, b, lower_is_better=lower)} |"
            )

    lines.extend(
        [
            "",
            "## Permission Diagnostics",
            "",
            "| Variant | Gate μ | Gate σ | Gold g | Random g | Gold/Random α | Gold/Random αg |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["display"]),
                    str(row["candidate_gate_mean"]),
                    str(row["candidate_gate_std"]),
                    str(row["gold_gate_mean"]),
                    str(row["random_gate_mean"]),
                    str(row["gold_random_attention_ratio"]),
                    str(row["gold_random_effective_ratio"]),
                ]
            )
            + " |"
        )

    synthetic_path = ROOT / "experiments" / "novelty_package" / "outputs" / "synthetic" / "same_aggregate_results.csv"
    if synthetic_path.exists():
        synthetic = pd.read_csv(synthetic_path)
        lines.extend(
            [
                "",
                "## Same-Aggregate Constructive Check",
                "",
                "| Variant | Item identity available | Attention renormalized | Expected accuracy | Interpretation |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for _, row in synthetic.iterrows():
            lines.append(
                f"| {row['variant']} | {row['item_identity_available']} | {row['attention_renormalized']} | {float(row['expected_accuracy']):.1f} | {row['interpretation']} |"
            )

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
