#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.common import PLOT_OUTPUT_DIR, save_figure, set_paper_style


DEFAULT_SUMMARY = (
    REPO_ROOT
    / "experiments"
    / "bert_hotpotqa_warrant"
    / "outputs"
    / "analysis"
    / "summary_means.csv"
)

VARIANT_ORDER = ["base", "attention_readout", "full_warrant"]
DISPLAY_NAMES = {
    "base": "Base",
    "attention_readout": "Attention\nreadout",
    "full_warrant": "Full\nWarrant",
}
PALETTE = {
    "base": "#7A869A",
    "attention_readout": "#5B8DB8",
    "full_warrant": "#1F9D8A",
}
RANKING_METRICS = {
    "support_mrr": "MRR",
    "recall_at_1": "R@1",
    "evidence_f1": "Evidence F1",
    "auprc": "AUPRC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RoBERTa HotpotQA Warrant main results.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=PLOT_OUTPUT_DIR)
    parser.add_argument("--stem", default="bert_hotpotqa_warrant_results")
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["variant"].isin(VARIANT_ORDER)].copy()
    frame["variant"] = pd.Categorical(frame["variant"], categories=VARIANT_ORDER, ordered=True)
    frame = frame.sort_values("variant")
    frame["variant_name"] = frame["variant"].astype(str).map(DISPLAY_NAMES)
    return frame


def ranking_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        for column, label in RANKING_METRICS.items():
            rows.append(
                {
                    "variant": str(row["variant"]),
                    "variant_name": row["variant_name"],
                    "metric": label,
                    "value": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def plot_results(frame: pd.DataFrame) -> plt.Figure:
    set_paper_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, 3.5),
        gridspec_kw={"width_ratios": [2.15, 1.0], "wspace": 0.28},
    )
    ax_rank, ax_failure = axes

    rank = ranking_long(frame)
    hue_order = [DISPLAY_NAMES[v] for v in VARIANT_ORDER]
    colors = [PALETTE[v] for v in VARIANT_ORDER]
    sns.barplot(
        data=rank,
        x="metric",
        y="value",
        hue="variant_name",
        hue_order=hue_order,
        palette=colors,
        ax=ax_rank,
        width=0.72,
        errorbar=None,
    )
    ax_rank.set_xlabel("")
    ax_rank.set_ylabel("Score")
    ax_rank.set_ylim(0.74, 0.93)
    ax_rank.set_yticks([0.75, 0.80, 0.85, 0.90])
    ax_rank.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax_rank.grid(axis="x", visible=False)
    ax_rank.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=3, title="")

    base = frame.set_index(frame["variant"].astype(str)).loc["base"]
    full = frame.set_index(frame["variant"].astype(str)).loc["full_warrant"]
    deltas = {
        "MRR": float(full["support_mrr"] - base["support_mrr"]),
        "R@1": float(full["recall_at_1"] - base["recall_at_1"]),
        "Evidence F1": float(full["evidence_f1"] - base["evidence_f1"]),
        "AUPRC": float(full["auprc"] - base["auprc"]),
    }
    metric_order = list(RANKING_METRICS.values())
    for idx, metric in enumerate(metric_order):
        best_value = float(rank.loc[rank["metric"] == metric, "value"].max())
        ax_rank.text(
            idx,
            min(0.928, best_value + 0.006),
            f"{deltas[metric]:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#0F766E",
        )

    failure_rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        failure_rows.extend(
            [
                {
                    "variant": str(row["variant"]),
                    "variant_name": row["variant_name"],
                    "metric": "Top-1 error",
                    "value": 1.0 - float(row["recall_at_1"]),
                },
                {
                    "variant": str(row["variant"]),
                    "variant_name": row["variant_name"],
                    "metric": "Unsupported\nevidence",
                    "value": float(row["unsupported_at_gold_count"]),
                },
            ]
        )
    failure = pd.DataFrame(failure_rows)
    sns.barplot(
        data=failure,
        x="metric",
        y="value",
        hue="variant_name",
        hue_order=hue_order,
        palette=colors,
        ax=ax_failure,
        width=0.72,
        errorbar=None,
        legend=False,
    )
    ax_failure.set_xlabel("")
    ax_failure.set_ylabel("Failure rate")
    ax_failure.set_ylim(0.145, 0.235)
    ax_failure.set_yticks([0.15, 0.18, 0.21, 0.23])
    ax_failure.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax_failure.grid(axis="x", visible=False)
    ax_failure.tick_params(axis="x", length=0)

    base_unsupported = float(base["unsupported_at_gold_count"])
    full_unsupported = float(full["unsupported_at_gold_count"])
    reduction = base_unsupported - full_unsupported
    rel_reduction = reduction / base_unsupported * 100.0
    base_top1_error = 1.0 - float(base["recall_at_1"])
    full_top1_error = 1.0 - float(full["recall_at_1"])
    top1_reduction = base_top1_error - full_top1_error
    top1_rel_reduction = top1_reduction / base_top1_error * 100.0
    ax_failure.text(
        0.0,
        0.232,
        f"-{top1_reduction:.4f}\n({top1_rel_reduction:.1f}% lower)",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#B91C1C",
    )
    ax_failure.text(
        1.0,
        0.232,
        f"-{reduction:.4f}\n({rel_reduction:.1f}% lower)",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#B91C1C",
    )

    fig.text(
        0.01,
        -0.015,
        "Full Warrant improves support ranking and reduces hallucination-adjacent failure signals: top-1 errors and unsupported evidence selected within the gold-count budget.",
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.32)
    return fig


def main() -> None:
    args = parse_args()
    frame = load_summary(args.summary)
    fig = plot_results(frame)
    png_path, pdf_path = save_figure(fig, args.stem, output_dir=args.output_dir)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
