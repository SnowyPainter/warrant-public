#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.common import PLOT_OUTPUT_DIR, save_figure, set_paper_style


DEFAULT_RAG_BASE = REPO_ROOT / "experiments/neural_dissection/outputs/rag/hotpotqa/LED/base/final_metrics.json"
DEFAULT_RAG_WARRANT = REPO_ROOT / "experiments/neural_dissection/outputs/rag/hotpotqa/LED/warrant/final_metrics.json"
DEFAULT_TKG_BASE = REPO_ROOT / "experiments/neural_dissection/outputs/tkg/icews18/xERTE/base/metrics_by_epoch.csv"
DEFAULT_TKG_WARRANT = REPO_ROOT / "experiments/neural_dissection/outputs/tkg/icews18/xERTE/warrant/metrics_by_epoch.csv"
DEFAULT_MASS_SUMMARY = REPO_ROOT / "experiments/mass_diagnostic/outputs/analysis/summary.csv"
DOMAIN_ORDER = ["ctdg", "rag", "tkg"]
DOMAIN_LABELS = {"ctdg": "CTDG", "rag": "RAG", "tkg": "TKG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot attention mass vs warranted contribution mass diagnostics.")
    parser.add_argument("--rag-base", type=Path, default=DEFAULT_RAG_BASE)
    parser.add_argument("--rag-warrant", type=Path, default=DEFAULT_RAG_WARRANT)
    parser.add_argument("--tkg-base", type=Path, default=DEFAULT_TKG_BASE)
    parser.add_argument("--tkg-warrant", type=Path, default=DEFAULT_TKG_WARRANT)
    parser.add_argument("--mass-summary", type=Path, default=DEFAULT_MASS_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=PLOT_OUTPUT_DIR)
    parser.add_argument("--stem", default="mass_separation")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_mass_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["variant"].eq("warrant")].copy()
    frame = frame[frame["domain"].isin(DOMAIN_ORDER)].copy()
    frame["domain"] = pd.Categorical(frame["domain"], categories=DOMAIN_ORDER, ordered=True)
    frame = frame.sort_values("domain")
    frame["domain_label"] = frame["domain"].astype(str).map(DOMAIN_LABELS)
    return frame


def plot_rag_mass(ax: plt.Axes, base: dict[str, float], warrant: dict[str, float]) -> None:
    frame = pd.DataFrame(
        [
            {
                "measure": "Support\nshare",
                "mass_type": "Attention",
                "value": warrant["support_attention_ratio"],
            },
            {
                "measure": "Support\nshare",
                "mass_type": "Warranted",
                "value": warrant["support_mass_ratio"],
            },
            {
                "measure": "Distractor\nmass",
                "mass_type": "Attention",
                "value": warrant["distractor_attention_mass"],
            },
            {
                "measure": "Distractor\nmass",
                "mass_type": "Warranted",
                "value": warrant["distractor_warrant_mass"],
            },
        ]
    )
    palette = {"Attention": "#9FB3C8", "Warranted": "#1F9D8A"}
    sns.barplot(
        data=frame,
        x="measure",
        y="value",
        hue="mass_type",
        palette=palette,
        ax=ax,
        width=0.68,
        errorbar=None,
    )
    ax.axhline(base["support_attention_ratio"], xmin=0.04, xmax=0.44, color="#7A869A", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.axhline(base["distractor_attention_mass"], xmin=0.56, xmax=0.96, color="#7A869A", linestyle=(0, (3, 3)), linewidth=1.0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    support_gain = warrant["support_mass_ratio"] - warrant["support_attention_ratio"]
    distractor_drop = warrant["distractor_warrant_mass"] - warrant["distractor_attention_mass"]
    ax.annotate(
        f"{support_gain:+.3f}",
        xy=(0.19, warrant["support_mass_ratio"]),
        xytext=(0.32, 0.36),
        arrowprops={"arrowstyle": "-|>", "color": "#0F766E", "linewidth": 1.0},
        ha="left",
        va="center",
        fontsize=8.5,
        color="#0F766E",
    )
    ax.annotate(
        f"{distractor_drop:+.3f}",
        xy=(1.19, warrant["distractor_warrant_mass"]),
        xytext=(1.30, 0.55),
        arrowprops={"arrowstyle": "-|>", "color": "#B91C1C", "linewidth": 1.0},
        ha="left",
        va="center",
        fontsize=8.5,
        color="#B91C1C",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mass / ratio")
    ax.set_ylim(0.0, 0.86)
    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 1.14), ncol=2, handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.text(0.02, 0.95, "RAG", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="top")
    ax.text(
        0.02,
        -0.22,
        "Attention relevance is nearly unchanged;\npermission reduces distractor contribution.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        ha="left",
        va="top",
    )


def plot_tkg_mass(ax: plt.Axes, base_path: Path, warrant_path: Path) -> None:
    base = pd.read_csv(base_path)
    warrant = pd.read_csv(warrant_path)
    ax.plot(base["epoch"], base["mrr"], marker="o", linewidth=1.8, markersize=3.5, color="#7A869A", label="Base MRR")
    ax.plot(warrant["epoch"], warrant["mrr"], marker="o", linewidth=1.8, markersize=3.5, color="#1F9D8A", label="Warrant MRR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MRR")
    ax.set_ylim(0.0, 0.082)
    ax.set_xlim(0.7, 10.55)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    twin = ax.twinx()
    twin.plot(
        warrant["epoch"],
        warrant["warrant_tail_mass_mean"],
        color="#D8A45D",
        linestyle=(0, (3, 2)),
        linewidth=1.6,
        label="Tail permission mass",
    )
    twin.set_ylabel("Tail permission mass")
    twin.set_ylim(0.0, 1.06)
    twin.tick_params(axis="y", labelsize=8)
    twin.spines["top"].set_visible(False)

    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="upper center", bbox_to_anchor=(0.58, 1.20), ncol=3, handlelength=1.6, columnspacing=0.85)

    final_base = float(base["mrr"].iloc[-1])
    final_warrant = float(warrant["mrr"].iloc[-1])
    final_mass = float(warrant["warrant_tail_mass_mean"].iloc[-1])
    ax.text(10.42, final_warrant + 0.001, f"{final_warrant:.3f}", color="#0F766E", fontsize=8.5, ha="right", va="bottom")
    ax.text(10.42, final_base + 0.001, f"{final_base:.3f}", color="#5E6C84", fontsize=8.5, ha="right", va="bottom")
    twin.text(10.42, final_mass - 0.055, f"{final_mass:.3f}", color="#9A6700", fontsize=8.5, ha="right", va="top")
    ax.text(0.02, 0.95, "TKG", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="top")
    ax.text(
        0.02,
        -0.22,
        "Permission opens the historical tail contribution path\nwhile ranking quality separates from the base model.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        ha="left",
        va="top",
    )


def plot_mass_diagnostic_mass(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary.melt(
        id_vars=["domain_label"],
        value_vars=["evidence_warrant_mass_mean", "non_evidence_warrant_mass_mean"],
        var_name="mass_type",
        value_name="mass",
    )
    frame["mass_type"] = frame["mass_type"].map(
        {
            "evidence_warrant_mass_mean": "Evidence",
            "non_evidence_warrant_mass_mean": "Non-evidence",
        }
    )
    palette = {"Evidence": "#1F9D8A", "Non-evidence": "#C44E52"}
    sns.barplot(
        data=frame,
        x="domain_label",
        y="mass",
        hue="mass_type",
        palette=palette,
        ax=ax,
        width=0.72,
        errorbar=None,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7.7)

    ax.set_xlabel("")
    ax.set_ylabel("Warrant mass")
    ax.set_ylim(0.0, max(1.55, float(frame["mass"].max()) * 1.14))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2, handlelength=1.2, columnspacing=0.9)
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.text(
        0.02,
        0.95,
        "Mass diagnostic",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )
    ax.text(
        0.02,
        -0.22,
        "Localized Warrant paths carry labeled evidence\nwhile retaining domain-specific context mass.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        ha="left",
        va="top",
    )


def plot_counterfactual_drops(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary.melt(
        id_vars=["domain_label"],
        value_vars=["metric_drop_zero_evidence_mean", "metric_drop_zero_non_evidence_mean"],
        var_name="intervention",
        value_name="drop",
    )
    frame["intervention"] = frame["intervention"].map(
        {
            "metric_drop_zero_evidence_mean": "Zero evidence",
            "metric_drop_zero_non_evidence_mean": "Zero non-evidence",
        }
    )
    palette = {"Zero evidence": "#1F9D8A", "Zero non-evidence": "#C44E52"}
    sns.barplot(
        data=frame,
        x="domain_label",
        y="drop",
        hue="intervention",
        palette=palette,
        ax=ax,
        width=0.72,
        errorbar=None,
    )
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    for container in ax.containers:
        labels = [f"{value.get_height():+.3f}" for value in container]
        ax.bar_label(container, labels=labels, padding=2, fontsize=7.7)

    limit = max(abs(float(frame["drop"].min())), abs(float(frame["drop"].max()))) * 1.22
    ax.set_ylim(-max(0.26, limit), max(0.14, limit * 0.62))
    ax.set_xlabel("")
    ax.set_ylabel("Metric drop")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2, handlelength=1.2, columnspacing=0.9)
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.text(
        0.02,
        0.95,
        "Counterfactual intervention",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )
    ax.text(
        0.02,
        -0.22,
        "Positive evidence drop means useful evidence mass;\nnegative non-evidence drop means distractor contribution.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        ha="left",
        va="top",
    )


def main() -> None:
    args = parse_args()
    set_paper_style()
    rag_base = load_json(args.rag_base)
    rag_warrant = load_json(args.rag_warrant)
    mass_summary = load_mass_summary(args.mass_summary)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.4, 7.1),
        gridspec_kw={"width_ratios": [1.0, 1.1], "height_ratios": [1.02, 1.0]},
    )
    plot_rag_mass(axes[0, 0], rag_base, rag_warrant)
    plot_tkg_mass(axes[0, 1], args.tkg_base, args.tkg_warrant)
    plot_mass_diagnostic_mass(axes[1, 0], mass_summary)
    plot_counterfactual_drops(axes[1, 1], mass_summary)
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.08, right=0.98, hspace=0.67, wspace=0.38)
    png_path, pdf_path = save_figure(fig, args.stem, output_dir=args.output_dir)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
