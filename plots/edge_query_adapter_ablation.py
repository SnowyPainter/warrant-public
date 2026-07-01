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


DEFAULT_SUMMARY = REPO_ROOT / "experiments" / "edge_query_adapter_ablation" / "outputs" / "analysis" / "summary.csv"
VARIANT_ORDER = [
    "base",
    "generic_qk_warrant",
    "param_control",
    "handcrafted_only_control",
    "edge_conditioned_warrant",
    "shuffled_edge_query",
]
DISPLAY_NAMES = {
    "base": "Base",
    "generic_qk_warrant": "Generic\nq-k",
    "param_control": "Param\ncontrol",
    "handcrafted_only_control": "Hand-crafted\ncontrol",
    "edge_conditioned_warrant": "Edge-conditioned\nWarrant",
    "shuffled_edge_query": "Shuffled\nedge query",
}
PALETTE = {
    "base": "#7A869A",
    "generic_qk_warrant": "#9FB3C8",
    "param_control": "#B8A88A",
    "handcrafted_only_control": "#D8A45D",
    "edge_conditioned_warrant": "#1F9D8A",
    "shuffled_edge_query": "#C44E52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the CTDG edge-query adapter ablation.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=PLOT_OUTPUT_DIR)
    parser.add_argument("--stem", default="edge_query_adapter_ablation")
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["variant"].isin(VARIANT_ORDER)].copy()
    frame["variant"] = pd.Categorical(frame["variant"], categories=VARIANT_ORDER, ordered=True)
    frame = frame.sort_values("variant")
    frame["plot_name"] = frame["variant"].map(DISPLAY_NAMES)
    return frame


def plot_auc(frame: pd.DataFrame) -> plt.Figure:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [PALETTE[str(variant)] for variant in frame["variant"]]

    sns.barplot(
        data=frame,
        x="plot_name",
        y="auc_mean",
        hue="plot_name",
        order=[DISPLAY_NAMES[variant] for variant in VARIANT_ORDER],
        hue_order=[DISPLAY_NAMES[variant] for variant in VARIANT_ORDER],
        palette=colors,
        ax=ax,
        width=0.72,
        errorbar=None,
        legend=False,
    )

    x_positions = list(range(len(frame)))
    ax.errorbar(
        x=x_positions,
        y=frame["auc_mean"],
        yerr=frame["auc_std"],
        fmt="none",
        ecolor="#1F2937",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=3,
    )

    base_auc = float(frame.loc[frame["variant"].astype(str) == "base", "auc_mean"].iloc[0])
    ax.axhline(base_auc, color="#7A869A", linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.75)
    ax.text(
        len(frame) - 0.45,
        base_auc + 0.004,
        "Base mean",
        ha="right",
        va="bottom",
        color="#5E6C84",
        fontsize=8.5,
    )

    for idx, row in enumerate(frame.itertuples(index=False)):
        auc = float(row.auc_mean)
        delta = float(row.delta_auc_mean)
        ax.text(idx, auc + 0.011, f"{auc:.3f}", ha="center", va="bottom", fontsize=8.5, color="#111827")
        if str(row.variant) != "base":
            delta_color = "#0F766E" if delta >= 0 else "#B91C1C"
            ax.text(idx, auc - 0.035, f"{delta:+.3f}", ha="center", va="top", fontsize=8, color=delta_color)

    ax.set_xlabel("")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.58, 0.93)
    ax.set_yticks([0.60, 0.70, 0.80, 0.90])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    frame = load_summary(args.summary)
    fig = plot_auc(frame)
    png_path, pdf_path = save_figure(fig, args.stem, output_dir=args.output_dir)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
