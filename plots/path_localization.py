#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.common import PLOT_OUTPUT_DIR, save_figure, set_paper_style


DEFAULT_RESULTS = REPO_ROOT / "experiments" / "path_localization" / "outputs" / "results.csv"
DOMAIN_ORDER = ["ctdg", "mtpp", "rag", "stpp", "tkg"]
CHECKS = [
    ("gain_vs_base", "Correct\n> Base"),
    ("beats_generic", "Correct\n> Generic"),
    ("beats_shuffled", "Correct\n> Shuffled"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot path-localization verdict checks.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=PLOT_OUTPUT_DIR)
    parser.add_argument("--stem", default="path_localization_checks")
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["status"].eq("completed")].copy()
    frame["primary_metric"] = pd.to_numeric(frame["primary_metric"], errors="coerce")
    return frame


def direction_sign(direction: str) -> float:
    return -1.0 if str(direction).lower() == "lower" else 1.0


def compute_checks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (domain, seed), subset in frame.groupby(["domain", "seed"]):
        by_variant = {row["variant"]: row for _, row in subset.iterrows()}
        correct = by_variant.get("correct_path_warrant")
        if correct is None:
            continue
        sign = direction_sign(correct.get("direction", "higher"))

        def delta_against(variant: str) -> float:
            other = by_variant.get(variant)
            if other is None:
                return float("nan")
            return sign * (float(correct["primary_metric"]) - float(other["primary_metric"]))

        rows.append(
            {
                "domain": domain,
                "seed": int(seed),
                "gain_vs_base": delta_against("base"),
                "beats_generic": delta_against("generic_qk_warrant"),
                "beats_shuffled": delta_against("shuffled_pairing"),
            }
        )

    paired = pd.DataFrame(rows)
    summary_rows = []
    for domain, subset in paired.groupby("domain"):
        row = {"domain": domain}
        for key, _ in CHECKS:
            values = pd.to_numeric(subset[key], errors="coerce").dropna()
            row[f"{key}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{key}_support"] = int((values > 0).sum()) if len(values) else 0
            row[f"{key}_n"] = int(len(values))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    present = [domain for domain in DOMAIN_ORDER if domain in set(summary["domain"])]
    summary = summary.set_index("domain").reindex(present).reset_index()

    score = pd.DataFrame(index=[domain.upper() for domain in summary["domain"]])
    delta = pd.DataFrame(index=[domain.upper() for domain in summary["domain"]])
    support = pd.DataFrame(index=[domain.upper() for domain in summary["domain"]])
    for key, label in CHECKS:
        score[label] = summary.apply(lambda row: verdict_score(row, key), axis=1).to_numpy()
        delta[label] = summary[f"{key}_mean"].to_numpy()
        support[label] = summary.apply(lambda row: f"{int(row[f'{key}_support'])}/{int(row[f'{key}_n'])}", axis=1).to_numpy()
    return score, delta, support


def verdict_score(row: pd.Series, key: str) -> float:
    n = int(row.get(f"{key}_n", 0))
    if n == 0 or not math.isfinite(float(row.get(f"{key}_mean", float("nan")))):
        return 0.0
    support = int(row.get(f"{key}_support", 0))
    if support == n and float(row[f"{key}_mean"]) > 0:
        return 1.0
    if support == 0 and float(row[f"{key}_mean"]) <= 0:
        return -1.0
    return 0.45 if float(row[f"{key}_mean"]) > 0 else -0.45


def fmt_delta(value: float) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):+.3f}"


def plot_checks(score: pd.DataFrame, delta: pd.DataFrame, support: pd.DataFrame) -> plt.Figure:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.45))
    cmap = ListedColormap(["#C44E52", "#F3F4F6", "#1F9D8A"])

    sns.heatmap(
        score,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cbar=False,
        linewidths=1.4,
        linecolor="white",
        annot=False,
    )

    for i, domain in enumerate(score.index):
        for j, check in enumerate(score.columns):
            value = float(score.iat[i, j])
            delta_text = fmt_delta(float(delta.iat[i, j]))
            support_text = str(support.iat[i, j])
            if value > 0:
                mark = "✓"
                color = "white" if value >= 1.0 else "#0F766E"
                weight = "bold"
            elif value < 0:
                mark = "×"
                color = "white" if value <= -1.0 else "#B91C1C"
                weight = "bold"
            else:
                mark = "n/a"
                color = "#6B7280"
                weight = "regular"
            ax.text(
                j + 0.5,
                i + 0.42,
                mark,
                ha="center",
                va="center",
                color=color,
                fontsize=14 if mark in {"✓", "×"} else 9,
                fontweight=weight,
            )
            ax.text(
                j + 0.5,
                i + 0.68,
                f"{delta_text} ({support_text})",
                ha="center",
                va="center",
                color=color,
                fontsize=7.6,
                fontweight="regular",
            )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9.5, fontweight="semibold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight="semibold")
    ax.tick_params(axis="both", length=0)

    fig.subplots_adjust(bottom=0.22)
    return fig


def main() -> None:
    args = parse_args()
    frame = load_results(args.results)
    score, delta, support = compute_checks(frame)
    fig = plot_checks(score, delta, support)
    png_path, pdf_path = save_figure(fig, args.stem, output_dir=args.output_dir)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
