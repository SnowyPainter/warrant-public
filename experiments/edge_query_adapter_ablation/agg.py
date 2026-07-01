#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path(__file__).resolve().parent / "outputs" / "results.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "analysis"
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
    "generic_qk_warrant": "Generic q-k Warrant",
    "param_control": "Param control",
    "handcrafted_only_control": "Hand-crafted-only control",
    "edge_conditioned_warrant": "Edge-conditioned Warrant",
    "shuffled_edge_query": "Shuffled edge query",
}
METRICS = ["auc", "accuracy", "eval_loss", "edge_warrant_gate_mean", "edge_warrant_attention_entropy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate edge-query adapter ablation results.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "completed":
                continue
            parsed: dict[str, Any] = dict(row)
            parsed["seed"] = int(row["seed"])
            for metric in METRICS + ["primary_metric"]:
                parsed[metric] = parse_float(row.get(metric, ""))
            rows.append(parsed)
    return rows


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = finite(values)
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    if not math.isfinite(mean):
        return ""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def variant_sort_key(variant: str) -> tuple[int, str]:
    try:
        return VARIANT_ORDER.index(variant), variant
    except ValueError:
        return len(VARIANT_ORDER), variant


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)

    base_by_seed = {row["seed"]: row for row in by_variant.get("base", [])}
    summary: list[dict[str, Any]] = []
    paired_deltas: list[dict[str, Any]] = []

    for variant in sorted(by_variant, key=variant_sort_key):
        variant_rows = sorted(by_variant[variant], key=lambda row: row["seed"])
        auc_values = [float(row["auc"]) for row in variant_rows]
        acc_values = [float(row["accuracy"]) for row in variant_rows]
        loss_values = [float(row["eval_loss"]) for row in variant_rows]
        auc_mean, auc_std = mean_std(auc_values)
        acc_mean, acc_std = mean_std(acc_values)
        loss_mean, loss_std = mean_std(loss_values)

        deltas = []
        acc_deltas = []
        for row in variant_rows:
            base = base_by_seed.get(row["seed"])
            if base is None:
                continue
            auc_delta = float(row["auc"]) - float(base["auc"])
            acc_delta = float(row["accuracy"]) - float(base["accuracy"])
            deltas.append(auc_delta)
            acc_deltas.append(acc_delta)
            paired_deltas.append(
                {
                    "variant": variant,
                    "seed": row["seed"],
                    "auc": row["auc"],
                    "base_auc": base["auc"],
                    "delta_auc": auc_delta,
                    "accuracy": row["accuracy"],
                    "base_accuracy": base["accuracy"],
                    "delta_accuracy": acc_delta,
                }
            )
        delta_mean, delta_std = mean_std(deltas)
        acc_delta_mean, acc_delta_std = mean_std(acc_deltas)

        edge_gate_mean, edge_gate_std = mean_std([float(row["edge_warrant_gate_mean"]) for row in variant_rows])
        edge_entropy_mean, edge_entropy_std = mean_std([float(row["edge_warrant_attention_entropy"]) for row in variant_rows])
        summary.append(
            {
                "variant": variant,
                "display_name": DISPLAY_NAMES.get(variant, variant),
                "seeds": len(variant_rows),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "delta_auc_mean": delta_mean,
                "delta_auc_std": delta_std,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "delta_accuracy_mean": acc_delta_mean,
                "delta_accuracy_std": acc_delta_std,
                "eval_loss_mean": loss_mean,
                "eval_loss_std": loss_std,
                "edge_warrant_gate_mean": edge_gate_mean,
                "edge_warrant_gate_std": edge_gate_std,
                "edge_warrant_attention_entropy_mean": edge_entropy_mean,
                "edge_warrant_attention_entropy_std": edge_entropy_std,
            }
        )
    return summary, paired_deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]], paired_deltas: list[dict[str, Any]]) -> None:
    lines = [
        "# Edge-Query Adapter Ablation",
        "",
        "Results are aggregated over completed seeds. AUC and accuracy are reported as mean ± sample standard deviation.",
        "",
        "| Variant | Seeds | AUC | Delta AUC vs Base | Accuracy | Delta Accuracy vs Base | Eval Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {name} | {seeds} | {auc} | {delta_auc} | {acc} | {delta_acc} | {loss} |".format(
                name=row["display_name"],
                seeds=row["seeds"],
                auc=fmt_pm(row["auc_mean"], row["auc_std"]),
                delta_auc=fmt_pm(row["delta_auc_mean"], row["delta_auc_std"]),
                acc=fmt_pm(row["accuracy_mean"], row["accuracy_std"]),
                delta_acc=fmt_pm(row["delta_accuracy_mean"], row["delta_accuracy_std"]),
                loss=fmt_pm(row["eval_loss_mean"], row["eval_loss_std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Seed Deltas",
            "",
            "| Variant | Seed | AUC | Base AUC | Delta AUC | Accuracy | Base Accuracy | Delta Accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(paired_deltas, key=lambda item: (variant_sort_key(str(item["variant"])), int(item["seed"]))):
        lines.append(
            "| {name} | {seed} | {auc} | {base_auc} | {delta_auc} | {acc} | {base_acc} | {delta_acc} |".format(
                name=DISPLAY_NAMES.get(str(row["variant"]), str(row["variant"])),
                seed=row["seed"],
                auc=fmt(float(row["auc"])),
                base_auc=fmt(float(row["base_auc"])),
                delta_auc=fmt(float(row["delta_auc"]), 4),
                acc=fmt(float(row["accuracy"])),
                base_acc=fmt(float(row["base_accuracy"])),
                delta_acc=fmt(float(row["delta_accuracy"]), 4),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    summary, paired_deltas = aggregate(rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_seed_deltas.csv", paired_deltas)
    write_markdown(output_dir / "summary.md", summary, paired_deltas)
    print(f"wrote {output_dir / 'summary.csv'}")
    print(f"wrote {output_dir / 'paired_seed_deltas.csv'}")
    print(f"wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
