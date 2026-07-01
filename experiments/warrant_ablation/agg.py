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
    "attention_copy_control",
    "logit_gate",
    "key_only_gate",
    "query_only_gate",
    "shuffled_gate",
    "full_value_gate",
]
DISPLAY_NAMES = {
    "base": "Base",
    "attention_copy_control": "Attention-copy control",
    "logit_gate": "Logit gate",
    "key_only_gate": "Key-only gate",
    "query_only_gate": "Query-only gate",
    "shuffled_gate": "Shuffled gate",
    "full_value_gate": "Full value gate",
}
METRICS = ["mrr", "accuracy", "eval_loss", "gate_mean", "warrant_logit_mean", "warrant_tail_mass_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TKG Warrant operator ablation results.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


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
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


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
    full_by_seed = {row["seed"]: row for row in by_variant.get("full_value_gate", [])}
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []

    for variant in sorted(by_variant, key=variant_sort_key):
        variant_rows = sorted(by_variant[variant], key=lambda row: row["seed"])
        mrr_mean, mrr_std = mean_std([float(row["mrr"]) for row in variant_rows])
        acc_mean, acc_std = mean_std([float(row["accuracy"]) for row in variant_rows])
        loss_mean, loss_std = mean_std([float(row["eval_loss"]) for row in variant_rows])
        gate_mean, gate_std = mean_std([float(row["gate_mean"]) for row in variant_rows])
        tail_mass_mean, tail_mass_std = mean_std([float(row["warrant_tail_mass_mean"]) for row in variant_rows])

        deltas_vs_base = []
        deltas_vs_full = []
        for row in variant_rows:
            base = base_by_seed.get(row["seed"])
            full = full_by_seed.get(row["seed"])
            if base is not None:
                deltas_vs_base.append(float(row["mrr"]) - float(base["mrr"]))
            if full is not None:
                deltas_vs_full.append(float(full["mrr"]) - float(row["mrr"]))
            paired.append(
                {
                    "variant": variant,
                    "seed": row["seed"],
                    "mrr": row["mrr"],
                    "base_mrr": float("nan") if base is None else base["mrr"],
                    "delta_mrr_vs_base": float("nan") if base is None else float(row["mrr"]) - float(base["mrr"]),
                    "full_value_gate_mrr": float("nan") if full is None else full["mrr"],
                    "full_minus_variant_mrr": float("nan") if full is None else float(full["mrr"]) - float(row["mrr"]),
                }
            )

        delta_base_mean, delta_base_std = mean_std(deltas_vs_base)
        full_gap_mean, full_gap_std = mean_std(deltas_vs_full)
        summary.append(
            {
                "variant": variant,
                "display_name": DISPLAY_NAMES.get(variant, variant),
                "seeds": len(variant_rows),
                "mrr_mean": mrr_mean,
                "mrr_std": mrr_std,
                "delta_mrr_vs_base_mean": delta_base_mean,
                "delta_mrr_vs_base_std": delta_base_std,
                "full_minus_variant_mrr_mean": full_gap_mean,
                "full_minus_variant_mrr_std": full_gap_std,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "eval_loss_mean": loss_mean,
                "eval_loss_std": loss_std,
                "gate_mean": gate_mean,
                "gate_std": gate_std,
                "warrant_tail_mass_mean": tail_mass_mean,
                "warrant_tail_mass_std": tail_mass_std,
            }
        )
    return summary, paired


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    lines = [
        "# TKG Warrant Operator Ablation",
        "",
        "Default setting: GDELT / xERTE. MRR is higher-is-better.",
        "",
        "| Variant | Seeds | MRR | Delta MRR vs Base | Full - Variant MRR | Accuracy | Eval Loss | Gate Mean | Tail Mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {name} | {seeds} | {mrr} | {delta} | {gap} | {acc} | {loss} | {gate} | {tail} |".format(
                name=row["display_name"],
                seeds=row["seeds"],
                mrr=fmt_pm(row["mrr_mean"], row["mrr_std"]),
                delta=fmt_pm(row["delta_mrr_vs_base_mean"], row["delta_mrr_vs_base_std"]),
                gap=fmt_pm(row["full_minus_variant_mrr_mean"], row["full_minus_variant_mrr_std"]),
                acc=fmt_pm(row["accuracy_mean"], row["accuracy_std"]),
                loss=fmt_pm(row["eval_loss_mean"], row["eval_loss_std"]),
                gate=fmt_pm(row["gate_mean"], row["gate_std"]),
                tail=fmt_pm(row["warrant_tail_mass_mean"], row["warrant_tail_mass_std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Seed Deltas",
            "",
            "| Variant | Seed | MRR | Base MRR | Delta MRR vs Base | Full MRR | Full - Variant MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(paired, key=lambda item: (variant_sort_key(str(item["variant"])), int(item["seed"]))):
        lines.append(
            "| {name} | {seed} | {mrr} | {base} | {delta} | {full} | {gap} |".format(
                name=DISPLAY_NAMES.get(str(row["variant"]), str(row["variant"])),
                seed=row["seed"],
                mrr=fmt(float(row["mrr"])),
                base=fmt(float(row["base_mrr"])),
                delta=fmt(float(row["delta_mrr_vs_base"])),
                full=fmt(float(row["full_value_gate_mrr"])),
                gap=fmt(float(row["full_minus_variant_mrr"])),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    summary, paired = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_seed_deltas.csv", paired)
    write_markdown(args.output_dir / "summary.md", summary, paired)
    print(f"wrote {args.output_dir / 'summary.csv'}")
    print(f"wrote {args.output_dir / 'paired_seed_deltas.csv'}")
    print(f"wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
