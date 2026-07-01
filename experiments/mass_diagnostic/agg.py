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
METRICS = [
    "primary_metric",
    "evidence_present_rate",
    "evidence_attention_mass",
    "evidence_warrant_mass",
    "non_evidence_warrant_mass",
    "counterfactual_zero_evidence_metric",
    "counterfactual_zero_non_evidence_metric",
    "metric_drop_zero_evidence",
    "metric_drop_zero_non_evidence",
]
DOMAIN_EVIDENCE = {
    "ctdg": "current counterpart appears in source/destination temporal history",
    "rag": "retrieved passage contains supporting evidence",
    "tkg": "target tail appears in historical facts",
}
DOMAIN_METRIC = {
    "ctdg": "AUC",
    "rag": "Support MRR",
    "tkg": "MRR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mass diagnostic results.")
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
            for metric in METRICS:
                parsed[metric] = parse_float(row.get(metric))
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


def signed(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:+.{digits}f}"


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["variant"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (domain, variant), subset in sorted(grouped.items()):
        out: dict[str, Any] = {
            "domain": domain,
            "dataset": subset[0].get("dataset", ""),
            "model": subset[0].get("model", ""),
            "variant": variant,
            "metric_name": subset[0].get("metric_name", ""),
            "seeds": len(subset),
        }
        for metric in METRICS:
            mean, std = mean_std([float(row[metric]) for row in subset])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Mass Diagnostic Summary",
        "",
        "This diagnostic asks whether the localized path carries labeled prediction evidence. It is not a new model leaderboard. The counterfactual columns are evaluation-time interventions that zero selected mass after the model has been trained.",
        "",
        "## How To Read This",
        "",
        "- `Evidence` means a labeled or task-derived support item: counterpart history for CTDG, supporting passage for RAG, and true-tail history for TKG.",
        "- `Evidence Warrant Mass` is the effective mass on that evidence after the Warrant path: attention mass times gate when a gate exists.",
        "- `Drop: Zero Evidence` is the metric decrease after zeroing evidence mass. A positive value means the path was carrying useful labeled evidence.",
        "- `Drop: Zero Non-Evidence` is the metric decrease after zeroing the remaining mass. A negative value means non-evidence mass was hurting this oracle counterfactual.",
        "- Base rows can still have attention mass, but they do not have the localized Warrant contribution path in the same sense; their zeroing columns are controls.",
        "",
        "## Column Meanings",
        "",
        "| Column | Meaning | What Supports The Claim |",
        "| --- | --- | --- |",
        "| `Primary` | The domain metric for the trained model. Higher is better for all rows here. | Warrant should improve over Base, but this table is mainly diagnostic. |",
        "| `Evidence Present` | Fraction of eval samples where labeled evidence is available in the input. | Shows how often the diagnostic can observe the evidence path. |",
        "| `Evidence Attention Mass` | Attention mass assigned to labeled evidence before permission scaling. | Indicates whether attention can find evidence at all. |",
        "| `Evidence Warrant Mass` | Effective mass assigned to labeled evidence after Warrant scaling. | Larger values mean evidence is being carried by the localized path. |",
        "| `Non-Evidence Warrant Mass` | Effective mass assigned to non-evidence items. | High values can still be useful context, but can also expose harmful mass. |",
        "| `Drop: Zero Evidence` | `Primary - metric_after_zeroing_evidence_mass`. | Positive and large means evidence mass is causally important. |",
        "| `Drop: Zero Non-Evidence` | `Primary - metric_after_zeroing_non_evidence_mass`. | Negative means removing non-evidence improves the oracle counterfactual. |",
        "",
        "## Main Table",
        "",
        "| Domain | Variant | Metric | Primary | Evidence Present | Evidence Attention Mass | Evidence Warrant Mass | Non-Evidence Warrant Mass | Drop: Zero Evidence | Drop: Zero Non-Evidence |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {variant} | {metric} | {primary} | {present} | {attn} | {warrant} | {non} | {drop_true} | {drop_non} |".format(
                domain=row["domain"],
                variant=row["variant"],
                metric=row["metric_name"],
                primary=fmt_pm(row["primary_metric_mean"], row["primary_metric_std"]),
                present=fmt_pm(row["evidence_present_rate_mean"], row["evidence_present_rate_std"]),
                attn=fmt_pm(row["evidence_attention_mass_mean"], row["evidence_attention_mass_std"]),
                warrant=fmt_pm(row["evidence_warrant_mass_mean"], row["evidence_warrant_mass_std"]),
                non=fmt_pm(row["non_evidence_warrant_mass_mean"], row["non_evidence_warrant_mass_std"]),
                drop_true=fmt_pm(row["metric_drop_zero_evidence_mean"], row["metric_drop_zero_evidence_std"]),
                drop_non=fmt_pm(row["metric_drop_zero_non_evidence_mean"], row["metric_drop_zero_non_evidence_std"]),
            )
        )
    lines.extend(["", "## Domain Notes", ""])
    for row in rows:
        if row["variant"] != "warrant":
            continue
        domain = str(row["domain"])
        metric = DOMAIN_METRIC.get(domain, str(row["metric_name"]))
        primary = float(row["primary_metric_mean"])
        evidence_mass = float(row["evidence_warrant_mass_mean"])
        non_evidence_mass = float(row["non_evidence_warrant_mass_mean"])
        drop_true = float(row["metric_drop_zero_evidence_mean"])
        drop_non = float(row["metric_drop_zero_non_evidence_mean"])
        evidence_def = DOMAIN_EVIDENCE.get(domain, "labeled evidence is present")
        if drop_true > 0 and abs(drop_true) >= abs(drop_non):
            verdict = "The evidence path is the stronger positive dependency."
        elif drop_true > 0:
            verdict = "The evidence path is useful, while non-evidence mass also changes the metric strongly."
        else:
            verdict = "The evidence path is weak in this run."
        if drop_non < 0:
            non_note = "Removing non-evidence improves the oracle counterfactual, so some remaining mass is harmful or distracting."
        elif drop_non > 0:
            non_note = "Removing non-evidence hurts, so remaining context still carries useful signal."
        else:
            non_note = "Removing non-evidence has little effect."
        lines.extend(
            [
                f"### {domain.upper()} / {row['variant']}",
                "",
                f"- Evidence definition: {evidence_def}.",
                f"- Primary {metric}: {fmt(primary)}.",
                f"- Evidence warrant mass: {fmt(evidence_mass)}; non-evidence warrant mass: {fmt(non_evidence_mass)}.",
                f"- Zeroing evidence changes the metric by {signed(-drop_true)} relative to normal evaluation (`Drop: Zero Evidence` = {signed(drop_true)}).",
                f"- Zeroing non-evidence changes the metric by {signed(-drop_non)} relative to normal evaluation (`Drop: Zero Non-Evidence` = {signed(drop_non)}).",
                f"- Reading: {verdict} {non_note}",
                "",
            ]
        )
    lines.extend(
        [
            "## Safe Wording",
            "",
            "Use this table as mechanism evidence, not as a replacement for the main benchmark. A good sentence is:",
            "",
            "> Mass diagnostics show that the localized Warrant path carries labeled prediction evidence: removing evidence mass reduces the task metric in CTDG, RAG, and especially TKG, while removing non-evidence mass reveals whether the remaining context is useful or distracting.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_markdown(args.output_dir / "summary.md", summary)
    print(f"wrote {args.output_dir / 'summary.csv'}")
    print(f"wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
