#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.path_localization.common import PATH_CLAIMS, VARIANT_ROLES  # noqa: E402


DEFAULT_RESULTS = Path(__file__).resolve().parent / "outputs" / "results.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "analysis"
VARIANT_ORDER = ["base", "generic_qk_warrant", "open_path_no_gate", "correct_path_warrant", "shuffled_pairing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Warrant path-localization results.")
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
            parsed["primary_metric"] = parse_float(row.get("primary_metric"))
            parsed["eval_loss"] = parse_float(row.get("eval_loss"))
            rows.append(parsed)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def direction_sign(direction: str) -> float:
    return -1.0 if direction == "lower" else 1.0


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["domain"]), str(row["dataset"]), str(row["model"]), str(row["variant"]))
        grouped[key].append(row)
        by_seed[(str(row["domain"]), str(row["dataset"]), str(row["model"]), int(row["seed"]))][str(row["variant"])] = row

    summary: list[dict[str, Any]] = []
    for (domain, dataset, model, variant), subset in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], variant_sort(item[0][3]))):
        mean, std = mean_std([float(row["primary_metric"]) for row in subset])
        loss_mean, loss_std = mean_std([float(row["eval_loss"]) for row in subset])
        claim = PATH_CLAIMS[domain]
        summary.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "path_role": VARIANT_ROLES.get(variant, variant),
                "seeds": len(subset),
                "metric": claim["metric_name"],
                "direction": claim["direction"],
                "primary_mean": mean,
                "primary_std": std,
                "eval_loss_mean": loss_mean,
                "eval_loss_std": loss_std,
            }
        )

    paired: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    for key, variant_rows in sorted(by_seed.items()):
        domain, dataset, model, seed = key
        base = variant_rows.get("base")
        correct = variant_rows.get("correct_path_warrant")
        if base is None:
            continue
        sign = direction_sign(PATH_CLAIMS[domain]["direction"])
        for variant in VARIANT_ORDER:
            row = variant_rows.get(variant)
            if row is None:
                continue
            delta = sign * (float(row["primary_metric"]) - float(base["primary_metric"]))
            paired.append(
                {
                    "domain": domain,
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "variant": variant,
                    "metric": PATH_CLAIMS[domain]["metric_name"],
                    "direction": PATH_CLAIMS[domain]["direction"],
                    "primary_metric": row["primary_metric"],
                    "base_primary_metric": base["primary_metric"],
                    "direction_aware_delta_vs_base": delta,
                }
            )
        if correct is None:
            continue
        generic = variant_rows.get("generic_qk_warrant")
        open_path = variant_rows.get("open_path_no_gate")
        shuffled = variant_rows.get("shuffled_pairing")
        correct_gain = sign * (float(correct["primary_metric"]) - float(base["primary_metric"]))
        generic_gap = float("nan") if generic is None else sign * (float(correct["primary_metric"]) - float(generic["primary_metric"]))
        open_path_gain = float("nan") if open_path is None else sign * (float(open_path["primary_metric"]) - float(base["primary_metric"]))
        permission_gain = float("nan") if open_path is None else sign * (float(correct["primary_metric"]) - float(open_path["primary_metric"]))
        shuffled_gap = float("nan") if shuffled is None else sign * (float(correct["primary_metric"]) - float(shuffled["primary_metric"]))
        verdicts.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "metric": PATH_CLAIMS[domain]["metric_name"],
                "correct_path_gain_vs_base": correct_gain,
                "open_path_gain_vs_base": open_path_gain,
                "correct_minus_open_path": permission_gain,
                "correct_minus_generic": generic_gap,
                "correct_minus_shuffled": shuffled_gap,
                "supports_path_localization": bool(
                    correct_gain > 0
                    and (not math.isfinite(generic_gap) or generic_gap > 0)
                    and (not math.isfinite(shuffled_gap) or shuffled_gap > 0)
                ),
            }
        )
    return summary, paired, verdicts


def variant_sort(variant: str) -> int:
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: Path, summary: list[dict[str, Any]], paired: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> None:
    def finite_average(values: list[float]) -> float:
        clean = [float(value) for value in values if math.isfinite(float(value))]
        return statistics.mean(clean) if clean else float("nan")

    verdict_summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in verdicts:
        grouped[(row["domain"], row["dataset"], row["model"])].append(row)
    for (domain, dataset, model), rows in sorted(grouped.items()):
        verdict_summary.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "metric": PATH_CLAIMS[domain]["metric_name"],
                "seeds": len(rows),
                "path_supported": sum(bool(row["supports_path_localization"]) for row in rows),
                "mean_correct_gain_vs_base": finite_average([float(row["correct_path_gain_vs_base"]) for row in rows]),
                "mean_open_path_gain_vs_base": finite_average([float(row["open_path_gain_vs_base"]) for row in rows]),
                "mean_correct_minus_open_path": finite_average([float(row["correct_minus_open_path"]) for row in rows]),
                "mean_correct_minus_generic": finite_average([float(row["correct_minus_generic"]) for row in rows]),
                "mean_correct_minus_shuffled": finite_average([float(row["correct_minus_shuffled"]) for row in rows]),
            }
        )

    lines = [
        "# Path Localization Results",
        "",
        "This report tests whether Warrant works best when placed on the metric-defining weighted value path.",
        "",
        "## Verdict",
        "",
        markdown_table(
            verdict_summary,
            [
                "domain",
                "dataset",
                "model",
                "metric",
                "seeds",
                "path_supported",
                "mean_correct_gain_vs_base",
                "mean_open_path_gain_vs_base",
                "mean_correct_minus_open_path",
                "mean_correct_minus_generic",
                "mean_correct_minus_shuffled",
            ],
        ),
        "",
        "## Variant Summary",
        "",
        markdown_table(
            summary,
            ["domain", "dataset", "model", "variant", "path_role", "seeds", "metric", "direction", "primary_mean", "primary_std"],
        ),
        "",
        "## Paired Deltas",
        "",
        markdown_table(
            paired,
            ["domain", "dataset", "model", "seed", "variant", "metric", "direction_aware_delta_vs_base"],
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    summary, paired, verdicts = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_deltas.csv", paired)
    write_csv(args.output_dir / "path_localization_verdicts.csv", verdicts)
    write_report(args.output_dir / "path_localization_report.md", summary, paired, verdicts)
    print(f"wrote {args.output_dir / 'summary.csv'}")
    print(f"wrote {args.output_dir / 'paired_deltas.csv'}")
    print(f"wrote {args.output_dir / 'path_localization_verdicts.csv'}")
    print(f"wrote {args.output_dir / 'path_localization_report.md'}")


if __name__ == "__main__":
    main()
