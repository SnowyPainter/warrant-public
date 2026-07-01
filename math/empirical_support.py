#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "math" / "outputs"


def first_float(value: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return float("nan")
    return float(match.group(0))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by(rows: list[dict[str, str]], **kwargs: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)


def bert_summary() -> dict[str, float | str]:
    rows = read_csv(REPO_ROOT / "experiments/bert_hotpotqa_warrant/outputs/analysis/summary.csv")
    base = row_by(rows, variant="base")
    warrant = row_by(rows, variant="full_warrant")
    return {
        "section": "bert_hotpotqa",
        "base_mrr": first_float(base["support_mrr"]),
        "warrant_mrr": first_float(warrant["support_mrr"]),
        "mrr_delta": first_float(warrant["support_mrr"]) - first_float(base["support_mrr"]),
        "mrr_deficit_reduction_pct": 100.0
        * (first_float(warrant["support_mrr"]) - first_float(base["support_mrr"]))
        / (1.0 - first_float(base["support_mrr"])),
        "base_r1": first_float(base["recall_at_1"]),
        "warrant_r1": first_float(warrant["recall_at_1"]),
        "r1_error_reduction_pct": 100.0
        * (first_float(warrant["recall_at_1"]) - first_float(base["recall_at_1"]))
        / (1.0 - first_float(base["recall_at_1"])),
        "base_unsupported": first_float(base["unsupported_at_gold_count"]),
        "warrant_unsupported": first_float(warrant["unsupported_at_gold_count"]),
        "unsupported_reduction_pct": 100.0
        * (first_float(base["unsupported_at_gold_count"]) - first_float(warrant["unsupported_at_gold_count"]))
        / first_float(base["unsupported_at_gold_count"]),
        "gold_random_attention_ratio": first_float(warrant["gold_random_attention_ratio"]),
        "gold_random_effective_ratio": first_float(warrant["gold_random_effective_ratio"]),
        "effective_ratio_gain": first_float(warrant["gold_random_effective_ratio"])
        / max(first_float(warrant["gold_random_attention_ratio"]), 1.0e-12),
    }


def mass_summary() -> list[dict[str, float | str]]:
    rows = read_csv(REPO_ROOT / "experiments/mass_diagnostic/outputs/analysis/summary.csv")
    out: list[dict[str, float | str]] = []
    for domain in ["ctdg", "rag", "tkg"]:
        warrant = row_by(rows, domain=domain, variant="warrant")
        out.append(
            {
                "section": f"mass_{domain}",
                "primary_metric": first_float(warrant["primary_metric_mean"]),
                "evidence_attention_mass": first_float(warrant["evidence_attention_mass_mean"]),
                "evidence_warrant_mass": first_float(warrant["evidence_warrant_mass_mean"]),
                "non_evidence_warrant_mass": first_float(warrant["non_evidence_warrant_mass_mean"]),
                "drop_zero_evidence": first_float(warrant["metric_drop_zero_evidence_mean"]),
                "drop_zero_non_evidence": first_float(warrant["metric_drop_zero_non_evidence_mean"]),
            }
        )
    return out


def path_summary() -> list[dict[str, float | str]]:
    rows = read_csv(REPO_ROOT / "experiments/path_localization/outputs/analysis/summary.csv")
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["domain"], row["dataset"], row["model"])
        grouped.setdefault(key, {})[row["variant"]] = row
    out: list[dict[str, float | str]] = []
    for (domain, dataset, model), variants in grouped.items():
        if "base" not in variants or "correct_path_warrant" not in variants:
            continue
        base = first_float(variants["base"]["primary_mean"])
        correct = first_float(variants["correct_path_warrant"]["primary_mean"])
        direction = variants["base"]["direction"]
        if direction == "lower":
            gain = base - correct
        else:
            gain = correct - base
        generic = variants.get("generic_qk_warrant")
        shuffled = variants.get("shuffled_pairing")
        out.append(
            {
                "section": f"path_{domain}",
                "dataset": dataset,
                "model": model,
                "metric": variants["base"]["metric"],
                "direction": direction,
                "base": base,
                "correct": correct,
                "direction_aware_gain": gain,
                "correct_minus_generic": (
                    (first_float(generic["primary_mean"]) - correct)
                    if direction == "lower"
                    else (correct - first_float(generic["primary_mean"]))
                )
                if generic
                else float("nan"),
                "correct_minus_shuffled": (
                    (first_float(shuffled["primary_mean"]) - correct)
                    if direction == "lower"
                    else (correct - first_float(shuffled["primary_mean"]))
                )
                if shuffled
                else float("nan"),
            }
        )
    return out


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(bert: dict[str, float | str], mass: list[dict[str, float | str]], path_rows: list[dict[str, float | str]], path: Path) -> None:
    lines = [
        "# Empirical Support for the Math Claims",
        "",
        "## BERT HotpotQA",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| MRR delta | {bert['mrr_delta']:.4f} |",
        f"| MRR-deficit reduction | {bert['mrr_deficit_reduction_pct']:.2f}% |",
        f"| R@1 error reduction | {bert['r1_error_reduction_pct']:.2f}% |",
        f"| Unsupported attribution reduction | {bert['unsupported_reduction_pct']:.2f}% |",
        f"| Gold/Random attention ratio | {bert['gold_random_attention_ratio']:.4f} |",
        f"| Gold/Random effective ratio | {bert['gold_random_effective_ratio']:.4f} |",
        f"| Effective ratio gain over attention ratio | {bert['effective_ratio_gain']:.2f}x |",
        "",
        "## Mass Diagnostics",
        "",
        "| Domain | Primary | Evidence attention | Evidence warrant | Non-evidence warrant | Drop zero evidence | Drop zero non-evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in mass:
        lines.append(
            "| {section} | {primary_metric:.4f} | {evidence_attention_mass:.4f} | {evidence_warrant_mass:.4f} | "
            "{non_evidence_warrant_mass:.4f} | {drop_zero_evidence:.4f} | {drop_zero_non_evidence:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Path Localization",
            "",
            "| Domain | Dataset | Model | Metric | Base | Correct | Gain | Correct-Generic | Correct-Shuffled |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in path_rows:
        lines.append(
            "| {section} | {dataset} | {model} | {metric} | {base:.4f} | {correct:.4f} | "
            "{direction_aware_gain:.4f} | {correct_minus_generic:.4f} | {correct_minus_shuffled:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bert = bert_summary()
    mass = mass_summary()
    paths = path_summary()
    rows = [bert] + mass + paths
    write_csv(rows, OUTPUT_DIR / "empirical_support.csv")
    write_markdown(bert, mass, paths, OUTPUT_DIR / "empirical_support.md")
    print(f"wrote {OUTPUT_DIR / 'empirical_support.csv'}")
    print(f"wrote {OUTPUT_DIR / 'empirical_support.md'}")


if __name__ == "__main__":
    main()
