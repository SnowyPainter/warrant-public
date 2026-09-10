#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec, evaluate_run


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
RESULT_COLUMNS = [
    "domain",
    "dataset",
    "model",
    "variant",
    "seed",
    "implementation",
    "effective_implementation",
    "status",
    "train_loss",
    "eval_loss",
    "joint_nll",
    "primary_metric",
    "accuracy",
    "mark_mrr",
    "mrr",
    "answer_accuracy",
    "support_precision",
    "support_recall",
    "support_f1",
    "support_recall_at_1",
    "support_recall_at_2",
    "support_recall_at_5",
    "support_mrr",
    "support_ap",
    "support_auc",
    "support_warrant_mass",
    "distractor_warrant_mass",
    "support_mass_ratio",
    "auc",
    "mae_time",
    "rmse_location",
    "examples",
    "warrant_active_blocks",
    "warrant_replacements",
    "elapsed_sec",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main Warrant benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default=None, help="Domain name, comma list, or all. Overrides config.")
    parser.add_argument("--dataset", default=None, help="Dataset name, comma list, or all. Overrides config.")
    parser.add_argument("--model", default=None, help="Model name, comma list, or all. Overrides config.")
    parser.add_argument("--variant", default=None, help="Variant name, comma list, or all. Overrides config.")
    parser.add_argument("--seed", default=None, help="Seed, comma list, or all. Overrides config.")
    parser.add_argument("--force", action="store_true", help="Re-run rows already marked completed in results.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected runs without training.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override output root for an isolated run shard.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def progress_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("progress", True))


def as_selection(value: Any, default: Iterable[Any]) -> set[str]:
    if value is None or value == "all":
        return {str(item) for item in default}
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}


def iter_specs(config: dict[str, Any], args: argparse.Namespace) -> list[RunSpec]:
    datasets = config["datasets"]
    models = config["models"]
    variants = config["variants"]
    selection = config.get("selection", {})
    training = config.get("training", {})

    domains = as_selection(args.domain or selection.get("domains", "all"), datasets.keys())
    variant_names = as_selection(args.variant or selection.get("variants", variants.keys()), variants.keys())
    seeds = config.get("experiment", {}).get("seeds", [2024, 2025, 2026])
    seed_values = [int(seed) for seed in as_selection(args.seed or "all", seeds)]

    specs: list[RunSpec] = []
    for domain in sorted(domains):
        if domain not in datasets:
            raise ValueError(f"Unknown domain {domain!r}")
        domain_datasets = datasets[domain]
        domain_models = models.get(domain, {})
        dataset_names = as_selection(args.dataset or selection.get("datasets", "all"), domain_datasets.keys())
        model_names = as_selection(args.model or selection.get("models", "all"), domain_models.keys())
        for dataset_name in sorted(dataset_names):
            if dataset_name not in domain_datasets:
                continue
            for model_name in sorted(model_names):
                if model_name not in domain_models:
                    continue
                for variant in sorted(variant_names):
                    if variant not in variants:
                        continue
                    for seed in seed_values:
                        specs.append(
                            RunSpec(
                                domain=domain,
                                dataset=dataset_name,
                                dataset_path=REPO_ROOT / domain_datasets[dataset_name],
                                model=model_name,
                                variant=variant,
                                seed=seed,
                                implementation=training.get("implementation", "reference"),
                                model_config=dict(domain_models[model_name]),
                                use_warrant=bool(variants[variant].get("use_warrant", False)),
                            )
                        )
    return specs


def load_completed(results_path: Path) -> set[tuple[str, str, str, str, int, str]]:
    if not results_path.exists():
        return set()
    completed = set()
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "completed":
                if (
                    "effective_implementation" not in row
                    or "warrant_active_blocks" not in row
                    or row.get("effective_implementation", "") == ""
                    or row.get("warrant_active_blocks", "") == ""
                ):
                    continue
                if row.get("domain") == "rag" and (
                    "support_f1" not in row
                    or row.get("support_f1", "") == ""
                    or "support_recall_at_2" not in row
                    or row.get("support_recall_at_1", "") == ""
                    or row.get("support_recall_at_2", "") == ""
                    or row.get("support_ap", "") == ""
                ):
                    continue
                if row.get("domain") == "mtpp" and ("mark_mrr" not in row or row.get("mark_mrr", "") == ""):
                    continue
                if row.get("domain") == "stpp" and (
                    "joint_nll" not in row
                    or row.get("joint_nll", "") == ""
                    or "mae_time" not in row
                    or row.get("mae_time", "") == ""
                ):
                    continue
                if row.get("domain") == "tkg" and ("mrr" not in row or row.get("mrr", "") == ""):
                    continue
                completed.add(
                    (
                        row["domain"],
                        row["dataset"],
                        row["model"],
                        row["variant"],
                        int(row["seed"]),
                        row.get("implementation", "reference"),
                    )
                )
    return completed


def result_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, str]:
    return (
        str(row["domain"]),
        str(row["dataset"]),
        str(row["model"]),
        str(row["variant"]),
        int(row["seed"]),
        str(row.get("implementation", "reference")),
    )


def legacy_missing_audit(row: dict[str, Any]) -> bool:
    effective = str(row.get("effective_implementation") or "")
    status = str(row.get("status") or "")
    if status == "completed" and (not effective or not row.get("warrant_active_blocks")):
        return True
    if effective == "completed" or effective.startswith("failed:"):
        return True
    try:
        float(status)
        return True
    except ValueError:
        return False


def normalize_existing_result(row: dict[str, Any]) -> dict[str, Any]:
    if not legacy_missing_audit(row):
        return {column: row.get(column, "") for column in RESULT_COLUMNS}
    output_dir = row.get("output_dir") or row.get("warrant_active_blocks") or ""
    normalized = {column: "" for column in RESULT_COLUMNS}
    normalized.update(
        {
            "domain": row.get("domain", ""),
            "dataset": row.get("dataset", ""),
            "model": row.get("model", ""),
            "variant": row.get("variant", ""),
            "seed": row.get("seed", ""),
            "implementation": row.get("implementation", "reference"),
            "status": "legacy_missing_audit",
            "output_dir": output_dir,
        }
    )
    return normalized


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    key = result_key(row)
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                normalize_existing_result(existing)
                for existing in csv.DictReader(handle)
                if result_key(existing) != key
            ]
    rows.append({column: row.get(column, "") for column in RESULT_COLUMNS})
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def output_dir(root: Path, spec: RunSpec) -> Path:
    return root / spec.model / str(spec.seed) / spec.domain / spec.dataset / spec.variant


def run_spec(spec: RunSpec, config: dict[str, Any], device: torch.device, output_root: Path) -> dict[str, Any]:
    started = time.time()
    set_seed(spec.seed)
    out_dir = output_dir(output_root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_run(spec, config, device)
    elapsed = time.time() - started

    error_file = out_dir / "error.txt"
    if error_file.exists():
        error_file.unlink()

    row = {
        "domain": spec.domain,
        "dataset": spec.dataset,
        "model": spec.model,
        "variant": spec.variant,
        "seed": spec.seed,
        "implementation": spec.implementation,
        "status": "completed",
        "elapsed_sec": round(elapsed, 3),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[key] = round(float(value), 6)
        else:
            row[key] = value

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "domain": spec.domain,
                "dataset": spec.dataset,
                "dataset_path": str(spec.dataset_path.relative_to(REPO_ROOT)),
                "model": spec.model,
                "variant": spec.variant,
                "seed": spec.seed,
                "implementation": spec.implementation,
                "model_config": spec.model_config,
                "use_warrant": spec.use_warrant,
                "warrant": config.get("warrant", {}),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else REPO_ROOT / config.get("experiment", {}).get("output_root", "experiments/main_benchmark/outputs")
    )
    results_path = output_root / "results.csv"
    device = resolve_device(config.get("experiment", {}).get("device", "auto"))
    specs = iter_specs(config, args)
    completed = set() if args.force else load_completed(results_path)

    print(f"selected_runs={len(specs)} device={device} output_root={output_root}")
    if args.dry_run:
        for spec in specs:
            print(f"{spec.domain}/{spec.dataset} {spec.model} {spec.variant} seed={spec.seed}")
        return

    run_bar = tqdm(specs, desc="benchmark runs", unit="run", disable=not progress_enabled(config))
    for spec in run_bar:
        key = (spec.domain, spec.dataset, spec.model, spec.variant, spec.seed, spec.implementation)
        label = f"{spec.domain}/{spec.dataset} {spec.model} {spec.variant} seed={spec.seed}"
        run_bar.set_postfix_str(label[:80])
        if key in completed and config.get("experiment", {}).get("skip_completed", True):
            tqdm.write(f"skip completed: {label}")
            continue
        tqdm.write(f"run: {label}")
        try:
            row = run_spec(spec, config, device, output_root)
        except Exception as exc:
            out_dir = output_dir(output_root, spec)
            out_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "domain": spec.domain,
                "dataset": spec.dataset,
                "model": spec.model,
                "variant": spec.variant,
                "seed": spec.seed,
                "implementation": spec.implementation,
                "status": f"failed:{type(exc).__name__}",
                "output_dir": str(out_dir.relative_to(REPO_ROOT)),
            }
            with (out_dir / "error.txt").open("w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            tqdm.write(f"failed: {label}: {exc}")
        append_result(results_path, row)


if __name__ == "__main__":
    main()
