#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import audit_warrant_application, build_benchmark_model  # noqa: E402
from experiments.neural_dissection.run import set_seed  # noqa: E402
from experiments.path_localization import ctdg, mtpp, rag, stpp, tkg  # noqa: E402
from experiments.path_localization.common import (  # noqa: E402
    RESULT_COLUMNS,
    append_result,
    configure_variant,
    iter_specs,
    load_config,
    output_dir,
    path_metadata,
    resolve_device,
    rounded,
    selected_domains,
    selected_seeds,
    selected_variants,
    should_skip,
    write_csv,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
RUNNER_BY_DOMAIN = {
    "ctdg": ctdg.PathCTDGRunner,
    "mtpp": mtpp.PathMTPPRunner,
    "rag": rag.PathRAGRunner,
    "stpp": stpp.PathSTPPRunner,
    "tkg": tkg.PathTKGRunner,
}
CONFIGURE_BY_DOMAIN = {
    "ctdg": ctdg.configure_model,
    "mtpp": mtpp.configure_model,
    "rag": rag.configure_model,
    "stpp": stpp.configure_model,
    "tkg": tkg.configure_model,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Warrant path-localization experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default=None, help="Domain name, comma list, or all.")
    parser.add_argument("--variant", default=None, help="Variant name, comma list, or all.")
    parser.add_argument("--seed", default=None, help="Seed, comma list, or all.")
    parser.add_argument("--force", action="store_true", help="Re-run completed folders.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_spec(config: dict[str, Any], spec, device: torch.device, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    if spec.domain == "ctdg" and spec.variant in ctdg.EDGE_VARIANT_BY_PATH_VARIANT:
        return ctdg.run_edge_ablation_source(spec, device, out_dir, force=False)

    started = time.time()
    set_seed(spec.seed)
    runner_cls = RUNNER_BY_DOMAIN[spec.domain]
    runner = runner_cls(spec, config, device)
    frame = runner.prepare()
    model = build_benchmark_model(spec, frame, config, device)
    configure_variant(model, spec)
    CONFIGURE_BY_DOMAIN[spec.domain](model, spec.variant)
    model.to(device)

    training = config["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epoch_rows: list[dict[str, Any]] = []
    epochs = int(training["epochs"])
    iterator = tqdm(
        range(1, epochs + 1),
        desc=f"{spec.domain}/{spec.model} {spec.variant} seed={spec.seed}",
        unit="epoch",
        disable=not bool(training.get("progress", True)),
    )
    last_train: dict[str, float] = {}
    last_eval: dict[str, float] = {}
    for epoch in iterator:
        last_train = runner.train_epoch(model, opt, epoch)
        last_eval = runner.evaluate(model)
        epoch_row = rounded({"epoch": epoch, **last_train, **last_eval})
        epoch_rows.append(epoch_row)
        iterator.set_postfix(metric=epoch_row.get("primary_metric", ""), loss=epoch_row.get("eval_loss", ""))

    try:
        audit = audit_warrant_application(model, spec) if spec.use_warrant else {
            "effective_implementation": model.__class__.__name__,
            "warrant_active_blocks": 0.0,
            "warrant_replacements": 0.0,
        }
    except Exception as exc:
        audit = {"audit_warning": str(exc)}

    row = {
        "domain": spec.domain,
        "dataset": spec.dataset,
        "model": spec.model,
        "variant": spec.variant,
        "seed": spec.seed,
        "status": "completed",
        "train_loss": last_train.get("train_loss"),
        "eval_loss": last_eval.get("eval_loss"),
        "primary_metric": last_eval.get("primary_metric"),
        "elapsed_sec": time.time() - started,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        **path_metadata(spec.domain, spec.variant),
        **audit,
    }
    metric_keys = [
        "accuracy",
        "auc",
        "mark_mrr",
        "mrr",
        "support_mrr",
        "support_ap",
        "support_auc",
        "support_attention_mass",
        "distractor_attention_mass",
        "support_attention_ratio",
        "support_warrant_mass",
        "distractor_warrant_mass",
        "support_mass_ratio",
        "joint_nll",
        "mae_time",
        "rmse_location",
        "gate_mean",
        "warrant_logit_mean",
        "edge_warrant_gate_mean",
        "edge_warrant_attention_entropy",
        "warrant_tail_mass_mean",
    ]
    for key in metric_keys:
        if key in last_eval:
            row[key] = last_eval[key]
    row = rounded(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics_by_epoch.csv", epoch_rows)
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"spec": spec.__dict__, "variant": config["variants"][spec.variant]}, handle, indent=2, sort_keys=True, default=str)
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    domains = selected_domains(config, args.domain or config.get("selection", {}).get("domains"))
    variants = selected_variants(config, args.variant or config.get("selection", {}).get("variants"))
    seeds = selected_seeds(config, args.seed)
    specs = iter_specs(config, domains=domains, variants=variants, seeds=seeds)
    device = resolve_device(config)
    output_root = REPO_ROOT / str(config.get("experiment", {}).get("output_root", "experiments/path_localization/outputs"))
    results_path = output_root / "results.csv"

    print(f"domains={domains} variants={variants} seeds={seeds} runs={len(specs)} device={device}")
    if args.dry_run:
        for spec in specs:
            print(f"dry_run: {spec.domain}/{spec.dataset}/{spec.model}/{spec.variant}/seed_{spec.seed}")
        return

    for spec in tqdm(specs, desc="path-localization runs", unit="run", disable=not bool(config["training"].get("progress", True))):
        out_dir = output_dir(output_root, spec)
        if should_skip(out_dir, args.force) and bool(config.get("experiment", {}).get("skip_completed", True)):
            tqdm.write(f"skip completed: {spec.domain}/{spec.model}/{spec.variant} seed={spec.seed}")
            continue
        try:
            row = run_spec(config, spec, device, out_dir, force=args.force)
        except Exception as exc:
            out_dir.mkdir(parents=True, exist_ok=True)
            row = rounded(
                {
                    "domain": spec.domain,
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "variant": spec.variant,
                    "seed": spec.seed,
                    "status": f"failed:{type(exc).__name__}",
                    "output_dir": str(out_dir.relative_to(REPO_ROOT)),
                    **path_metadata(spec.domain, spec.variant),
                }
            )
            with (out_dir / "error.txt").open("w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            tqdm.write(f"failed: {spec.domain}/{spec.model}/{spec.variant} seed={spec.seed}: {exc}")
        append_result(results_path, {column: row.get(column, "") for column in RESULT_COLUMNS})


if __name__ == "__main__":
    main()
