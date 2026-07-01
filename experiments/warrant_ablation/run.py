#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec, audit_warrant_application, build_benchmark_model  # noqa: E402
from experiments.neural_dissection.run import TKGRunner, finite_mean, gate_stats, set_seed  # noqa: E402
from models.attention import WarrantedAttention  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
RESULT_COLUMNS = [
    "variant",
    "seed",
    "dataset",
    "model",
    "status",
    "train_loss",
    "eval_loss",
    "primary_metric",
    "mrr",
    "accuracy",
    "gate_mean",
    "warrant_logit_mean",
    "warrant_tail_mass_mean",
    "warrant_active_blocks",
    "warrant_replacements",
    "elapsed_sec",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TKG Warrant operator ablations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", default=None, help="Variant name, comma list, or all.")
    parser.add_argument("--seed", default=None, help="Seed, comma list, or all.")
    parser.add_argument("--force", action="store_true", help="Re-run completed variants.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(config: dict[str, Any]) -> torch.device:
    value = str(config.get("experiment", {}).get("device", "auto"))
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def selected_variants(config: dict[str, Any], value: str | None) -> list[str]:
    names = list(config.get("variants", {}).keys())
    if value is None or value == "all":
        return names
    wanted = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return wanted


def selected_seeds(config: dict[str, Any], value: str | None) -> list[int]:
    configured = config.get("experiment", {}).get("seeds", [7])
    seeds = [int(seed) for seed in configured]
    if value is None or value == "all":
        return seeds
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def output_dir(output_root: Path, variant: str, seed: int) -> Path:
    return output_root / variant / f"seed_{seed}"


def display_path(path: Path) -> str:
    return os.path.relpath(path, REPO_ROOT)


def build_spec(config: dict[str, Any], variant: str, seed: int) -> RunSpec:
    dataset = config["dataset"]
    model = config["model"]
    return RunSpec(
        domain="tkg",
        dataset=str(dataset["name"]),
        dataset_path=REPO_ROOT / str(dataset["path"]),
        model=str(model["name"]),
        variant=variant,
        seed=seed,
        implementation=str(config.get("training", {}).get("implementation", "reference")),
        model_config=dict(model.get("config", {})),
        use_warrant=variant != "base",
    )


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        pair_mask = mask.unsqueeze(1) if mask.ndim == 2 else mask
        logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
        empty_rows = ~pair_mask.any(dim=-1, keepdim=True)
        logits = torch.where(empty_rows, torch.zeros_like(logits), logits)
    weights = torch.softmax(logits, dim=-1)
    if mask is not None:
        weights = weights * pair_mask.to(weights.dtype)
    return weights


def _patched_attention_forward(self: WarrantedAttention, query: torch.Tensor, memory: torch.Tensor, *, mask: torch.Tensor | None = None):
    mode = str(getattr(self, "_warrant_ablation_mode", "full_value_gate"))
    if query.ndim == 2:
        query_3d = query.unsqueeze(1)
        squeeze = True
    else:
        query_3d = query
        squeeze = False

    q = self.query_proj(query_3d)
    k = self.key_proj(memory)
    v = self.value_proj(memory)
    logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(k.shape[-1]))
    base_weights = _masked_softmax(logits, mask)
    pair_mask = None if mask is None else (mask.unsqueeze(1) if mask.ndim == 2 else mask)

    if mode == "attention_copy_control":
        weights = base_weights
        gate = torch.ones_like(weights)
        warrant_logits = torch.zeros_like(weights)
        out = torch.matmul(self.dropout(weights), v)
    else:
        if mode == "key_only_gate":
            gate_query = torch.zeros_like(q)
            gate_key = k
        elif mode == "query_only_gate":
            gate_query = q
            gate_key = torch.zeros_like(k)
        else:
            gate_query = q
            gate_key = k

        warrant_logits = self.warrant._gate_logits(gate_query, gate_key)
        gate = self.warrant.gate_leak + (1.0 - self.warrant.gate_leak) * torch.sigmoid(warrant_logits)
        if pair_mask is not None:
            gate = torch.where(pair_mask, gate, torch.zeros_like(gate))
            warrant_logits = torch.where(pair_mask, warrant_logits, torch.zeros_like(warrant_logits))

        if mode == "shuffled_gate" and not self.training and gate.shape[0] > 1:
            order = torch.roll(torch.arange(gate.shape[0], device=gate.device), shifts=max(1, gate.shape[0] // 2))
            gate = gate[order]
            warrant_logits = warrant_logits[order]

        if mode == "logit_gate":
            weights = _masked_softmax(logits + gate.clamp_min(1.0e-8).log(), mask)
            out = torch.matmul(self.dropout(weights), v)
            gate = torch.ones_like(weights)
        else:
            weights = self.dropout(base_weights)
            out = torch.sum(weights.unsqueeze(-1) * gate.unsqueeze(-1) * v.unsqueeze(1), dim=2)

    if squeeze:
        out = out.squeeze(1)
        weights = weights.squeeze(1)
        gate = gate.squeeze(1)
        warrant_logits = warrant_logits.squeeze(1)
    return out, {"attention": weights, "warrant_gate": gate, "warrant_logits": warrant_logits}


def configure_ablation_model(model: torch.nn.Module, variant: str) -> None:
    if variant == "base":
        return
    supported = {
        "attention_copy_control",
        "full_value_gate",
        "logit_gate",
        "key_only_gate",
        "query_only_gate",
        "shuffled_gate",
    }
    if variant not in supported:
        raise ValueError(f"Unsupported variant {variant!r}")
    if variant == "full_value_gate":
        return
    for module in model.modules():
        if isinstance(module, WarrantedAttention):
            module._warrant_ablation_mode = variant
            module.forward = types.MethodType(_patched_attention_forward, module)


def should_skip(out_dir: Path, force: bool) -> bool:
    if force:
        return False
    metrics_path = out_dir / "final_metrics.json"
    if not metrics_path.exists():
        return False
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                if not (existing.get("variant") == str(row["variant"]) and int(existing.get("seed", -1)) == int(row["seed"])):
                    rows.append(existing)
    rows.append({column: row.get(column, "") for column in RESULT_COLUMNS})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def rounded(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            if math.isnan(value):
                continue
            out[key] = round(value, 6)
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = round(float(value), 6)
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_variant(config: dict[str, Any], variant: str, seed: int, device: torch.device, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    set_seed(seed)
    spec = build_spec(config, variant, seed)
    runner = TKGRunner(spec, config, device)
    frame = runner.prepare()
    model = build_benchmark_model(spec, frame, config, device)
    configure_ablation_model(model, variant)
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    epoch_rows: list[dict[str, Any]] = []
    last_train: dict[str, float] = {}
    last_eval: dict[str, float] = {}
    iterator = tqdm(
        range(1, int(config["training"]["epochs"]) + 1),
        desc=f"{variant} seed={seed}",
        unit="epoch",
        disable=not bool(config["training"].get("progress", True)),
    )
    for epoch in iterator:
        last_train = runner.train_epoch(model, opt, epoch)
        last_eval = runner.evaluate(model)
        epoch_row = rounded({"epoch": epoch, **last_train, **last_eval})
        epoch_rows.append(epoch_row)
        iterator.set_postfix(mrr=epoch_row.get("mrr", ""), loss=epoch_row.get("eval_loss", ""))

    try:
        audit = audit_warrant_application(model, spec) if spec.use_warrant else {
            "effective_implementation": model.__class__.__name__,
            "warrant_active_blocks": 0.0,
            "warrant_replacements": 0.0,
        }
    except Exception as exc:
        audit = {"audit_warning": str(exc)}

    row = rounded(
        {
            "variant": variant,
            "seed": seed,
            "dataset": spec.dataset,
            "model": spec.model,
            "status": "completed",
            "train_loss": last_train.get("train_loss"),
            "eval_loss": last_eval.get("eval_loss"),
            "primary_metric": last_eval.get("primary_metric"),
            "mrr": last_eval.get("mrr"),
            "accuracy": last_eval.get("accuracy"),
            "gate_mean": last_eval.get("gate_mean"),
            "warrant_logit_mean": last_eval.get("warrant_logit_mean"),
            "warrant_tail_mass_mean": last_eval.get("warrant_tail_mass_mean"),
            "elapsed_sec": time.time() - started,
            "output_dir": display_path(out_dir),
            **audit,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics_by_epoch.csv", epoch_rows)
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"spec": spec.__dict__, "variant_description": config["variants"][variant]}, handle, indent=2, sort_keys=True, default=str)
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    variants = selected_variants(config, args.variant)
    seeds = selected_seeds(config, args.seed)
    output_root = REPO_ROOT / str(config.get("experiment", {}).get("output_root", "experiments/warrant_ablation/outputs"))
    results_path = output_root / "results.csv"
    device = resolve_device(config)
    run_items = [(variant, seed) for seed in seeds for variant in variants]

    print(f"selected_variants={variants} selected_seeds={seeds} runs={len(run_items)} device={device} output_root={output_root}")
    if args.dry_run:
        for variant, seed in run_items:
            print(f"dry_run: variant={variant} seed={seed} out_dir={output_dir(output_root, variant, seed)}")
        return

    for variant, seed in tqdm(run_items, desc="warrant ablations", unit="run", disable=not bool(config["training"].get("progress", True))):
        out_dir = output_dir(output_root, variant, seed)
        if should_skip(out_dir, args.force) and bool(config.get("experiment", {}).get("skip_completed", True)):
            tqdm.write(f"skip completed: {variant} seed={seed}")
            continue
        try:
            row = run_variant(config, variant, seed, device, out_dir)
        except Exception as exc:
            out_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "variant": variant,
                "seed": seed,
                "dataset": config["dataset"]["name"],
                "model": config["model"]["name"],
                "status": f"failed:{type(exc).__name__}",
                "output_dir": display_path(out_dir),
            }
            with (out_dir / "error.txt").open("w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            tqdm.write(f"failed: {variant}: {exc}")
        append_result(results_path, row)


if __name__ == "__main__":
    main()
