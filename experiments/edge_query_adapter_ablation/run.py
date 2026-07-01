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
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec, audit_warrant_application, binary_auc, build_benchmark_model
from experiments.neural_dissection.run import CTDGRunner, batches, finite_mean, gate_stats, set_seed


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
    "auc",
    "accuracy",
    "warrant_gate_mean",
    "warrant_logit_mean",
    "edge_warrant_gate_mean",
    "edge_warrant_attention_entropy",
    "warrant_active_blocks",
    "warrant_replacements",
    "elapsed_sec",
    "output_dir",
]


class HandcraftedOnlyEdgeAdapter(nn.Module):
    """Fixed heuristic gate control for the CTDG edge-query adapter ablation."""

    def __init__(self, hidden_dim: int, *, context_scale_init: float = 0.10) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init), dtype=torch.float32))

    def _side_context(self, memory: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask.to(memory.dtype)
        # stats[..., 1] is the raw recency/counterpart heuristic produced by the CTDG data path.
        gate = (0.05 + 0.95 * stats[..., 1].float()).clamp(0.05, 1.0) * valid
        weights = gate / gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (memory * weights.unsqueeze(-1)).sum(dim=1), gate

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        time_h: torch.Tensor,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        src_delta_t_enc: torch.Tensor,
        dst_delta_t_enc: torch.Tensor,
        src_stats: torch.Tensor,
        dst_stats: torch.Tensor,
        src_mask: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        src_context, src_gate = self._side_context(src_memory, src_stats, src_mask)
        dst_context, dst_gate = self._side_context(dst_memory, dst_stats, dst_mask)
        delta = self.context_proj(torch.cat([src_context, dst_context], dim=-1))
        src_delta, dst_delta = delta.chunk(2, dim=-1)
        valid_count = src_mask.float().sum() + dst_mask.float().sum()
        gate_mean = (src_gate.sum() + dst_gate.sum()) / valid_count.clamp_min(1.0)
        info = {
            "edge_warrant_gate_mean": gate_mean.detach(),
            "edge_warrant_attention_entropy": src_delta.new_zeros(()),
        }
        return self.context_scale * src_delta, self.context_scale * dst_delta, info


class ParamControlEdgeAdapter(nn.Module):
    """Learned edge adapter with no permission gate."""

    def __init__(self, hidden_dim: int, *, context_scale_init: float = 0.10) -> None:
        super().__init__()
        self.query_proj = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.context_proj = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init), dtype=torch.float32))

    @staticmethod
    def _mean_context(memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(memory.dtype)
        return (memory * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        time_h: torch.Tensor,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        src_delta_t_enc: torch.Tensor,
        dst_delta_t_enc: torch.Tensor,
        src_stats: torch.Tensor,
        dst_stats: torch.Tensor,
        src_mask: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        edge_query = self.query_proj(torch.cat([src_h, dst_h], dim=-1))
        src_context = self._mean_context(src_memory, src_mask)
        dst_context = self._mean_context(dst_memory, dst_mask)
        delta = self.context_proj(torch.cat([edge_query, src_context, dst_context, src_h * dst_h], dim=-1))
        src_delta, dst_delta = delta.chunk(2, dim=-1)
        return self.context_scale * src_delta, self.context_scale * dst_delta, {}


class ShuffledEdgeQueryAdapter(nn.Module):
    """Edge-conditioned adapter with the edge query mismatched across the batch."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.training:
            src_delta, dst_delta, info = self.inner(**kwargs)
            info = dict(info)
            info["shuffled_edge_query_fraction"] = kwargs["src_h"].new_tensor(0.0)
            return src_delta, dst_delta, info

        batch_size = kwargs["src_h"].shape[0]
        if batch_size > 1:
            order = torch.roll(
                torch.arange(batch_size, device=kwargs["src_h"].device),
                shifts=max(1, batch_size // 2),
            )
            kwargs = dict(kwargs)
            # Shuffle every tensor that defines the edge-level prediction request.
            # Keep source/destination memories and masks fixed so the control tests
            # whether the edge query still matches the history terms it gates.
            for key in (
                "src_h",
                "dst_h",
                "time_h",
                "src_delta_t_enc",
                "dst_delta_t_enc",
                "src_stats",
                "dst_stats",
            ):
                kwargs[key] = kwargs[key][order]
            src_delta, dst_delta, info = self.inner(**kwargs)
            info = dict(info)
            info["shuffled_edge_query_fraction"] = kwargs["src_h"].new_tensor(1.0)
            return src_delta, dst_delta, info
        src_delta, dst_delta, info = self.inner(**kwargs)
        info = dict(info)
        info["shuffled_edge_query_fraction"] = kwargs["src_h"].new_tensor(0.0)
        return src_delta, dst_delta, info


class CTDGAblationRunner(CTDGRunner):
    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.spec.variant != "shuffled_edge_query":
            return super().evaluate(model)

        model.eval()
        labels_all, scores_all, losses, aux_rows = [], [], [], []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                pos = self.make_batch(rows, negative=False)
                neg = self.make_batch(rows, negative=True)
                combined = tuple(torch.cat([pos_item, neg_item], dim=0) for pos_item, neg_item in zip(pos, neg))
                result = model(*combined)
                labels = torch.cat(
                    [
                        torch.ones(len(rows), device=self.device),
                        torch.zeros(len(rows), device=self.device),
                    ]
                )
                logits = result.logits
                losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
                labels_all.extend(labels.cpu().tolist())
                scores_all.extend(torch.sigmoid(logits).cpu().tolist())
                aux_rows.append(result.aux)
        preds = [1.0 if score >= 0.5 else 0.0 for score in scores_all]
        acc = float(np.mean([pred == label for pred, label in zip(preds, labels_all)])) if labels_all else float("nan")
        auc = binary_auc(labels_all, scores_all)
        return {"eval_loss": finite_mean(losses), "primary_metric": auc, "auc": auc, "accuracy": acc, **gate_stats(aux_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTDG edge-query adapter ablations.")
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
    experiment = config.get("experiment", {})
    configured = experiment.get("seeds")
    if configured is None:
        configured = [experiment.get("seed", 7)]
    seeds = [int(seed) for seed in configured]
    if value is None or value == "all":
        return seeds
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def output_dir(output_root: Path, variant: str, seed: int) -> Path:
    return output_root / variant / f"seed_{seed}"


def build_spec(config: dict[str, Any], variant: str, seed: int) -> RunSpec:
    dataset = config["dataset"]
    model = config["model"]
    use_warrant = variant in {"generic_qk_warrant", "edge_conditioned_warrant", "shuffled_edge_query"}
    return RunSpec(
        domain="ctdg",
        dataset=str(dataset["name"]),
        dataset_path=REPO_ROOT / str(dataset["path"]),
        model=str(model["name"]),
        variant=variant,
        seed=seed,
        implementation=str(config.get("training", {}).get("implementation", "reference")),
        model_config=dict(model.get("config", {})),
        use_warrant=use_warrant,
    )


def configure_ablation_model(model: nn.Module, variant: str, hidden_dim: int) -> None:
    if variant == "base":
        model.edge_warrant = None
        return
    if variant == "generic_qk_warrant":
        model.edge_warrant = None
        return
    if variant == "edge_conditioned_warrant":
        return
    if variant == "handcrafted_only_control":
        model.edge_warrant = HandcraftedOnlyEdgeAdapter(hidden_dim)
        return
    if variant == "param_control":
        model.edge_warrant = ParamControlEdgeAdapter(hidden_dim)
        return
    if variant == "shuffled_edge_query":
        if model.edge_warrant is None:
            raise ValueError("shuffled_edge_query requires an edge Warrant adapter")
        model.edge_warrant = ShuffledEdgeQueryAdapter(model.edge_warrant)
        return
    raise ValueError(f"Unsupported variant {variant!r}")


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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


def rounded_metrics(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            if math.isnan(value):
                continue
            rounded[key] = round(value, 6)
        elif isinstance(value, (np.floating, np.integer)):
            rounded[key] = round(float(value), 6)
        else:
            rounded[key] = value
    return rounded


def run_variant(config: dict[str, Any], variant: str, seed: int, device: torch.device, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    set_seed(seed)
    spec = build_spec(config, variant, seed)
    runner = CTDGAblationRunner(spec, config, device)
    frame = runner.prepare()
    model = build_benchmark_model(spec, frame, config, device)
    configure_ablation_model(model, variant, int(spec.model_config.get("hidden_dim", 128)))
    model.to(device)

    training = config["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epoch_rows: list[dict[str, Any]] = []
    epochs = int(training["epochs"])
    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc=f"{variant} seed={seed}",
        unit="epoch",
        disable=not bool(training.get("progress", True)),
    )
    last_train: dict[str, float] = {}
    last_eval: dict[str, float] = {}
    for epoch in epoch_iter:
        last_train = runner.train_epoch(model, opt, epoch)
        last_eval = runner.evaluate(model)
        epoch_row = rounded_metrics({"epoch": epoch, **last_train, **last_eval})
        epoch_rows.append(epoch_row)
        epoch_iter.set_postfix(auc=epoch_row.get("auc", ""), loss=epoch_row.get("eval_loss", ""))

    audit = {}
    try:
        if variant in {"generic_qk_warrant", "edge_conditioned_warrant", "shuffled_edge_query"}:
            audit = audit_warrant_application(model, spec)
        else:
            audit = {
                "effective_implementation": model.__class__.__name__,
                "warrant_active_blocks": 0.0,
                "warrant_replacements": 0.0,
            }
    except Exception as exc:
        audit = {"audit_warning": str(exc)}

    elapsed = time.time() - started
    row = {
        "variant": variant,
        "seed": seed,
        "dataset": spec.dataset,
        "model": spec.model,
        "status": "completed",
        "train_loss": last_train.get("train_loss"),
        "eval_loss": last_eval.get("eval_loss"),
        "primary_metric": last_eval.get("primary_metric"),
        "auc": last_eval.get("auc"),
        "accuracy": last_eval.get("accuracy"),
        "elapsed_sec": elapsed,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        **audit,
    }
    for key in ("warrant_gate_mean", "warrant_logit_mean", "edge_warrant_gate_mean", "edge_warrant_attention_entropy"):
        if key in last_eval:
            row[key] = last_eval[key]
    row = rounded_metrics(row)

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
    seeds = selected_seeds(config, args.seed)
    variants = selected_variants(config, args.variant)
    output_root = REPO_ROOT / str(config.get("experiment", {}).get("output_root", "experiments/edge_query_adapter_ablation/outputs"))
    results_path = output_root / "results.csv"
    device = resolve_device(config)
    run_items = [(variant, seed) for seed in seeds for variant in variants]

    print(f"selected_variants={variants} selected_seeds={seeds} runs={len(run_items)} device={device} output_root={output_root}")
    if args.dry_run:
        for variant, seed in run_items:
            print(f"dry_run: variant={variant} seed={seed} out_dir={output_dir(output_root, variant, seed)}")
        return

    for variant, seed in tqdm(run_items, desc="edge-query ablations", unit="run", disable=not bool(config["training"].get("progress", True))):
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
                "output_dir": str(out_dir.relative_to(REPO_ROOT)),
            }
            with (out_dir / "error.txt").open("w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            tqdm.write(f"failed: {variant}: {exc}")
        append_result(results_path, row)


if __name__ == "__main__":
    main()
