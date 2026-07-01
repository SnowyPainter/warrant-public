#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import (  # noqa: E402
    RunSpec,
    audit_warrant_application,
    binary_auc,
    build_benchmark_model,
    masked_support_nll,
    support_metrics,
)
from experiments.neural_dissection.run import (  # noqa: E402
    CTDGRunner,
    RAGRunner,
    TKGRunner,
    batches,
    finite_mean,
    gate_stats,
    set_seed,
)
from models.warrant_block import EdgeConditionedWarrantBlock  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
RESULT_COLUMNS = [
    "domain",
    "dataset",
    "model",
    "variant",
    "seed",
    "status",
    "train_loss",
    "eval_loss",
    "primary_metric",
    "metric_name",
    "auc",
    "mrr",
    "support_mrr",
    "accuracy",
    "evidence_present_rate",
    "evidence_attention_mass",
    "evidence_warrant_mass",
    "non_evidence_warrant_mass",
    "counterfactual_zero_evidence_metric",
    "counterfactual_zero_non_evidence_metric",
    "metric_drop_zero_evidence",
    "metric_drop_zero_non_evidence",
    "warrant_active_blocks",
    "warrant_replacements",
    "elapsed_sec",
    "output_dir",
]


RUNNERS = {
    "ctdg": CTDGRunner,
    "rag": RAGRunner,
    "tkg": TKGRunner,
}
METRIC_NAMES = {
    "ctdg": "auc",
    "rag": "support_mrr",
    "tkg": "mrr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mass diagnostics for Warrant evidence paths.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default=None, help="Domain name, comma list, or all.")
    parser.add_argument("--variant", default=None, help="Variant name, comma list, or all.")
    parser.add_argument("--seed", default=None, help="Seed, comma list, or all.")
    parser.add_argument("--force", action="store_true")
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


def selected(config_values: list[Any], override: str | None) -> list[str]:
    values = [str(value) for value in config_values]
    if override is None or override == "all":
        return values
    wanted = [item.strip() for item in override.split(",") if item.strip()]
    unknown = sorted(set(wanted) - set(values))
    if unknown:
        raise ValueError(f"Unknown selection: {unknown}")
    return wanted


def selected_seeds(config: dict[str, Any], override: str | None) -> list[int]:
    seeds = [int(seed) for seed in config.get("experiment", {}).get("seeds", [7])]
    if override is None or override == "all":
        return seeds
    return [int(item.strip()) for item in override.split(",") if item.strip()]


def display_path(path: Path) -> str:
    return os.path.relpath(path, REPO_ROOT)


def output_dir(root: Path, domain: str, variant: str, seed: int) -> Path:
    return root / domain / variant / f"seed_{seed}"


def build_spec(config: dict[str, Any], domain: str, variant: str, seed: int) -> RunSpec:
    dataset = config["datasets"][domain]
    model = config["models"][domain]
    return RunSpec(
        domain=domain,
        dataset=str(dataset["name"]),
        dataset_path=REPO_ROOT / str(dataset["path"]),
        model=str(model["name"]),
        variant=variant,
        seed=seed,
        implementation=str(config.get("training", {}).get("implementation", "reference")),
        model_config=dict(model.get("config", {})),
        use_warrant=variant == "warrant",
    )


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


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    key = (str(row["domain"]), str(row["variant"]), int(row["seed"]))
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                existing_key = (existing.get("domain"), existing.get("variant"), int(existing.get("seed", -1)))
                if existing_key != key:
                    rows.append(existing)
    rows.append({column: row.get(column, "") for column in RESULT_COLUMNS})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def train_model(runner, model: torch.nn.Module, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    rows: list[dict[str, Any]] = []
    last_train: dict[str, float] = {}
    last_eval: dict[str, float] = {}
    iterator = tqdm(
        range(1, int(config["training"]["epochs"]) + 1),
        desc=f"{runner.spec.domain}/{runner.spec.variant} seed={runner.spec.seed}",
        unit="epoch",
        disable=not bool(config["training"].get("progress", True)),
    )
    for epoch in iterator:
        last_train = runner.train_epoch(model, opt, epoch)
        last_eval = runner.evaluate(model)
        row = rounded({"epoch": epoch, **last_train, **last_eval})
        rows.append(row)
        iterator.set_postfix(primary=row.get("primary_metric", ""))
    return rows, last_train, last_eval


def rank_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(1, target.unsqueeze(1))
    return (logits > target_logits).sum(dim=1).float() + 1.0


def tkg_forward_with_mass(model, inputs: list[torch.Tensor], target: torch.Tensor, mode: str):
    head, relation, timestamp, history_head, history_relation, history_tail, history_time, history_mask = inputs
    if not hasattr(model, "hop_attn"):
        raise ValueError("TKG mass diagnostic currently expects an xERTE-style model with hop_attn")

    query = model.query_state(head, relation, timestamp)
    frontier = model.encode_events(history_head, history_relation, history_tail, history_time, timestamp)
    infos: list[dict[str, torch.Tensor]] = []
    state = query
    for attn, update in zip(model.hop_attn, model.hop_update):
        context, info = attn(state, frontier, mask=history_mask)
        state = state + update(torch.cat([state, context], dim=-1))
        frontier = frontier + state.unsqueeze(1)
        infos.append(info)

    base_logits = model.score_tails(state, relation)
    if not infos:
        return base_logits, base_logits, {}
    info = infos[-1]
    attention = info["attention"]
    gate = info.get("warrant_gate", torch.ones_like(attention))
    mass = attention * gate
    true_mask = history_mask.to(torch.bool) & history_tail.long().eq(target.long().unsqueeze(1))
    non_true_mask = history_mask.to(torch.bool) & ~true_mask
    effective_mass = mass
    if mode == "zero_evidence":
        effective_mass = mass.masked_fill(true_mask, 0.0)
    elif mode == "zero_non_evidence":
        effective_mass = mass.masked_fill(non_true_mask, 0.0)

    if getattr(model, "warrant_copy_scale", None) is None:
        copy_mass = torch.zeros((target.shape[0], model.num_entities), device=target.device, dtype=base_logits.dtype)
        tail_bias = torch.zeros_like(copy_mass)
    else:
        copy_mass = model.history_copy_mass(history_tail, history_time, timestamp, history_mask, evidence_weight=effective_mass)
        scale = torch.nn.functional.softplus(model.warrant_copy_scale)
        tail_bias = scale * torch.log1p(copy_mass * float(model.num_entities))
    logits = base_logits + tail_bias
    diag = {
        "attention": attention,
        "mass": mass,
        "true_mask": true_mask,
        "copy_mass": copy_mass,
        "tail_bias": tail_bias,
        "base_logits": base_logits,
    }
    return logits, base_logits, diag


def diagnose_tkg(runner: TKGRunner, model: torch.nn.Module, config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    mode_scores = {"normal": [], "zero_evidence": [], "zero_non_evidence": []}
    labels: list[int] = []
    losses, sample_rows = [], []
    normal_ranks, base_ranks = [], []
    with torch.no_grad():
        for rows in batches(runner.eval_idx, runner.batch_size, shuffle=False, desc="tkg mass diagnostic", config=config):
            *inputs, target = runner.make_batch(rows)
            input_list = list(inputs)
            logits, base_logits, diag = tkg_forward_with_mass(model, input_list, target, "normal")
            logits_zero, _, _ = tkg_forward_with_mass(model, input_list, target, "zero_evidence")
            logits_nonzero, _, _ = tkg_forward_with_mass(model, input_list, target, "zero_non_evidence")
            losses.append(float(F.cross_entropy(logits, target).cpu()))
            for name, item in (("normal", logits), ("zero_evidence", logits_zero), ("zero_non_evidence", logits_nonzero)):
                ranks = rank_from_logits(item, target)
                mode_scores[name].extend((1.0 / ranks).cpu().tolist())
            labels.extend(target.cpu().tolist())

            true_mask = diag["true_mask"]
            attention = diag["attention"]
            mass = diag["mass"]
            copy_mass = diag["copy_mass"]
            tail_bias = diag["tail_bias"]
            ranks_after = rank_from_logits(logits, target)
            ranks_before = rank_from_logits(base_logits, target)
            normal_ranks.extend(ranks_after.cpu().tolist())
            base_ranks.extend(ranks_before.cpu().tolist())
            row_ids = np.asarray(rows, dtype=np.int64)
            for i in range(target.shape[0]):
                target_id = int(target[i].detach().cpu())
                sample_rows.append(
                    {
                        "sample_id": int(row_ids[i]),
                        "true_tail_in_history": bool(true_mask[i].any().detach().cpu()),
                        "true_tail_attention_mass": float((attention[i] * true_mask[i].to(attention.dtype)).sum().detach().cpu()),
                        "true_tail_warrant_mass": float((mass[i] * true_mask[i].to(mass.dtype)).sum().detach().cpu()),
                        "non_true_tail_warrant_mass": float((mass[i] * (~true_mask[i] & input_list[-1].bool()[i]).to(mass.dtype)).sum().detach().cpu()),
                        "true_tail_copy_mass": float(copy_mass[i, target_id].detach().cpu()),
                        "true_tail_bias": float(tail_bias[i, target_id].detach().cpu()),
                        "rank_before_tail_bias": float(ranks_before[i].detach().cpu()),
                        "rank_after_tail_bias": float(ranks_after[i].detach().cpu()),
                        "rank_gain_from_tail_path": float((ranks_before[i] - ranks_after[i]).detach().cpu()),
                    }
                )

    normal = finite_mean(mode_scores["normal"])
    zero = finite_mean(mode_scores["zero_evidence"])
    zero_non = finite_mean(mode_scores["zero_non_evidence"])
    metrics = {
        "eval_loss": finite_mean(losses),
        "primary_metric": normal,
        "mrr": normal,
        "evidence_present_rate": finite_mean([1.0 if row["true_tail_in_history"] else 0.0 for row in sample_rows]),
        "evidence_attention_mass": finite_mean([row["true_tail_attention_mass"] for row in sample_rows]),
        "evidence_warrant_mass": finite_mean([row["true_tail_warrant_mass"] for row in sample_rows]),
        "non_evidence_warrant_mass": finite_mean([row["non_true_tail_warrant_mass"] for row in sample_rows]),
        "counterfactual_zero_evidence_metric": zero,
        "counterfactual_zero_non_evidence_metric": zero_non,
        "metric_drop_zero_evidence": normal - zero,
        "metric_drop_zero_non_evidence": normal - zero_non,
    }
    return metrics, sample_rows


def diagnose_rag(runner: RAGRunner, model: torch.nn.Module, config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    losses, metric_rows, zero_rows, zero_non_rows, sample_rows = [], [], [], [], []
    with torch.no_grad():
        for rows in batches(runner.eval_idx, runner.eval_batch_size, shuffle=False, desc="rag mass diagnostic", config=config):
            question, passages, mask, target = runner.make_batch(rows)
            query = model.encode_question(question)
            memory = model.encode_passages(passages)
            valid_mask = model._passage_valid_mask(passages, mask)
            context, info = model.attn(query, memory, mask=valid_mask)
            logits = model._support_logits(query, memory, context, info, valid_mask)
            mass = info["attention"] * info.get("warrant_gate", torch.ones_like(info["attention"]))
            support = target.to(torch.bool) & valid_mask
            distractor = valid_mask & ~support
            ones = torch.ones_like(mass)
            info_zero = {"attention": mass.masked_fill(support, 0.0), "warrant_gate": ones}
            info_zero_non = {"attention": mass.masked_fill(distractor, 0.0), "warrant_gate": ones}
            logits_zero = model._support_logits(query, memory, context, info_zero, valid_mask)
            logits_zero_non = model._support_logits(query, memory, context, info_zero_non, valid_mask)

            losses.append(float(masked_support_nll(logits, target, valid_mask).cpu()))
            metric_rows.append(support_metrics(logits, target, valid_mask, mass))
            zero_rows.append(support_metrics(logits_zero, target, valid_mask, mass.masked_fill(support, 0.0)))
            zero_non_rows.append(support_metrics(logits_zero_non, target, valid_mask, mass.masked_fill(distractor, 0.0)))
            row_ids = np.asarray(rows, dtype=np.int64)
            for i in range(target.shape[0]):
                sample_rows.append(
                    {
                        "sample_id": int(row_ids[i]),
                        "support_present": bool(support[i].any().detach().cpu()),
                        "support_attention_mass": float((info["attention"][i] * support[i].to(mass.dtype)).sum().detach().cpu()),
                        "support_warrant_mass": float((mass[i] * support[i].to(mass.dtype)).sum().detach().cpu()),
                        "distractor_warrant_mass": float((mass[i] * distractor[i].to(mass.dtype)).sum().detach().cpu()),
                    }
                )
    normal = finite_mean([row["support_mrr"] for row in metric_rows])
    zero = finite_mean([row["support_mrr"] for row in zero_rows])
    zero_non = finite_mean([row["support_mrr"] for row in zero_non_rows])
    return (
        {
            "eval_loss": finite_mean(losses),
            "primary_metric": normal,
            "support_mrr": normal,
            "evidence_present_rate": finite_mean([1.0 if row["support_present"] else 0.0 for row in sample_rows]),
            "evidence_attention_mass": finite_mean([row["support_attention_mass"] for row in sample_rows]),
            "evidence_warrant_mass": finite_mean([row["support_warrant_mass"] for row in sample_rows]),
            "non_evidence_warrant_mass": finite_mean([row["distractor_warrant_mass"] for row in sample_rows]),
            "counterfactual_zero_evidence_metric": zero,
            "counterfactual_zero_non_evidence_metric": zero_non,
            "metric_drop_zero_evidence": normal - zero,
            "metric_drop_zero_non_evidence": normal - zero_non,
        },
        sample_rows,
    )


def patch_edge_warrant(block: EdgeConditionedWarrantBlock, mode: str, side_records: list[dict[str, torch.Tensor]]):
    def diagnostic_side_context(self, query, memory, delta_t_enc, stats, valid_mask):
        tokens = self.token_encoder(torch.cat([memory, stats], dim=-1))
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        weights = self._semantic_attention(query, tokens, delta_t_enc, valid_mask)
        gate, _ = self.validity(query, tokens, delta_t_enc, stats, valid_mask)
        evidence_mask = stats[..., 0].gt(0.5) & valid_mask
        non_evidence_mask = valid_mask & ~evidence_mask
        effective_gate = gate
        if mode == "zero_evidence":
            effective_gate = gate.masked_fill(evidence_mask, 0.0)
        elif mode == "zero_non_evidence":
            effective_gate = gate.masked_fill(non_evidence_mask, 0.0)
        mass = weights * gate
        side_records.append(
            {
                "evidence_present": evidence_mask.any(dim=1).detach(),
                "evidence_attention_mass": (weights * evidence_mask.to(weights.dtype)).sum(dim=1).detach(),
                "evidence_warrant_mass": (mass * evidence_mask.to(mass.dtype)).sum(dim=1).detach(),
                "non_evidence_warrant_mass": (mass * non_evidence_mask.to(mass.dtype)).sum(dim=1).detach(),
            }
        )
        context = (tokens * weights.unsqueeze(-1) * effective_gate.unsqueeze(-1)).sum(dim=1)
        return context, gate, weights

    block._side_context = types.MethodType(diagnostic_side_context, block)


def diagnose_ctdg(runner: CTDGRunner, model: torch.nn.Module, config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    scores_by_mode = {"normal": [], "zero_evidence": [], "zero_non_evidence": []}
    labels_all: list[float] = []
    losses, sample_rows = [], []
    has_edge_warrant = isinstance(getattr(model, "edge_warrant", None), EdgeConditionedWarrantBlock)
    original_side_context = model.edge_warrant._side_context if has_edge_warrant else None

    with torch.no_grad():
        for rows in batches(runner.eval_idx, runner.batch_size, shuffle=False, desc="ctdg mass diagnostic", config=config):
            pos = runner.make_batch(rows, negative=False)
            neg = runner.make_batch(rows, negative=True)
            combined = tuple(torch.cat([pos_item, neg_item], dim=0) for pos_item, neg_item in zip(pos, neg))
            labels = torch.cat([torch.ones(len(rows), device=runner.device), torch.zeros(len(rows), device=runner.device)])
            labels_all.extend(labels.cpu().tolist())
            mode_records: dict[str, list[dict[str, torch.Tensor]]] = {}
            for mode in scores_by_mode:
                side_records: list[dict[str, torch.Tensor]] = []
                if has_edge_warrant:
                    patch_edge_warrant(model.edge_warrant, mode, side_records)
                result = model(*combined)
                if mode == "normal":
                    losses.append(float(F.binary_cross_entropy_with_logits(result.logits, labels).cpu()))
                scores_by_mode[mode].extend(torch.sigmoid(result.logits).cpu().tolist())
                mode_records[mode] = side_records
            if has_edge_warrant and original_side_context is not None:
                model.edge_warrant._side_context = original_side_context

            normal_sides = mode_records.get("normal", [])
            if len(normal_sides) >= 2:
                first, second = normal_sides[0], normal_sides[1]
                evidence_present = first["evidence_present"] | second["evidence_present"]
                evidence_attention = first["evidence_attention_mass"] + second["evidence_attention_mass"]
                evidence_warrant = first["evidence_warrant_mass"] + second["evidence_warrant_mass"]
                non_evidence_warrant = first["non_evidence_warrant_mass"] + second["non_evidence_warrant_mass"]
            else:
                src, dst = combined[0], combined[1]
                src_history_nodes, src_mask = combined[3], combined[6].to(torch.bool)
                dst_history_nodes, dst_mask = combined[7], combined[10].to(torch.bool)
                src_evidence = src_history_nodes.long().eq(dst.long().unsqueeze(1)) & src_mask
                dst_evidence = dst_history_nodes.long().eq(src.long().unsqueeze(1)) & dst_mask
                evidence_present = (src_evidence.any(dim=1) | dst_evidence.any(dim=1)).detach().cpu()
                n = labels.shape[0]
                evidence_attention = torch.zeros(n)
                evidence_warrant = torch.zeros(n)
                non_evidence_warrant = torch.zeros(n)
            row_ids = np.concatenate([np.asarray(rows, dtype=np.int64), np.asarray(rows, dtype=np.int64)])
            for i in range(labels.shape[0]):
                sample_rows.append(
                    {
                        "sample_id": int(row_ids[i]),
                        "label": float(labels[i].cpu()),
                        "score": float(scores_by_mode["normal"][-labels.shape[0] + i]),
                        "counterpart_evidence_present": bool(evidence_present[i].cpu()),
                        "counterpart_attention_mass": float(evidence_attention[i].cpu()),
                        "counterpart_warrant_mass": float(evidence_warrant[i].cpu()),
                        "non_counterpart_warrant_mass": float(non_evidence_warrant[i].cpu()),
                    }
                )

    normal = binary_auc(labels_all, scores_by_mode["normal"])
    zero = binary_auc(labels_all, scores_by_mode["zero_evidence"])
    zero_non = binary_auc(labels_all, scores_by_mode["zero_non_evidence"])
    preds = [1.0 if score >= 0.5 else 0.0 for score in scores_by_mode["normal"]]
    accuracy = float(np.mean([pred == label for pred, label in zip(preds, labels_all)])) if labels_all else float("nan")
    return (
        {
            "eval_loss": finite_mean(losses),
            "primary_metric": normal,
            "auc": normal,
            "accuracy": accuracy,
            "evidence_present_rate": finite_mean([1.0 if row["counterpart_evidence_present"] else 0.0 for row in sample_rows]),
            "evidence_attention_mass": finite_mean([row["counterpart_attention_mass"] for row in sample_rows]),
            "evidence_warrant_mass": finite_mean([row["counterpart_warrant_mass"] for row in sample_rows]),
            "non_evidence_warrant_mass": finite_mean([row["non_counterpart_warrant_mass"] for row in sample_rows]),
            "counterfactual_zero_evidence_metric": zero,
            "counterfactual_zero_non_evidence_metric": zero_non,
            "metric_drop_zero_evidence": normal - zero,
            "metric_drop_zero_non_evidence": normal - zero_non,
        },
        sample_rows,
    )


def run_diagnostic(runner, model: torch.nn.Module, config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if runner.spec.domain == "tkg":
        return diagnose_tkg(runner, model, config)
    if runner.spec.domain == "rag":
        return diagnose_rag(runner, model, config)
    if runner.spec.domain == "ctdg":
        return diagnose_ctdg(runner, model, config)
    raise ValueError(f"Unsupported domain {runner.spec.domain!r}")


def run_one(config: dict[str, Any], domain: str, variant: str, seed: int, device: torch.device, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    set_seed(seed)
    spec = build_spec(config, domain, variant, seed)
    runner = RUNNERS[domain](spec, config, device)
    frame = runner.prepare()
    model = build_benchmark_model(spec, frame, config, device)
    model.to(device)
    epoch_rows, last_train, _ = train_model(runner, model, config)
    diagnostic_metrics, sample_rows = run_diagnostic(runner, model, config)
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
            "domain": domain,
            "dataset": spec.dataset,
            "model": spec.model,
            "variant": variant,
            "seed": seed,
            "status": "completed",
            "train_loss": last_train.get("train_loss"),
            "metric_name": METRIC_NAMES[domain],
            "elapsed_sec": time.time() - started,
            "output_dir": display_path(out_dir),
            **diagnostic_metrics,
            **audit,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics_by_epoch.csv", epoch_rows)
    write_csv(out_dir / "sample_diagnostics.csv", sample_rows)
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"spec": spec.__dict__, "config": config}, handle, indent=2, sort_keys=True, default=str)
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    domains = selected(config.get("experiment", {}).get("domains", list(RUNNERS)), args.domain)
    variants = selected(config.get("experiment", {}).get("variants", ["base", "warrant"]), args.variant)
    seeds = selected_seeds(config, args.seed)
    output_root = REPO_ROOT / str(config.get("experiment", {}).get("output_root", "experiments/mass_diagnostic/outputs"))
    results_path = output_root / "results.csv"
    device = resolve_device(config)
    run_items = [(domain, variant, seed) for seed in seeds for domain in domains for variant in variants]
    print(f"selected_domains={domains} selected_variants={variants} selected_seeds={seeds} runs={len(run_items)} device={device}")
    if args.dry_run:
        for domain, variant, seed in run_items:
            print(f"dry_run: {domain}/{variant} seed={seed} out_dir={output_dir(output_root, domain, variant, seed)}")
        return
    for domain, variant, seed in tqdm(run_items, desc="mass diagnostics", unit="run", disable=not bool(config["training"].get("progress", True))):
        out_dir = output_dir(output_root, domain, variant, seed)
        if should_skip(out_dir, args.force) and bool(config.get("experiment", {}).get("skip_completed", True)):
            tqdm.write(f"skip completed: {domain}/{variant} seed={seed}")
            continue
        try:
            row = run_one(config, domain, variant, seed, device, out_dir)
        except Exception as exc:
            out_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "domain": domain,
                "dataset": config["datasets"][domain]["name"],
                "model": config["models"][domain]["name"],
                "variant": variant,
                "seed": seed,
                "status": f"failed:{type(exc).__name__}",
                "output_dir": display_path(out_dir),
            }
            with (out_dir / "error.txt").open("w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            tqdm.write(f"failed: {domain}/{variant} seed={seed}: {exc}")
        append_result(results_path, row)


if __name__ == "__main__":
    main()
