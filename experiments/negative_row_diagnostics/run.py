#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import audit_warrant_application, build_benchmark_model, label_encode, split_indices, stpp_joint_nll  # noqa: E402
from experiments.neural_dissection.run import batches, finite_mean, gate_stats, set_seed  # noqa: E402
from experiments.path_localization import stpp, tkg  # noqa: E402
from experiments.path_localization.common import (  # noqa: E402
    RESULT_COLUMNS,
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
from experiments.path_localization.rag import _nontrivial_permutation  # noqa: E402
from experiments.warrant_need_score.run import (  # noqa: E402
    checkpoint_name,
    save_checkpoint_enabled,
    write_checkpoint,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"

EXTRA_COLUMNS = [
    "stpp_revisit_rate",
    "stpp_rmse_revisit",
    "stpp_rmse_new_place",
    "stpp_rmse_short_move",
    "stpp_rmse_long_move",
    "stpp_attention_prior_rmse",
    "stpp_warrant_prior_rmse",
    "stpp_prior_rmse_delta",
    "stpp_useful_gate_mean",
    "stpp_nonuseful_gate_mean",
    "stpp_useful_mass_retention",
    "stpp_nonuseful_mass_retention",
    "stpp_false_suppression_index",
    "tkg_true_tail_seen_rate",
    "tkg_mrr_seen",
    "tkg_mrr_unseen",
    "tkg_copy_gate_mean_seen",
    "tkg_copy_gate_mean_unseen",
    "tkg_true_copy_mass_attention",
    "tkg_true_copy_mass_warrant",
    "tkg_false_top_copy_mass_attention",
    "tkg_false_top_copy_mass_warrant",
    "tkg_true_copy_retention",
    "tkg_false_copy_retention",
    "tkg_copy_saturation_attention",
    "tkg_copy_saturation_warrant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run micro-WNS and failure diagnostics for negative benchmark rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default=None, help="Domain name, comma list, or all.")
    parser.add_argument("--variant", default=None, help="Variant name, comma list, or all.")
    parser.add_argument("--seed", default=None, help="Seed, comma list, or all.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite_tensor_mean(values: list[torch.Tensor]) -> float:
    if not values:
        return float("nan")
    flat = torch.cat([value.detach().float().flatten().cpu() for value in values if value.numel() > 0])
    if flat.numel() == 0:
        return float("nan")
    flat = flat[torch.isfinite(flat)]
    return float(flat.mean()) if flat.numel() else float("nan")


def masked_mean_tensor(value: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = mask.to(value.dtype)
    return (value * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_quantile_midpoint(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = value.masked_fill(~mask, float("nan"))
    clean = torch.where(torch.isfinite(masked), masked, torch.full_like(masked, float("inf")))
    sorted_value, _ = torch.sort(clean, dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    midpoint_idx = ((counts - 1) // 2).long()
    return sorted_value.gather(1, midpoint_idx.unsqueeze(1)).squeeze(1)


def append_result_dynamic(results_path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    key = (str(row["domain"]), str(row["dataset"]), str(row["model"]), str(row["variant"]), int(row["seed"]))
    existing_columns: list[str] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_columns = list(reader.fieldnames or [])
            for existing in reader:
                existing_key = (
                    existing.get("domain", ""),
                    existing.get("dataset", ""),
                    existing.get("model", ""),
                    existing.get("variant", ""),
                    int(existing.get("seed", -1)),
                )
                if existing_key != key:
                    rows.append(existing)
    preferred = RESULT_COLUMNS + EXTRA_COLUMNS + ["checkpoint_path"]
    columns = [column for column in preferred if column in set(preferred) | set(row) | set(existing_columns)]
    for column in existing_columns:
        if column not in columns:
            columns.append(column)
    for column in row:
        if column not in columns:
            columns.append(column)
    rows.append({column: row.get(column, "") for column in columns})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class NegativeSTPPRunner(stpp.PathSTPPRunner):
    def prepare(self) -> Any:
        frame = self._prepare_recent_stpp_frame(self.spec.dataset_path, int(self.training["max_examples"]))
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float64")
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce").fillna(0.0).astype("float32")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce").fillna(0.0).astype("float32")
        for column in ("lat", "lon"):
            std = float(frame[column].std() or 1.0)
            frame[column] = (frame[column] - float(frame[column].mean())) / std
        time_origin = float(frame["timestamp"].min())
        time_scale = max(float(frame["timestamp"].std() or 1.0), 1.0)
        frame["time_value"] = ((frame["timestamp"] - time_origin) / time_scale).astype("float32")
        deltas = []
        for _, group in frame.groupby("sequence_id", sort=False):
            ts = group.sort_values("timestamp")["timestamp"].to_numpy(dtype=np.float64)
            if len(ts) > 1:
                deltas.extend(np.log1p(np.maximum(0.0, np.diff(ts))).tolist())
        self.delta_scale = max(float(np.mean(deltas) or 1.0), 1.0)
        self.frame = frame
        self.windows = self.make_windows(frame)
        if len(self.windows) < 2:
            raise ValueError("Not enough STPP windows")
        self.train_idx, self.eval_idx = split_indices(len(self.windows), float(self.training["eval_ratio"]))
        self.num_marks = int(frame["mark_id"].max()) + 1
        return frame

    def _prepare_recent_stpp_frame(self, path: Path, max_examples: int) -> pd.DataFrame:
        """Read only the recent slice needed for this negative-row diagnostic.

        Gowalla is a multi-million-row processed CSV.  The regular STPP loader
        parses the whole file before taking a chronological tail sample, which
        makes this small diagnostic appear stuck before the first epoch.  The
        processed STPP files are written in timestamp order, so this diagnostic
        reads the file header plus the last `max_examples` rows.
        """

        header = list(pd.read_csv(path, nrows=0).columns)
        completed = subprocess.run(
            ["tail", "-n", str(max(2, int(max_examples))), str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        frame = pd.read_csv(io.StringIO(completed.stdout), names=header)
        if not frame.empty:
            header_like = np.ones(len(frame), dtype=bool)
            for column in header:
                header_like &= frame[column].astype(str).eq(str(column)).to_numpy()
            frame = frame.loc[~header_like].copy()
        if "sequence_id" not in frame.columns:
            frame["sequence_id"] = "sequence_0"
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0)
        frame = frame.sort_values("timestamp").tail(max_examples).reset_index(drop=True)
        frame["mark_id"], _ = label_encode(frame.get("mark", pd.Series(["event"] * len(frame))))
        return frame

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        losses, correct, total, rmses, maes, aux_rows = [], 0, 0, [], [], []
        records: list[dict[str, float | bool]] = []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                times, coords, marks, targets, target_coords, dtime, mask = self.make_batch(rows)
                if self.spec.variant == "shuffled_pairing" and targets.shape[0] > 1:
                    order = _nontrivial_permutation(targets.shape[0], targets.device)
                    times = times[order]
                    coords = coords[order]
                    marks = marks[order]
                    mask = mask[order]
                result = model(times, coords, marks, mask)
                loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=self.num_marks)
                losses.append(float(loss.cpu()))
                if self.num_marks > 1:
                    correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                    total += int(targets.numel())
                sample_rmse = torch.sqrt(((result.aux["next_location"] - target_coords) ** 2).sum(dim=-1))
                rmses.extend(sample_rmse.cpu().tolist())
                maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
                aux_rows.append(result.aux)
                records.extend(self._stpp_records(result, coords, marks, targets, target_coords, mask, sample_rmse))

        joint_nll = finite_mean(losses)
        rmse_location = finite_mean(rmses)
        return {
            "eval_loss": joint_nll,
            "primary_metric": rmse_location,
            "joint_nll": joint_nll,
            "accuracy": correct / max(1, total) if total else float("nan"),
            "rmse_location": rmse_location,
            "mae_time": finite_mean(maes),
            **gate_stats(aux_rows),
            **self._aggregate_stpp_records(records),
        }

    def _stpp_records(
        self,
        result: Any,
        coords: torch.Tensor,
        marks: torch.Tensor,
        targets: torch.Tensor,
        target_coords: torch.Tensor,
        mask: torch.Tensor,
        sample_rmse: torch.Tensor,
    ) -> list[dict[str, float | bool]]:
        attention = result.aux.get("prediction_warrant_attention")
        gate = result.aux.get("prediction_warrant_gate", result.aux.get("warrant_gate"))
        if attention is None or gate is None:
            return [
                {
                    "rmse": float(sample_rmse[idx].detach().cpu()),
                    "revisit": bool(((marks[idx] == targets[idx]) & mask[idx]).any().detach().cpu()),
                    "last_dist": float(torch.sqrt(((coords[idx, -1] - target_coords[idx]) ** 2).sum()).detach().cpu()),
                }
                for idx in range(coords.shape[0])
            ]
        if attention.ndim == 3:
            attention = attention.squeeze(1)
        if gate.ndim == 3:
            gate = gate.squeeze(1)
        attention = attention * mask.to(attention.dtype)
        gate = gate * mask.to(gate.dtype)
        effective = attention * gate
        target_delta = coords - target_coords.unsqueeze(1)
        history_dist = torch.sqrt((target_delta**2).sum(dim=-1)).masked_fill(~mask, float("inf"))
        threshold = masked_quantile_midpoint(history_dist, mask)
        useful = mask & (history_dist <= threshold.unsqueeze(1))
        nonuseful = mask & ~useful

        attention_sum = attention.sum(dim=1).clamp_min(1.0e-9)
        effective_sum = effective.sum(dim=1).clamp_min(1.0e-9)
        attn_prior = (attention.unsqueeze(-1) * coords).sum(dim=1) / attention_sum.unsqueeze(1)
        warrant_prior = (effective.unsqueeze(-1) * coords).sum(dim=1) / effective_sum.unsqueeze(1)
        attn_prior_rmse = torch.sqrt(((attn_prior - target_coords) ** 2).sum(dim=-1))
        warrant_prior_rmse = torch.sqrt(((warrant_prior - target_coords) ** 2).sum(dim=-1))

        useful_attention = (attention * useful.to(attention.dtype)).sum(dim=1)
        useful_effective = (effective * useful.to(effective.dtype)).sum(dim=1)
        nonuseful_attention = (attention * nonuseful.to(attention.dtype)).sum(dim=1)
        nonuseful_effective = (effective * nonuseful.to(effective.dtype)).sum(dim=1)
        useful_retention = useful_effective / useful_attention.clamp_min(1.0e-9)
        nonuseful_retention = nonuseful_effective / nonuseful_attention.clamp_min(1.0e-9)
        useful_gate = masked_mean_tensor(gate, useful)
        nonuseful_gate = masked_mean_tensor(gate, nonuseful)
        last_idx = mask.long().sum(dim=1).sub(1).clamp_min(0)
        last_coords = coords.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, coords.shape[-1])).squeeze(1)
        last_dist = torch.sqrt(((last_coords - target_coords) ** 2).sum(dim=-1))
        revisit = ((marks == targets.unsqueeze(1)) & mask).any(dim=1)

        rows: list[dict[str, float | bool]] = []
        for idx in range(coords.shape[0]):
            rows.append(
                {
                    "rmse": float(sample_rmse[idx].detach().cpu()),
                    "revisit": bool(revisit[idx].detach().cpu()),
                    "last_dist": float(last_dist[idx].detach().cpu()),
                    "attention_prior_rmse": float(attn_prior_rmse[idx].detach().cpu()),
                    "warrant_prior_rmse": float(warrant_prior_rmse[idx].detach().cpu()),
                    "useful_gate": float(useful_gate[idx].detach().cpu()),
                    "nonuseful_gate": float(nonuseful_gate[idx].detach().cpu()),
                    "useful_retention": float(useful_retention[idx].detach().cpu()),
                    "nonuseful_retention": float(nonuseful_retention[idx].detach().cpu()),
                    "false_suppression": float((nonuseful_retention[idx] - useful_retention[idx]).detach().cpu()),
                }
            )
        return rows

    def _aggregate_stpp_records(self, records: list[dict[str, float | bool]]) -> dict[str, float]:
        if not records:
            return {}
        last_dist = np.asarray([parse_float(row.get("last_dist")) for row in records], dtype=float)
        finite = np.isfinite(last_dist)
        short_threshold = float(np.nanmedian(last_dist[finite])) if finite.any() else float("nan")

        def mean_where(key: str, predicate) -> float:
            values = [parse_float(row.get(key)) for row in records if predicate(row)]
            return finite_mean(values)

        return {
            "stpp_revisit_rate": float(np.mean([bool(row.get("revisit", False)) for row in records])),
            "stpp_rmse_revisit": mean_where("rmse", lambda row: bool(row.get("revisit", False))),
            "stpp_rmse_new_place": mean_where("rmse", lambda row: not bool(row.get("revisit", False))),
            "stpp_rmse_short_move": mean_where("rmse", lambda row: parse_float(row.get("last_dist")) <= short_threshold),
            "stpp_rmse_long_move": mean_where("rmse", lambda row: parse_float(row.get("last_dist")) > short_threshold),
            "stpp_attention_prior_rmse": finite_mean([parse_float(row.get("attention_prior_rmse")) for row in records]),
            "stpp_warrant_prior_rmse": finite_mean([parse_float(row.get("warrant_prior_rmse")) for row in records]),
            "stpp_prior_rmse_delta": finite_mean([parse_float(row.get("warrant_prior_rmse")) - parse_float(row.get("attention_prior_rmse")) for row in records]),
            "stpp_useful_gate_mean": finite_mean([parse_float(row.get("useful_gate")) for row in records]),
            "stpp_nonuseful_gate_mean": finite_mean([parse_float(row.get("nonuseful_gate")) for row in records]),
            "stpp_useful_mass_retention": finite_mean([parse_float(row.get("useful_retention")) for row in records]),
            "stpp_nonuseful_mass_retention": finite_mean([parse_float(row.get("nonuseful_retention")) for row in records]),
            "stpp_false_suppression_index": finite_mean([parse_float(row.get("false_suppression")) for row in records]),
        }


class NegativeTKGRunner(tkg.PathTKGRunner):
    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        losses, correct, total, rr, aux_rows = [], 0, 0, [], []
        records: list[dict[str, float | bool]] = []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                *inputs, target = self.make_batch(rows)
                if self.spec.variant == "shuffled_pairing" and target.shape[0] > 1:
                    order = _nontrivial_permutation(target.shape[0], target.device)
                    inputs[0] = inputs[0][order]
                    inputs[1] = inputs[1][order]
                    inputs[2] = inputs[2][order]
                result = model(*inputs)
                losses.append(float(F.cross_entropy(result.logits, target).cpu()))
                correct += int((result.logits.argmax(dim=-1) == target).sum().cpu())
                total += int(target.numel())
                target_logits = result.logits.gather(1, target.unsqueeze(1))
                ranks = (result.logits > target_logits).sum(dim=1).float() + 1.0
                reciprocal = 1.0 / ranks
                rr.extend(reciprocal.cpu().tolist())
                aux_rows.append(result.aux)
                records.extend(self._tkg_records(model, inputs, target, reciprocal))
        mrr = finite_mean(rr)
        return {
            "eval_loss": finite_mean(losses),
            "primary_metric": mrr,
            "mrr": mrr,
            "accuracy": correct / max(1, total),
            **gate_stats(aux_rows),
            **self._aggregate_tkg_records(records),
        }

    def _tkg_records(
        self,
        model: torch.nn.Module,
        inputs: list[torch.Tensor],
        target: torch.Tensor,
        reciprocal: torch.Tensor,
    ) -> list[dict[str, float | bool]]:
        if not hasattr(model, "copy_attn"):
            return []
        head, relation, timestamp, history_head, history_relation, history_tail, history_time, history_mask = inputs
        query = model.query_state(head, relation, timestamp)
        events = model.encode_events(history_head, history_relation, history_tail, history_time, timestamp)
        context, info = model.copy_attn(query, events, mask=history_mask)
        state = query + context
        attention = info["attention"]
        gate = info["warrant_gate"]
        if attention.ndim == 3:
            attention = attention.squeeze(1)
        if gate.ndim == 3:
            gate = gate.squeeze(1)
        attention_weight = attention
        warrant_weight = attention * gate
        attention_copy = model.history_copy_mass(history_tail, history_time, timestamp, history_mask, evidence_weight=attention_weight)
        warrant_copy = model.history_copy_mass(history_tail, history_time, timestamp, history_mask, evidence_weight=warrant_weight)
        true_attention = attention_copy.gather(1, target.unsqueeze(1)).squeeze(1)
        true_warrant = warrant_copy.gather(1, target.unsqueeze(1)).squeeze(1)
        false_attention = attention_copy.clone()
        false_warrant = warrant_copy.clone()
        false_attention.scatter_(1, target.unsqueeze(1), 0.0)
        false_warrant.scatter_(1, target.unsqueeze(1), 0.0)
        false_attention_top = false_attention.max(dim=1).values
        false_warrant_top = false_warrant.max(dim=1).values
        true_seen = ((history_tail == target.unsqueeze(1)) & history_mask).any(dim=1)
        copy_gate = model.copy_gate(state).squeeze(-1) if hasattr(model, "copy_gate") else torch.full_like(true_attention, float("nan"))
        rows: list[dict[str, float | bool]] = []
        for idx in range(target.shape[0]):
            true_attn = true_attention[idx]
            true_eff = true_warrant[idx]
            false_attn = false_attention_top[idx]
            false_eff = false_warrant_top[idx]
            rows.append(
                {
                    "true_seen": bool(true_seen[idx].detach().cpu()),
                    "rr": float(reciprocal[idx].detach().cpu()),
                    "copy_gate": float(copy_gate[idx].detach().cpu()),
                    "true_attention": float(true_attn.detach().cpu()),
                    "true_warrant": float(true_eff.detach().cpu()),
                    "false_attention": float(false_attn.detach().cpu()),
                    "false_warrant": float(false_eff.detach().cpu()),
                    "true_retention": float((true_eff / true_attn.clamp_min(1.0e-9)).detach().cpu()),
                    "false_retention": float((false_eff / false_attn.clamp_min(1.0e-9)).detach().cpu()),
                    "saturation_attention": float((false_attn / true_attn.clamp_min(1.0e-9)).detach().cpu()),
                    "saturation_warrant": float((false_eff / true_eff.clamp_min(1.0e-9)).detach().cpu()),
                }
            )
        return rows

    def _aggregate_tkg_records(self, records: list[dict[str, float | bool]]) -> dict[str, float]:
        if not records:
            return {}

        def mean_where(key: str, predicate) -> float:
            return finite_mean([parse_float(row.get(key)) for row in records if predicate(row)])

        seen_rate = float(np.mean([bool(row.get("true_seen", False)) for row in records]))
        return {
            "tkg_true_tail_seen_rate": seen_rate,
            "tkg_mrr_seen": mean_where("rr", lambda row: bool(row.get("true_seen", False))),
            "tkg_mrr_unseen": mean_where("rr", lambda row: not bool(row.get("true_seen", False))),
            "tkg_copy_gate_mean_seen": mean_where("copy_gate", lambda row: bool(row.get("true_seen", False))),
            "tkg_copy_gate_mean_unseen": mean_where("copy_gate", lambda row: not bool(row.get("true_seen", False))),
            "tkg_true_copy_mass_attention": finite_mean([parse_float(row.get("true_attention")) for row in records]),
            "tkg_true_copy_mass_warrant": finite_mean([parse_float(row.get("true_warrant")) for row in records]),
            "tkg_false_top_copy_mass_attention": finite_mean([parse_float(row.get("false_attention")) for row in records]),
            "tkg_false_top_copy_mass_warrant": finite_mean([parse_float(row.get("false_warrant")) for row in records]),
            "tkg_true_copy_retention": mean_where("true_retention", lambda row: bool(row.get("true_seen", False))),
            "tkg_false_copy_retention": mean_where("false_retention", lambda row: bool(row.get("true_seen", False))),
            "tkg_copy_saturation_attention": mean_where("saturation_attention", lambda row: bool(row.get("true_seen", False))),
            "tkg_copy_saturation_warrant": mean_where("saturation_warrant", lambda row: bool(row.get("true_seen", False))),
        }


RUNNER_BY_DOMAIN = {
    "stpp": NegativeSTPPRunner,
    "tkg": NegativeTKGRunner,
}

CONFIGURE_BY_DOMAIN = {
    "stpp": stpp.configure_model,
    "tkg": tkg.configure_model,
}


def run_spec(config: dict[str, Any], spec, device: torch.device, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    set_seed(spec.seed)
    runner = RUNNER_BY_DOMAIN[spec.domain](spec, config, device)
    frame = runner.prepare()
    model = build_benchmark_model(spec, frame, config, device)
    if spec.domain == "stpp":
        setattr(model, "emit_prediction_warrant_tensors", True)
    configure_variant(model, spec)
    CONFIGURE_BY_DOMAIN[spec.domain](model, spec.variant)
    model.to(device)

    training = config["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epoch_rows: list[dict[str, Any]] = []
    iterator = tqdm(
        range(1, int(training["epochs"]) + 1),
        desc=f"negative/{spec.domain}/{spec.model} {spec.variant} seed={spec.seed}",
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
    for key, value in last_eval.items():
        if key not in {"primary_metric", "eval_loss"}:
            row[key] = value
    row = rounded(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics_by_epoch.csv", epoch_rows)
    if save_checkpoint_enabled(config):
        checkpoint_path = out_dir / checkpoint_name(config)
        row["checkpoint_path"] = str(checkpoint_path.relative_to(REPO_ROOT))
        write_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=opt,
            config=config,
            spec=spec,
            final_metrics=row,
            epoch_rows=epoch_rows,
        )
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"spec": spec.__dict__, "variant": config["variants"][spec.variant]}, handle, indent=2, sort_keys=True, default=str)
    return row


def aggregate(config_path: Path, output_root: Path) -> None:
    from experiments.negative_row_diagnostics.agg import run_aggregation

    config = load_config(config_path)
    configured = config.get("inputs", {}).get("main_base_vs_warrant")
    main_path = REPO_ROOT / str(configured) if configured else None
    run_aggregation(
        results_path=output_root / "results.csv",
        output_dir=output_root / "analysis",
        config=config,
        main_path=main_path if main_path and main_path.exists() else None,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    domains = selected_domains(config, args.domain or config.get("selection", {}).get("domains"))
    variants = selected_variants(config, args.variant or config.get("selection", {}).get("variants"))
    seeds = selected_seeds(config, args.seed)
    specs = iter_specs(config, domains=domains, variants=variants, seeds=seeds)
    device = resolve_device(config)
    output_root = REPO_ROOT / str(config.get("experiment", {}).get("output_root", "experiments/negative_row_diagnostics/outputs"))
    results_path = output_root / "results.csv"

    print(f"domains={domains} variants={variants} seeds={seeds} runs={len(specs)} device={device}")
    if args.dry_run:
        for spec in specs:
            print(f"dry_run: {spec.domain}/{spec.dataset}/{spec.model}/{spec.variant}/seed_{spec.seed}")
        return

    if not args.aggregate_only:
        for spec in tqdm(specs, desc="negative-row runs", unit="run", disable=not bool(config["training"].get("progress", True))):
            out_dir = output_dir(output_root, spec)
            checkpoint_path = out_dir / checkpoint_name(config)
            checkpoint_required = save_checkpoint_enabled(config)
            checkpoint_ready = checkpoint_path.exists() if checkpoint_required else True
            if should_skip(out_dir, args.force) and checkpoint_ready and bool(config.get("experiment", {}).get("skip_completed", True)):
                tqdm.write(f"skip completed: {spec.domain}/{spec.model}/{spec.variant} seed={spec.seed}")
                continue
            try:
                row = run_spec(config, spec, device, out_dir)
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
            append_result_dynamic(results_path, row)

    if not args.no_aggregate:
        aggregate(args.config, output_root)


if __name__ == "__main__":
    main()
